"""Canonical plans, events, assertions, and reports for adapter qualification."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from membench.contract import Mechanism, WritePolicy
from membench.memory.models import (
    AdapterProvenance,
    CanonicalModel,
    EmptyStateResult,
    FlushResult,
    JsonObject,
    MemoryMessage,
    MemoryScope,
    MemorySnapshot,
    RetrievalResult,
)
from membench.memory.policy import PolicyDispatchResult


def _utc_now() -> datetime:
    return datetime.now(UTC)


class QualificationStage(StrEnum):
    """Ordered lifecycle stages emitted by the qualification runner."""

    PREFLIGHT = "preflight"
    INITIALIZE = "initialize"
    VERIFY_EMPTY = "verify_empty"
    INSPECT_BEFORE = "inspect_before"
    WRITE = "write"
    FLUSH = "flush"
    QUIESCENCE = "quiescence"
    RETRIEVE = "retrieve"
    VERIFY_RETRIEVAL_UNCHANGED = "verify_retrieval_unchanged"
    CONTEXT = "context"
    RESPOND = "respond"
    VERIFY_RESPONSE_UNCHANGED = "verify_response_unchanged"
    INSPECT_AFTER = "inspect_after"
    ISOLATION = "isolation"
    PROVENANCE = "provenance"
    RESET = "reset"
    CLOSE = "close"
    COMPLETE = "complete"


class QualificationEventStatus(StrEnum):
    """State of a single emitted lifecycle event."""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class QualificationFailureClass(StrEnum):
    """Stable operational classification for failed qualifications."""

    CONFIGURATION_INVALID = "configuration_invalid"
    SERVICE_UNAVAILABLE = "service_unavailable"
    NAMESPACE_NOT_EMPTY = "namespace_not_empty"
    WRITE_TRANSPORT_FAILED = "write_transport_failed"
    WRITE_VISIBILITY_TIMEOUT = "write_visibility_timeout"
    STATE_STABILIZATION_TIMEOUT = "state_stabilization_timeout"
    RETRIEVAL_TRANSPORT_FAILED = "retrieval_transport_failed"
    RESPONSE_TRANSPORT_FAILED = "response_transport_failed"
    STATE_INSPECTION_FAILED = "state_inspection_failed"
    RESET_VERIFICATION_FAILED = "reset_verification_failed"
    POLICY_FAILED = "policy_failed"
    ASSERTION_FAILED = "assertion_failed"
    INSTRUMENTATION_INCOMPLETE = "instrumentation_incomplete"


class QuiescenceConfiguration(CanonicalModel):
    """Polling contract used to establish a stable canonical memory baseline."""

    timeout_seconds: float = Field(default=30.0, gt=0)
    poll_interval_seconds: float = Field(default=0.25, gt=0)
    stable_window_seconds: float = Field(default=2.0, ge=0)
    minimum_stable_samples: int = Field(default=3, ge=1)
    enumeration_policy: Literal["exact", "bounded"] = "exact"

    @model_validator(mode="after")
    def timing_must_fit_timeout(self) -> Self:
        if self.poll_interval_seconds > self.timeout_seconds:
            raise ValueError("quiescence poll interval cannot exceed timeout")
        if self.stable_window_seconds > self.timeout_seconds:
            raise ValueError("quiescence stable window cannot exceed timeout")
        return self


class QuiescenceObservation(CanonicalModel):
    """One normalized state sample taken while waiting for stabilization."""

    sample_index: int = Field(ge=1)
    elapsed_seconds: float = Field(ge=0)
    changed_since_previous: bool
    complete: bool
    record_count: int = Field(ge=0)
    record_ids: tuple[str, ...] = Field(default_factory=tuple)
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reported_total_count: int | None = Field(default=None, ge=0)
    unobserved_count: int | None = Field(default=None, ge=0)
    coverage_ratio: float | None = Field(default=None, ge=0, le=1)


class QuiescenceEvidence(CanonicalModel):
    """Auditable proof that a scope stabilized, or timed out while changing."""

    phase: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    scope: MemoryScope
    stabilized: bool
    enumeration_policy: Literal["exact", "bounded"] = "exact"
    timeout_seconds: float = Field(gt=0)
    poll_interval_seconds: float = Field(gt=0)
    stable_window_seconds: float = Field(ge=0)
    minimum_stable_samples: int = Field(ge=1)
    stable_for_seconds: float = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    observations: tuple[QuiescenceObservation, ...] = Field(min_length=1)
    final_snapshot: MemorySnapshot

    @model_validator(mode="after")
    def final_observation_must_match_snapshot(self) -> Self:
        if self.final_snapshot.scope != self.scope:
            raise ValueError("quiescence final snapshot has the wrong scope")
        if self.observations[-1].record_count != self.final_snapshot.record_count:
            raise ValueError(
                "quiescence final observation count does not match snapshot"
            )
        expected_record_ids = tuple(
            sorted(record.canonical_id for record in self.final_snapshot.records)
        )
        if self.observations[-1].record_ids != expected_record_ids:
            raise ValueError(
                "quiescence final observation IDs do not match snapshot"
            )
        expected_indexes = tuple(range(1, len(self.observations) + 1))
        observed_indexes = tuple(item.sample_index for item in self.observations)
        if observed_indexes != expected_indexes:
            raise ValueError("quiescence sample indexes must be consecutive")
        elapsed = tuple(item.elapsed_seconds for item in self.observations)
        if elapsed != tuple(sorted(elapsed)):
            raise ValueError("quiescence observations must be chronological")
        return self


class QualificationPlan(CanonicalModel):
    """Resolved, mechanism-neutral instructions for one qualification run."""

    qualification_id: str = Field(pattern=r"^[a-z][a-z0-9_]*_v\d+$")
    mechanism: Mechanism
    policy: WritePolicy
    scope: MemoryScope
    secondary_scope: MemoryScope | None = None
    top_k: int = Field(gt=0)
    threshold: float | None = None
    context_token_budget: int = Field(default=2048, gt=0)
    quiescence: QuiescenceConfiguration = Field(
        default_factory=QuiescenceConfiguration
    )
    responder_required: bool = False
    responder_provider: Literal["ollama", "openai"] | None = None
    responder_model: str | None = Field(default=None, min_length=1)
    responder_model_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    cleanup: bool = True
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    native_options_by_canary: dict[str, JsonObject] = Field(default_factory=dict)

    @model_validator(mode="after")
    def scope_and_responder_requirements_must_be_valid(self) -> Self:
        responder_identity = (self.responder_provider, self.responder_model)
        if self.responder_required and any(
            item is None for item in responder_identity
        ):
            raise ValueError(
                "responder-required plans need provider and model identity"
            )
        if not self.responder_required and any(
            item is not None
            for item in (*responder_identity, self.responder_model_digest)
        ):
            raise ValueError(
                "responder identity is only valid when responder_required is true"
            )
        if self.secondary_scope is None:
            return self
        if self.secondary_scope == self.scope:
            raise ValueError("secondary qualification scope must differ from primary")
        if self.secondary_scope.experiment_id != self.scope.experiment_id:
            raise ValueError(
                "primary and secondary scopes must belong to the same experiment"
            )
        if self.secondary_scope.run_id == self.scope.run_id:
            raise ValueError("secondary qualification scope must use another run_id")
        if self.secondary_scope.namespace == self.scope.namespace:
            raise ValueError(
                "secondary qualification scope must use another namespace"
            )
        return self

    @field_validator("native_options_by_canary")
    @classmethod
    def canary_option_keys_must_not_be_blank(
        cls,
        value: dict[str, JsonObject],
    ) -> dict[str, JsonObject]:
        if any(not key.strip() for key in value):
            raise ValueError("native-option canary IDs must not be blank")
        return value


class QualificationAssertion(CanonicalModel):
    """One explicit behavioral claim and its JSON-safe evidence."""

    assertion_id: str = Field(pattern=r"^[a-z][a-z0-9_.:-]*$")
    stage: QualificationStage
    passed: bool
    required: bool = True
    expected: JsonValue
    observed: JsonValue
    canary_id: str | None = None
    message: str | None = None

    @field_validator("canary_id", "message")
    @classmethod
    def optional_strings_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional assertion strings must not be blank")
        return value


class QualificationEvent(CanonicalModel):
    """Append-only event suitable for the future JSONL event sink."""

    schema_version: Literal[1] = 1
    sequence: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    experiment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    mechanism: Mechanism
    policy: WritePolicy
    qualification_id: str = Field(min_length=1)
    stage: QualificationStage
    status: QualificationEventStatus
    operation_id: str | None = None
    canary_id: str | None = None
    latency_seconds: float = Field(default=0.0, ge=0)
    payload: JsonObject = Field(default_factory=dict)
    failure_class: QualificationFailureClass | None = None
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("qualification event timestamp must include a timezone")
        return value

    @field_validator(
        "operation_id",
        "canary_id",
        "error_type",
        "error_message",
    )
    @classmethod
    def optional_event_strings_must_not_be_blank(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional event strings must not be blank")
        return value

    @model_validator(mode="after")
    def failure_fields_must_match_status(self) -> Self:
        failure_values = (
            self.failure_class,
            self.error_type,
            self.error_message,
        )
        if self.status == QualificationEventStatus.FAILED:
            if any(item is None for item in failure_values):
                raise ValueError("failed events require complete failure information")
        elif any(item is not None for item in failure_values):
            raise ValueError("successful events cannot contain failure information")
        return self


class CanaryDispatchRecord(CanonicalModel):
    """Join a canary identity to its canonical policy result."""

    canary_id: str = Field(min_length=1)
    scope_role: Literal["primary", "secondary"] = "primary"
    result: PolicyDispatchResult


class CanaryRetrievalRecord(CanonicalModel):
    """Join a canary query to its canonical retrieval result."""

    canary_id: str = Field(min_length=1)
    scope_role: Literal["primary", "secondary"] = "primary"
    phase: str = Field(default="own", pattern=r"^[a-z][a-z0-9_]*$")
    expectation_index: int = Field(ge=0)
    required_fragments: tuple[str, ...] = Field(min_length=1)
    result: RetrievalResult


class SnapshotRecordChange(CanonicalModel):
    """Changed canonical fields for one record present in both snapshots."""

    canonical_id: str = Field(min_length=1)
    changed_fields: tuple[str, ...] = Field(min_length=1)

    @field_validator("changed_fields")
    @classmethod
    def changed_fields_must_be_sorted_and_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not field.strip() for field in value):
            raise ValueError("changed field paths must not be blank")
        if value != tuple(sorted(set(value))):
            raise ValueError("changed field paths must be sorted and unique")
        return value


class ImportanceChange(CanonicalModel):
    """Before/after importance values for one canonically identified record."""

    canonical_id: str = Field(min_length=1)
    before: float | None
    after: float | None
    delta: float | None

    @model_validator(mode="after")
    def delta_must_match_values(self) -> Self:
        if self.before == self.after:
            raise ValueError("importance change requires different values")
        expected = (
            None
            if self.before is None or self.after is None
            else self.after - self.before
        )
        if expected is None:
            if self.delta is not None:
                raise ValueError(
                    "importance delta must be null when either value is absent"
                )
        elif self.delta is None or not math.isclose(
            self.delta,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("importance delta must equal after minus before")
        return self


class RetrievalStateDelta(CanonicalModel):
    """Structured content and operational changes caused during retrieval."""

    scope: MemoryScope
    before_complete: bool
    after_complete: bool
    comparison_level: Literal["exact", "bounded"]
    before_reported_total_count: int | None = Field(default=None, ge=0)
    after_reported_total_count: int | None = Field(default=None, ge=0)
    content_unchanged: bool
    full_state_unchanged: bool
    observed_content_unchanged: bool
    observed_full_state_unchanged: bool
    created_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    deleted_record_ids: tuple[str, ...] = Field(default_factory=tuple)
    content_changes: tuple[SnapshotRecordChange, ...] = Field(
        default_factory=tuple
    )
    operational_changes: tuple[SnapshotRecordChange, ...] = Field(
        default_factory=tuple
    )
    importance_changes: tuple[ImportanceChange, ...] = Field(
        default_factory=tuple
    )

    @model_validator(mode="after")
    def flags_and_record_ids_must_match_changes(self) -> Self:
        created = self.created_record_ids
        deleted = self.deleted_record_ids
        if created != tuple(sorted(set(created))):
            raise ValueError("created record IDs must be sorted and unique")
        if deleted != tuple(sorted(set(deleted))):
            raise ValueError("deleted record IDs must be sorted and unique")
        if set(created) & set(deleted):
            raise ValueError("a record cannot be both created and deleted")

        content_ids = tuple(item.canonical_id for item in self.content_changes)
        operational_ids = tuple(
            item.canonical_id for item in self.operational_changes
        )
        importance_ids = tuple(
            item.canonical_id for item in self.importance_changes
        )
        for label, values in (
            ("content change", content_ids),
            ("operational change", operational_ids),
            ("importance change", importance_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{label} record IDs must be sorted and unique")
        created_or_deleted = set(created) | set(deleted)
        if created_or_deleted.intersection(
            (*content_ids, *operational_ids, *importance_ids)
        ):
            raise ValueError(
                "created or deleted records cannot also contain field changes"
            )

        content_changed = bool(created or deleted or content_ids)
        exact_comparison = self.before_complete and self.after_complete
        expected_comparison_level = (
            "exact" if exact_comparison else "bounded"
        )
        if self.comparison_level != expected_comparison_level:
            raise ValueError(
                "comparison_level does not match snapshot completeness"
            )

        reported_total_unchanged = (
            self.before_reported_total_count is not None
            and self.after_reported_total_count is not None
            and self.before_reported_total_count
            == self.after_reported_total_count
        )
        expected_content_unchanged = (
            exact_comparison and not content_changed
        )
        full_changed = content_changed or bool(operational_ids)
        expected_full_unchanged = (
            exact_comparison and not full_changed
        )
        expected_observed_content_unchanged = (
            not content_changed
            and (exact_comparison or reported_total_unchanged)
        )
        expected_observed_full_state_unchanged = (
            expected_observed_content_unchanged and not full_changed
        )
        if self.content_unchanged != expected_content_unchanged:
            raise ValueError("content_unchanged does not match recorded changes")
        if self.full_state_unchanged != expected_full_unchanged:
            raise ValueError("full_state_unchanged does not match recorded changes")
        if (
            self.observed_content_unchanged
            != expected_observed_content_unchanged
        ):
            raise ValueError(
                "observed_content_unchanged does not match recorded changes"
            )
        if (
            self.observed_full_state_unchanged
            != expected_observed_full_state_unchanged
        ):
            raise ValueError(
                "observed_full_state_unchanged does not match recorded changes"
            )
        return self


class RetrievalReadOnlyEvidence(CanonicalModel):
    """Before/after state and visibility barrier for one retrieval phase."""

    phase: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    scope_role: Literal["primary", "secondary"]
    scope: MemoryScope
    operation_ids: tuple[str, ...] = Field(min_length=1)
    snapshot_before: MemorySnapshot
    visibility_barrier: FlushResult
    snapshot_after: MemorySnapshot
    state_delta: RetrievalStateDelta

    @model_validator(mode="after")
    def evidence_must_reference_one_scope(self) -> Self:
        if len(self.operation_ids) != len(set(self.operation_ids)):
            raise ValueError("retrieval operation IDs must be unique")
        if self.snapshot_before.scope != self.scope:
            raise ValueError("retrieval snapshot_before has the wrong scope")
        if self.snapshot_after.scope != self.scope:
            raise ValueError("retrieval snapshot_after has the wrong scope")
        if self.state_delta.scope != self.scope:
            raise ValueError("retrieval state_delta has the wrong scope")
        return self


class ResponderStageEvidence(CanonicalModel):
    """Exact context, prompt, response, and state proof for one probe."""

    operation_id: str = Field(min_length=1)
    canary_id: str = Field(min_length=1)
    expectation_index: int = Field(ge=0)
    retrieval_operation_id: str = Field(min_length=1)
    context_record_ids: tuple[str, ...]
    context: str
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_token_count: int = Field(ge=0)
    context_token_counter: str = Field(min_length=1)
    context_truncated: bool
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_messages: tuple[MemoryMessage, ...] = Field(min_length=1)
    predicted_prompt_tokens: int = Field(ge=0)
    prompt_token_counter: str = Field(min_length=1)
    response_content: str = Field(min_length=1)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_provider: str = Field(min_length=1)
    response_model: str = Field(min_length=1)
    response_model_digest: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    prompt_token_delta: int | None = None
    completion_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    latency_seconds: float = Field(ge=0)
    raw_response: JsonObject
    snapshot_before: MemorySnapshot
    visibility_barrier: FlushResult
    snapshot_after: MemorySnapshot

    @field_validator("context_record_ids")
    @classmethod
    def context_record_ids_must_be_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("context record IDs must be unique")
        return value

    @model_validator(mode="after")
    def hashes_and_scopes_must_match(self) -> Self:
        if hashlib.sha256(self.context.encode("utf-8")).hexdigest() != (
            self.context_sha256
        ):
            raise ValueError("context_sha256 does not match exact context")
        if hashlib.sha256(self.response_content.encode("utf-8")).hexdigest() != (
            self.response_sha256
        ):
            raise ValueError("response_sha256 does not match exact response")
        prompt_payload = "\n\n".join(
            f"{message.role.value}:{message.content}"
            for message in self.prompt_messages
        )
        if hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest() != (
            self.prompt_sha256
        ):
            raise ValueError("prompt_sha256 does not match exact prompt messages")
        if self.snapshot_before.scope != self.snapshot_after.scope:
            raise ValueError("responder snapshots must use the same scope")
        expected_delta = (
            None
            if self.prompt_tokens is None
            else self.prompt_tokens - self.predicted_prompt_tokens
        )
        if self.prompt_token_delta != expected_delta:
            raise ValueError(
                "prompt_token_delta must equal observed minus predicted tokens"
            )
        return self


class QualificationSummary(CanonicalModel):
    """Compact final status used by CLI exit handling and later reporting."""

    experiment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    mechanism: Mechanism
    policy: WritePolicy
    qualification_id: str = Field(min_length=1)
    passed: bool
    completed_stages: tuple[QualificationStage, ...] = Field(default_factory=tuple)
    assertions: tuple[QualificationAssertion, ...] = Field(default_factory=tuple)
    failure_class: QualificationFailureClass | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    started_at: datetime
    finished_at: datetime

    @field_validator("started_at", "finished_at")
    @classmethod
    def summary_timestamps_must_have_timezones(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("qualification summary timestamps require timezones")
        return value

    @model_validator(mode="after")
    def status_must_match_evidence(self) -> Self:
        if len(self.completed_stages) != len(set(self.completed_stages)):
            raise ValueError("completed qualification stages must be unique")
        if self.finished_at < self.started_at:
            raise ValueError("qualification cannot finish before it starts")
        failure_values = (
            self.failure_class,
            self.failure_type,
            self.failure_message,
        )
        has_failure = any(item is not None for item in failure_values)
        if has_failure and any(item is None for item in failure_values):
            raise ValueError("qualification failure information must be complete")
        required_assertions_passed = all(
            item.passed or not item.required for item in self.assertions
        )
        if self.passed != (not has_failure and required_assertions_passed):
            raise ValueError("passed must agree with failures and assertions")
        return self


class QualificationReport(CanonicalModel):
    """Complete in-memory result ready for artifact persistence."""

    plan: QualificationPlan
    summary: QualificationSummary
    dispatches: tuple[CanaryDispatchRecord, ...] = Field(default_factory=tuple)
    flush_result: FlushResult | None = None
    retrievals: tuple[CanaryRetrievalRecord, ...] = Field(default_factory=tuple)
    retrieval_read_only: tuple[RetrievalReadOnlyEvidence, ...] = Field(
        default_factory=tuple
    )
    quiescence: tuple[QuiescenceEvidence, ...] = Field(default_factory=tuple)
    responder_stages: tuple[ResponderStageEvidence, ...] = Field(
        default_factory=tuple
    )
    snapshot_before: MemorySnapshot | None = None
    snapshot_after: MemorySnapshot | None = None
    secondary_snapshot_before: MemorySnapshot | None = None
    secondary_snapshot_after: MemorySnapshot | None = None
    secondary_flush_result: FlushResult | None = None
    secondary_provenance: AdapterProvenance | None = None
    provenance: AdapterProvenance | None = None
    reset_result: EmptyStateResult | None = None
    secondary_reset_result: EmptyStateResult | None = None

    @model_validator(mode="after")
    def secondary_evidence_must_match_plan(self) -> Self:
        secondary_evidence = (
            self.secondary_snapshot_before,
            self.secondary_snapshot_after,
            self.secondary_flush_result,
            self.secondary_provenance,
            self.secondary_reset_result,
        )
        if self.plan.secondary_scope is None and any(
            item is not None for item in secondary_evidence
        ):
            raise ValueError(
                "secondary evidence requires a secondary qualification scope"
            )
        if self.summary.passed and self.plan.secondary_scope is not None and any(
            item is None for item in secondary_evidence
        ):
            raise ValueError(
                "a passed secondary-scope qualification requires complete evidence"
            )
        if self.summary.passed and self.retrievals and not self.retrieval_read_only:
            raise ValueError(
                "a passed qualification with retrievals requires read-only evidence"
            )
        if self.summary.passed and self.plan.responder_required:
            if not self.responder_stages:
                raise ValueError(
                    "a passed responder qualification requires responder evidence"
                )
            if not self.quiescence or any(
                not item.stabilized for item in self.quiescence
            ):
                raise ValueError(
                    "a passed responder qualification requires stable quiescence "
                    "evidence"
                )
        phases = tuple(item.phase for item in self.quiescence)
        if len(phases) != len(set(phases)):
            raise ValueError("quiescence evidence phases must be unique")
        return self
