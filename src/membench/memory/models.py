"""Canonical inputs and outputs shared by every memory adapter.

These models form the stable boundary between an agent/campaign runner and a
vendor-specific memory mechanism. Adapters translate native requests and
responses at this boundary; callers never need to understand vendor objects.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

JsonObject = dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CanonicalModel(BaseModel):
    """Strict and immutable base for adapter-boundary data."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class MemoryType(StrEnum):
    """Cross-mechanism memory families understood by the experiment harness."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    UNSPECIFIED = "unspecified"


class MemoryState(StrEnum):
    """Lifecycle state of a normalized memory record."""

    ACTIVE = "active"
    DELETED = "deleted"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class MessageRole(StrEnum):
    """Conversation roles accepted by the common write contract."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class WriteStatus(StrEnum):
    """Aggregate result of one adapter write request."""

    QUEUED = "queued"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    NO_OP = "no_op"


class WriteAction(StrEnum):
    """Effect a mechanism reported for one candidate memory."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    SKIPPED = "skipped"


class ScoreKind(StrEnum):
    """Meaning of an unmodified score returned by a mechanism."""

    SIMILARITY = "similarity"
    DISTANCE = "distance"
    RELEVANCE = "relevance"
    RERANK = "rerank"
    UNKNOWN = "unknown"


class MemoryScope(CanonicalModel):
    """Immutable experiment, run, user, and backend namespace boundary."""

    experiment_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    namespace: tuple[str, ...] = Field(min_length=1)

    @field_validator("namespace")
    @classmethod
    def namespace_parts_must_not_be_blank(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not part.strip() for part in value):
            raise ValueError("namespace components must not be blank")
        return value

    @property
    def key(self) -> str:
        """Return a readable stable scope identifier for logs and errors."""

        return "/".join(self.namespace)


class MemoryMessage(CanonicalModel):
    """One normalized message supplied to a native conversation writer."""

    role: MessageRole
    content: str = Field(min_length=1)
    metadata: JsonObject = Field(default_factory=dict)


class DirectWriteRequest(CanonicalModel):
    """Canonical input for storing one already-approved memory candidate."""

    operation_id: str = Field(min_length=1)
    scope: MemoryScope
    text: str = Field(min_length=1)
    memory_type: MemoryType = MemoryType.UNSPECIFIED
    source_event_id: str = Field(min_length=1)
    metadata: JsonObject = Field(default_factory=dict)


class ConversationWriteRequest(CanonicalModel):
    """Canonical input for mechanism-native extraction from a conversation."""

    operation_id: str = Field(min_length=1)
    scope: MemoryScope
    messages: tuple[MemoryMessage, ...] = Field(min_length=1)
    source_event_id: str = Field(min_length=1)
    native_options: JsonObject = Field(default_factory=dict)


class WriteOutcome(CanonicalModel):
    """Normalized effect of a write on one memory candidate or record."""

    action: WriteAction
    canonical_id: str | None = None
    vendor_id: str | None = None
    text: str | None = None

    @field_validator("canonical_id", "vendor_id", "text")
    @classmethod
    def optional_strings_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional strings must be null or non-blank")
        return value


class WriteResult(CanonicalModel):
    """Canonical output for either direct or conversation-based writing."""

    operation_id: str = Field(min_length=1)
    status: WriteStatus
    outcomes: tuple[WriteOutcome, ...] = Field(default_factory=tuple)
    latency_seconds: float = Field(default=0.0, ge=0)
    raw_vendor: JsonObject | None = None

    @model_validator(mode="after")
    def status_must_match_outcomes(self) -> Self:
        if self.status == WriteStatus.QUEUED and self.outcomes:
            raise ValueError("queued write results cannot contain outcomes")
        if self.status == WriteStatus.NO_OP and self.outcomes:
            if any(item.action != WriteAction.SKIPPED for item in self.outcomes):
                raise ValueError(
                    "no_op write results may contain only skipped outcomes"
                )
        return self


class MemoryRecord(CanonicalModel):
    """Backend-neutral representation of one stored memory."""

    canonical_id: str = Field(min_length=1)
    vendor_id: str | None = None
    scope: MemoryScope
    text: str = Field(min_length=1)
    memory_type: MemoryType = MemoryType.UNSPECIFIED
    state: MemoryState = MemoryState.ACTIVE
    source_event_ids: tuple[str, ...] = Field(default_factory=tuple)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    attributes: JsonObject = Field(default_factory=dict)
    raw_vendor: JsonObject | None = None

    @field_validator("vendor_id")
    @classmethod
    def vendor_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("vendor_id must be null or non-blank")
        return value

    @field_validator("source_event_ids")
    @classmethod
    def source_event_ids_must_be_unique(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("source_event_ids must not contain blank values")
        if len(value) != len(set(value)):
            raise ValueError("source_event_ids must be unique")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def timestamps_must_have_timezones(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("memory timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def update_cannot_precede_creation(self) -> Self:
        if (
            self.created_at is not None
            and self.updated_at is not None
            and self.updated_at < self.created_at
        ):
            raise ValueError("updated_at cannot precede created_at")
        return self


class RetrievalRequest(CanonicalModel):
    """Canonical query, ranking limit, filters, and optional threshold."""

    operation_id: str = Field(min_length=1)
    scope: MemoryScope
    query: str = Field(min_length=1)
    top_k: int = Field(gt=0)
    threshold: float | None = None
    memory_types: tuple[MemoryType, ...] = Field(default_factory=tuple)
    filters: JsonObject = Field(default_factory=dict)
    native_options: JsonObject = Field(default_factory=dict)

    @field_validator("memory_types")
    @classmethod
    def memory_types_must_be_unique(
        cls, value: tuple[MemoryType, ...]
    ) -> tuple[MemoryType, ...]:
        if len(value) != len(set(value)):
            raise ValueError("memory_types must be unique")
        return value

    @property
    def query_sha256(self) -> str:
        """Hash the exact query without copying private query text into results."""

        return hashlib.sha256(self.query.encode("utf-8")).hexdigest()


class RetrievalHit(CanonicalModel):
    """One ranked record with its unmodified vendor score semantics."""

    record: MemoryRecord
    rank: int = Field(ge=1)
    raw_score: float | None = None
    score_kind: ScoreKind = ScoreKind.UNKNOWN
    passed_threshold: bool = True


class RetrievalResult(CanonicalModel):
    """Canonical ranked output returned for a retrieval request."""

    operation_id: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_top_k: int = Field(gt=0)
    threshold: float | None = None
    hits: tuple[RetrievalHit, ...] = Field(default_factory=tuple)
    latency_seconds: float = Field(ge=0)
    raw_vendor: JsonObject | None = None

    @model_validator(mode="after")
    def hits_must_be_ranked_and_bounded(self) -> Self:
        if len(self.hits) > self.request_top_k:
            raise ValueError("retrieval hits cannot exceed request_top_k")
        ranks = [hit.rank for hit in self.hits]
        if ranks != list(range(1, len(self.hits) + 1)):
            raise ValueError("retrieval hit ranks must be consecutive and one-based")
        return self


class FlushResult(CanonicalModel):
    """Visibility-barrier result for writes acknowledged by the mechanism."""

    completed: bool
    visible_records: tuple[MemoryRecord, ...] = Field(default_factory=tuple)
    pending_ids: tuple[str, ...] = Field(default_factory=tuple)
    latency_seconds: float = Field(default=0.0, ge=0)
    raw_vendor: JsonObject | None = None

    @model_validator(mode="after")
    def completed_flush_cannot_have_pending_ids(self) -> Self:
        if self.completed and self.pending_ids:
            raise ValueError("a completed flush cannot have pending IDs")
        return self


class MemorySnapshot(CanonicalModel):
    """Evaluator-only enumeration of the records visible in one scope."""

    scope: MemoryScope
    records: tuple[MemoryRecord, ...] = Field(default_factory=tuple)
    captured_at: datetime = Field(default_factory=_utc_now)
    complete: bool = True
    raw_vendor: JsonObject | None = None
    reported_total_count: int | None = Field(default=None, ge=0)
    inspection_limit: int | None = Field(default=None, gt=0)

    @property
    def unobserved_count(self) -> int | None:
        if self.reported_total_count is None:
            return None
        return max(0, self.reported_total_count - self.record_count)

    @property
    def coverage_ratio(self) -> float | None:
        if self.reported_total_count is None:
            return None
        if self.reported_total_count == 0:
            return 1.0
        return self.record_count / self.reported_total_count

    @field_validator("captured_at")
    @classmethod
    def captured_at_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("captured_at must include a timezone")
        return value

    @property
    def record_count(self) -> int:
        return len(self.records)


class EmptyStateResult(CanonicalModel):
    """Evidence that inspection, retrieval, and pending writes found no state."""

    scope: MemoryScope
    empty: bool
    inspected_count: int = Field(ge=0)
    retrieved_count: int = Field(ge=0)
    pending_count: int = Field(default=0, ge=0)
    checked_at: datetime = Field(default_factory=_utc_now)
    raw_vendor: JsonObject | None = None

    @field_validator("checked_at")
    @classmethod
    def checked_at_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("checked_at must include a timezone")
        return value

    @model_validator(mode="after")
    def empty_flag_must_match_evidence(self) -> Self:
        observed_empty = not (
            self.inspected_count or self.retrieved_count or self.pending_count
        )
        if self.empty != observed_empty:
            raise ValueError("empty must agree with all observed state counts")
        return self


class AdapterProvenance(CanonicalModel):
    """Non-secret versions and resolved configuration for reproducibility."""

    mechanism: str = Field(min_length=1)
    mechanism_version: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    storage_backend: str = Field(min_length=1)
    model_identifiers: dict[str, str] = Field(default_factory=dict)
    package_versions: dict[str, str] = Field(default_factory=dict)
    namespace: tuple[str, ...] = Field(min_length=1)
    native_configuration: JsonObject = Field(default_factory=dict)
    captured_at: datetime = Field(default_factory=_utc_now)

    @field_validator("namespace")
    @classmethod
    def provenance_namespace_must_not_be_blank(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not part.strip() for part in value):
            raise ValueError("namespace components must not be blank")
        return value

    @field_validator("captured_at")
    @classmethod
    def provenance_timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("captured_at must include a timezone")
        return value
