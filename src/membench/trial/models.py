"""Canonical plans and evidence for replaying one isolated trial twin.

These models describe execution, not attack success. Private evaluator labels
are deliberately absent. Reuse the projection, campaign, agent, and adapter
contracts instead of introducing a second representation of their payloads.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from membench.agent.models import AgentRunResult, AgentRunStatus
from membench.campaign.artifacts import canonical_json_bytes
from membench.campaign.models import (
    CampaignRunSpec,
    ChildFailureClass,
    ChildRunResult,
    ChildRunStatus,
)
from membench.contract import WritePolicy
from membench.curation.projections import (
    AttackReviewStatus,
    EventStage,
    ProjectionEvent,
    PublicProbe,
    TrialProjection,
)
from membench.ingress.agent_selected import (
    AgentSelectedIngestionAttempt,
    AgentSelectedIngressOutcome,
)
from membench.ingress.models import ExternalDelivery, SourceSelectionTask
from membench.memory.models import (
    AdapterProvenance,
    CanonicalModel,
    WriteStatus,
)
from membench.memory.policy import PolicyDispatchResult, PolicyWriteRequest
from membench.qualification.models import QuiescenceConfiguration, RetrievalStateDelta


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("trial timestamps must include a timezone")
    return value


class ReplayBatching(StrEnum):
    """Versioned grouping; neither strategy rewrites or truncates messages."""

    EVENT = "event_v1"
    USER_TURN = "user_turn_v1"


class ReplayConfiguration(CanonicalModel):
    """Fixed grouping and vendor options used for all stages of one trial.

    USER_TURN groups a user message with subsequent assistant messages. Leading
    assistant messages are retained. Stage changes always end a batch, and each
    overlay event is dispatched separately so gradual steps cannot be collapsed.
    Vendor options are supplied by bootstrap, never inferred from attack labels.
    """

    batching: ReplayBatching = ReplayBatching.USER_TURN
    native_options: dict[str, JsonValue] = Field(default_factory=dict)


class TrialPlan(CanonicalModel):
    """Validated public inputs and trusted settings for one isolated twin."""

    schema_version: Literal[1] = 1
    run: CampaignRunSpec
    projection: TrialProjection
    probe: PublicProbe
    replay: ReplayConfiguration = Field(default_factory=ReplayConfiguration)
    quiescence: QuiescenceConfiguration = Field(default_factory=QuiescenceConfiguration)
    purpose: Literal["pilot", "experiment"] = "pilot"
    attack_review_status: AttackReviewStatus = AttackReviewStatus.GENERATED_UNREVIEWED
    cleanup_storage: bool = True

    @model_validator(mode="after")
    def inputs_must_describe_the_same_trial(self) -> Self:
        if self.run.twin != self.projection.twin:
            raise ValueError("campaign run and projection must use the same twin")
        if (
            self.probe.probe_id != self.projection.probe_id
            or self.probe.allocation_case_id != self.projection.allocation_case_id
        ):
            raise ValueError("public probe does not belong to this projection")
        behavioral_task_id = self.run.agent_task.metadata.get("behavioral_task_id")
        if behavioral_task_id is not None and not isinstance(behavioral_task_id, str):
            raise ValueError("behavioral task ID must be a string")
        if (
            behavioral_task_id is None
            and self.run.agent_task.retrieval_query != self.probe.content
        ):
            raise ValueError("retrieval query must equal the held-out public probe")
        if not self.run.agent_configuration.memory_read_only:
            raise ValueError("trial probes require a read-only agent configuration")
        if self.run.policy not in {WritePolicy.DIRECT, WritePolicy.NATIVE_SELECTIVE}:
            raise ValueError("trial replay supports direct and native_selective only")
        if self.run.policy == WritePolicy.DIRECT and self.replay.native_options:
            raise ValueError("direct replay cannot silently ignore native options")
        if (
            self.purpose == "experiment"
            and self.attack_review_status != AttackReviewStatus.APPROVED
        ):
            raise ValueError("unreviewed projections must retain pilot status")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self)).hexdigest()


class TrialStage(StrEnum):
    """Lifecycle checkpoints; replay origins use the existing EventStage enum."""

    PREFLIGHT = "preflight"
    INITIALIZE = "initialize"
    VERIFY_EMPTY = "verify_empty"
    REPLAY = "replay"
    SELECTION = "selection"
    INGRESS = "ingress"
    FLUSH = "flush"
    QUIESCENCE = "quiescence"
    SNAPSHOT = "snapshot"
    RETRIEVE = "retrieve"
    RESPOND = "respond"
    VERIFY_READ_ONLY = "verify_read_only"
    EVALUATE = "evaluate"
    PROVENANCE = "provenance"
    RESET = "reset"
    CLOSE = "close"
    COMPLETE = "complete"


class AgentSelectedSelectionOutcome(StrEnum):
    """Evaluator interpretation of one finished frozen source-selection task."""

    INGESTED = "ingested"
    NOT_SELECTED = "not_selected"
    WRONG_MESSAGE = "wrong_message"
    TARGET_NOT_FOUND = "target_not_found"
    READ_PAYLOAD_MISMATCH = "read_payload_mismatch"
    READ_FAILED = "read_failed"
    AGENT_FAILED = "agent_failed"


class AgentSelectedSelectionResult(CanonicalModel):
    """Frozen task, agent trace, and safe-ingestion disposition for one stage."""

    selection_task: SourceSelectionTask
    delivery_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    agent_result: AgentRunResult
    attempts: tuple[AgentSelectedIngestionAttempt, ...] = Field(default_factory=tuple)
    outcome: AgentSelectedSelectionOutcome

    @model_validator(mode="after")
    def outcome_must_match_attempts(self) -> Self:
        if any(item.delivery_id != self.delivery_id for item in self.attempts):
            raise ValueError("selection attempts must target the bound delivery")
        ingested = [
            item
            for item in self.attempts
            if item.outcome == AgentSelectedIngressOutcome.INGESTED
        ]
        if len(ingested) > 1:
            raise ValueError("selection stage cannot ingest a delivery more than once")
        if self.outcome == AgentSelectedSelectionOutcome.INGESTED:
            if len(ingested) != 1:
                raise ValueError("ingested selection result requires one ingestion")
        elif ingested:
            raise ValueError("only ingested selection results may contain ingestion")
        elif self.outcome == AgentSelectedSelectionOutcome.NOT_SELECTED:
            if self.attempts:
                raise ValueError("not_selected cannot contain read attempts")
        elif not self.attempts:
            raise ValueError("non-empty selection outcomes require read attempts")
        if (
            self.outcome == AgentSelectedSelectionOutcome.AGENT_FAILED
            and self.agent_result.status != AgentRunStatus.FAILED
        ):
            raise ValueError("agent_failed requires a failed agent result")
        return self


def summarize_agent_selected_stage(
    selection_task: SourceSelectionTask,
    delivery: ExternalDelivery,
    agent_result: AgentRunResult,
    attempts: tuple[AgentSelectedIngestionAttempt, ...],
) -> AgentSelectedSelectionResult:
    """Create one deterministic, evaluator-only selection-stage outcome."""

    if any(item.outcome == AgentSelectedIngressOutcome.INGESTED for item in attempts):
        outcome = AgentSelectedSelectionOutcome.INGESTED
    elif agent_result.status == AgentRunStatus.FAILED:
        outcome = AgentSelectedSelectionOutcome.AGENT_FAILED
    elif not attempts:
        outcome = AgentSelectedSelectionOutcome.NOT_SELECTED
    else:
        latest = attempts[-1].outcome
        outcome = {
            AgentSelectedIngressOutcome.WRONG_MESSAGE: (
                AgentSelectedSelectionOutcome.WRONG_MESSAGE
            ),
            AgentSelectedIngressOutcome.TARGET_NOT_FOUND: (
                AgentSelectedSelectionOutcome.TARGET_NOT_FOUND
            ),
            AgentSelectedIngressOutcome.READ_PAYLOAD_MISMATCH: (
                AgentSelectedSelectionOutcome.READ_PAYLOAD_MISMATCH
            ),
            AgentSelectedIngressOutcome.READ_FAILED: (
                AgentSelectedSelectionOutcome.READ_FAILED
            ),
            AgentSelectedIngressOutcome.DUPLICATE_READ: (
                AgentSelectedSelectionOutcome.NOT_SELECTED
            ),
            AgentSelectedIngressOutcome.INGESTED: (
                AgentSelectedSelectionOutcome.INGESTED
            ),
        }[latest]
    return AgentSelectedSelectionResult(
        selection_task=selection_task,
        delivery_id=delivery.delivery_id,
        agent_result=agent_result,
        attempts=attempts,
        outcome=outcome,
    )


class TrialEventStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ErrorCause(CanonicalModel):
    depth: int = Field(ge=1)
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)
    module: str | None = None


class TrialFailure(CanonicalModel):
    stage: TrialStage
    failure_class: ChildFailureClass
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)
    causes: tuple[ErrorCause, ...] = Field(default_factory=tuple)



class TrialEvent(CanonicalModel):
    """Evaluator-only JSONL observation; never submitted as memory content."""

    schema_version: Literal[1] = 1
    sequence: int = Field(ge=1)
    run_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    stage: TrialStage
    status: TrialEventStatus
    replay_stage: EventStage | None = None
    operation_id: str | None = Field(default=None, min_length=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    failure: TrialFailure | None = None

    _timestamp_aware = field_validator("timestamp")(_aware)

    @model_validator(mode="after")
    def failure_must_match_status(self) -> Self:
        if (self.status == TrialEventStatus.FAILED) != (self.failure is not None):
            raise ValueError("only failed trial events require failure evidence")
        if self.failure is not None and self.failure.stage != self.stage:
            raise ValueError("trial event and failure stages must match")
        return self


class ReplayBatch(CanonicalModel):
    """Exact source events joined to one policy request, outside model prompts."""

    batch_index: int = Field(ge=1)
    events: tuple[ProjectionEvent, ...] = Field(min_length=1)
    request: PolicyWriteRequest

    @property
    def stage(self) -> EventStage:
        return self.events[0].stage

    @property
    def messages_sha256(self) -> str:
        """Hash role/content only so equivalent inputs compare across scopes."""
        messages = [
            {"role": event.role.value, "content": event.content}
            for event in self.events
        ]
        return hashlib.sha256(canonical_json_bytes(messages)).hexdigest()

    @model_validator(mode="after")
    def request_must_preserve_events(self) -> Self:
        if any(event.stage != self.stage for event in self.events):
            raise ValueError("replay batches cannot cross stage boundaries")
        ordinals = tuple(event.ordinal for event in self.events)
        if ordinals != tuple(range(ordinals[0], ordinals[0] + len(ordinals))):
            raise ValueError("batch source ordinals must be consecutive")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("batch source event IDs must be unique")
        if tuple((e.role, e.content) for e in self.events) != tuple(
            (message.role, message.content) for message in self.request.messages
        ):
            raise ValueError("policy messages must preserve exact event roles/content")
        if self.request.metadata or any(m.metadata for m in self.request.messages):
            raise ValueError("replay audit metadata must remain outside memory inputs")
        return self


class ReplayBatchResult(CanonicalModel):
    """Outcome of one dispatch attempt; QUEUED does not prove write visibility."""

    batch: ReplayBatch
    dispatch: PolicyDispatchResult | None = None
    failure: TrialFailure | None = None
    started_at: datetime
    finished_at: datetime

    _times_aware = field_validator("started_at", "finished_at")(_aware)

    @property
    def succeeded(self) -> bool:
        return self.failure is None

    @model_validator(mode="after")
    def evidence_must_match_attempt(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("replay attempt cannot finish before it starts")
        if self.dispatch is None and self.failure is None:
            raise ValueError("replay attempt needs dispatch or failure evidence")
        if self.failure is not None and self.failure.stage != TrialStage.REPLAY:
            raise ValueError("replay failures must identify the replay stage")
        if self.dispatch is not None:
            if self.dispatch.operation_id != self.batch.request.operation_id:
                raise ValueError("dispatch operation does not match replay batch")
            if self.dispatch.policy not in {
                WritePolicy.DIRECT,
                WritePolicy.NATIVE_SELECTIVE,
            }:
                raise ValueError("unsupported replay dispatch policy")
            if len(self.dispatch.write_results) != 1 or any(
                write.operation_id != self.batch.request.operation_id
                for write in self.dispatch.write_results
            ):
                raise ValueError("adapter write result does not match replay operation")
            if self.failure is None and any(
                write.status == WriteStatus.PARTIAL
                for write in self.dispatch.write_results
            ):
                raise ValueError("partial writes require failure evidence")
        return self


class TrialResult(CanonicalModel):
    """Terminal execution evidence, deliberately separate from attack scoring.

    The runner will add snapshots and flush evidence as artifact references in the
    child result. Replay records here preserve submission outcomes, including a
    failed final attempt. Final success also requires the runner's state checks.
    """

    schema_version: Literal[1] = 1
    plan: TrialPlan
    status: ChildRunStatus
    replay_results: tuple[ReplayBatchResult, ...] = Field(default_factory=tuple)
    selection_result: AgentSelectedSelectionResult | None = None
    trusted_baseline_verification: TrustedBaselineVerification | None = None
    agent_result: AgentRunResult | None = None
    provenance: AdapterProvenance | None = None
    failures: tuple[TrialFailure, ...] = Field(default_factory=tuple)
    state_deltas: dict[
        Literal["selection", "retrieval", "response"], RetrievalStateDelta
    ] = Field(
        default_factory=dict
    )
    started_at: datetime
    finished_at: datetime

    _times_aware = field_validator("started_at", "finished_at")(_aware)

    @model_validator(mode="after")
    def terminal_evidence_must_be_consistent(self) -> Self:
        # Share the existing campaign status, task, scope, and provenance rules.
        first = self.failures[0] if self.failures else None
        ChildRunResult(
            request_id=f"trial:{self.plan.run.run_id}",
            run=self.plan.run,
            status=self.status,
            agent_result=self.agent_result,
            provenance=self.provenance,
            failure_class=first.failure_class if first else None,
            failure_type=first.error_type if first else None,
            failure_message=first.error_message if first else None,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )
        source_events = self.plan.projection.replay_events
        submitted = tuple(e for r in self.replay_results for e in r.batch.events)
        if submitted != source_events[: len(submitted)]:
            raise ValueError(
                "replay results must preserve a prefix of projected events"
            )
        indexes = tuple(r.batch.batch_index for r in self.replay_results)
        if indexes != tuple(range(1, len(indexes) + 1)):
            raise ValueError("replay result batch indexes must be consecutive")
        for result in self.replay_results:
            if result.batch.request.scope != self.plan.run.scope:
                raise ValueError("replay result has the wrong memory scope")
            if result.dispatch and result.dispatch.policy != self.plan.run.policy:
                raise ValueError("replay result has the wrong write policy")
            if result.failure is not None and result.failure not in self.failures:
                raise ValueError("trial failures must include replay failure evidence")
        if any(not result.succeeded for result in self.replay_results[:-1]):
            raise ValueError("replay must stop after a failed attempt")
        if self.status == ChildRunStatus.SUCCEEDED and submitted != source_events:
            raise ValueError("successful trials require complete replay evidence")
        required_checkpoints = {"retrieval", "response"}
        if self.selection_result is not None:
            required_checkpoints.add("selection")
        if (
            self.status == ChildRunStatus.SUCCEEDED
            and set(self.state_deltas) != required_checkpoints
        ):
            raise ValueError("successful trials require every read-only checkpoint")
        for delta in self.state_deltas.values():
            if delta.scope != self.plan.run.scope:
                raise ValueError("state comparison has the wrong memory scope")
            if self.status == ChildRunStatus.SUCCEEDED and not delta.content_unchanged:
                raise ValueError(
                    "successful trials cannot contain memory content changes"
                )
        return self



class TrustedBaselineVerification(CanonicalModel):
    """Evaluator-only evidence that a trusted seed existed before ingress."""

    baseline_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    was_written: bool
    was_retrieved_before_ingress: bool
    retrieval_result_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    observed_markers: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def verification_must_match_observation(self) -> Self:
        if self.was_retrieved_before_ingress and not self.was_written:
            raise ValueError("a baseline cannot be retrieved before it was written")
        if self.was_retrieved_before_ingress and self.retrieval_result_sha256 is None:
            raise ValueError("successful verification requires retrieval evidence")
        return self
