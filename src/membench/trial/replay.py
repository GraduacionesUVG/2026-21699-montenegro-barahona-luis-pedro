"""Deterministic replay preparation and stage-at-a-time policy dispatch.

No initialization, flushing, retrieval, retries, or cleanup occurs here. The
lifecycle runner owns those barriers. Each replay instance is for one fresh
scope; deterministic operation IDs are audit IDs, not a backend dedup guarantee.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from copy import deepcopy
from datetime import UTC, datetime

from membench.campaign.models import ChildFailureClass
from membench.curation.projections import EventStage, ProjectionEvent
from membench.memory.models import MemoryMessage, MessageRole, WriteStatus
from membench.memory.policy import PolicyDispatcher, PolicyWriteRequest
from membench.trial.failures import exception_causes
from membench.trial.models import (
    ReplayBatch,
    ReplayBatching,
    ReplayBatchResult,
    TrialFailure,
    TrialPlan,
    TrialStage,
)

_OVERLAY_STAGES = {
    EventStage.MATCHED_BENIGN_OVERLAY,
    EventStage.POISONED_OVERLAY,
}


class ReplayExecutionError(RuntimeError):
    """Invalid sequencing or a mismatch between replay and its dispatcher."""


class PartialReplayWriteError(RuntimeError):
    """An adapter reported a partial write; later replay must not continue."""


def build_replay_batches(
    plan: TrialPlan,
    *,
    split_after_source_message_indexes: Iterable[int] = (),
) -> tuple[ReplayBatch, ...]:
    """Build the complete ordered write schedule without calling an adapter.

    Only projection.replay_events supply memory content. The probe, private-label
    reference, attack review status, and stage labels never become write text or
    metadata. The same grouping is used for direct and native policies.
    """
    plan = TrialPlan.model_validate(plan.model_dump(mode="json"))
    split_indexes = tuple(split_after_source_message_indexes)
    if any(not isinstance(index, int) or index < 0 for index in split_indexes):
        raise ReplayExecutionError("replay split indexes must be nonnegative integers")
    if split_indexes != tuple(sorted(set(split_indexes))):
        raise ReplayExecutionError("replay split indexes must be sorted and unique")
    source_indexes = {
        event.source_message_index for event in plan.projection.replay_events
    }
    unknown_splits = sorted(set(split_indexes) - source_indexes)
    if unknown_splits:
        raise ReplayExecutionError(
            "replay split index is not present in projected events: "
            + ", ".join(str(index) for index in unknown_splits)
        )
    groups: list[list[ProjectionEvent]] = []
    for event in plan.projection.replay_events:
        new_batch = (
            not groups
            or plan.replay.batching == ReplayBatching.EVENT
            or event.stage != groups[-1][0].stage
            or event.stage in _OVERLAY_STAGES
            or event.role == MessageRole.USER
            or groups[-1][-1].source_message_index in split_indexes
        )
        if new_batch:
            groups.append([])
        groups[-1].append(event)

    batches = []
    for index, events in enumerate(groups, start=1):
        operation_id = f"{plan.run.run_id}:replay:{index:06d}"
        batches.append(
            ReplayBatch(
                batch_index=index,
                events=tuple(events),
                request=PolicyWriteRequest(
                    operation_id=operation_id,
                    scope=plan.run.scope,
                    messages=tuple(
                        MemoryMessage(role=event.role, content=event.content)
                        for event in events
                    ),
                    source_event_id=operation_id,
                    native_options=deepcopy(plan.replay.native_options),
                ),
            )
        )
    return tuple(batches)


class ReplayExecutor:
    """Single-use ordered dispatcher that yields evidence after each attempt.

    Consume replay_stage(stage), flush/checkpoint in the lifecycle runner, then
    move to the next stage. A failed attempt is yielded once and permanently stops
    this executor. Queued and no-op results remain successful submissions; only
    the lifecycle's subsequent flush/quiescence can establish memory visibility.
    """

    def __init__(
        self,
        plan: TrialPlan,
        dispatcher: PolicyDispatcher,
        *,
        split_after_source_message_indexes: Iterable[int] = (),
    ) -> None:
        if dispatcher.policy != plan.run.policy:
            raise ReplayExecutionError("dispatcher policy differs from trial plan")
        self._policy = plan.run.policy
        self._batches = build_replay_batches(
            plan,
            split_after_source_message_indexes=split_after_source_message_indexes,
        )
        self._dispatcher = dispatcher
        self._cursor = 0
        self._results: list[ReplayBatchResult] = []
        self._failed = False
        self._active = False

    @property
    def stages(self) -> tuple[EventStage, ...]:
        """Non-empty stages in source order; the clean twin has no overlay."""
        return tuple(dict.fromkeys(batch.stage for batch in self._batches))

    @property
    def results(self) -> tuple[ReplayBatchResult, ...]:
        return tuple(result.model_copy(deep=True) for result in self._results)

    @property
    def complete(self) -> bool:
        """Whether every batch was submitted successfully, not necessarily stored."""
        return self._cursor == len(self._batches) and not self._failed

    def replay_stage(self, stage: EventStage) -> Iterator[ReplayBatchResult]:
        """Yield submission evidence; stop at the stage boundary or first failure.

        Sequence errors raise before any write. After an adapter error, the yielded
        record retains the failed batch so callers can persist partial-run evidence.
        """
        if self._active:
            raise ReplayExecutionError("another replay stage iterator is active")
        if self._failed:
            raise ReplayExecutionError("replay stopped after a failed attempt")
        if self._cursor == len(self._batches):
            raise ReplayExecutionError("replay is already complete")
        if self._batches[self._cursor].stage != stage:
            raise ReplayExecutionError("requested stage is not the next replay stage")
        if self._dispatcher.policy != self._policy:
            raise ReplayExecutionError("dispatcher policy changed during replay")

        self._active = True
        try:
            while self._cursor < len(self._batches):
                batch = self._batches[self._cursor]
                if batch.stage != stage:
                    break
                result = self._dispatch(batch)
                self._cursor += 1
                self._results.append(result)
                self._failed = not result.succeeded
                yield result.model_copy(deep=True)
                if self._failed:
                    break
        finally:
            self._active = False

    def _dispatch(self, batch: ReplayBatch) -> ReplayBatchResult:
        started_at = datetime.now(UTC)
        dispatch = None
        failure = None
        try:
            if self._dispatcher.policy != self._policy:
                raise ReplayExecutionError("dispatcher policy changed during replay")
            # Vendor code cannot mutate the frozen request retained as evidence.
            observed = self._dispatcher.dispatch(batch.request.model_copy(deep=True))
            if observed.policy != self._policy:
                raise ReplayExecutionError("dispatch returned the wrong write policy")
            if any(w.status == WriteStatus.PARTIAL for w in observed.write_results):
                # Validate identity before retaining the observed partial result.
                failure = TrialFailure(
                    stage=TrialStage.REPLAY,
                    failure_class=ChildFailureClass.ADAPTER_FAILURE,
                    error_type=PartialReplayWriteError.__name__,
                    error_message="adapter reported a partial replay write",
                )
            ReplayBatchResult(
                batch=batch,
                dispatch=observed,
                failure=failure,
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
            dispatch = observed.model_copy(deep=True)
        except Exception as exc:
            failure = TrialFailure(
                stage=TrialStage.REPLAY,
                failure_class=ChildFailureClass.ADAPTER_FAILURE,
                error_type=type(exc).__name__,
                error_message=str(exc) or type(exc).__name__,
                causes=exception_causes(exc)
            )
        return ReplayBatchResult(
            batch=batch,
            dispatch=dispatch,
            failure=failure,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
