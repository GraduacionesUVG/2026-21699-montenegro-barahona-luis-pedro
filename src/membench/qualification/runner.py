"""Mechanism-neutral single- and secondary-scope qualification lifecycle."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from membench.canary_data import (
    CanaryCategory,
    CanarySuite,
    CommonQualification,
    SyntheticCanary,
)
from membench.memory.errors import (
    AdapterConfigurationError,
    AdapterUnavailableError,
    NamespaceNotEmptyError,
    PolicyDependencyError,
    ResetVerificationError,
    RetrievalTransportError,
    StateInspectionError,
    UnsupportedCapabilityError,
    WriteTransportError,
    WriteVisibilityTimeout,
)
from membench.memory.models import (
    AdapterProvenance,
    EmptyStateResult,
    FlushResult,
    MemoryMessage,
    MemoryScope,
    MemorySnapshot,
    RetrievalRequest,
)
from membench.memory.policy import AdmissionGuard, PolicyDispatcher, PolicyWriteRequest
from membench.memory.protocol import MemoryAdapter
from membench.qualification.assertions import (
    assert_context_fragments,
    assert_dispatch_action,
    assert_empty_state,
    assert_flush_completed,
    assert_provenance_mechanism,
    assert_quiescence_reached,
    assert_responder_identity,
    assert_retrieval_excludes_canary,
    assert_retrieval_fragments,
    assert_retrieval_scope,
    assert_snapshot_content_unchanged,
    assert_snapshot_empty,
    assert_snapshot_excludes_canaries,
    assert_snapshot_full_state_unchanged,
    assert_snapshot_preserved,
    assert_snapshot_unchanged,
    compare_snapshot_state,
    guard_rejected,
    snapshot_canary_assertions,
)
from membench.qualification.events import (
    NullQualificationEventSink,
    QualificationEventSink,
)
from membench.qualification.models import (
    CanaryDispatchRecord,
    CanaryRetrievalRecord,
    QualificationAssertion,
    QualificationEvent,
    QualificationEventStatus,
    QualificationFailureClass,
    QualificationPlan,
    QualificationReport,
    QualificationStage,
    QualificationSummary,
    QuiescenceEvidence,
    ResponderStageEvidence,
    RetrievalReadOnlyEvidence,
)
from membench.qualification.quiescence import (
    QualificationQuiescenceTimeout,
    wait_for_quiescence,
)
from membench.qualification.responder_stage import (
    QualificationContextAssembler,
    QualificationPromptBuilder,
    QualificationResponder,
)
from membench.responder import ResponderError, ResponderResult


class QualificationInstrumentationError(RuntimeError):
    """Raised internally when the configured event sink stops accepting events."""


@dataclass(frozen=True, slots=True)
class _IsolationResult:
    primary_flush: FlushResult
    primary_snapshot_after: MemorySnapshot
    primary_reset: EmptyStateResult
    secondary_flush: FlushResult
    secondary_snapshot_before: MemorySnapshot
    secondary_snapshot_after: MemorySnapshot
    secondary_provenance: AdapterProvenance
    secondary_reset: EmptyStateResult
    retrieval_read_only: tuple[RetrievalReadOnlyEvidence, ...]


class QualificationRunner:
    """Execute one qualification using only the common adapter protocol."""

    def __init__(
        self,
        adapter: MemoryAdapter,
        suite: CanarySuite,
        *,
        event_sink: QualificationEventSink | None = None,
        guard: AdmissionGuard | None = None,
        secondary_adapter_factory: Callable[[], MemoryAdapter] | None = None,
        responder: QualificationResponder | None = None,
        context_assembler: QualificationContextAssembler | None = None,
        prompt_builder: QualificationPromptBuilder | None = None,
        quiescence_clock: Callable[[], float] = time.perf_counter,
        quiescence_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.adapter = adapter
        self.suite = suite
        self.event_sink = event_sink or NullQualificationEventSink()
        self.guard = guard
        self.secondary_adapter_factory = secondary_adapter_factory
        self.responder = responder
        self.context_assembler = context_assembler or QualificationContextAssembler()
        self.prompt_builder = prompt_builder or QualificationPromptBuilder()
        self.quiescence_clock = quiescence_clock
        self.quiescence_sleep = quiescence_sleep
        self._sequence = 0
        self._active_stage = QualificationStage.PREFLIGHT
        self._event_sink_error: Exception | None = None

    def run(self, plan: QualificationPlan) -> QualificationReport:
        """Run primary-scope qualification and always close the adapter."""
        started_at = datetime.now(UTC)
        completed: list[QualificationStage] = []
        assertions: list[QualificationAssertion] = []
        dispatches: list[CanaryDispatchRecord] = []
        retrievals: list[CanaryRetrievalRecord] = []
        retrieval_read_only: list[RetrievalReadOnlyEvidence] = []
        quiescence: list[QuiescenceEvidence] = []
        responder_stages: list[ResponderStageEvidence] = []
        flush_result = None
        snapshot_before = None
        snapshot_after = None
        secondary_snapshot_before = None
        secondary_snapshot_after = None
        secondary_flush_result = None
        secondary_provenance = None
        secondary_reset_result = None
        provenance = None
        reset_result = None
        failure: tuple[QualificationFailureClass, str, str] | None = None
        initialized = False
        owns_clean_scope = False
        reset_attempted = False

        try:
            qualification, canaries, dispatcher = self._preflight(plan)
            self._complete_stage(
                plan,
                QualificationStage.PREFLIGHT,
                completed,
                payload={
                    "suite_version": self.suite.manifest.suite_version,
                    "canary_ids": [item.canary_id for item in canaries],
                    "required_capabilities": list(
                        qualification.required_capabilities
                    ),
                },
            )

            self._start_stage(plan, QualificationStage.INITIALIZE)
            stage_started = time.perf_counter()
            self.adapter.initialize(plan.scope)
            initialized = True
            self._complete_stage(
                plan,
                QualificationStage.INITIALIZE,
                completed,
                latency_seconds=time.perf_counter() - stage_started,
            )

            self._start_stage(plan, QualificationStage.VERIFY_EMPTY)
            stage_started = time.perf_counter()
            empty = self.adapter.verify_empty()
            owns_clean_scope = empty.empty
            assertions.append(
                assert_empty_state(
                    empty,
                    assertion_id="initial_scope_empty",
                    stage=QualificationStage.VERIFY_EMPTY,
                )
            )
            if not empty.empty:
                raise NamespaceNotEmptyError(
                    "qualification scope was not empty before writing"
                )
            self._complete_stage(
                plan,
                QualificationStage.VERIFY_EMPTY,
                completed,
                latency_seconds=time.perf_counter() - stage_started,
                payload={"empty_state": empty.model_dump(mode="json")},
            )

            self._start_stage(plan, QualificationStage.INSPECT_BEFORE)
            stage_started = time.perf_counter()
            snapshot_before = self.adapter.inspect_state()
            assertions.append(assert_snapshot_empty(snapshot_before))
            if snapshot_before.record_count:
                owns_clean_scope = False
                raise NamespaceNotEmptyError(
                    "qualification snapshot was not empty before writing"
                )
            self._complete_stage(
                plan,
                QualificationStage.INSPECT_BEFORE,
                completed,
                latency_seconds=time.perf_counter() - stage_started,
                payload={"snapshot": snapshot_before.model_dump(mode="json")},
            )

            if qualification.requires_secondary_scope:
                isolation = self._run_namespace_isolation(
                    plan,
                    canaries,
                    dispatcher,
                    completed,
                    assertions,
                    dispatches,
                    retrievals,
                )
                flush_result = isolation.primary_flush
                snapshot_after = isolation.primary_snapshot_after
                reset_result = isolation.primary_reset
                reset_attempted = True
                secondary_flush_result = isolation.secondary_flush
                secondary_snapshot_before = isolation.secondary_snapshot_before
                secondary_snapshot_after = isolation.secondary_snapshot_after
                secondary_provenance = isolation.secondary_provenance
                secondary_reset_result = isolation.secondary_reset
                retrieval_read_only.extend(isolation.retrieval_read_only)
            else:
                (
                    flush_result,
                    snapshot_after,
                    read_only_evidence,
                ) = self._run_primary_data_path(
                    plan,
                    canaries,
                    dispatcher,
                    completed,
                    assertions,
                    dispatches,
                    retrievals,
                    quiescence,
                )
                retrieval_read_only.append(read_only_evidence)

            if qualification.requires_responder:
                responder_stages.extend(
                    self._run_responder_stages(
                        plan,
                        canaries,
                        retrievals,
                        completed,
                        assertions,
                        quiescence,
                        initial_snapshot=snapshot_after,
                    )
                )
                snapshot_after = responder_stages[-1].snapshot_after

            self._start_stage(plan, QualificationStage.PROVENANCE)
            stage_started = time.perf_counter()
            provenance = self.adapter.export_provenance()
            assertions.append(
                assert_provenance_mechanism(provenance, plan.mechanism)
            )
            self._complete_stage(
                plan,
                QualificationStage.PROVENANCE,
                completed,
                latency_seconds=time.perf_counter() - stage_started,
                payload={"provenance": provenance.model_dump(mode="json")},
            )

            if plan.cleanup and not reset_attempted:
                reset_attempted = True
                reset_result = self._reset(plan, completed, assertions)

        except Exception as exc:
            failure_class = _classify_failure(exc, self._active_stage)
            error_message = _error_message(exc)
            failure = (failure_class, type(exc).__name__, error_message)
            self._emit(
                plan,
                stage=self._active_stage,
                status=QualificationEventStatus.FAILED,
                failure_class=failure_class,
                error_type=type(exc).__name__,
                error_message=error_message,
            )
        finally:
            if (
                initialized
                and owns_clean_scope
                and plan.cleanup
                and not reset_attempted
            ):
                try:
                    reset_attempted = True
                    reset_result = self._reset(plan, completed, assertions)
                except Exception as cleanup_exc:
                    if failure is None:
                        cleanup_class = _classify_failure(
                            cleanup_exc,
                            QualificationStage.RESET,
                        )
                        failure = (
                            cleanup_class,
                            type(cleanup_exc).__name__,
                            _error_message(cleanup_exc),
                        )
            try:
                self._start_stage(plan, QualificationStage.CLOSE)
                stage_started = time.perf_counter()
                self.adapter.close()
                self._complete_stage(
                    plan,
                    QualificationStage.CLOSE,
                    completed,
                    latency_seconds=time.perf_counter() - stage_started,
                )
            except Exception as close_exc:
                self._emit(
                    plan,
                    stage=QualificationStage.CLOSE,
                    status=QualificationEventStatus.FAILED,
                    failure_class=(
                        QualificationFailureClass.INSTRUMENTATION_INCOMPLETE
                    ),
                    error_type=type(close_exc).__name__,
                    error_message=_error_message(close_exc),
                )
                if failure is None:
                    failure = (
                        QualificationFailureClass.INSTRUMENTATION_INCOMPLETE,
                        type(close_exc).__name__,
                        _error_message(close_exc),
                    )

        if self._event_sink_error is not None and failure is None:
            failure = (
                QualificationFailureClass.INSTRUMENTATION_INCOMPLETE,
                type(self._event_sink_error).__name__,
                _error_message(self._event_sink_error),
            )

        if failure is None and any(
            item.required and not item.passed for item in assertions
        ):
            failure = (
                QualificationFailureClass.ASSERTION_FAILED,
                "QualificationAssertionError",
                "one or more required qualification assertions failed",
            )

        passed = failure is None
        if passed:
            self._complete_stage(plan, QualificationStage.COMPLETE, completed)
        else:
            assert failure is not None
            self._emit(
                plan,
                stage=QualificationStage.COMPLETE,
                status=QualificationEventStatus.FAILED,
                failure_class=failure[0],
                error_type=failure[1],
                error_message=failure[2],
            )

        finished_at = datetime.now(UTC)
        summary = QualificationSummary(
            experiment_id=plan.scope.experiment_id,
            run_id=plan.scope.run_id,
            mechanism=plan.mechanism,
            policy=plan.policy,
            qualification_id=plan.qualification_id,
            passed=passed,
            completed_stages=tuple(completed),
            assertions=tuple(assertions),
            failure_class=failure[0] if failure else None,
            failure_type=failure[1] if failure else None,
            failure_message=failure[2] if failure else None,
            started_at=started_at,
            finished_at=finished_at,
        )
        return QualificationReport(
            plan=plan,
            summary=summary,
            dispatches=tuple(dispatches),
            flush_result=flush_result,
            retrievals=tuple(retrievals),
            retrieval_read_only=tuple(retrieval_read_only),
            quiescence=tuple(quiescence),
            responder_stages=tuple(responder_stages),
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
            secondary_snapshot_before=secondary_snapshot_before,
            secondary_snapshot_after=secondary_snapshot_after,
            secondary_flush_result=secondary_flush_result,
            secondary_provenance=secondary_provenance,
            provenance=provenance,
            reset_result=reset_result,
            secondary_reset_result=secondary_reset_result,
        )

    def _run_primary_data_path(
        self,
        plan: QualificationPlan,
        canaries: tuple[SyntheticCanary, ...],
        dispatcher: PolicyDispatcher,
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        dispatches: list[CanaryDispatchRecord],
        retrievals: list[CanaryRetrievalRecord],
        quiescence: list[QuiescenceEvidence],
    ) -> tuple[FlushResult, MemorySnapshot, RetrievalReadOnlyEvidence]:
        for canary in canaries:
            self._dispatch_canary(
                plan,
                plan.scope,
                canary,
                dispatcher,
                completed,
                assertions,
                dispatches,
                scope_role="primary",
            )
        flush_result = self._flush_adapter(
            plan,
            self.adapter,
            completed,
            assertions,
            scope_role="primary",
        )
        if plan.responder_required:
            retrieval_snapshot_before = self._wait_for_quiescence(
                plan,
                self.adapter,
                completed,
                assertions,
                quiescence,
                phase="pre_retrieval",
                scope=plan.scope,
            ).final_snapshot
        else:
            retrieval_snapshot_before = self.adapter.inspect_state()
        retrieval_operation_ids: list[str] = []
        for canary in canaries:
            if not guard_rejected(plan.policy, canary):
                retrieval_operation_ids.extend(
                    self._retrieve_expected(
                        plan,
                        self.adapter,
                        plan.scope,
                        canary,
                        completed,
                        assertions,
                        retrievals,
                        phase="own",
                    )
                )

        retrieval_barrier = self._flush_adapter(
            plan,
            self.adapter,
            completed,
            assertions,
            scope_role="primary",
            phase="post_retrieval",
            assertion_id="post_retrieval_flush_completed",
        )

        self._start_stage(plan, QualificationStage.INSPECT_AFTER)
        stage_started = time.perf_counter()
        if plan.responder_required:
            snapshot = self._wait_for_quiescence(
                plan,
                self.adapter,
                completed,
                assertions,
                quiescence,
                phase="post_retrieval",
                scope=plan.scope,
            ).final_snapshot
        else:
            snapshot = self.adapter.inspect_state()
        assertions.extend(
            snapshot_canary_assertions(plan.policy, canaries, snapshot)
        )
        self._complete_stage(
            plan,
            QualificationStage.INSPECT_AFTER,
            completed,
            latency_seconds=time.perf_counter() - stage_started,
            payload={
                "scope_role": "primary",
                "snapshot": snapshot.model_dump(mode="json"),
            },
        )
        read_only_evidence = self._verify_retrieval_unchanged(
            plan,
            completed,
            assertions,
            phase="primary_probe_context",
            scope_role="primary",
            scope=plan.scope,
            operation_ids=tuple(retrieval_operation_ids),
            snapshot_before=retrieval_snapshot_before,
            visibility_barrier=retrieval_barrier,
            snapshot_after=snapshot,
        )
        return flush_result, snapshot, read_only_evidence

    def _run_namespace_isolation(
        self,
        plan: QualificationPlan,
        canaries: tuple[SyntheticCanary, ...],
        primary_dispatcher: PolicyDispatcher,
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        dispatches: list[CanaryDispatchRecord],
        retrievals: list[CanaryRetrievalRecord],
    ) -> _IsolationResult:
        secondary_scope = plan.secondary_scope
        factory = self.secondary_adapter_factory
        if secondary_scope is None or factory is None:
            raise AdapterConfigurationError(
                "secondary-scope dependencies disappeared after preflight"
            )
        primary_canaries = tuple(
            canary
            for canary in canaries
            if canary.category != CanaryCategory.ISOLATION
        )
        secondary_canaries = tuple(
            canary
            for canary in canaries
            if canary.category == CanaryCategory.ISOLATION
        )
        if not primary_canaries or not secondary_canaries:
            raise AdapterConfigurationError(
                "namespace isolation requires both ordinary primary canaries and "
                "isolation-category secondary canaries"
            )

        self._start_stage(plan, QualificationStage.ISOLATION)
        secondary = factory()
        if not isinstance(secondary, MemoryAdapter):
            raise AdapterConfigurationError(
                "secondary adapter factory did not return a MemoryAdapter"
            )
        if secondary is self.adapter:
            raise AdapterConfigurationError(
                "secondary adapter factory must return a fresh adapter"
            )

        secondary_initialized = False
        secondary_owns_clean_scope = False
        secondary_reset: EmptyStateResult | None = None
        secondary_closed = False
        try:
            qualification = self.suite.qualification(plan.qualification_id)
            missing = tuple(
                name
                for name in qualification.required_capabilities
                if not secondary.capabilities[name]
            )
            if missing:
                raise UnsupportedCapabilityError(
                    "secondary adapter does not satisfy qualification "
                    "capabilities: " + ", ".join(missing),
                    missing=missing,
                )
            secondary_dispatcher = PolicyDispatcher(
                secondary,
                plan.policy,
                guard=self.guard,
            )

            secondary.initialize(secondary_scope)
            secondary_initialized = True
            secondary_empty = secondary.verify_empty()
            secondary_owns_clean_scope = secondary_empty.empty
            assertions.append(
                assert_empty_state(
                    secondary_empty,
                    assertion_id="secondary_initial_scope_empty",
                    stage=QualificationStage.ISOLATION,
                )
            )
            if not secondary_empty.empty:
                raise NamespaceNotEmptyError(
                    "secondary qualification scope was not empty before writing"
                )
            secondary_snapshot_before = secondary.inspect_state()
            assertions.append(
                assert_snapshot_empty(
                    secondary_snapshot_before,
                    assertion_id="secondary_initial_snapshot_empty",
                )
            )
            if secondary_snapshot_before.record_count:
                secondary_owns_clean_scope = False
                raise NamespaceNotEmptyError(
                    "secondary qualification snapshot was not empty before writing"
                )

            for canary in primary_canaries:
                self._dispatch_canary(
                    plan,
                    plan.scope,
                    canary,
                    primary_dispatcher,
                    completed,
                    assertions,
                    dispatches,
                    scope_role="primary",
                )
            for canary in secondary_canaries:
                self._dispatch_canary(
                    plan,
                    secondary_scope,
                    canary,
                    secondary_dispatcher,
                    completed,
                    assertions,
                    dispatches,
                    scope_role="secondary",
                )

            primary_flush = self._flush_adapter(
                plan,
                self.adapter,
                completed,
                assertions,
                scope_role="primary",
            )
            secondary_flush = self._flush_adapter(
                plan,
                secondary,
                completed,
                assertions,
                scope_role="secondary",
            )

            primary_retrieval_before = self.adapter.inspect_state()
            secondary_retrieval_before = secondary.inspect_state()
            primary_retrieval_operations: list[str] = []
            secondary_retrieval_operations: list[str] = []
            for canary in primary_canaries:
                primary_retrieval_operations.extend(
                    self._retrieve_expected(
                        plan,
                        self.adapter,
                        plan.scope,
                        canary,
                        completed,
                        assertions,
                        retrievals,
                        phase="primary_own",
                    )
                )
                secondary_retrieval_operations.extend(
                    self._retrieve_exclusion(
                        plan,
                        secondary,
                        secondary_scope,
                        canary,
                        completed,
                        assertions,
                        phase="primary_from_secondary",
                    )
                )
            for canary in secondary_canaries:
                secondary_retrieval_operations.extend(
                    self._retrieve_expected(
                        plan,
                        secondary,
                        secondary_scope,
                        canary,
                        completed,
                        assertions,
                        retrievals,
                        phase="secondary_own",
                        scope_role="secondary",
                    )
                )
                primary_retrieval_operations.extend(
                    self._retrieve_exclusion(
                        plan,
                        self.adapter,
                        plan.scope,
                        canary,
                        completed,
                        assertions,
                        phase="secondary_from_primary",
                    )
                )

            primary_retrieval_barrier = self._flush_adapter(
                plan,
                self.adapter,
                completed,
                assertions,
                scope_role="primary",
                phase="post_retrieval",
                assertion_id="primary_post_retrieval_flush_completed",
            )
            secondary_retrieval_barrier = self._flush_adapter(
                plan,
                secondary,
                completed,
                assertions,
                scope_role="secondary",
                phase="post_retrieval",
                assertion_id="secondary_post_retrieval_flush_completed",
            )
            primary_snapshot_after = self.adapter.inspect_state()
            secondary_snapshot_written = secondary.inspect_state()
            read_only_evidence = [
                self._verify_retrieval_unchanged(
                    plan,
                    completed,
                    assertions,
                    phase="primary_isolation_queries",
                    scope_role="primary",
                    scope=plan.scope,
                    operation_ids=tuple(primary_retrieval_operations),
                    snapshot_before=primary_retrieval_before,
                    visibility_barrier=primary_retrieval_barrier,
                    snapshot_after=primary_snapshot_after,
                ),
                self._verify_retrieval_unchanged(
                    plan,
                    completed,
                    assertions,
                    phase="secondary_isolation_queries",
                    scope_role="secondary",
                    scope=secondary_scope,
                    operation_ids=tuple(secondary_retrieval_operations),
                    snapshot_before=secondary_retrieval_before,
                    visibility_barrier=secondary_retrieval_barrier,
                    snapshot_after=secondary_snapshot_written,
                ),
            ]
            assertions.extend(
                snapshot_canary_assertions(
                    plan.policy,
                    primary_canaries,
                    primary_snapshot_after,
                )
            )
            assertions.extend(
                snapshot_canary_assertions(
                    plan.policy,
                    secondary_canaries,
                    secondary_snapshot_written,
                )
            )
            assertions.append(
                assert_snapshot_excludes_canaries(
                    primary_snapshot_after,
                    secondary_canaries,
                    assertion_id="primary_snapshot_excludes_secondary_canaries",
                )
            )
            assertions.append(
                assert_snapshot_excludes_canaries(
                    secondary_snapshot_written,
                    primary_canaries,
                    assertion_id="secondary_snapshot_excludes_primary_canaries",
                )
            )

            secondary_provenance = secondary.export_provenance()
            assertions.append(
                assert_provenance_mechanism(
                    secondary_provenance,
                    plan.mechanism,
                    assertion_id="secondary_provenance_mechanism_matches",
                )
            )

            primary_reset = self._reset(plan, completed, assertions)
            secondary_after_primary_reset = secondary.inspect_state()
            assertions.append(
                assert_snapshot_preserved(
                    secondary_snapshot_written,
                    secondary_after_primary_reset,
                    assertion_id="secondary_unchanged_after_primary_reset",
                )
            )
            survival_operations: list[str] = []
            for canary in secondary_canaries:
                survival_operations.extend(
                    self._retrieve_expected(
                        plan,
                        secondary,
                        secondary_scope,
                        canary,
                        completed,
                        assertions,
                        retrievals,
                        phase="secondary_after_primary_reset",
                        scope_role="secondary",
                    )
                )
            survival_barrier = self._flush_adapter(
                plan,
                secondary,
                completed,
                assertions,
                scope_role="secondary",
                phase="post_primary_reset_retrieval",
                assertion_id=(
                    "secondary_post_primary_reset_retrieval_flush_completed"
                ),
            )
            secondary_snapshot_after = secondary.inspect_state()
            read_only_evidence.append(
                self._verify_retrieval_unchanged(
                    plan,
                    completed,
                    assertions,
                    phase="secondary_post_primary_reset_probe",
                    scope_role="secondary",
                    scope=secondary_scope,
                    operation_ids=tuple(survival_operations),
                    snapshot_before=secondary_after_primary_reset,
                    visibility_barrier=survival_barrier,
                    snapshot_after=secondary_snapshot_after,
                )
            )

            secondary_reset = self._reset(
                plan,
                completed,
                assertions,
                adapter=secondary,
                assertion_id="secondary_reset_verified_empty",
                operation_id=f"{secondary_scope.run_id}:reset",
            )
            secondary.close()
            secondary_closed = True
            self._complete_stage(
                plan,
                QualificationStage.ISOLATION,
                completed,
                payload={
                    "primary_scope": plan.scope.model_dump(mode="json"),
                    "secondary_scope": secondary_scope.model_dump(mode="json"),
                    "primary_reset": primary_reset.model_dump(mode="json"),
                    "secondary_reset": secondary_reset.model_dump(mode="json"),
                },
                deduplicate_stage=True,
            )
            return _IsolationResult(
                primary_flush=primary_flush,
                primary_snapshot_after=primary_snapshot_after,
                primary_reset=primary_reset,
                secondary_flush=secondary_flush,
                secondary_snapshot_before=secondary_snapshot_before,
                secondary_snapshot_after=secondary_snapshot_after,
                secondary_provenance=secondary_provenance,
                secondary_reset=secondary_reset,
                retrieval_read_only=tuple(read_only_evidence),
            )
        except Exception:
            if (
                secondary_initialized
                and secondary_owns_clean_scope
                and secondary_reset is None
            ):
                try:
                    secondary.reset_and_verify_empty()
                except Exception:
                    pass
            if not secondary_closed:
                try:
                    secondary.close()
                except Exception:
                    pass
            raise

    def _dispatch_canary(
        self,
        plan: QualificationPlan,
        scope: MemoryScope,
        canary: SyntheticCanary,
        dispatcher: PolicyDispatcher,
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        dispatches: list[CanaryDispatchRecord],
        *,
        scope_role: Literal["primary", "secondary"],
    ) -> None:
        self._start_stage(
            plan,
            QualificationStage.WRITE,
            canary_id=canary.canary_id,
        )
        stage_started = time.perf_counter()
        result = dispatcher.dispatch(self._write_request(plan, canary, scope))
        dispatches.append(
            CanaryDispatchRecord(
                canary_id=canary.canary_id,
                scope_role=scope_role,
                result=result,
            )
        )
        assertions.append(assert_dispatch_action(plan.policy, canary, result.action))
        self._complete_stage(
            plan,
            QualificationStage.WRITE,
            completed,
            latency_seconds=time.perf_counter() - stage_started,
            operation_id=result.operation_id,
            canary_id=canary.canary_id,
            payload={
                "scope_role": scope_role,
                "dispatch": result.model_dump(mode="json"),
            },
            deduplicate_stage=True,
        )

    def _flush_adapter(
        self,
        plan: QualificationPlan,
        adapter: MemoryAdapter,
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        *,
        scope_role: Literal["primary", "secondary"],
        phase: str = "write_visibility",
        assertion_id: str | None = None,
    ) -> FlushResult:
        operation_id = f"{scope_role}:flush:{phase}"
        self._start_stage(
            plan,
            QualificationStage.FLUSH,
            operation_id=operation_id,
        )
        stage_started = time.perf_counter()
        result = adapter.flush_and_wait()
        resolved_assertion_id = assertion_id or (
            "flush_completed"
            if scope_role == "primary"
            else f"{scope_role}_flush_completed"
        )
        assertions.append(
            assert_flush_completed(result, assertion_id=resolved_assertion_id)
        )
        self._complete_stage(
            plan,
            QualificationStage.FLUSH,
            completed,
            latency_seconds=time.perf_counter() - stage_started,
            operation_id=operation_id,
            payload={
                "scope_role": scope_role,
                "phase": phase,
                "flush": result.model_dump(mode="json"),
            },
            deduplicate_stage=True,
        )
        return result

    def _wait_for_quiescence(
        self,
        plan: QualificationPlan,
        adapter: MemoryAdapter,
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        evidence: list[QuiescenceEvidence],
        *,
        phase: str,
        scope: MemoryScope,
    ) -> QuiescenceEvidence:
        operation_id = f"primary:quiescence:{phase}"
        self._start_stage(
            plan,
            QualificationStage.QUIESCENCE,
            operation_id=operation_id,
        )
        result = wait_for_quiescence(
            adapter.inspect_state,
            plan.quiescence,
            phase=phase,
            expected_scope=scope,
            clock=self.quiescence_clock,
            sleep=self.quiescence_sleep,
        )
        evidence.append(result)
        assertion = assert_quiescence_reached(
            result,
            assertion_id=f"quiescence_reached:{phase}",
        )
        assertions.append(assertion)
        if not result.stabilized:
            raise QualificationQuiescenceTimeout(
                f"memory state did not stabilize during {phase!r} within "
                f"{result.timeout_seconds:g} seconds"
            )
        self._complete_stage(
            plan,
            QualificationStage.QUIESCENCE,
            completed,
            latency_seconds=result.latency_seconds,
            operation_id=operation_id,
            payload={
                "phase": phase,
                "scope": scope.model_dump(mode="json"),
                "stabilized": result.stabilized,
                "sample_count": len(result.observations),
                "record_count": result.final_snapshot.record_count,
                "normalized_sha256": (
                    result.observations[-1].normalized_sha256
                ),
            },
            deduplicate_stage=True,
        )
        return result

    def _retrieve_expected(
        self,
        plan: QualificationPlan,
        adapter: MemoryAdapter,
        scope: MemoryScope,
        canary: SyntheticCanary,
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        retrievals: list[CanaryRetrievalRecord],
        *,
        phase: str,
        scope_role: Literal["primary", "secondary"] = "primary",
    ) -> tuple[str, ...]:
        operation_ids: list[str] = []
        for index, expectation in enumerate(canary.retrieval_expectations):
            operation_id = (
                f"{scope.run_id}:retrieve:{phase}:{canary.canary_id}:{index:03d}"
            )
            operation_ids.append(operation_id)
            self._start_stage(
                plan,
                QualificationStage.RETRIEVE,
                operation_id=operation_id,
                canary_id=canary.canary_id,
            )
            stage_started = time.perf_counter()
            result = adapter.retrieve(
                RetrievalRequest(
                    operation_id=operation_id,
                    scope=scope,
                    query=expectation.query,
                    top_k=plan.top_k,
                    threshold=plan.threshold,
                )
            )
            retrievals.append(
                CanaryRetrievalRecord(
                    canary_id=canary.canary_id,
                    scope_role=scope_role,
                    phase=phase,
                    expectation_index=index,
                    required_fragments=expectation.required_fragments,
                    result=result,
                )
            )
            assertion_id = None
            if phase != "own":
                assertion_id = (
                    f"retrieval_fragments:{phase}:{canary.canary_id}:{index:03d}"
                )
            assertions.append(
                assert_retrieval_fragments(
                    canary,
                    index,
                    result,
                    assertion_id=assertion_id,
                )
            )
            assertions.append(
                assert_retrieval_scope(
                    result,
                    scope,
                    assertion_id=(
                        f"retrieval_scope:{phase}:{canary.canary_id}:{index:03d}"
                    ),
                )
            )
            self._complete_stage(
                plan,
                QualificationStage.RETRIEVE,
                completed,
                latency_seconds=time.perf_counter() - stage_started,
                operation_id=operation_id,
                canary_id=canary.canary_id,
                payload={
                    "phase": phase,
                    "retrieval": result.model_dump(mode="json"),
                },
                deduplicate_stage=True,
            )
        return tuple(operation_ids)

    def _verify_retrieval_unchanged(
        self,
        plan: QualificationPlan,
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        *,
        phase: str,
        scope_role: Literal["primary", "secondary"],
        scope: MemoryScope,
        operation_ids: tuple[str, ...],
        snapshot_before: MemorySnapshot,
        visibility_barrier: FlushResult,
        snapshot_after: MemorySnapshot,
    ) -> RetrievalReadOnlyEvidence:
        self._start_stage(
            plan,
            QualificationStage.VERIFY_RETRIEVAL_UNCHANGED,
        )
        content_assertion = assert_snapshot_content_unchanged(
            snapshot_before,
            snapshot_after,
            assertion_id=f"retrieval_content_unchanged:{phase}",
            stage=QualificationStage.VERIFY_RETRIEVAL_UNCHANGED,
        )
        operational_assertion = assert_snapshot_full_state_unchanged(
            snapshot_before,
            snapshot_after,
            assertion_id=f"retrieval_operational_unchanged:{phase}",
            stage=QualificationStage.VERIFY_RETRIEVAL_UNCHANGED,
            required=False,
        )
        assertions.extend((content_assertion, operational_assertion))
        state_delta = compare_snapshot_state(snapshot_before, snapshot_after)
        if (
            content_assertion.passed != state_delta.content_unchanged
            or operational_assertion.passed
            != state_delta.full_state_unchanged
        ):
            raise QualificationInstrumentationError(
                "retrieval assertions disagree with structured state delta"
            )
        evidence = RetrievalReadOnlyEvidence(
            phase=phase,
            scope_role=scope_role,
            scope=scope,
            operation_ids=operation_ids,
            snapshot_before=snapshot_before,
            visibility_barrier=visibility_barrier,
            snapshot_after=snapshot_after,
            state_delta=state_delta,
        )
        self._complete_stage(
            plan,
            QualificationStage.VERIFY_RETRIEVAL_UNCHANGED,
            completed,
            payload={
                "phase": phase,
                "scope_role": scope_role,
                "operation_ids": list(operation_ids),
                "assertions": [
                    content_assertion.model_dump(mode="json"),
                    operational_assertion.model_dump(mode="json"),
                ],
                "state_delta": state_delta.model_dump(mode="json"),
            },
            deduplicate_stage=True,
        )
        return evidence

    def _retrieve_exclusion(
        self,
        plan: QualificationPlan,
        adapter: MemoryAdapter,
        scope: MemoryScope,
        foreign_canary: SyntheticCanary,
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        *,
        phase: str,
    ) -> tuple[str, ...]:
        operation_ids: list[str] = []
        for index, expectation in enumerate(foreign_canary.retrieval_expectations):
            operation_id = (
                f"{scope.run_id}:isolation:{phase}:"
                f"{foreign_canary.canary_id}:{index:03d}"
            )
            operation_ids.append(operation_id)
            self._start_stage(
                plan,
                QualificationStage.ISOLATION,
                operation_id=operation_id,
                canary_id=foreign_canary.canary_id,
            )
            stage_started = time.perf_counter()
            result = adapter.retrieve(
                RetrievalRequest(
                    operation_id=operation_id,
                    scope=scope,
                    query=expectation.query,
                    top_k=plan.top_k,
                    threshold=plan.threshold,
                )
            )
            assertions.append(
                assert_retrieval_excludes_canary(
                    foreign_canary,
                    result,
                    assertion_id=(
                        f"cross_scope_exclusion:{phase}:"
                        f"{foreign_canary.canary_id}:{index:03d}"
                    ),
                )
            )
            assertions.append(
                assert_retrieval_scope(
                    result,
                    scope,
                    assertion_id=(
                        f"cross_retrieval_scope:{phase}:"
                        f"{foreign_canary.canary_id}:{index:03d}"
                    ),
                )
            )
            self._complete_stage(
                plan,
                QualificationStage.ISOLATION,
                completed,
                latency_seconds=time.perf_counter() - stage_started,
                operation_id=operation_id,
                canary_id=foreign_canary.canary_id,
                payload={
                    "phase": phase,
                    "retrieval": result.model_dump(mode="json"),
                },
                deduplicate_stage=True,
            )
        return tuple(operation_ids)

    def _run_responder_stages(
        self,
        plan: QualificationPlan,
        canaries: tuple[SyntheticCanary, ...],
        retrievals: list[CanaryRetrievalRecord],
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        quiescence: list[QuiescenceEvidence],
        *,
        initial_snapshot: MemorySnapshot,
    ) -> tuple[ResponderStageEvidence, ...]:
        responder = self.responder
        if responder is None:
            raise AdapterConfigurationError(
                "responder dependency disappeared after preflight"
            )
        canary_by_id = {canary.canary_id: canary for canary in canaries}
        probe_retrievals = tuple(
            item
            for item in retrievals
            if item.scope_role == "primary" and item.phase == "own"
        )
        if not probe_retrievals:
            raise AdapterConfigurationError(
                "responder qualification has no primary probe retrieval"
            )

        evidence: list[ResponderStageEvidence] = []
        stable_snapshot = initial_snapshot
        for retrieval in probe_retrievals:
            canary = canary_by_id[retrieval.canary_id]
            expectation = canary.retrieval_expectations[
                retrieval.expectation_index
            ]
            operation_id = (
                f"{plan.scope.run_id}:respond:{canary.canary_id}:"
                f"{retrieval.expectation_index:03d}"
            )

            self._start_stage(
                plan,
                QualificationStage.CONTEXT,
                operation_id=operation_id,
                canary_id=canary.canary_id,
            )
            stage_started = time.perf_counter()
            context = self.context_assembler.assemble(
                retrieval.result.hits,
                query=expectation.query,
                token_budget=plan.context_token_budget,
                prompt_builder=self.prompt_builder,
            )
            assertions.append(
                assert_context_fragments(
                    canary,
                    retrieval.expectation_index,
                    context.content,
                )
            )
            prompt = self.prompt_builder.build(expectation.query, context)
            self._complete_stage(
                plan,
                QualificationStage.CONTEXT,
                completed,
                latency_seconds=time.perf_counter() - stage_started,
                operation_id=operation_id,
                canary_id=canary.canary_id,
                payload={
                    "context_record_ids": list(context.record_ids),
                    "context": context.content,
                    "context_sha256": context.sha256,
                    "context_token_count": context.token_count,
                    "context_token_counter": context.token_counter,
                    "context_truncated": context.truncated,
                    "prompt_version": prompt.version,
                    "prompt_sha256": prompt.sha256,
                    "predicted_prompt_tokens": prompt.input_token_count,
                    "prompt_token_counter": prompt.token_counter,
                },
                deduplicate_stage=True,
            )

            snapshot_before = stable_snapshot
            self._start_stage(
                plan,
                QualificationStage.RESPOND,
                operation_id=operation_id,
                canary_id=canary.canary_id,
            )
            stage_started = time.perf_counter()
            response = responder.respond(prompt.messages)
            if not isinstance(response, ResponderResult):
                raise ResponderError(
                    "qualification responder returned an invalid result"
                )
            if not response.content.strip():
                raise ResponderError(
                    "qualification responder returned an empty response"
                )
            if plan.responder_provider is None or plan.responder_model is None:
                raise AdapterConfigurationError(
                    "responder identity disappeared after preflight"
                )
            assertions.append(
                assert_responder_identity(
                    response,
                    provider=plan.responder_provider,
                    model=plan.responder_model,
                    model_digest=plan.responder_model_digest,
                    assertion_id=(
                        f"responder_identity_matches:{canary.canary_id}:"
                        f"{retrieval.expectation_index:03d}"
                    ),
                )
            )
            self._complete_stage(
                plan,
                QualificationStage.RESPOND,
                completed,
                latency_seconds=time.perf_counter() - stage_started,
                operation_id=operation_id,
                canary_id=canary.canary_id,
                payload={
                    "response": response.as_dict(),
                    "response_sha256": hashlib.sha256(
                        response.content.encode("utf-8")
                    ).hexdigest(),
                },
                deduplicate_stage=True,
            )

            visibility_barrier = self._flush_adapter(
                plan,
                self.adapter,
                completed,
                assertions,
                scope_role="primary",
                phase=(
                    f"post_response_{canary.canary_id}_"
                    f"{retrieval.expectation_index:03d}"
                ),
                assertion_id=(
                    f"post_response_flush_completed:{canary.canary_id}:"
                    f"{retrieval.expectation_index:03d}"
                ),
            )
            snapshot_after = self._wait_for_quiescence(
                plan,
                self.adapter,
                completed,
                assertions,
                quiescence,
                phase=(
                    f"post_response_{canary.canary_id}_"
                    f"{retrieval.expectation_index:03d}"
                ),
                scope=plan.scope,
            ).final_snapshot
            self._start_stage(
                plan,
                QualificationStage.VERIFY_RESPONSE_UNCHANGED,
                operation_id=operation_id,
                canary_id=canary.canary_id,
            )
            unchanged = assert_snapshot_unchanged(
                snapshot_before,
                snapshot_after,
                assertion_id=(
                    f"response_state_unchanged:{canary.canary_id}:"
                    f"{retrieval.expectation_index:03d}"
                ),
                stage=QualificationStage.VERIFY_RESPONSE_UNCHANGED,
            )
            assertions.append(unchanged)
            prompt_messages = tuple(
                MemoryMessage(role=message.role, content=message.content)
                for message in prompt.messages
            )
            item = ResponderStageEvidence(
                operation_id=operation_id,
                canary_id=canary.canary_id,
                expectation_index=retrieval.expectation_index,
                retrieval_operation_id=retrieval.result.operation_id,
                context_record_ids=context.record_ids,
                context=context.content,
                context_sha256=context.sha256,
                context_token_count=context.token_count,
                context_token_counter=context.token_counter,
                context_truncated=context.truncated,
                prompt_version=prompt.version,
                prompt_sha256=prompt.sha256,
                prompt_messages=prompt_messages,
                predicted_prompt_tokens=prompt.input_token_count,
                prompt_token_counter=prompt.token_counter,
                response_content=response.content,
                response_sha256=hashlib.sha256(
                    response.content.encode("utf-8")
                ).hexdigest(),
                response_provider=response.provider,
                response_model=response.model,
                response_model_digest=response.model_digest,
                prompt_tokens=response.prompt_tokens,
                prompt_token_delta=(
                    None
                    if response.prompt_tokens is None
                    else response.prompt_tokens - prompt.input_token_count
                ),
                completion_tokens=response.completion_tokens,
                finish_reason=response.finish_reason,
                latency_seconds=response.latency_seconds,
                raw_response=response.raw_response,
                snapshot_before=snapshot_before,
                visibility_barrier=visibility_barrier,
                snapshot_after=snapshot_after,
            )
            evidence.append(item)
            stable_snapshot = snapshot_after
            self._complete_stage(
                plan,
                QualificationStage.VERIFY_RESPONSE_UNCHANGED,
                completed,
                operation_id=operation_id,
                canary_id=canary.canary_id,
                payload={
                    "assertion": unchanged.model_dump(mode="json"),
                    "response_sha256": item.response_sha256,
                },
                deduplicate_stage=True,
            )
        return tuple(evidence)

    def _preflight(
        self,
        plan: QualificationPlan,
    ) -> tuple[
        CommonQualification,
        tuple[SyntheticCanary, ...],
        PolicyDispatcher,
    ]:
        self._start_stage(plan, QualificationStage.PREFLIGHT)
        qualification = self.suite.qualification(plan.qualification_id)
        if plan.policy not in qualification.supported_policies:
            raise AdapterConfigurationError(
                f"qualification {plan.qualification_id!r} does not support "
                f"policy {plan.policy.value!r}"
            )
        if qualification.requires_responder != plan.responder_required:
            raise AdapterConfigurationError(
                "qualification responder requirement does not match its plan"
            )
        if qualification.requires_responder:
            if self.responder is None:
                raise AdapterConfigurationError(
                    "responder qualification requires a responder"
                )
            if not isinstance(self.responder, QualificationResponder):
                raise AdapterConfigurationError(
                    "responder must implement verify() and respond(messages)"
                )
            try:
                self.responder.verify()
            except ResponderError:
                raise
            except Exception as exc:
                raise ResponderError(
                    f"qualification responder verification failed: {exc}"
                ) from exc
        elif self.responder is not None:
            raise AdapterConfigurationError(
                "a responder may only be supplied to a responder qualification"
            )
        if qualification.requires_secondary_scope:
            if plan.secondary_scope is None:
                raise AdapterConfigurationError(
                    "secondary-scope qualification requires secondary_scope"
                )
            if self.secondary_adapter_factory is None:
                raise AdapterConfigurationError(
                    "secondary-scope qualification requires an adapter factory"
                )
            if not plan.cleanup:
                raise AdapterConfigurationError(
                    "secondary-scope qualification requires cleanup so reset "
                    "containment can be tested"
                )
        elif plan.secondary_scope is not None:
            raise AdapterConfigurationError(
                "secondary_scope is only valid for a secondary-scope qualification"
            )
        unknown_options = set(plan.native_options_by_canary) - set(
            qualification.canary_ids
        )
        if unknown_options:
            raise AdapterConfigurationError(
                "native options reference canaries outside the qualification: "
                + ", ".join(sorted(unknown_options))
            )
        missing = tuple(
            capability
            for capability in qualification.required_capabilities
            if not self.adapter.capabilities[capability]
        )
        if missing:
            raise UnsupportedCapabilityError(
                "adapter does not satisfy qualification capabilities: "
                + ", ".join(missing),
                missing=missing,
            )
        dispatcher = PolicyDispatcher(
            self.adapter,
            plan.policy,
            guard=self.guard,
        )
        canaries = self.suite.qualification_canaries(plan.qualification_id)
        return qualification, canaries, dispatcher

    @staticmethod
    def _write_request(
        plan: QualificationPlan,
        canary: SyntheticCanary,
        scope: MemoryScope | None = None,
    ) -> PolicyWriteRequest:
        target_scope = scope or plan.scope
        return PolicyWriteRequest(
            operation_id=f"{target_scope.run_id}:write:{canary.canary_id}",
            scope=target_scope,
            messages=canary.messages,
            source_event_id=canary.source_event_id,
            memory_type=canary.memory_type,
            metadata={
                **canary.metadata,
                "qualification_canary_id": canary.canary_id,
                "qualification_marker": canary.marker,
            },
            native_options=plan.native_options_by_canary.get(canary.canary_id, {}),
        )

    def _reset(
        self,
        plan: QualificationPlan,
        completed: list[QualificationStage],
        assertions: list[QualificationAssertion],
        *,
        adapter: MemoryAdapter | None = None,
        assertion_id: str = "reset_verified_empty",
        operation_id: str | None = None,
    ) -> EmptyStateResult:
        target = adapter or self.adapter
        self._start_stage(
            plan,
            QualificationStage.RESET,
            operation_id=operation_id,
        )
        stage_started = time.perf_counter()
        result = target.reset_and_verify_empty()
        assertions.append(
            assert_empty_state(
                result,
                assertion_id=assertion_id,
                stage=QualificationStage.RESET,
            )
        )
        self._complete_stage(
            plan,
            QualificationStage.RESET,
            completed,
            latency_seconds=time.perf_counter() - stage_started,
            operation_id=operation_id,
            payload={"empty_state": result.model_dump(mode="json")},
            deduplicate_stage=True,
        )
        return result

    def _start_stage(
        self,
        plan: QualificationPlan,
        stage: QualificationStage,
        *,
        operation_id: str | None = None,
        canary_id: str | None = None,
    ) -> None:
        self._active_stage = stage
        self._emit(
            plan,
            stage=stage,
            status=QualificationEventStatus.STARTED,
            operation_id=operation_id,
            canary_id=canary_id,
        )

    def _complete_stage(
        self,
        plan: QualificationPlan,
        stage: QualificationStage,
        completed: list[QualificationStage],
        *,
        latency_seconds: float = 0.0,
        operation_id: str | None = None,
        canary_id: str | None = None,
        payload: dict[str, Any] | None = None,
        deduplicate_stage: bool = False,
    ) -> None:
        if not deduplicate_stage or stage not in completed:
            completed.append(stage)
        self._emit(
            plan,
            stage=stage,
            status=QualificationEventStatus.SUCCEEDED,
            operation_id=operation_id,
            canary_id=canary_id,
            latency_seconds=latency_seconds,
            payload=payload or {},
        )

    def _emit(
        self,
        plan: QualificationPlan,
        *,
        stage: QualificationStage,
        status: QualificationEventStatus,
        operation_id: str | None = None,
        canary_id: str | None = None,
        latency_seconds: float = 0.0,
        payload: dict[str, Any] | None = None,
        failure_class: QualificationFailureClass | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self._sequence += 1
        if self._event_sink_error is not None:
            return
        event = QualificationEvent(
            sequence=self._sequence,
            experiment_id=plan.scope.experiment_id,
            run_id=plan.scope.run_id,
            mechanism=plan.mechanism,
            policy=plan.policy,
            qualification_id=plan.qualification_id,
            stage=stage,
            status=status,
            operation_id=operation_id,
            canary_id=canary_id,
            latency_seconds=latency_seconds,
            payload=payload or {},
            failure_class=failure_class,
            error_type=error_type,
            error_message=error_message,
        )
        try:
            self.event_sink.emit(event)
        except Exception as exc:
            self._event_sink_error = exc
            raise QualificationInstrumentationError(
                "qualification event sink failed"
            ) from exc


def _error_message(exc: Exception) -> str:
    return str(exc).strip() or "exception did not provide an error message"


def _classify_failure(
    exc: Exception,
    stage: QualificationStage,
) -> QualificationFailureClass:
    if isinstance(
        exc,
        (
            AdapterConfigurationError,
            UnsupportedCapabilityError,
            PolicyDependencyError,
            KeyError,
            ValueError,
        ),
    ):
        return QualificationFailureClass.CONFIGURATION_INVALID
    if isinstance(exc, AdapterUnavailableError):
        return QualificationFailureClass.SERVICE_UNAVAILABLE
    if isinstance(exc, NamespaceNotEmptyError):
        return QualificationFailureClass.NAMESPACE_NOT_EMPTY
    if isinstance(exc, WriteVisibilityTimeout):
        return QualificationFailureClass.WRITE_VISIBILITY_TIMEOUT
    if isinstance(exc, QualificationQuiescenceTimeout):
        return QualificationFailureClass.STATE_STABILIZATION_TIMEOUT
    if isinstance(exc, WriteTransportError):
        return QualificationFailureClass.WRITE_TRANSPORT_FAILED
    if isinstance(exc, RetrievalTransportError):
        return QualificationFailureClass.RETRIEVAL_TRANSPORT_FAILED
    if isinstance(exc, ResponderError):
        return QualificationFailureClass.RESPONSE_TRANSPORT_FAILED
    if isinstance(exc, StateInspectionError):
        return QualificationFailureClass.STATE_INSPECTION_FAILED
    if isinstance(exc, ResetVerificationError):
        return QualificationFailureClass.RESET_VERIFICATION_FAILED
    if isinstance(exc, QualificationInstrumentationError):
        return QualificationFailureClass.INSTRUMENTATION_INCOMPLETE
    if stage == QualificationStage.WRITE:
        return QualificationFailureClass.POLICY_FAILED
    if stage == QualificationStage.RESPOND:
        return QualificationFailureClass.RESPONSE_TRANSPORT_FAILED
    return QualificationFailureClass.INSTRUMENTATION_INCOMPLETE
