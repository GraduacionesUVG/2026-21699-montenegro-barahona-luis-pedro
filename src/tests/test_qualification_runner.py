import hashlib
from collections.abc import Sequence
from typing import Any

import pytest

from membench.canary_data import CanarySuite
from membench.contract import Mechanism, WritePolicy
from membench.memory.capabilities import AdapterCapabilities
from membench.memory.models import (
    AdapterProvenance,
    ConversationWriteRequest,
    DirectWriteRequest,
    EmptyStateResult,
    FlushResult,
    MemoryRecord,
    MemoryScope,
    MemorySnapshot,
    MemoryType,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResult,
    ScoreKind,
    WriteAction,
    WriteOutcome,
    WriteResult,
    WriteStatus,
)
from membench.qualification.events import InMemoryQualificationEventSink
from membench.qualification.models import (
    QualificationEventStatus,
    QualificationFailureClass,
    QualificationPlan,
    QualificationStage,
)
from membench.qualification.runner import QualificationRunner
from membench.responder import ChatMessage, ResponderResult


def _scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="qualification-tests",
        run_id="run-001",
        user_id="user-001",
        namespace=("membench", "qualification", "run-001"),
    )


def _secondary_scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="qualification-tests",
        run_id="run-002",
        user_id="user-002",
        namespace=("membench", "qualification", "run-002"),
    )


def _capabilities(*, semantic: bool = True) -> AdapterCapabilities:
    return AdapterCapabilities(
        direct_write=True,
        native_conversation_write=True,
        semantic_memory=semantic,
        episodic_memory=False,
        procedural_memory=False,
        working_memory=False,
        updates=False,
        deletion=True,
        ttl=False,
        metadata_filters=False,
        threshold=False,
        reranking=False,
        hybrid_search=False,
        associations=False,
    )


class RecordingQualificationAdapter:
    def __init__(
        self,
        *,
        semantic: bool = True,
        drop_retrieval: bool = False,
        preexisting: bool = False,
    ) -> None:
        self._capabilities = _capabilities(semantic=semantic)
        self.drop_retrieval = drop_retrieval
        self.preexisting = preexisting
        self.scope: MemoryScope | None = None
        self.records: list[MemoryRecord] = []
        self.initialized = False
        self.reset_called = False
        self.closed = False

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def initialize(self, scope: MemoryScope) -> None:
        self.scope = scope
        self.initialized = True
        if self.preexisting:
            self._store(
                "preexisting-write",
                "preexisting-event",
                "Pre-existing state must not be deleted.",
                MemoryType.SEMANTIC,
            )

    def verify_empty(self) -> EmptyStateResult:
        assert self.scope is not None
        count = len(self.records)
        return EmptyStateResult(
            scope=self.scope,
            empty=count == 0,
            inspected_count=count,
            retrieved_count=0,
        )

    def _store(
        self,
        operation_id: str,
        source_event_id: str,
        text: str,
        memory_type: MemoryType,
    ) -> WriteResult:
        assert self.scope is not None
        vendor_id = f"record-{len(self.records) + 1}"
        record = MemoryRecord(
            canonical_id=f"fake-{vendor_id}",
            vendor_id=vendor_id,
            scope=self.scope,
            text=text,
            memory_type=memory_type,
            source_event_ids=(source_event_id,),
        )
        self.records.append(record)
        return WriteResult(
            operation_id=operation_id,
            status=WriteStatus.SUCCEEDED,
            outcomes=(
                WriteOutcome(
                    action=WriteAction.CREATED,
                    canonical_id=record.canonical_id,
                    vendor_id=vendor_id,
                    text=text,
                ),
            ),
        )

    def write_direct(self, request: DirectWriteRequest) -> WriteResult:
        return self._store(
            request.operation_id,
            request.source_event_id,
            request.text,
            request.memory_type,
        )

    def write_conversation(self, request: ConversationWriteRequest) -> WriteResult:
        text = "\n".join(message.content for message in request.messages)
        return self._store(
            request.operation_id,
            request.source_event_id,
            text,
            MemoryType.SEMANTIC,
        )

    def flush_and_wait(self) -> FlushResult:
        return FlushResult(completed=True, visible_records=tuple(self.records))

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        visible = [] if self.drop_retrieval else self.records[: request.top_k]
        hits = tuple(
            RetrievalHit(
                record=record,
                rank=index,
                raw_score=1.0,
                score_kind=ScoreKind.SIMILARITY,
            )
            for index, record in enumerate(visible, start=1)
        )
        return RetrievalResult(
            operation_id=request.operation_id,
            query_sha256=request.query_sha256,
            request_top_k=request.top_k,
            threshold=request.threshold,
            hits=hits,
            latency_seconds=0,
        )

    def inspect_state(self) -> MemorySnapshot:
        assert self.scope is not None
        return MemorySnapshot(scope=self.scope, records=tuple(self.records))

    def reset_and_verify_empty(self) -> EmptyStateResult:
        assert self.scope is not None
        self.reset_called = True
        self.records.clear()
        return EmptyStateResult(
            scope=self.scope,
            empty=True,
            inspected_count=0,
            retrieved_count=0,
        )

    def export_provenance(self) -> AdapterProvenance:
        assert self.scope is not None
        return AdapterProvenance(
            mechanism="telemem",
            mechanism_version="test",
            adapter_version="test",
            storage_backend="fake",
            namespace=self.scope.namespace,
        )

    def close(self) -> None:
        self.closed = True


class RecordingResponder:
    def __init__(self) -> None:
        self.verified = False
        self.messages: tuple[ChatMessage, ...] = ()

    def verify(self) -> None:
        self.verified = True

    def respond(self, messages: Sequence[ChatMessage]) -> ResponderResult:
        self.messages = tuple(messages)
        return ResponderResult(
            content="The field notebook marker is HELIOTROPE-7Q2M.",
            provider="ollama",
            model="qualification-model",
            model_digest="a" * 64,
            prompt_tokens=32,
            completion_tokens=8,
            finish_reason="stop",
            latency_seconds=0.01,
            raw_response={"done": True},
        )


class AdvancingClock:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += seconds


def _plan(**updates: Any) -> QualificationPlan:
    values: dict[str, Any] = {
        "qualification_id": "baseline_semantic_v1",
        "mechanism": Mechanism.TELEMEM,
        "policy": WritePolicy.DIRECT,
        "scope": _scope(),
        "top_k": 5,
        "quiescence": {
            "timeout_seconds": 0.01,
            "poll_interval_seconds": 0.001,
            "stable_window_seconds": 0,
            "minimum_stable_samples": 1,
        },
    }
    values.update(updates)
    if values.get("responder_required"):
        values.setdefault("responder_provider", "ollama")
        values.setdefault("responder_model", "qualification-model")
        values.setdefault("responder_model_digest", "a" * 64)
    return QualificationPlan(**values)


def test_runner_executes_and_records_complete_primary_lifecycle() -> None:
    adapter = RecordingQualificationAdapter()
    events = InMemoryQualificationEventSink()
    runner = QualificationRunner(
        adapter,
        CanarySuite.load(),
        event_sink=events,
    )

    report = runner.run(_plan())

    assert report.summary.passed
    assert all(item.passed for item in report.summary.assertions)
    assert len(report.dispatches) == 1
    assert len(report.retrievals) == 1
    assert len(report.retrieval_read_only) == 1
    assert (
        report.retrieval_read_only[0].snapshot_before.records
        == report.retrieval_read_only[0].snapshot_after.records
    )
    assert report.snapshot_before is not None
    assert report.snapshot_before.record_count == 0
    assert report.snapshot_after is not None
    assert report.snapshot_after.record_count == 1
    assert report.provenance is not None
    assert report.reset_result is not None and report.reset_result.empty
    assert adapter.reset_called
    assert adapter.closed
    assert [event.sequence for event in events.events] == list(
        range(1, len(events.events) + 1)
    )
    assert events.events[-1].stage == QualificationStage.COMPLETE
    assert events.events[-1].status == QualificationEventStatus.SUCCEEDED


def test_runner_rejects_missing_capability_before_initialization() -> None:
    adapter = RecordingQualificationAdapter(semantic=False)
    events = InMemoryQualificationEventSink()
    runner = QualificationRunner(
        adapter,
        CanarySuite.load(),
        event_sink=events,
    )

    report = runner.run(_plan())

    assert not report.summary.passed
    assert (
        report.summary.failure_class
        == QualificationFailureClass.CONFIGURATION_INVALID
    )
    assert not adapter.initialized
    assert not adapter.reset_called
    assert adapter.closed


def test_runner_records_assertion_failure_and_still_resets() -> None:
    adapter = RecordingQualificationAdapter(drop_retrieval=True)
    runner = QualificationRunner(adapter, CanarySuite.load())

    report = runner.run(_plan())

    assert not report.summary.passed
    assert report.summary.failure_class == QualificationFailureClass.ASSERTION_FAILED
    assert any(
        item.assertion_id.startswith("retrieval_fragments") and not item.passed
        for item in report.summary.assertions
    )
    assert adapter.reset_called
    assert adapter.closed


def test_runner_detects_memory_created_during_retrieval() -> None:
    class RetrievalMutatingAdapter(RecordingQualificationAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.mutated = False

        def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
            result = super().retrieve(request)
            if not self.mutated:
                self.mutated = True
                self._store(
                    "retrieval-side-effect",
                    "probe-writeback-event",
                    "A retrieval side effect must invalidate qualification.",
                    MemoryType.SEMANTIC,
                )
            return result

    adapter = RetrievalMutatingAdapter()
    runner = QualificationRunner(adapter, CanarySuite.load())

    report = runner.run(_plan())

    assert not report.summary.passed
    assert report.summary.failure_class == QualificationFailureClass.ASSERTION_FAILED
    assert any(
        item.assertion_id == "retrieval_content_unchanged:primary_probe_context"
        and not item.passed
        and item.required
        for item in report.summary.assertions
    )
    assert report.retrieval_read_only[0].state_delta.created_record_ids == (
        "fake-record-2",
    )
    assert adapter.reset_called
    assert adapter.closed


def test_runner_records_retrieval_reinforcement_without_failing() -> None:
    class ReinforcingRetrievalAdapter(RecordingQualificationAdapter):
        def write_direct(self, request: DirectWriteRequest) -> WriteResult:
            result = super().write_direct(request)
            record = self.records[0]
            self.records[0] = record.model_copy(
                update={"attributes": {"importance": 0.5}}
            )
            return result

        def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
            result = super().retrieve(request)
            record = self.records[0]
            self.records[0] = record.model_copy(
                update={"attributes": {"importance": 0.575}}
            )
            return result

    adapter = ReinforcingRetrievalAdapter()
    runner = QualificationRunner(adapter, CanarySuite.load())

    report = runner.run(_plan())

    content = next(
        item
        for item in report.summary.assertions
        if item.assertion_id
        == "retrieval_content_unchanged:primary_probe_context"
    )
    operational = next(
        item
        for item in report.summary.assertions
        if item.assertion_id
        == "retrieval_operational_unchanged:primary_probe_context"
    )
    assert report.summary.passed
    assert content.passed and content.required
    assert not operational.passed and not operational.required
    state_delta = report.retrieval_read_only[0].state_delta
    assert state_delta.content_unchanged
    assert not state_delta.full_state_unchanged
    assert state_delta.operational_changes[0].changed_fields == (
        "attributes.importance",
    )
    assert state_delta.importance_changes[0].delta == pytest.approx(0.075)
    assert adapter.reset_called
    assert adapter.closed


def test_runner_assembles_hashed_context_and_proves_response_is_read_only() -> None:
    adapter = RecordingQualificationAdapter()
    responder = RecordingResponder()
    runner = QualificationRunner(
        adapter,
        CanarySuite.load(),
        responder=responder,
    )

    report = runner.run(
        _plan(
            qualification_id="responder_read_only_v1",
            responder_required=True,
            context_token_budget=256,
        )
    )

    assert report.summary.passed
    assert responder.verified
    assert len(report.responder_stages) == 1
    assert [item.phase for item in report.quiescence] == [
        "pre_retrieval",
        "post_retrieval",
        "post_response_semantic_preference_v1_000",
    ]
    assert all(item.stabilized for item in report.quiescence)
    evidence = report.responder_stages[0]
    assert "HELIOTROPE-7Q2M" in evidence.context
    assert evidence.context_sha256 == hashlib.sha256(
        evidence.context.encode("utf-8")
    ).hexdigest()
    assert evidence.response_sha256 == hashlib.sha256(
        evidence.response_content.encode("utf-8")
    ).hexdigest()
    assert evidence.snapshot_before.records == evidence.snapshot_after.records
    assert evidence.predicted_prompt_tokens <= 256
    assert evidence.prompt_token_counter == "regex_lexical_v1"
    assert evidence.prompt_token_delta == (
        evidence.prompt_tokens - evidence.predicted_prompt_tokens
    )
    assert "HELIOTROPE-7Q2M" in responder.messages[1].content
    assert QualificationStage.CONTEXT in report.summary.completed_stages
    assert QualificationStage.RESPOND in report.summary.completed_stages
    assert (
        QualificationStage.VERIFY_RESPONSE_UNCHANGED
        in report.summary.completed_stages
    )
    assert QualificationStage.QUIESCENCE in report.summary.completed_stages


def test_runner_detects_memory_created_during_response() -> None:
    adapter = RecordingQualificationAdapter()

    class MutatingResponder(RecordingResponder):
        def respond(self, messages: Sequence[ChatMessage]) -> ResponderResult:
            result = super().respond(messages)
            adapter._store(
                "response-side-effect",
                "response-writeback-event",
                result.content,
                MemoryType.SEMANTIC,
            )
            return result

    runner = QualificationRunner(
        adapter,
        CanarySuite.load(),
        responder=MutatingResponder(),
    )

    report = runner.run(
        _plan(
            qualification_id="responder_read_only_v1",
            responder_required=True,
        )
    )

    assert not report.summary.passed
    assert report.summary.failure_class == QualificationFailureClass.ASSERTION_FAILED
    assert any(
        item.assertion_id
        == "response_state_unchanged:semantic_preference_v1:000"
        and not item.passed
        for item in report.summary.assertions
    )
    assert adapter.reset_called
    assert adapter.closed


def test_runner_stabilizes_delayed_write_derivatives_before_response() -> None:
    class DelayedDerivativeAdapter(RecordingQualificationAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.inspect_calls = 0
            self.derivative_created = False

        def inspect_state(self) -> MemorySnapshot:
            self.inspect_calls += 1
            if (
                self.records
                and not self.derivative_created
                and self.inspect_calls == 3
            ):
                self.derivative_created = True
                self._store(
                    "native-derivative",
                    "native-derivative-event",
                    "Derived from the original field notebook conversation.",
                    MemoryType.SEMANTIC,
                )
            return super().inspect_state()

    clock = AdvancingClock()
    adapter = DelayedDerivativeAdapter()
    responder = RecordingResponder()
    runner = QualificationRunner(
        adapter,
        CanarySuite.load(),
        responder=responder,
        quiescence_clock=clock,
        quiescence_sleep=clock.sleep,
    )

    report = runner.run(
        _plan(
            qualification_id="responder_read_only_v1",
            responder_required=True,
            quiescence={
                "timeout_seconds": 3,
                "poll_interval_seconds": 0.25,
                "stable_window_seconds": 0.5,
                "minimum_stable_samples": 3,
            },
        )
    )

    assert report.summary.passed
    assert adapter.derivative_created
    assert report.quiescence[0].phase == "pre_retrieval"
    assert any(
        item.changed_since_previous
        for item in report.quiescence[0].observations
    )
    assert report.quiescence[0].final_snapshot.record_count == 2
    assert report.responder_stages[0].snapshot_before.record_count == 2
    assert report.responder_stages[0].snapshot_after.record_count == 2


def test_runner_fails_when_memory_state_never_reaches_quiescence() -> None:
    class ContinuouslyMutatingAdapter(RecordingQualificationAdapter):
        def inspect_state(self) -> MemorySnapshot:
            if self.records:
                self._store(
                    f"background-write-{len(self.records)}",
                    f"background-event-{len(self.records)}",
                    f"Background mutation {len(self.records)}",
                    MemoryType.SEMANTIC,
                )
            return super().inspect_state()

    clock = AdvancingClock()
    adapter = ContinuouslyMutatingAdapter()
    responder = RecordingResponder()
    runner = QualificationRunner(
        adapter,
        CanarySuite.load(),
        responder=responder,
        quiescence_clock=clock,
        quiescence_sleep=clock.sleep,
    )

    report = runner.run(
        _plan(
            qualification_id="responder_read_only_v1",
            responder_required=True,
            quiescence={
                "timeout_seconds": 0.5,
                "poll_interval_seconds": 0.1,
                "stable_window_seconds": 0.2,
                "minimum_stable_samples": 2,
            },
        )
    )

    assert not report.summary.passed
    assert (
        report.summary.failure_class
        == QualificationFailureClass.STATE_STABILIZATION_TIMEOUT
    )
    assert len(report.quiescence) == 1
    assert not report.quiescence[0].stabilized
    assert responder.messages == ()
    assert adapter.reset_called
    assert adapter.closed


def test_responder_qualification_requires_responder_dependency() -> None:
    adapter = RecordingQualificationAdapter()
    runner = QualificationRunner(adapter, CanarySuite.load())

    report = runner.run(
        _plan(
            qualification_id="responder_read_only_v1",
            responder_required=True,
        )
    )

    assert not report.summary.passed
    assert (
        report.summary.failure_class
        == QualificationFailureClass.CONFIGURATION_INVALID
    )
    assert not adapter.initialized


def test_responder_transport_failure_is_classified_and_cleaned_up() -> None:
    class FailingResponder(RecordingResponder):
        def respond(self, messages: Sequence[ChatMessage]) -> ResponderResult:
            del messages
            raise RuntimeError("synthetic responder transport failure")

    adapter = RecordingQualificationAdapter()
    runner = QualificationRunner(
        adapter,
        CanarySuite.load(),
        responder=FailingResponder(),
    )

    report = runner.run(
        _plan(
            qualification_id="responder_read_only_v1",
            responder_required=True,
        )
    )

    assert not report.summary.passed
    assert (
        report.summary.failure_class
        == QualificationFailureClass.RESPONSE_TRANSPORT_FAILED
    )
    assert adapter.reset_called
    assert adapter.closed


def test_runner_fails_fast_for_secondary_scope_profile() -> None:
    adapter = RecordingQualificationAdapter()
    runner = QualificationRunner(adapter, CanarySuite.load())

    report = runner.run(
        _plan(qualification_id="namespace_isolation_v1")
    )

    assert not report.summary.passed
    assert (
        report.summary.failure_class
        == QualificationFailureClass.CONFIGURATION_INVALID
    )
    assert "secondary-scope" in (report.summary.failure_message or "")
    assert not adapter.initialized


def test_runner_executes_bidirectional_secondary_scope_isolation() -> None:
    primary = RecordingQualificationAdapter()
    secondary_adapters: list[RecordingQualificationAdapter] = []

    def create_secondary() -> RecordingQualificationAdapter:
        adapter = RecordingQualificationAdapter()
        secondary_adapters.append(adapter)
        return adapter

    runner = QualificationRunner(
        primary,
        CanarySuite.load(),
        secondary_adapter_factory=create_secondary,
    )

    report = runner.run(
        _plan(
            qualification_id="namespace_isolation_v1",
            secondary_scope=_secondary_scope(),
        )
    )

    assert report.summary.passed
    assert all(item.passed for item in report.summary.assertions)
    assert report.snapshot_after is not None
    assert report.snapshot_after.record_count == 1
    assert report.secondary_snapshot_before is not None
    assert report.secondary_snapshot_before.record_count == 0
    assert report.secondary_snapshot_after is not None
    assert report.secondary_snapshot_after.record_count == 1
    assert report.reset_result is not None and report.reset_result.empty
    assert (
        report.secondary_reset_result is not None
        and report.secondary_reset_result.empty
    )
    assert report.secondary_provenance is not None
    assert {
        item.phase for item in report.retrieval_read_only
    } == {
        "primary_isolation_queries",
        "secondary_isolation_queries",
        "secondary_post_primary_reset_probe",
    }
    assert {item.scope_role for item in report.dispatches} == {
        "primary",
        "secondary",
    }
    assert any(
        item.scope_role == "secondary"
        and item.phase == "secondary_after_primary_reset"
        for item in report.retrievals
    )
    assert len(secondary_adapters) == 1
    assert secondary_adapters[0].reset_called
    assert secondary_adapters[0].closed
    assert primary.reset_called
    assert primary.closed


def test_runner_detects_cross_scope_storage_leakage() -> None:
    shared_records: list[MemoryRecord] = []
    primary = RecordingQualificationAdapter()
    primary.records = shared_records

    def create_leaky_secondary() -> RecordingQualificationAdapter:
        adapter = RecordingQualificationAdapter()
        adapter.records = shared_records
        return adapter

    runner = QualificationRunner(
        primary,
        CanarySuite.load(),
        secondary_adapter_factory=create_leaky_secondary,
    )

    report = runner.run(
        _plan(
            qualification_id="namespace_isolation_v1",
            secondary_scope=_secondary_scope(),
        )
    )

    assert not report.summary.passed
    assert report.summary.failure_class == QualificationFailureClass.ASSERTION_FAILED
    assert any(
        item.assertion_id.startswith("cross_scope_exclusion")
        and not item.passed
        for item in report.summary.assertions
    )


def test_runner_does_not_reset_a_scope_that_started_nonempty() -> None:
    adapter = RecordingQualificationAdapter(preexisting=True)
    runner = QualificationRunner(adapter, CanarySuite.load())

    report = runner.run(_plan())

    assert not report.summary.passed
    assert report.summary.failure_class == QualificationFailureClass.NAMESPACE_NOT_EMPTY
    assert not adapter.reset_called
    assert len(adapter.records) == 1
    assert adapter.closed


def test_event_sink_failure_still_cleans_written_state() -> None:
    class FailingSink:
        def __init__(self) -> None:
            self.calls = 0

        def emit(self, _event) -> None:
            self.calls += 1
            # The tenth event is WRITE/succeeded, after the adapter side effect.
            if self.calls == 10:
                raise OSError("synthetic event sink failure")

    adapter = RecordingQualificationAdapter()
    runner = QualificationRunner(
        adapter,
        CanarySuite.load(),
        event_sink=FailingSink(),
    )

    report = runner.run(_plan())

    assert not report.summary.passed
    assert (
        report.summary.failure_class
        == QualificationFailureClass.INSTRUMENTATION_INCOMPLETE
    )
    assert adapter.reset_called
    assert adapter.records == []
    assert adapter.closed
