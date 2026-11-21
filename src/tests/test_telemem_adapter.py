from types import SimpleNamespace
from typing import Any

import pytest

from membench.memory.adapters import telemem_adapter as adapter_module
from membench.memory.adapters.telemem_adapter import TeleMemAdapter
from membench.memory.errors import (
    AdapterConfigurationError,
    NamespaceNotEmptyError,
    WriteTransportError,
)
from membench.memory.models import (
    ConversationWriteRequest,
    DirectWriteRequest,
    MemoryMessage,
    MemoryScope,
    MemoryType,
    RetrievalRequest,
    ScoreKind,
    WriteAction,
    WriteStatus,
)
from membench.memory.protocol import MemoryAdapter
from membench.telemem import TeleMemRecord


def _scope(run_id: str = "run-001") -> MemoryScope:
    return MemoryScope(
        experiment_id="experiment-001",
        run_id=run_id,
        user_id="persona-001",
        namespace=("membench", "telemem", run_id, "persona-001"),
    )


def _record(
    vendor_id: str = "vendor-001",
    *,
    event: str | None = "ADD",
    metadata: dict[str, Any] | None = None,
    score: float | None = None,
) -> TeleMemRecord:
    return TeleMemRecord(
        vendor_id=vendor_id,
        text="The user prefers dark mode.",
        score=score,
        event=event,
        user_id="persona-001",
        run_id="run-001",
        role="user",
        created_at="2026-09-13T00:00:00Z",
        updated_at=None,
        metadata=metadata or {},
        raw_value={"id": vendor_id, "event": event},
    )


class FakeRuntime:
    instances: list["FakeRuntime"] = []

    def __init__(self, settings: Any, run_id: str, *, user_id: str) -> None:
        self.settings = settings
        self.run_id = run_id
        self.user_id = user_id
        self.initialize_require_empty: bool | None = None
        self.direct_calls: list[dict[str, Any]] = []
        self.conversation_calls: list[dict[str, Any]] = []
        self.direct_records = [_record()]
        self.conversation_records = [_record("vendor-002", event="UPDATE")]
        self.flush_records = [_record()]
        self.retrieval_records = [
            _record(
                metadata={
                    "membench_memory_type": "semantic",
                    "membench_source_event_id": "event-001",
                },
                score=0.9,
            )
        ]
        self.inspection_records = list(self.retrieval_records)
        self.retrieve_calls: list[dict[str, Any]] = []
        self.inspect_limits: list[int] = []
        self.reset_calls = 0
        self.pending_count_value = 0
        self.closed = False
        self.__class__.instances.append(self)

    @property
    def pending_count(self) -> int:
        return self.pending_count_value

    def initialize(self, *, require_empty: bool) -> "FakeRuntime":
        self.initialize_require_empty = require_empty
        return self

    def store_memory(self, content: str, **kwargs: Any) -> list[TeleMemRecord]:
        self.direct_calls.append({"content": content, **kwargs})
        return self.direct_records

    def submit_conversation(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> list[TeleMemRecord]:
        self.conversation_calls.append({"messages": messages, **kwargs})
        return self.conversation_records

    def flush_and_wait(self) -> list[TeleMemRecord]:
        return self.flush_records

    def retrieve(self, query: str, *, top_k: int) -> list[TeleMemRecord]:
        self.retrieve_calls.append({"query": query, "top_k": top_k})
        return self.retrieval_records[:top_k]

    def inspect_state(self, *, limit: int) -> list[TeleMemRecord]:
        self.inspect_limits.append(limit)
        return self.inspection_records[:limit]

    def reset_and_verify_empty(self) -> None:
        self.reset_calls += 1
        self.inspection_records = []
        self.retrieval_records = []

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def initialized_adapter(monkeypatch: pytest.MonkeyPatch) -> TeleMemAdapter:
    FakeRuntime.instances.clear()
    monkeypatch.setattr(adapter_module, "TeleMemRuntime", FakeRuntime)
    settings = SimpleNamespace(retrieval_threshold=0.2, rerank=False)
    adapter = TeleMemAdapter(settings=settings)  # type: ignore[arg-type]
    adapter.initialize(_scope())
    return adapter


def test_direct_write_maps_request_and_native_result(
    initialized_adapter: TeleMemAdapter,
) -> None:
    request = DirectWriteRequest(
        operation_id="write-direct-001",
        scope=_scope(),
        text="The user prefers dark mode.",
        memory_type=MemoryType.SEMANTIC,
        source_event_id="event-001",
        metadata={"condition": "clean"},
    )

    result = initialized_adapter.write_direct(request)
    runtime = FakeRuntime.instances[0]
    call = runtime.direct_calls[0]

    assert result.status == WriteStatus.SUCCEEDED
    assert result.outcomes[0].action == WriteAction.CREATED
    assert result.outcomes[0].canonical_id.startswith("telemem-")
    assert result.outcomes[0].vendor_id == "vendor-001"
    assert result.latency_seconds >= 0
    assert call["content"] == request.text
    assert call["source_event_id"] == request.source_event_id
    assert call["metadata"]["condition"] == "clean"
    assert call["metadata"]["membench_memory_type"] == "semantic"
    assert call["metadata"]["membench_operation_id"] == request.operation_id


def test_conversation_write_uses_native_inference_and_preserves_metadata(
    initialized_adapter: TeleMemAdapter,
) -> None:
    request = ConversationWriteRequest(
        operation_id="write-native-001",
        scope=_scope(),
        messages=(
            MemoryMessage(
                role="user",
                content="Remember that I prefer dark mode.",
                metadata={"turn": 1},
            ),
            MemoryMessage(role="assistant", content="Understood."),
        ),
        source_event_id="event-002",
        native_options={"prompt": "Extract stable preferences."},
    )

    result = initialized_adapter.write_conversation(request)
    call = FakeRuntime.instances[0].conversation_calls[0]

    assert result.status == WriteStatus.SUCCEEDED
    assert result.outcomes[0].action == WriteAction.UPDATED
    assert call["infer"] is True
    assert call["messages"][0] == {
        "role": "user",
        "content": "Remember that I prefer dark mode.",
    }
    assert call["prompt"] == "Extract stable preferences."
    assert call["memory_type"] is None
    assert call["metadata"]["membench_message_metadata"] == [
        {"turn": 1},
        {},
    ]


def test_conversation_no_records_is_a_no_op(
    initialized_adapter: TeleMemAdapter,
) -> None:
    FakeRuntime.instances[0].conversation_records = []
    request = ConversationWriteRequest(
        operation_id="write-native-002",
        scope=_scope(),
        messages=(MemoryMessage(role="user", content="Ordinary greeting."),),
        source_event_id="event-003",
    )

    result = initialized_adapter.write_conversation(request)

    assert result.status == WriteStatus.NO_OP
    assert result.outcomes == ()
    assert result.raw_vendor == {"results": []}


def test_write_rejects_scope_mismatch_and_unknown_native_options(
    initialized_adapter: TeleMemAdapter,
) -> None:
    with pytest.raises(AdapterConfigurationError, match="scope"):
        initialized_adapter.write_direct(
            DirectWriteRequest(
                operation_id="write-wrong-scope",
                scope=_scope("other-run"),
                text="Do not write this.",
                source_event_id="event-004",
            )
        )

    with pytest.raises(AdapterConfigurationError, match="unsupported"):
        initialized_adapter.write_conversation(
            ConversationWriteRequest(
                operation_id="write-bad-option",
                scope=_scope(),
                messages=(MemoryMessage(role="user", content="Hello."),),
                source_event_id="event-005",
                native_options={"infer": False},
            )
        )


def test_write_wraps_native_failure_without_including_memory_text(
    initialized_adapter: TeleMemAdapter,
) -> None:
    secret_text = "synthetic secret should not appear in the error"

    def fail(*args: Any, **kwargs: Any) -> list[TeleMemRecord]:
        raise RuntimeError("native failure")

    FakeRuntime.instances[0].store_memory = fail  # type: ignore[method-assign]
    request = DirectWriteRequest(
        operation_id="write-failure-001",
        scope=_scope(),
        text=secret_text,
        source_event_id="event-006",
    )

    with pytest.raises(WriteTransportError) as captured:
        initialized_adapter.write_direct(request)

    assert request.operation_id in str(captured.value)
    assert secret_text not in str(captured.value)


def test_close_is_idempotent_and_unbinds_the_adapter(
    initialized_adapter: TeleMemAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]

    initialized_adapter.close()
    initialized_adapter.close()

    assert runtime.closed
    assert initialized_adapter._runtime is None
    assert initialized_adapter._scope is None
    with pytest.raises(AdapterConfigurationError, match="not initialized"):
        initialized_adapter.write_direct(
            DirectWriteRequest(
                operation_id="write-after-close",
                scope=_scope(),
                text="Do not write this.",
                source_event_id="event-007",
            )
        )


def test_flush_normalizes_visible_records(
    initialized_adapter: TeleMemAdapter,
) -> None:
    result = initialized_adapter.flush_and_wait()

    assert result.completed
    assert result.pending_ids == ()
    assert len(result.visible_records) == 1
    assert result.visible_records[0].vendor_id == "vendor-001"
    assert result.visible_records[0].canonical_id.startswith("telemem-")
    assert result.latency_seconds >= 0


def test_retrieve_normalizes_ranking_scores_and_query_hash(
    initialized_adapter: TeleMemAdapter,
) -> None:
    request = RetrievalRequest(
        operation_id="retrieve-001",
        scope=_scope(),
        query="What display theme does the user prefer?",
        top_k=5,
        threshold=0.2,
    )

    result = initialized_adapter.retrieve(request)
    call = FakeRuntime.instances[0].retrieve_calls[0]

    assert result.query_sha256 == request.query_sha256
    assert result.threshold == 0.2
    assert result.hits[0].rank == 1
    assert result.hits[0].raw_score == 0.9
    assert result.hits[0].score_kind == ScoreKind.SIMILARITY
    assert result.hits[0].passed_threshold
    assert result.hits[0].record.memory_type == MemoryType.SEMANTIC
    assert result.hits[0].record.source_event_ids == ("event-001",)
    assert call == {"query": request.query, "top_k": 5}


def test_retrieve_rejects_options_the_runtime_cannot_honor(
    initialized_adapter: TeleMemAdapter,
) -> None:
    with pytest.raises(AdapterConfigurationError, match="threshold"):
        initialized_adapter.retrieve(
            RetrievalRequest(
                operation_id="retrieve-threshold",
                scope=_scope(),
                query="query",
                top_k=5,
                threshold=0.3,
            )
        )
    with pytest.raises(AdapterConfigurationError, match="filters"):
        initialized_adapter.retrieve(
            RetrievalRequest(
                operation_id="retrieve-filter",
                scope=_scope(),
                query="query",
                top_k=5,
                filters={"topic": "security"},
            )
        )


def test_inspect_state_returns_non_query_snapshot(
    initialized_adapter: TeleMemAdapter,
) -> None:
    snapshot = initialized_adapter.inspect_state()

    assert snapshot.scope == _scope()
    assert snapshot.complete
    assert snapshot.record_count == 1
    assert snapshot.records[0].vendor_id == "vendor-001"
    assert FakeRuntime.instances[0].inspect_limits == [10_000]
    assert FakeRuntime.instances[0].retrieve_calls == []


def test_verify_empty_returns_evidence_without_resetting(
    initialized_adapter: TeleMemAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]
    runtime.inspection_records = []
    runtime.retrieval_records = []

    result = initialized_adapter.verify_empty()

    assert isinstance(initialized_adapter, MemoryAdapter)
    assert result.empty
    assert result.inspected_count == 0
    assert result.retrieved_count == 0
    assert result.pending_count == 0
    assert runtime.reset_calls == 0
    assert runtime.inspect_limits[-1] == 1
    assert runtime.retrieve_calls[-1]["top_k"] == 1


def test_verify_empty_rejects_visible_or_pending_state(
    initialized_adapter: TeleMemAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]

    with pytest.raises(NamespaceNotEmptyError, match="inspected=1"):
        initialized_adapter.verify_empty()

    runtime.inspection_records = []
    runtime.retrieval_records = []
    runtime.pending_count_value = 1
    with pytest.raises(NamespaceNotEmptyError, match="pending=1"):
        initialized_adapter.verify_empty()

    assert runtime.reset_calls == 0


def test_reset_returns_verified_empty_evidence(
    initialized_adapter: TeleMemAdapter,
) -> None:
    result = initialized_adapter.reset_and_verify_empty()
    runtime = FakeRuntime.instances[0]

    assert result.empty
    assert result.inspected_count == 0
    assert result.retrieved_count == 0
    assert result.pending_count == 0
    assert runtime.reset_calls == 1
    assert runtime.inspect_limits[-1] == 1
    assert runtime.retrieve_calls[-1]["top_k"] == 1


def test_export_provenance_returns_non_secret_canonical_data(
    initialized_adapter: TeleMemAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = initialized_adapter.settings
    settings.llm_model = "qwen2.5:7b"
    settings.embedding_model = "nomic-embed-text:137m-v1.5-fp16"
    settings.public_config = lambda: {"credential_configured": False}
    runtime = FakeRuntime.instances[0]
    runtime.paths = SimpleNamespace(
        run_dir="run",
        mem0_dir="mem0",
        vector_dir="vectors",
        history_db="history.sqlite",
        collection_name="collection",
    )
    monkeypatch.setattr(
        adapter_module,
        "telemem_package_versions",
        lambda: {"telemem": "1.10.0", "mem0ai": "1.0.0"},
    )

    result = initialized_adapter.export_provenance()

    assert result.mechanism == "telemem"
    assert result.mechanism_version == "1.10.0"
    assert result.namespace == _scope().namespace
    assert result.model_identifiers["writer"] == "qwen2.5:7b"
    assert result.native_configuration["paths"]["history_db"] == "history.sqlite"
