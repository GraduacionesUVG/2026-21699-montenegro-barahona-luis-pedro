from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from membench.langmem import LangMemRecord
from membench.langmem import MemoryType as LangMemMemoryType
from membench.memory.adapters import langmem_adapter as adapter_module
from membench.memory.adapters.langmem_adapter import LangMemAdapter
from membench.memory.errors import (
    AdapterConfigurationError,
    NamespaceNotEmptyError,
    ResetVerificationError,
    RetrievalTransportError,
    StateInspectionError,
    WriteVisibilityTimeout,
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


def _scope(run_id: str = "run-001") -> MemoryScope:
    return MemoryScope(
        experiment_id="experiment-001",
        run_id=run_id,
        user_id="persona-001",
        namespace=("membench", "langmem", run_id, "persona-001"),
    )


class FakeRuntime:
    instances: list["FakeRuntime"] = []

    def __init__(
        self,
        settings: Any,
        run_id: str,
        *,
        namespace_prefix: tuple[str, ...],
    ) -> None:
        self.settings = settings
        self.run_id = run_id
        self.namespace_prefix = tuple(namespace_prefix)
        self.initialize_require_empty: bool | None = None
        self.direct_calls: list[dict[str, Any]] = []
        self.conversation_calls: list[dict[str, Any]] = []
        self.records: list[LangMemRecord] = []
        self.retrieve_calls: list[dict[str, Any]] = []
        self.inspect_limits: list[int] = []
        self.reset_calls = 0
        self.closed = False
        self.keep_records_on_reset = False
        self.invisible_flush = False
        self.pending_count_value = 0
        self.__class__.instances.append(self)

    @property
    def pending_count(self) -> int:
        return self.pending_count_value

    def initialize(self, *, require_empty: bool) -> "FakeRuntime":
        self.initialize_require_empty = require_empty
        return self

    def close(self) -> None:
        self.closed = True

    def store_memory(
        self,
        content: str,
        *,
        memory_type: LangMemMemoryType,
        key: str,
        metadata: dict[str, Any],
        source_event_id: str,
    ) -> LangMemRecord:
        self.direct_calls.append(
            {
                "content": content,
                "memory_type": memory_type,
                "key": key,
                "metadata": metadata,
                "source_event_id": source_event_id,
            }
        )
        existing = next((item for item in self.records if item.key == key), None)
        raw_value = {
            "kind": "SemanticMemory",
            "content": {"text": content},
            "metadata": {
                **metadata,
                "membench_source_event_id": source_event_id,
            },
        }
        record = LangMemRecord(
            memory_type=memory_type,
            namespace=(*self.namespace_prefix, memory_type.value),
            key=key,
            text=content,
            score=None,
            raw_value=raw_value,
            event="UPDATE" if existing else "ADD",
        )
        self.records = [item for item in self.records if item.key != key]
        self.records.append(record)
        return record

    def submit_conversation(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> None:
        self.conversation_calls.append({"messages": messages, **kwargs})

    def flush_and_wait(self) -> dict[str, list[dict[str, Any]]]:
        writes = {item.value: [] for item in self.settings.memory_types}
        for call_number, call in enumerate(self.conversation_calls, start=1):
            for memory_type in call["memory_types"]:
                key = f"native-{call_number}-{memory_type.value}"
                namespace = (*self.namespace_prefix, memory_type.value)
                raw_value = {
                    "kind": f"{memory_type.value.title()}Memory",
                    "content": {"text": f"Extracted {memory_type.value} memory"},
                    "metadata": {
                        **call["metadata"],
                        "membench_source_event_id": call["source_event_id"],
                    },
                }
                writes[memory_type.value].append(
                    {"namespace": namespace, "key": key, "value": raw_value}
                )
                if not self.invisible_flush:
                    self.records.append(
                        LangMemRecord(
                            memory_type=memory_type,
                            namespace=namespace,
                            key=key,
                            text=raw_value["content"]["text"],
                            score=None,
                            raw_value=raw_value,
                        )
                    )
        self.conversation_calls.clear()
        return writes

    def inspect_state(self, *, limit_per_type: int) -> list[LangMemRecord]:
        self.inspect_limits.append(limit_per_type)
        return self.records[: limit_per_type * len(self.settings.memory_types)]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        memory_types: tuple[LangMemMemoryType, ...] | None = None,
    ) -> list[LangMemRecord]:
        selected = memory_types or self.settings.memory_types
        self.retrieve_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "memory_types": selected,
            }
        )
        records = [item for item in self.records if item.memory_type in selected]
        records.sort(
            key=lambda item: item.score if item.score is not None else float("-inf"),
            reverse=True,
        )
        return records[:top_k]

    def reset_and_verify_empty(self) -> None:
        self.reset_calls += 1
        self.conversation_calls.clear()
        if not self.keep_records_on_reset:
            self.records.clear()


@pytest.fixture
def initialized_adapter(monkeypatch: pytest.MonkeyPatch) -> LangMemAdapter:
    FakeRuntime.instances.clear()
    monkeypatch.setattr(adapter_module, "LangMemRuntime", FakeRuntime)
    settings = SimpleNamespace(
        memory_types=(LangMemMemoryType.SEMANTIC, LangMemMemoryType.EPISODIC),
        enable_deletes=False,
        storage_backend="sqlite",
    )
    adapter = LangMemAdapter(settings)  # type: ignore[arg-type]
    adapter.initialize(_scope())
    return adapter


def test_initialize_binds_the_runtime_to_the_canonical_namespace(
    initialized_adapter: LangMemAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]

    assert runtime.run_id == "run-001"
    assert runtime.namespace_prefix == _scope().namespace
    assert runtime.initialize_require_empty is False


def test_direct_write_uses_store_and_flush_verifies_visibility(
    initialized_adapter: LangMemAdapter,
) -> None:
    request = DirectWriteRequest(
        operation_id="direct-001",
        scope=_scope(),
        text="The user prefers dark mode.",
        memory_type=MemoryType.SEMANTIC,
        source_event_id="event-001",
        metadata={"condition": "clean"},
    )

    result = initialized_adapter.write_direct(request)
    call = FakeRuntime.instances[0].direct_calls[0]
    flushed = initialized_adapter.flush_and_wait()

    assert result.status == WriteStatus.SUCCEEDED
    assert result.outcomes[0].action == WriteAction.CREATED
    assert result.outcomes[0].canonical_id.startswith("langmem-")
    assert call["memory_type"] == LangMemMemoryType.SEMANTIC
    assert call["metadata"]["condition"] == "clean"
    assert call["metadata"]["membench_user_id"] == "persona-001"
    assert flushed.completed
    assert flushed.pending_ids == ()
    assert flushed.visible_records[0].source_event_ids == ("event-001",)


def test_conversation_write_queues_selected_families_until_flush(
    initialized_adapter: LangMemAdapter,
) -> None:
    request = ConversationWriteRequest(
        operation_id="conversation-001",
        scope=_scope(),
        messages=(
            MemoryMessage(
                role="user",
                content="Remember my preference.",
                metadata={"turn": 1},
            ),
            MemoryMessage(role="assistant", content="Understood."),
        ),
        source_event_id="event-002",
        native_options={"memory_types": ["semantic", "episodic"]},
    )

    queued = initialized_adapter.write_conversation(request)
    call = FakeRuntime.instances[0].conversation_calls[0]
    flushed = initialized_adapter.flush_and_wait()

    assert queued.status == WriteStatus.QUEUED
    assert queued.outcomes == ()
    assert call["memory_types"] == (
        LangMemMemoryType.SEMANTIC,
        LangMemMemoryType.EPISODIC,
    )
    assert call["metadata"]["membench_message_metadata"] == [{"turn": 1}, {}]
    assert len(flushed.visible_records) == 2
    assert {record.memory_type for record in flushed.visible_records} == {
        MemoryType.SEMANTIC,
        MemoryType.EPISODIC,
    }


def test_conversation_rejects_unavailable_prompt_override(
    initialized_adapter: LangMemAdapter,
) -> None:
    request = ConversationWriteRequest(
        operation_id="conversation-prompt",
        scope=_scope(),
        messages=(MemoryMessage(role="user", content="Remember this."),),
        source_event_id="event-003",
        native_options={"prompt": "Use different extraction instructions."},
    )

    with pytest.raises(AdapterConfigurationError, match="fixed during initialize"):
        initialized_adapter.write_conversation(request)


def test_flush_raises_when_langmem_reports_an_invisible_write(
    initialized_adapter: LangMemAdapter,
) -> None:
    initialized_adapter.write_conversation(
        ConversationWriteRequest(
            operation_id="conversation-invisible",
            scope=_scope(),
            messages=(MemoryMessage(role="user", content="Remember this."),),
            source_event_id="event-004",
            native_options={"memory_type": "semantic"},
        )
    )
    FakeRuntime.instances[0].invisible_flush = True

    with pytest.raises(WriteVisibilityTimeout, match="invisible"):
        initialized_adapter.flush_and_wait()


def test_retrieve_normalizes_selected_families_scores_and_timestamps(
    initialized_adapter: LangMemAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]
    created_at = datetime(2026, 9, 15, tzinfo=UTC)
    runtime.records.extend(
        [
            LangMemRecord(
                memory_type=LangMemMemoryType.SEMANTIC,
                namespace=(*_scope().namespace, "semantic"),
                key="semantic-high",
                text="The user prefers dark mode.",
                score=0.91,
                raw_value={"content": {"text": "The user prefers dark mode."}},
                created_at=created_at,
                updated_at=created_at,
            ),
            LangMemRecord(
                memory_type=LangMemMemoryType.SEMANTIC,
                namespace=(*_scope().namespace, "semantic"),
                key="semantic-low",
                text="The user likes tea.",
                score=0.72,
                raw_value={"content": {"text": "The user likes tea."}},
            ),
            LangMemRecord(
                memory_type=LangMemMemoryType.EPISODIC,
                namespace=(*_scope().namespace, "episodic"),
                key="episode-hidden",
                text="The user visited an observatory.",
                score=0.99,
                raw_value={
                    "content": {"text": "The user visited an observatory."}
                },
            ),
        ]
    )
    request = RetrievalRequest(
        operation_id="retrieve-001",
        scope=_scope(),
        query="What interface theme does the user prefer?",
        top_k=5,
        threshold=0.8,
        memory_types=(MemoryType.SEMANTIC,),
    )

    result = initialized_adapter.retrieve(request)

    assert result.query_sha256 == request.query_sha256
    assert [hit.record.vendor_id for hit in result.hits] == [
        "semantic-high",
        "semantic-low",
    ]
    assert result.hits[0].score_kind == ScoreKind.SIMILARITY
    assert result.hits[0].passed_threshold
    assert not result.hits[1].passed_threshold
    assert result.hits[0].record.created_at == created_at
    assert runtime.retrieve_calls[0]["memory_types"] == (
        LangMemMemoryType.SEMANTIC,
    )


@pytest.mark.parametrize(
    "native_timestamp",
    [
        datetime(2026, 9, 21, 0, 36, 30),
        "2026-09-21 00:36:30",
    ],
)
def test_sqlite_naive_timestamps_are_annotated_as_utc(
    initialized_adapter: LangMemAdapter,
    native_timestamp: str | datetime,
) -> None:
    runtime = FakeRuntime.instances[0]
    runtime.records.append(
        LangMemRecord(
            memory_type=LangMemMemoryType.SEMANTIC,
            namespace=(*_scope().namespace, "semantic"),
            key="sqlite-naive-timestamp",
            text="The user prefers dark mode.",
            score=0.91,
            raw_value={"content": {"text": "The user prefers dark mode."}},
            created_at=native_timestamp,
            updated_at=native_timestamp,
        )
    )

    result = initialized_adapter.retrieve(
        RetrievalRequest(
            operation_id="retrieve-sqlite-timestamp",
            scope=_scope(),
            query="What interface theme does the user prefer?",
            top_k=1,
        )
    )

    expected = datetime(2026, 9, 21, 0, 36, 30, tzinfo=UTC)
    assert result.hits[0].record.created_at == expected
    assert result.hits[0].record.updated_at == expected


def test_timestamp_normalization_preserves_aware_values_and_none() -> None:
    aware = datetime(2026, 9, 21, 0, 36, 30, tzinfo=UTC)

    assert (
        adapter_module.normalize_langmem_timestamp(
            aware,
            storage_backend="sqlite",
        )
        is aware
    )
    assert (
        adapter_module.normalize_langmem_timestamp(
            None,
            storage_backend="sqlite",
        )
        is None
    )


def test_non_sqlite_naive_timestamp_remains_strictly_rejected(
    initialized_adapter: LangMemAdapter,
) -> None:
    initialized_adapter.settings.storage_backend = "postgres"
    runtime = FakeRuntime.instances[0]
    runtime.records.append(
        LangMemRecord(
            memory_type=LangMemMemoryType.SEMANTIC,
            namespace=(*_scope().namespace, "semantic"),
            key="postgres-naive-timestamp",
            text="The user prefers dark mode.",
            score=0.91,
            raw_value={"content": {"text": "The user prefers dark mode."}},
            created_at=datetime(2026, 9, 21, 0, 36, 30),
            updated_at=datetime(2026, 9, 21, 0, 36, 30),
        )
    )

    with pytest.raises(RetrievalTransportError) as captured:
        initialized_adapter.retrieve(
            RetrievalRequest(
                operation_id="retrieve-postgres-timestamp",
                scope=_scope(),
                query="What interface theme does the user prefer?",
                top_k=1,
            )
        )

    assert isinstance(captured.value.__cause__, ValueError)
    assert "timezone-free created_at" in str(captured.value.__cause__)


def test_retrieve_rejects_unsupported_options_and_wraps_transport_errors(
    initialized_adapter: LangMemAdapter,
) -> None:
    with pytest.raises(AdapterConfigurationError, match="filters"):
        initialized_adapter.retrieve(
            RetrievalRequest(
                operation_id="retrieve-filter",
                scope=_scope(),
                query="query",
                top_k=1,
                filters={"topic": "security"},
            )
        )
    with pytest.raises(AdapterConfigurationError, match="working"):
        initialized_adapter.retrieve(
            RetrievalRequest(
                operation_id="retrieve-working",
                scope=_scope(),
                query="query",
                top_k=1,
                memory_types=(MemoryType.WORKING,),
            )
        )

    def fail(*_args: Any, **_kwargs: Any) -> list[LangMemRecord]:
        raise RuntimeError("transport failed")

    FakeRuntime.instances[0].retrieve = fail  # type: ignore[method-assign]
    with pytest.raises(RetrievalTransportError, match="retrieve-failure"):
        initialized_adapter.retrieve(
            RetrievalRequest(
                operation_id="retrieve-failure",
                scope=_scope(),
                query="query",
                top_k=1,
            )
        )


def test_inspect_state_normalizes_records_without_semantic_retrieval(
    initialized_adapter: LangMemAdapter,
) -> None:
    initialized_adapter.write_direct(
        DirectWriteRequest(
            operation_id="inspect-direct",
            scope=_scope(),
            text="The user prefers dark mode.",
            memory_type=MemoryType.SEMANTIC,
            source_event_id="event-inspect",
        )
    )
    runtime = FakeRuntime.instances[0]

    snapshot = initialized_adapter.inspect_state()

    assert snapshot.scope == _scope()
    assert snapshot.complete
    assert snapshot.record_count == 1
    assert snapshot.records[0].vendor_id == runtime.records[0].key
    assert snapshot.raw_vendor["pending_ids"] == ["inspect-direct"]
    assert runtime.inspect_limits == [10_000]
    assert runtime.retrieve_calls == []


def test_inspect_state_wraps_native_failures(
    initialized_adapter: LangMemAdapter,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> list[LangMemRecord]:
        raise RuntimeError("inspection failed")

    FakeRuntime.instances[0].inspect_state = fail  # type: ignore[method-assign]

    with pytest.raises(StateInspectionError, match="inspection"):
        initialized_adapter.inspect_state()


def test_verify_empty_returns_evidence_without_resetting(
    initialized_adapter: LangMemAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]

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
    initialized_adapter: LangMemAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]
    runtime.records.append(
        LangMemRecord(
            memory_type=LangMemMemoryType.SEMANTIC,
            namespace=(*_scope().namespace, "semantic"),
            key="existing-memory",
            text="Existing memory.",
            score=0.9,
            raw_value={"content": {"text": "Existing memory."}},
        )
    )

    with pytest.raises(NamespaceNotEmptyError, match="inspected=1"):
        initialized_adapter.verify_empty()

    runtime.records.clear()
    runtime.pending_count_value = 1
    with pytest.raises(NamespaceNotEmptyError, match="pending=1"):
        initialized_adapter.verify_empty()

    assert runtime.reset_calls == 0


def test_reset_verifies_empty_and_clears_adapter_pending_state(
    initialized_adapter: LangMemAdapter,
) -> None:
    initialized_adapter.write_direct(
        DirectWriteRequest(
            operation_id="reset-direct",
            scope=_scope(),
            text="Temporary memory.",
            memory_type=MemoryType.SEMANTIC,
            source_event_id="event-reset",
        )
    )
    runtime = FakeRuntime.instances[0]

    result = initialized_adapter.reset_and_verify_empty()

    assert result.empty
    assert result.pending_count == 0
    assert runtime.reset_calls == 1
    assert runtime.inspect_limits[-1] == 1
    assert runtime.retrieve_calls[-1]["top_k"] == 1
    assert initialized_adapter._pending_operation_ids == []
    assert initialized_adapter._pending_record_locators == []


def test_reset_raises_if_records_remain_visible(
    initialized_adapter: LangMemAdapter,
) -> None:
    initialized_adapter.write_direct(
        DirectWriteRequest(
            operation_id="reset-failure",
            scope=_scope(),
            text="Memory that remains visible.",
            memory_type=MemoryType.SEMANTIC,
            source_event_id="event-reset-failure",
        )
    )
    FakeRuntime.instances[0].keep_records_on_reset = True

    with pytest.raises(ResetVerificationError, match="not empty"):
        initialized_adapter.reset_and_verify_empty()


def test_close_is_idempotent_and_unbinds_the_adapter(
    initialized_adapter: LangMemAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]

    initialized_adapter.close()
    initialized_adapter.close()

    assert runtime.closed
    assert initialized_adapter._runtime is None
    assert initialized_adapter._scope is None
    with pytest.raises(AdapterConfigurationError, match="not initialized"):
        initialized_adapter.inspect_state()


def test_export_provenance_returns_storage_and_model_details(
    initialized_adapter: LangMemAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = initialized_adapter.settings
    settings.enable_deletes = False
    settings.storage_backend = "sqlite"
    settings.writer_model = "qwen2.5:7b"
    settings.embedding_model = "nomic-embed-text:137m-v1.5-fp16"
    settings.public_config = lambda: {"credential_configured": False}
    FakeRuntime.instances[0].storage_path = "store.sqlite"
    monkeypatch.setattr(
        adapter_module,
        "langmem_package_versions",
        lambda: {"langmem": "0.0.30", "langgraph": "1.2.11"},
    )

    result = initialized_adapter.export_provenance()

    assert result.mechanism == "langmem"
    assert result.mechanism_version == "0.0.30"
    assert result.storage_backend == "sqlite"
    assert result.native_configuration["storage_path"] == "store.sqlite"
