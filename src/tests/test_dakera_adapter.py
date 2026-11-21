from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from membench.dakera import DakeraMemoryType, DakeraRecord, DakeraRouting
from membench.memory.adapters import dakera_adapter as adapter_module
from membench.memory.adapters.dakera_adapter import DakeraAdapter
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


def _scope(
    run_id: str = "run-001",
    user_id: str = "persona-001",
) -> MemoryScope:
    return MemoryScope(
        experiment_id="experiment-001",
        run_id=run_id,
        user_id=user_id,
        namespace=("membench", "dakera", run_id, user_id),
    )


def _record(
    vendor_id: str,
    *,
    memory_type: DakeraMemoryType = DakeraMemoryType.SEMANTIC,
    score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> DakeraRecord:
    raw_value = {
        "id": vendor_id,
        "content": f"Memory {vendor_id}",
        "memory_type": memory_type.value,
        "importance": 0.5,
        "score": score,
        "metadata": metadata or {},
        "created_at": "2026-09-15T00:00:00Z",
    }
    return DakeraRecord(
        vendor_id=vendor_id,
        text=raw_value["content"],
        memory_type=memory_type,
        importance=0.5,
        score=score,
        metadata=dict(metadata or {}),
        created_at=raw_value["created_at"],
        raw_value=raw_value,
    )


class FakeRuntime:
    instances: list["FakeRuntime"] = []

    def __init__(
        self,
        settings: Any,
        run_id: str,
        *,
        agent_id: str,
    ) -> None:
        self.settings = settings
        self.run_id = run_id
        self.agent_id = agent_id
        self.initialize_require_empty: bool | None = None
        self.records: list[DakeraRecord] = []
        self.pending: list[DakeraRecord] = []
        self.store_calls: list[dict[str, Any]] = []
        self.conversation_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[dict[str, Any]] = []
        self.inspect_limits: list[int] = []
        self.reset_calls = 0
        self.keep_records_on_reset = False
        self.closed = False
        self.__class__.instances.append(self)

    def initialize(self, *, require_empty: bool) -> "FakeRuntime":
        self.initialize_require_empty = require_empty
        return self

    def store_memory(
        self,
        content: str,
        *,
        memory_type: DakeraMemoryType,
        importance: float,
        metadata: dict[str, Any],
        source_event_id: str,
    ) -> DakeraRecord:
        call = {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "metadata": metadata,
            "source_event_id": source_event_id,
        }
        self.store_calls.append(call)
        resolved_metadata = {
            **metadata,
            "membench_run_id": self.run_id,
            "membench_source_event_id": source_event_id,
        }
        record = _record(
            f"memory-{len(self.records) + 1}",
            memory_type=memory_type,
            metadata=resolved_metadata,
        )
        record = replace(record, text=content)
        self.records.append(record)
        self.pending.append(record)
        return record

    def submit_conversation(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> DakeraRecord:
        self.conversation_calls.append({"messages": messages, **kwargs})
        content = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        return self.store_memory(content, **kwargs)

    def flush_and_wait(self) -> list[DakeraRecord]:
        visible = list(self.pending)
        self.pending.clear()
        return visible

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        memory_type: DakeraMemoryType | None = None,
    ) -> list[DakeraRecord]:
        self.retrieve_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "memory_type": memory_type,
            }
        )
        selected = [
            record
            for record in self.records
            if memory_type is None or record.memory_type == memory_type
        ]
        selected.sort(
            key=lambda record: (
                record.score is not None,
                record.score if record.score is not None else float("-inf"),
            ),
            reverse=True,
        )
        return selected[:top_k]

    def memory_count(self) -> int:
        return len(self.records)

    def inspect_state(self, *, limit: int) -> list[DakeraRecord]:
        self.inspect_limits.append(limit)
        return self.records[:limit]

    def reset_and_verify_empty(self) -> None:
        self.reset_calls += 1
        self.pending.clear()
        if not self.keep_records_on_reset:
            self.records.clear()

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def initialized_adapter(monkeypatch: pytest.MonkeyPatch) -> DakeraAdapter:
    FakeRuntime.instances.clear()
    monkeypatch.setattr(adapter_module, "DakeraRuntime", FakeRuntime)
    settings = SimpleNamespace(
        agent_prefix="membench",
        memory_type=DakeraMemoryType.SEMANTIC,
        importance=0.5,
        rerank=False,
        routing=DakeraRouting.VECTOR,
        ttl_seconds=None,
        include_associated=False,
    )
    adapter = DakeraAdapter(settings)  # type: ignore[arg-type]
    adapter.initialize(_scope())
    return adapter


def test_initialize_uses_a_scope_derived_agent_id(
    initialized_adapter: DakeraAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]

    assert runtime.run_id == "run-001"
    assert runtime.agent_id.startswith("membench-")
    assert runtime.agent_id != "membench-run-001"
    assert runtime.initialize_require_empty is False


def test_direct_and_conversation_writes_flush_as_canonical_records(
    initialized_adapter: DakeraAdapter,
) -> None:
    direct = initialized_adapter.write_direct(
        DirectWriteRequest(
            operation_id="direct-001",
            scope=_scope(),
            text="The user prefers dark mode.",
            memory_type=MemoryType.SEMANTIC,
            source_event_id="event-001",
            metadata={"condition": "clean"},
        )
    )
    conversation = initialized_adapter.write_conversation(
        ConversationWriteRequest(
            operation_id="conversation-001",
            scope=_scope(),
            messages=(
                MemoryMessage(
                    role="user",
                    content="I visited the observatory.",
                    metadata={"turn": 1},
                ),
                MemoryMessage(role="assistant", content="That sounds enjoyable."),
            ),
            source_event_id="event-002",
            native_options={"memory_type": "episodic", "importance": 0.8},
        )
    )
    flushed = initialized_adapter.flush_and_wait()
    runtime = FakeRuntime.instances[0]

    assert direct.status == WriteStatus.SUCCEEDED
    assert direct.outcomes[0].action == WriteAction.CREATED
    assert conversation.status == WriteStatus.SUCCEEDED
    assert runtime.store_calls[0]["metadata"]["condition"] == "clean"
    assert runtime.conversation_calls[0]["memory_type"] == (
        DakeraMemoryType.EPISODIC
    )
    assert flushed.completed
    assert len(flushed.visible_records) == 2
    assert flushed.visible_records[0].source_event_ids == ("event-001",)
    assert flushed.visible_records[0].created_at is not None


def test_flush_maps_native_timeout(
    initialized_adapter: DakeraAdapter,
) -> None:
    def fail() -> list[DakeraRecord]:
        raise RuntimeError("visibility timed out")

    FakeRuntime.instances[0].flush_and_wait = fail  # type: ignore[method-assign]

    with pytest.raises(WriteVisibilityTimeout, match="timed out"):
        initialized_adapter.flush_and_wait()


def test_retrieve_merges_selected_memory_families_and_applies_threshold(
    initialized_adapter: DakeraAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]
    runtime.records = [
        _record("semantic", memory_type=DakeraMemoryType.SEMANTIC, score=0.91),
        _record("episodic", memory_type=DakeraMemoryType.EPISODIC, score=0.72),
        _record("working", memory_type=DakeraMemoryType.WORKING, score=0.99),
    ]
    request = RetrievalRequest(
        operation_id="retrieve-001",
        scope=_scope(),
        query="What does the user remember?",
        top_k=3,
        threshold=0.8,
        memory_types=(MemoryType.SEMANTIC, MemoryType.EPISODIC),
    )

    result = initialized_adapter.retrieve(request)

    assert [hit.record.vendor_id for hit in result.hits] == [
        "semantic",
        "episodic",
    ]
    assert result.hits[0].score_kind == ScoreKind.SIMILARITY
    assert result.hits[0].passed_threshold
    assert not result.hits[1].passed_threshold
    assert [call["memory_type"] for call in runtime.retrieve_calls] == [
        DakeraMemoryType.SEMANTIC,
        DakeraMemoryType.EPISODIC,
    ]


def test_retrieve_rejects_options_and_wraps_native_failures(
    initialized_adapter: DakeraAdapter,
) -> None:
    with pytest.raises(AdapterConfigurationError, match="filters"):
        initialized_adapter.retrieve(
            RetrievalRequest(
                operation_id="retrieve-filter",
                scope=_scope(),
                query="query",
                top_k=1,
                filters={"condition": "clean"},
            )
        )
    with pytest.raises(AdapterConfigurationError, match="unspecified"):
        initialized_adapter.retrieve(
            RetrievalRequest(
                operation_id="retrieve-unspecified",
                scope=_scope(),
                query="query",
                top_k=1,
                memory_types=(MemoryType.UNSPECIFIED,),
            )
        )

    def fail(*_args: Any, **_kwargs: Any) -> list[DakeraRecord]:
        raise RuntimeError("recall failed")

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


def test_inspect_state_does_not_issue_a_semantic_query(
    initialized_adapter: DakeraAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]
    runtime.records = [_record("memory-001")]

    snapshot = initialized_adapter.inspect_state()

    assert snapshot.scope == _scope()
    assert snapshot.complete
    assert snapshot.record_count == 1
    assert snapshot.records[0].vendor_id == "memory-001"
    assert runtime.inspect_limits == [10_000]
    assert runtime.retrieve_calls == []


def test_inspect_state_wraps_native_failures(
    initialized_adapter: DakeraAdapter,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> list[DakeraRecord]:
        raise RuntimeError("inspection failed")

    FakeRuntime.instances[0].inspect_state = fail  # type: ignore[method-assign]

    with pytest.raises(StateInspectionError, match="inspection"):
        initialized_adapter.inspect_state()


def test_verify_empty_returns_evidence_without_resetting(
    initialized_adapter: DakeraAdapter,
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
    initialized_adapter: DakeraAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]
    runtime.records = [_record("existing-memory", score=0.9)]

    with pytest.raises(NamespaceNotEmptyError, match="inspected=1"):
        initialized_adapter.verify_empty()

    runtime.records.clear()
    runtime.pending = [_record("pending-memory")]
    with pytest.raises(NamespaceNotEmptyError, match="pending=1"):
        initialized_adapter.verify_empty()

    assert runtime.reset_calls == 0


def test_reset_verifies_empty_and_detects_remaining_records(
    initialized_adapter: DakeraAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]
    runtime.records = [_record("memory-001")]

    result = initialized_adapter.reset_and_verify_empty()

    assert result.empty
    assert result.pending_count == 0
    assert runtime.reset_calls == 1
    assert runtime.inspect_limits[-1] == 1
    assert runtime.retrieve_calls[-1]["top_k"] == 1

    runtime.records = [_record("memory-remaining")]
    runtime.keep_records_on_reset = True
    with pytest.raises(ResetVerificationError, match="not empty"):
        initialized_adapter.reset_and_verify_empty()


def test_close_is_idempotent_and_unbinds_the_adapter(
    initialized_adapter: DakeraAdapter,
) -> None:
    runtime = FakeRuntime.instances[0]

    initialized_adapter.close()
    initialized_adapter.close()

    assert runtime.closed
    with pytest.raises(AdapterConfigurationError, match="not initialized"):
        initialized_adapter.inspect_state()


def test_export_provenance_distinguishes_sdk_and_server_versions(
    initialized_adapter: DakeraAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = initialized_adapter.settings
    settings.ttl_seconds = None
    settings.include_associated = False
    settings.expected_server_version = "2.0.0"
    settings.public_config = lambda: {"credential_configured": True}
    runtime = FakeRuntime.instances[0]
    runtime.health = {"status": "healthy", "version": "2.1.0"}
    monkeypatch.setattr(
        adapter_module,
        "dakera_package_version",
        lambda: "0.12.12",
    )

    result = initialized_adapter.export_provenance()

    assert result.mechanism == "dakera"
    assert result.mechanism_version == "2.1.0"
    assert result.package_versions == {"dakera": "0.12.12"}
    assert result.native_configuration["agent_id"] == runtime.agent_id
