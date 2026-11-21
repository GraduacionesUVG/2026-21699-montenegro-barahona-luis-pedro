from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from membench.dakera import (
    DakeraError,
    DakeraMemoryType,
    DakeraRouting,
    DakeraRuntime,
    DakeraSettings,
    _normalize_dakera_timestamp,
)


def _set_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAKERA_URL", "http://127.0.0.1:3300")
    monkeypatch.setenv("DAKERA_API_KEY", "test-dakera-secret")
    monkeypatch.setenv("DAKERA_MEMORY_TYPE", "semantic")
    monkeypatch.setenv("DAKERA_IMPORTANCE", "0.5")
    monkeypatch.setenv("DAKERA_TOP_K", "5")
    monkeypatch.setenv("DAKERA_ROUTING", "vector")
    monkeypatch.setenv("DAKERA_RERANK", "false")
    monkeypatch.setenv("DAKERA_NEIGHBORHOOD", "false")
    monkeypatch.setenv("DAKERA_INCLUDE_ASSOCIATED", "false")
    monkeypatch.setenv("DAKERA_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("DAKERA_VISIBILITY_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("DAKERA_POLL_INTERVAL_SECONDS", "0.001")
    monkeypatch.setenv("DAKERA_MAX_RETRIES", "3")
    monkeypatch.setenv("DAKERA_AGENT_PREFIX", "membench")
    monkeypatch.setenv("DAKERA_TTL_SECONDS", "")
    monkeypatch.setenv("DAKERA_SERVER_VERSION", "0.11.107")
    monkeypatch.setenv(
        "DAKERA_SERVER_IMAGE",
        "ghcr.io/dakera-ai/dakera@sha256:"
        "00caa29c818c0706a031ddb6cdfe3580ced4b8f056543a7925256012a3be54bc",
    )


class FakeDakeraClient:
    def __init__(self) -> None:
        self.memories: dict[str, list[dict[str, Any]]] = {}
        self.next_id = 1
        self.agent_memories_error: Exception | None = None
        self.recall_error: Exception | None = None

    @staticmethod
    def health() -> dict[str, str]:
        return {"status": "healthy", "version": "0.11.107"}

    def agent_memories(self, agent_id: str, limit: int) -> list[dict[str, Any]]:
        if self.agent_memories_error is not None:
            raise self.agent_memories_error
        return self.memories.get(agent_id, [])[:limit]

    def agent_stats(self, agent_id: str) -> dict[str, Any]:
        memories = self.memories.get(agent_id, [])
        return {
            "agent_id": agent_id,
            "total_memories": len(memories),
        }

    def store_memory(self, **kwargs: Any) -> dict[str, Any]:
        memory = {
            "id": f"memory-{self.next_id}",
            "content": kwargs["content"],
            "memory_type": kwargs["memory_type"],
            "importance": kwargs["importance"],
            "metadata": kwargs["metadata"],
            "created_at": "2026-09-07T00:00:00Z",
        }
        self.next_id += 1
        self.memories.setdefault(kwargs["agent_id"], []).append(memory)
        return memory

    def recall(self, **kwargs: Any) -> SimpleNamespace:
        if self.recall_error is not None:
            raise self.recall_error
        memories = self.memories.get(kwargs["agent_id"], [])
        selected = [
            {**memory, "score": 0.9}
            for memory in memories
            if kwargs["memory_type"] is None
            or memory["memory_type"] == kwargs["memory_type"]
        ][: kwargs["top_k"]]
        return SimpleNamespace(memories=selected)

    def forget(self, agent_id: str, memory_id: str) -> dict[str, int]:
        self.memories[agent_id] = [
            memory
            for memory in self.memories.get(agent_id, [])
            if memory["id"] != memory_id
        ]
        return {"deleted_count": 1}


class NotFoundError(Exception):
    """Test double matching the pinned Dakera SDK exception surface."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = SimpleNamespace(value=code)


@pytest.mark.parametrize(
    ("raw_timestamp", "expected"),
    [
        (1789959653, datetime.fromtimestamp(1789959653, tz=UTC).isoformat()),
        (1789959653.5, datetime.fromtimestamp(1789959653.5, tz=UTC).isoformat()),
        ("2026-09-21T03:00:53+00:00", "2026-09-21T03:00:53+00:00"),
        (None, None),
    ],
)
def test_normalize_dakera_timestamp(
    raw_timestamp: object,
    expected: str | None,
) -> None:
    assert _normalize_dakera_timestamp(raw_timestamp) == expected


@pytest.mark.parametrize("raw_timestamp", [True, False, object()])
def test_normalize_dakera_timestamp_rejects_invalid_values(
    raw_timestamp: object,
) -> None:
    with pytest.raises(DakeraError, match="invalid created_at timestamp"):
        _normalize_dakera_timestamp(raw_timestamp)


def test_settings_are_explicit_and_secret_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_defaults(monkeypatch)

    settings = DakeraSettings.from_env()

    assert settings.memory_type == DakeraMemoryType.SEMANTIC
    assert settings.routing == DakeraRouting.VECTOR
    assert settings.agent_id_for("run-001") == "membench-run-001"
    assert settings.credential_configured
    assert "test-dakera-secret" not in repr(settings)
    assert "api_key" not in settings.public_config()


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_defaults(monkeypatch)
    monkeypatch.setenv("DAKERA_API_KEY", "")

    with pytest.raises(DakeraError, match="DAKERA_API_KEY"):
        DakeraSettings.from_env()


def test_run_id_rejects_path_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_defaults(monkeypatch)
    settings = DakeraSettings.from_env()

    with pytest.raises(ValueError, match="run_id"):
        settings.agent_id_for("../another-run")


def test_runtime_accepts_a_valid_scope_resolved_agent_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_defaults(monkeypatch)
    settings = DakeraSettings.from_env()
    runtime = DakeraRuntime(
        settings,
        "run-001",
        agent_id="membench-scoped-agent-001",
        client=FakeDakeraClient(),
    ).initialize(require_empty=False)

    assert runtime.run_id == "run-001"
    assert runtime.agent_id == "membench-scoped-agent-001"

    runtime.close()


def test_runtime_isolates_writes_and_supports_full_direct_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_defaults(monkeypatch)
    settings = DakeraSettings.from_env()
    client = FakeDakeraClient()
    runtime = DakeraRuntime(settings, "run-001", client=client).initialize()

    stored = runtime.submit_conversation(
        [
            {"role": "user", "content": "My field notebook is amber."},
            {"role": "assistant", "content": "Understood."},
        ],
        source_event_id="event-001",
    )
    visible = runtime.flush_and_wait()
    retrieved = runtime.retrieve("Which notebook color is preferred?")

    assert [record.vendor_id for record in visible] == [stored.vendor_id]
    assert retrieved[0].score == 0.9
    assert retrieved[0].metadata["membench_run_id"] == "run-001"
    assert retrieved[0].metadata["membench_source_event_id"] == "event-001"
    assert "[semantic] user: My field notebook is amber." in (
        runtime.assemble_context(retrieved)
    )
    assert client.agent_memories("membench-run-001-isolated", limit=1) == []

    runtime.reset_and_verify_empty()

    assert runtime.inspect_state() == []
    runtime.close()


def test_runtime_treats_missing_agent_namespace_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_defaults(monkeypatch)
    settings = DakeraSettings.from_env()
    client = FakeDakeraClient()
    client.agent_memories_error = NotFoundError("NAMESPACE_NOT_FOUND")
    client.recall_error = NotFoundError("NAMESPACE_NOT_FOUND")

    runtime = DakeraRuntime(settings, "run-missing", client=client).initialize()

    assert runtime.inspect_state() == []
    assert runtime.retrieve("empty namespace probe", top_k=1) == []
    runtime.close()


def test_runtime_does_not_suppress_other_not_found_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_defaults(monkeypatch)
    settings = DakeraSettings.from_env()
    client = FakeDakeraClient()
    runtime = DakeraRuntime(
        settings,
        "run-other-not-found",
        client=client,
    ).initialize(require_empty=False)

    client.recall_error = NotFoundError("VECTOR_NOT_FOUND")
    with pytest.raises(NotFoundError, match="VECTOR_NOT_FOUND"):
        runtime.retrieve("missing vector probe", top_k=1)

    client.agent_memories_error = NotFoundError("VECTOR_NOT_FOUND")
    with pytest.raises(NotFoundError, match="VECTOR_NOT_FOUND"):
        runtime.inspect_state()
    runtime.close()


def test_reserved_provenance_metadata_cannot_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_defaults(monkeypatch)
    settings = DakeraSettings.from_env()
    runtime = DakeraRuntime(
        settings, "run-001", client=FakeDakeraClient()
    ).initialize()

    with pytest.raises(ValueError, match="reserved Dakera metadata"):
        runtime.store_memory(
            "candidate", metadata={"membench_run_id": "another-run"}
        )

def test_reset_drains_multiple_inspection_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_defaults(monkeypatch)
    settings = DakeraSettings.from_env()
    client = FakeDakeraClient()

    agent_id = settings.agent_id_for("run-large")
    client.memories[agent_id] = [
        {
            "id": f"memory-{index}",
            "content": f"content-{index}",
            "memory_type": "semantic",
            "importance": 0.5,
            "metadata": {},
            "created_at": "2026-10-11T00:00:00Z",
        }
        for index in range(1_205)
    ]

    runtime = DakeraRuntime(
        settings,
        "run-large",
        client=client,
    ).initialize(require_empty=False)

    runtime.reset_and_verify_empty()

    assert runtime.memory_count() == 0
    assert runtime.inspect_state(limit=1) == []