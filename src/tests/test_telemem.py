import sys
from pathlib import Path
from typing import Any

import pytest

from membench.telemem import (
    TeleMemError,
    TeleMemRuntime,
    TeleMemSettings,
    create_memory,
)


def _set_runtime_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TELEMEM_PROVIDER", "ollama")
    monkeypatch.setenv("TELEMEM_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL_DIGEST", "chat-digest")
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:137m-v1.5-fp16"
    )
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL_DIGEST", "embedding-digest")
    monkeypatch.setenv("OLLAMA_EMBEDDING_DIMENSIONS", "768")
    monkeypatch.setenv("TELEMEM_BUFFER_SIZE", "64")
    monkeypatch.setenv("TELEMEM_SIMILARITY_THRESHOLD", "0.95")
    monkeypatch.setenv("TELEMEM_RETRIEVAL_THRESHOLD", "0.2")
    monkeypatch.setenv("TELEMEM_RERANK", "false")
    monkeypatch.setenv("TELEMEM_VISIBILITY_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("TELEMEM_POLL_INTERVAL_SECONDS", "0.001")


class FakeTeleMemory:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.memory_buffer: dict[str, list[dict[str, Any]]] = {}
        self.next_id = 1
        self.closed = False

    def add(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        if kwargs["infer"]:
            contents = [
                "The user takes astronomy notes in amber ink."
            ]
        else:
            contents = [
                message["content"]
                for message in messages
                if message["role"] != "system"
            ]

        results: list[dict[str, Any]] = []
        for content in contents:
            record = {
                "id": f"memory-{self.next_id}",
                "memory": content,
                "user_id": kwargs["user_id"],
                "run_id": kwargs["run_id"],
                "role": "user",
                "metadata": dict(kwargs["metadata"]),
                "created_at": "2026-09-07T00:00:00Z",
            }
            self.next_id += 1
            self.records.append(record)
            results.append(
                {"id": record["id"], "memory": content, "event": "ADD"}
            )
        return {"results": results}

    def search(self, _query: str, **kwargs: Any) -> dict[str, Any]:
        records = [
            {**record, "score": 0.9}
            for record in self.records
            if record["user_id"] == kwargs["user_id"]
            and record["run_id"] == kwargs["run_id"]
        ]
        return {"results": records[: kwargs["limit"]]}

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        filters = kwargs["filters"]
        records = [
            record
            for record in self.records
            if record["user_id"] == filters["user_id"]
            and record["run_id"] == filters["run_id"]
        ]
        return {"results": records[: kwargs["top_k"]]}

    def delete_all(self, **kwargs: Any) -> dict[str, str]:
        self.records = [
            record
            for record in self.records
            if not (
                record["user_id"] == kwargs["user_id"]
                and record["run_id"] == kwargs["run_id"]
            )
        ]
        return {"message": "Memories deleted successfully!"}

    def close(self) -> None:
        self.closed = True


def test_ollama_settings_build_isolated_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TELEMEM_PROVIDER", "ollama")
    monkeypatch.setenv("TELEMEM_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    monkeypatch.setenv(
        "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:137m-v1.5-fp16"
    )
    monkeypatch.setenv("OLLAMA_EMBEDDING_DIMENSIONS", "768")

    settings = TeleMemSettings.from_env()
    config = settings.memory_config("run-001")

    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["config"]["model"] == "qwen2.5:7b"
    assert config["embedder"]["config"]["embedding_dims"] == 768
    assert config["vector_store"]["config"]["path"] == str(
        tmp_path / "run-001" / "faiss"
    )
    assert config["history_db_path"] == str(tmp_path / "run-001" / "history.db")


def test_openai_settings_are_independent_from_ollama(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TELEMEM_PROVIDER", "openai")
    monkeypatch.setenv("TELEMEM_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("TELEMEM_OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("TELEMEM_OPENAI_CHAT_MODEL", "test-chat-model")
    monkeypatch.setenv("TELEMEM_OPENAI_EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setenv("TELEMEM_OPENAI_EMBEDDING_DIMENSIONS", "1536")

    settings = TeleMemSettings.from_env()
    config = settings.memory_config("run-openai")

    assert settings.provider == "openai"
    assert settings.credential_configured
    assert config["llm"]["config"]["model"] == "test-chat-model"
    assert config["embedder"]["config"]["model"] == "test-embedding-model"
    assert config["llm"]["config"]["api_key"] == "test-secret"


def test_openai_runtime_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEMEM_PROVIDER", "openai")
    # An explicit empty value prevents python-dotenv from loading a developer's
    # real key into this negative test.
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = TeleMemSettings.from_env()

    with pytest.raises(TeleMemError, match="OPENAI_API_KEY"):
        settings.memory_config("run-openai")


def test_run_id_rejects_path_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEMEM_PROVIDER", "ollama")
    settings = TeleMemSettings.from_env()

    with pytest.raises(ValueError, match="run_id"):
        settings.paths_for("../another-run")


def test_runtime_settings_freeze_retrieval_and_model_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_runtime_defaults(monkeypatch, tmp_path)

    settings = TeleMemSettings.from_env()

    assert settings.retrieval_threshold == 0.2
    assert not settings.rerank
    assert settings.llm_model_digest == "chat-digest"
    assert settings.embedding_model_digest == "embedding-digest"
    assert "api_key" not in settings.public_config()


def test_runtime_supports_direct_and_native_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_runtime_defaults(monkeypatch, tmp_path)
    settings = TeleMemSettings.from_env()
    memory = FakeTeleMemory()
    runtime = TeleMemRuntime(
        settings,
        "run-001",
        user_id="persona-001",
        memory=memory,
    ).initialize()

    direct = runtime.store_memory(
        "My field notebook is amber.", source_event_id="event-direct"
    )
    visible_direct = runtime.flush_and_wait()
    native = runtime.submit_conversation(
        [
            {
                "role": "user",
                "content": "Remember that I use amber ink for astronomy notes.",
            },
            {"role": "assistant", "content": "Understood."},
        ],
        infer=True,
        source_event_id="event-native",
    )
    visible_native = runtime.flush_and_wait()
    retrieved = runtime.retrieve("Which ink and notebook color do I use?")

    assert [record.vendor_id for record in visible_direct] == [
        direct[0].vendor_id
    ]
    assert [record.vendor_id for record in visible_native] == [native[0].vendor_id]
    assert all(record.score == 0.9 for record in retrieved)
    assert len(runtime.inspect_state()) == 2
    assert "[memory] My field notebook is amber." in (
        runtime.assemble_context(retrieved)
    )
    assert memory.search(
        "isolation",
        user_id="persona-001",
        run_id="run-001-isolated",
        limit=1,
    )["results"] == []

    runtime.reset_and_verify_empty()

    assert runtime.inspect_state() == []
    runtime.close()


def test_runtime_rejects_reserved_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_runtime_defaults(monkeypatch, tmp_path)
    settings = TeleMemSettings.from_env()
    runtime = TeleMemRuntime(
        settings, "run-001", memory=FakeTeleMemory()
    ).initialize()

    with pytest.raises(ValueError, match="reserved TeleMem metadata"):
        runtime.store_memory(
            "candidate", metadata={"membench_run_id": "another-run"}
        )


def test_owned_memory_is_closed_on_context_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_runtime_defaults(monkeypatch, tmp_path)
    settings = TeleMemSettings.from_env()
    memory = FakeTeleMemory()
    monkeypatch.setattr("membench.telemem.create_memory", lambda *_args: memory)

    with TeleMemRuntime(settings, "run-001"):
        pass

    assert memory.closed


def test_imported_telemem_cannot_switch_mem0_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_runtime_defaults(monkeypatch, tmp_path)
    settings = TeleMemSettings.from_env()
    monkeypatch.setattr("membench.telemem.package_version", lambda: "1.10.0")
    monkeypatch.setitem(sys.modules, "telemem", object())
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "another-run" / "mem0"))

    with pytest.raises(TeleMemError, match="fresh worker process"):
        create_memory(settings, "run-001")
