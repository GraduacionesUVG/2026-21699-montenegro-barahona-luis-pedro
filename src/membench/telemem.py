"""Provider-selectable TeleMem configuration and qualification helpers."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class TeleMemError(RuntimeError):
    """Raised when TeleMem cannot be configured or qualified."""


_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_RESERVED_METADATA = frozenset(
    {
        "user_id",
        "run_id",
        "membench_run_id",
        "membench_source_event_id",
        "membench_infer",
    }
)
_EMPTY_STATE_QUERY = "membench empty-state isolation canary"


def _parse_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _parse_optional_threshold(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    threshold = float(value)
    if not 0 <= threshold <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return threshold


@dataclass(frozen=True)
class TeleMemPaths:
    """Run-specific storage paths required for TeleMem isolation."""

    run_dir: Path
    mem0_dir: Path
    vector_dir: Path
    history_db: Path
    collection_name: str


@dataclass(frozen=True)
class TeleMemSettings:
    """Resolved TeleMem model, provider, and local-storage settings."""

    provider: str
    base_url: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    storage_root: Path
    buffer_size: int
    similarity_threshold: float
    retrieval_threshold: float | None
    rerank: bool
    visibility_timeout_seconds: float
    poll_interval_seconds: float
    llm_model_digest: str | None
    embedding_model_digest: str | None
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> TeleMemSettings:
        load_dotenv(override=False)
        provider = os.getenv("TELEMEM_PROVIDER", "ollama").strip().lower()
        if provider == "ollama":
            base_url = os.getenv(
                "OLLAMA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"
            )
            api_key = os.getenv("OLLAMA_API_KEY", "ollama")
            llm_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
            embedding_model = os.getenv(
                "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:137m-v1.5-fp16"
            )
            dimensions_text = os.getenv("OLLAMA_EMBEDDING_DIMENSIONS", "768")
            llm_model_digest = (
                os.getenv("OLLAMA_CHAT_MODEL_DIGEST") or ""
            ).strip() or None
            embedding_model_digest = (
                os.getenv("OLLAMA_EMBEDDING_MODEL_DIGEST") or ""
            ).strip() or None
        elif provider == "openai":
            base_url = (
                os.getenv("TELEMEM_OPENAI_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )
            api_key = os.getenv("OPENAI_API_KEY") or None
            llm_model = os.getenv("TELEMEM_OPENAI_CHAT_MODEL", "gpt-5-mini")
            embedding_model = os.getenv(
                "TELEMEM_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            )
            dimensions_text = os.getenv(
                "TELEMEM_OPENAI_EMBEDDING_DIMENSIONS", "1536"
            )
            llm_model_digest = None
            embedding_model_digest = None
        else:
            raise ValueError("TELEMEM_PROVIDER must be 'ollama' or 'openai'")

        dimensions = int(dimensions_text)
        buffer_size = int(os.getenv("TELEMEM_BUFFER_SIZE", "64"))
        similarity_threshold = float(
            os.getenv("TELEMEM_SIMILARITY_THRESHOLD", "0.95")
        )
        visibility_timeout_seconds = float(
            os.getenv("TELEMEM_VISIBILITY_TIMEOUT_SECONDS", "30")
        )
        poll_interval_seconds = float(
            os.getenv("TELEMEM_POLL_INTERVAL_SECONDS", "0.25")
        )
        if dimensions <= 0:
            raise ValueError("TeleMem embedding dimensions must be positive")
        if buffer_size <= 0:
            raise ValueError("TELEMEM_BUFFER_SIZE must be positive")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(
                "TELEMEM_SIMILARITY_THRESHOLD must be between zero and one"
            )
        if visibility_timeout_seconds <= 0:
            raise ValueError("TELEMEM_VISIBILITY_TIMEOUT_SECONDS must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("TELEMEM_POLL_INTERVAL_SECONDS must be positive")

        return cls(
            provider=provider,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_dimensions=dimensions,
            storage_root=Path(
                os.getenv("TELEMEM_STORAGE_ROOT", "artifacts/memory/telemem")
            ),
            buffer_size=buffer_size,
            similarity_threshold=similarity_threshold,
            retrieval_threshold=_parse_optional_threshold(
                "TELEMEM_RETRIEVAL_THRESHOLD"
            ),
            rerank=_parse_bool("TELEMEM_RERANK", False),
            visibility_timeout_seconds=visibility_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            llm_model_digest=llm_model_digest,
            embedding_model_digest=embedding_model_digest,
        )

    @property
    def credential_configured(self) -> bool:
        """Report credential presence without exposing its value."""
        return bool(self.api_key)

    def validate_runtime(self) -> None:
        """Reject configurations that would silently fall back or fail remotely."""
        if not self.llm_model:
            raise TeleMemError("TeleMem chat model is empty")
        if not self.embedding_model:
            raise TeleMemError("TeleMem embedding model is empty")
        if self.provider == "openai" and not self.api_key:
            raise TeleMemError(
                "OPENAI_API_KEY is required when TELEMEM_PROVIDER=openai"
            )

    def paths_for(self, run_id: str) -> TeleMemPaths:
        """Resolve storage for one run without touching another run's state."""
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                "run_id must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_' or '-'"
            )
        run_dir = self.storage_root / run_id
        return TeleMemPaths(
            run_dir=run_dir,
            mem0_dir=run_dir / "mem0",
            vector_dir=run_dir / "faiss",
            history_db=run_dir / "history.db",
            collection_name=f"telemem_{run_id}",
        )

    def memory_config(self, run_id: str) -> dict[str, Any]:
        """Build TeleMem's Mem0-compatible configuration for one isolated run."""
        self.validate_runtime()
        paths = self.paths_for(run_id)
        provider_config = {
            "model": self.llm_model,
            "openai_base_url": self.base_url,
            "api_key": self.api_key,
            "temperature": 0,
        }
        embedder_config = {
            "model": self.embedding_model,
            "openai_base_url": self.base_url,
            "api_key": self.api_key,
            "embedding_dims": self.embedding_dimensions,
        }
        return {
            # TeleMem's supported Ollama example intentionally uses the OpenAI
            # provider against Ollama's /v1 compatibility endpoint.
            "llm": {"provider": "openai", "config": provider_config},
            "embedder": {"provider": "openai", "config": embedder_config},
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": paths.collection_name,
                    "path": str(paths.vector_dir),
                    "embedding_model_dims": self.embedding_dimensions,
                },
            },
            "history_db_path": str(paths.history_db),
            "buffer_size": self.buffer_size,
            "similarity_threshold": self.similarity_threshold,
        }

    def public_config(self) -> dict[str, Any]:
        """Return resolved experiment settings without credential values."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "credential_configured": self.credential_configured,
            "llm_model": self.llm_model,
            "llm_model_digest": self.llm_model_digest,
            "embedding_model": self.embedding_model,
            "embedding_model_digest": self.embedding_model_digest,
            "embedding_dimensions": self.embedding_dimensions,
            "storage_root": str(self.storage_root),
            "buffer_size": self.buffer_size,
            "similarity_threshold": self.similarity_threshold,
            "retrieval_threshold": self.retrieval_threshold,
            "rerank": self.rerank,
            "visibility_timeout_seconds": self.visibility_timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
        }


@dataclass(frozen=True)
class TeleMemQualification:
    """Non-secret summary of one isolated TeleMem qualification run."""

    run_id: str
    storage_path: Path
    direct_write_count: int
    direct_search_count: int
    native_write_count: int
    native_search_count: int
    isolated_run_empty: bool = True


@dataclass(frozen=True)
class TeleMemRecord:
    """Canonical TeleMem memory while retaining its complete vendor value."""

    vendor_id: str
    text: str
    score: float | None
    event: str | None
    user_id: str | None
    run_id: str | None
    role: str | None
    created_at: str | None
    updated_at: str | None
    metadata: dict[str, Any]
    raw_value: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "vendor_id": self.vendor_id,
            "text": self.text,
            "score": self.score,
            "event": self.event,
            "user_id": self.user_id,
            "run_id": self.run_id,
            "role": self.role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "raw_value": self.raw_value,
        }


def package_version() -> str:
    """Return the installed TeleMem version without importing the package."""
    try:
        return importlib.metadata.version("telemem")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TeleMemError(
            "TeleMem is not installed in this environment; run this command from "
            ".venvs/telemem"
        ) from exc


def package_versions() -> dict[str, str]:
    """Return the installed TeleMem storage/runtime package versions."""
    packages = ("telemem", "mem0ai", "faiss-cpu")
    try:
        return {package: importlib.metadata.version(package) for package in packages}
    except importlib.metadata.PackageNotFoundError as exc:
        raise TeleMemError(
            "TeleMem dependencies are incomplete; run this command from "
            ".venvs/telemem"
        ) from exc


def create_memory(settings: TeleMemSettings, run_id: str) -> Any:
    """Create TeleMem only after its run-specific MEM0_DIR has been selected."""
    package_version()
    settings.validate_runtime()
    paths = settings.paths_for(run_id)
    target_mem0_dir = paths.mem0_dir.resolve()
    if "telemem" in sys.modules:
        active_mem0_dir = os.getenv("MEM0_DIR")
        wrong_mem0_dir = (
            active_mem0_dir is None
            or Path(active_mem0_dir).resolve() != target_mem0_dir
        )
        if wrong_mem0_dir:
            raise TeleMemError(
                "TeleMem is already imported with a different MEM0_DIR; start a "
                "fresh worker process for each run"
            )
    paths.mem0_dir.mkdir(parents=True, exist_ok=True)
    paths.vector_dir.mkdir(parents=True, exist_ok=True)
    paths.history_db.parent.mkdir(parents=True, exist_ok=True)

    os.environ["MEM0_DIR"] = str(target_mem0_dir)
    os.environ["MEM0_TELEMETRY"] = "False"

    try:
        import telemem
        from telemem.configs import TeleMemoryConfig

        config = TeleMemoryConfig.model_validate(settings.memory_config(run_id))
        return telemem.Memory(config=config)
    except Exception as exc:
        raise TeleMemError(f"TeleMem initialization failed: {exc}") from exc


def _record_from_vendor(value: Any) -> TeleMemRecord:
    if not isinstance(value, Mapping):
        raise TeleMemError("TeleMem returned a memory in an unsupported shape")
    raw = dict(value)
    vendor_id = raw.get("id", raw.get("memory_id"))
    text = raw.get("memory", raw.get("data", raw.get("content")))
    if not isinstance(vendor_id, str) or not vendor_id:
        raise TeleMemError("TeleMem memory response does not contain an ID")
    if not isinstance(text, str) or not text:
        raise TeleMemError("TeleMem memory response does not contain text")
    metadata_value = raw.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    score = raw.get("score")
    return TeleMemRecord(
        vendor_id=vendor_id,
        text=text,
        score=float(score) if score is not None else None,
        event=str(raw["event"]) if raw.get("event") is not None else None,
        user_id=(str(raw["user_id"]) if raw.get("user_id") is not None else None),
        run_id=str(raw["run_id"]) if raw.get("run_id") is not None else None,
        role=str(raw["role"]) if raw.get("role") is not None else None,
        created_at=(
            str(raw["created_at"]) if raw.get("created_at") is not None else None
        ),
        updated_at=(
            str(raw["updated_at"]) if raw.get("updated_at") is not None else None
        ),
        metadata=metadata,
        raw_value=raw,
    )


class TeleMemRuntime:
    """Deterministic direct/native lifecycle for one isolated TeleMem run."""

    def __init__(
        self,
        settings: TeleMemSettings,
        run_id: str,
        *,
        user_id: str = "membench-user",
        memory: Any | None = None,
    ) -> None:
        self.settings = settings
        self.run_id = run_id
        self.paths = settings.paths_for(run_id)
        if not _USER_ID_PATTERN.fullmatch(user_id):
            raise ValueError(
                "user_id must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_' or '-'"
            )
        self.user_id = user_id
        self._memory = memory
        self._owns_memory = memory is None
        self._initialized = False
        self._pending: dict[str, TeleMemRecord] = {}
        self._write_responses: list[dict[str, Any]] = []

    @property
    def pending_count(self) -> int:
        """Return acknowledged writes still awaiting the visibility barrier."""
        return len(self._pending)

    def initialize(self, *, require_empty: bool = True) -> TeleMemRuntime:
        """Create TeleMem after selecting its run path and prove initial state."""
        if self._initialized:
            raise TeleMemError("TeleMem runtime is already initialized")
        self.settings.validate_runtime()
        if self._memory is None:
            self._memory = create_memory(self.settings, self.run_id)
        self._initialized = True
        try:
            if require_empty:
                inspected = self.inspect_state(limit=1)
                retrieved = self.retrieve(_EMPTY_STATE_QUERY, top_k=1)
                if inspected or retrieved:
                    raise TeleMemError(
                        "TeleMem storage is not empty for "
                        f"run_id={self.run_id!r}"
                    )
        except Exception:
            self.close()
            raise
        return self

    def __enter__(self) -> TeleMemRuntime:
        return self.initialize()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_initialized(self) -> Any:
        if not self._initialized or self._memory is None:
            raise TeleMemError("TeleMem runtime has not been initialized")
        return self._memory

    @staticmethod
    def _normalize_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError("each message role must be system, user, or assistant")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("each message content must be a non-empty string")
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("at least one conversation message is required")
        return normalized

    def submit_conversation(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        infer: bool,
        metadata: Mapping[str, Any] | None = None,
        source_event_id: str | None = None,
        memory_type: str | None = None,
        prompt: str | None = None,
    ) -> list[TeleMemRecord]:
        """Submit ordinary conversation with an explicit direct/native choice."""
        memory = self._require_initialized()
        if not isinstance(infer, bool):
            raise ValueError("infer must be an explicit boolean")
        if memory_type not in {None, "procedural_memory"}:
            raise ValueError("memory_type must be null or 'procedural_memory'")
        if prompt is not None and not infer:
            raise ValueError("a TeleMem extraction prompt requires infer=True")
        normalized = self._normalize_messages(messages)
        resolved_metadata = dict(metadata or {})
        conflict = _RESERVED_METADATA.intersection(resolved_metadata)
        if conflict:
            names = ", ".join(sorted(conflict))
            raise ValueError(f"reserved TeleMem metadata cannot be overridden: {names}")
        resolved_metadata["membench_run_id"] = self.run_id
        resolved_metadata["membench_infer"] = infer
        if source_event_id is not None:
            if not source_event_id.strip():
                raise ValueError("source_event_id must not be empty")
            resolved_metadata["membench_source_event_id"] = source_event_id

        response = memory.add(
            normalized,
            user_id=self.user_id,
            run_id=self.run_id,
            metadata=resolved_metadata,
            infer=infer,
            memory_type=memory_type,
            prompt=prompt,
            batch=False,
        )
        if not isinstance(response, Mapping):
            raise TeleMemError("TeleMem add returned an invalid response")
        raw_response = dict(response)
        self._write_responses.append(raw_response)
        results = raw_response.get("results", [])
        if not isinstance(results, list):
            raise TeleMemError("TeleMem add results are not a list")
        records = [_record_from_vendor(item) for item in results]
        for record in records:
            if (record.event or "ADD").upper() != "DELETE":
                self._pending[record.vendor_id] = record
        return records

    def ingest_benign(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        infer: bool,
        metadata: Mapping[str, Any] | None = None,
        source_event_id: str | None = None,
    ) -> list[TeleMemRecord]:
        """Ingest benign replay events through the selected experiment policy."""
        return self.submit_conversation(
            messages,
            infer=infer,
            metadata=metadata,
            source_event_id=source_event_id,
        )

    def store_memory(
        self,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        source_event_id: str | None = None,
    ) -> list[TeleMemRecord]:
        """Store one user-authored candidate through the unfiltered direct path."""
        return self.submit_conversation(
            [{"role": "user", "content": content}],
            infer=False,
            metadata=metadata,
            source_event_id=source_event_id,
        )

    def flush_and_wait(self) -> list[TeleMemRecord]:
        """Wait until every acknowledged non-deletion result is inspectable."""
        self._require_initialized()
        expected = list(self._pending)
        if not expected:
            return []
        deadline = time.monotonic() + self.settings.visibility_timeout_seconds
        while True:
            records = self.inspect_state(limit=max(1000, len(expected)))
            visible = {record.vendor_id: record for record in records}
            if all(vendor_id in visible for vendor_id in expected):
                for vendor_id in expected:
                    self._pending.pop(vendor_id, None)
                return [visible[vendor_id] for vendor_id in expected]
            if time.monotonic() >= deadline:
                missing = ", ".join(
                    vendor_id for vendor_id in expected if vendor_id not in visible
                )
                raise TeleMemError(
                    "TeleMem write visibility timed out; missing memory IDs: "
                    + missing
                )
            time.sleep(self.settings.poll_interval_seconds)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[TeleMemRecord]:
        """Search only the current user/run scope with fixed retrieval options."""
        memory = self._require_initialized()
        if not query.strip():
            raise ValueError("TeleMem retrieval query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        response = memory.search(
            query,
            user_id=self.user_id,
            run_id=self.run_id,
            limit=top_k,
            threshold=self.settings.retrieval_threshold,
            rerank=self.settings.rerank,
        )
        if not isinstance(response, Mapping):
            raise TeleMemError("TeleMem search returned an invalid response")
        results = response.get("results", [])
        if not isinstance(results, list):
            raise TeleMemError("TeleMem search results are not a list")
        return [_record_from_vendor(item) for item in results]

    def inspect_state(self, *, limit: int = 1000) -> list[TeleMemRecord]:
        """Enumerate evaluator-visible state without semantic retrieval."""
        memory = self._require_initialized()
        if limit <= 0:
            raise ValueError("inspection limit must be positive")
        response = memory.get_all(
            filters={"user_id": self.user_id, "run_id": self.run_id},
            top_k=limit,
            show_expired=True,
        )
        if not isinstance(response, Mapping):
            raise TeleMemError("TeleMem get_all returned an invalid response")
        results = response.get("results", [])
        if not isinstance(results, list):
            raise TeleMemError("TeleMem get_all results are not a list")
        records = [_record_from_vendor(item) for item in results]
        return sorted(records, key=lambda record: record.vendor_id)

    def reset_and_verify_empty(self) -> None:
        """Delete only this user/run scope and verify inspection and search."""
        memory = self._require_initialized()
        memory.delete_all(user_id=self.user_id, run_id=self.run_id)
        if hasattr(memory, "memory_buffer"):
            buffer_keys = (
                f"{self.run_id}_person_{self.user_id}",
                f"{self.run_id}_events",
            )
            for key in buffer_keys:
                memory.memory_buffer.pop(key, None)
        self._pending.clear()
        deadline = time.monotonic() + self.settings.visibility_timeout_seconds
        while True:
            inspected = self.inspect_state(limit=1)
            retrieved = self.retrieve(_EMPTY_STATE_QUERY, top_k=1)
            if not inspected and not retrieved:
                return
            if time.monotonic() >= deadline:
                raise TeleMemError(
                    "failed to empty TeleMem storage for "
                    f"run_id={self.run_id!r}"
                )
            time.sleep(self.settings.poll_interval_seconds)

    @staticmethod
    def assemble_context(records: Sequence[TeleMemRecord]) -> str:
        """Render retrieved records without vendor-only fields."""
        return "\n".join(f"[memory] {record.text}" for record in records)

    def export_provenance(self, path: Path) -> None:
        """Export non-secret configuration, raw write results, and current state."""
        payload = {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "packages": package_versions(),
            "configuration": self.settings.public_config(),
            "paths": {
                "run_dir": str(self.paths.run_dir),
                "mem0_dir": str(self.paths.mem0_dir),
                "vector_dir": str(self.paths.vector_dir),
                "history_db": str(self.paths.history_db),
                "collection_name": self.paths.collection_name,
            },
            "write_responses": self._write_responses,
            "records": [record.as_dict() for record in self.inspect_state()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def close(self) -> None:
        memory = self._memory
        self._initialized = False
        self._pending.clear()
        self._memory = None
        if self._owns_memory and memory is not None and hasattr(memory, "close"):
            memory.close()


def qualify(settings: TeleMemSettings, run_id: str) -> TeleMemQualification:
    """Exercise empty state, direct memory, and native-selective memory paths."""
    try:
        with TeleMemRuntime(
            settings, run_id, user_id="qualification-user"
        ) as runtime:
            direct_write = runtime.store_memory(
                "My field notebook color is amber.",
                metadata={"qualification": "direct"},
                source_event_id="telemem-qualification-direct",
            )
            runtime.flush_and_wait()
            direct_results = runtime.retrieve(
                "What field notebook color do I prefer?", top_k=5
            )
            if not any("amber" in record.text.lower() for record in direct_results):
                raise TeleMemError("TeleMem direct write was not retrieved")

            native_write = runtime.submit_conversation(
                [
                    {
                        "role": "user",
                        "content": (
                            "Please remember that I take astronomy notes in amber "
                            "ink."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "I will remember your preference for amber ink.",
                    },
                ],
                infer=True,
                metadata={"qualification": "native_selective"},
                source_event_id="telemem-qualification-native",
            )
            if not native_write:
                raise TeleMemError(
                    "TeleMem native-selective write produced no memory"
                )
            runtime.flush_and_wait()
            native_results = runtime.retrieve(
                "What ink do I use for astronomy notes?", top_k=5
            )
            if not any("amber" in record.text.lower() for record in native_results):
                raise TeleMemError(
                    "TeleMem native-selective memory was not retrieved"
                )

            isolated = runtime._require_initialized().search(
                _EMPTY_STATE_QUERY,
                user_id=runtime.user_id,
                run_id=f"{run_id}-isolated",
                limit=1,
                threshold=settings.retrieval_threshold,
                rerank=settings.rerank,
            )
            isolated_results = (
                isolated.get("results", [])
                if isinstance(isolated, Mapping)
                else ["invalid"]
            )
            if isolated_results:
                raise TeleMemError("TeleMem run namespace isolation failed")

            return TeleMemQualification(
                run_id=run_id,
                storage_path=runtime.paths.run_dir,
                direct_write_count=len(direct_write),
                direct_search_count=len(direct_results),
                native_write_count=len(native_write),
                native_search_count=len(native_results),
                isolated_run_empty=True,
            )
    except TeleMemError:
        raise
    except Exception as exc:
        raise TeleMemError(f"TeleMem qualification failed: {exc}") from exc
