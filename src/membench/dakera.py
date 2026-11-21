"""Run-isolated Dakera memory configuration and qualification helpers."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv

DAKERA_SDK_VERSION = "0.12.12"


class DakeraError(RuntimeError):
    """Raised when Dakera cannot be configured, reached, or isolated."""


class DakeraMemoryType(StrEnum):
    """Memory families accepted by Dakera's public memory endpoint."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class DakeraRouting(StrEnum):
    """Explicit retrieval routes supported by Dakera recall."""

    AUTO = "auto"
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_AGENT_PREFIX_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_RESERVED_METADATA = frozenset({"membench_run_id", "membench_source_event_id"})


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


def _parse_optional_positive_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when configured")
    return parsed


def _jsonable(value: Any) -> Any:
    """Convert SDK dataclasses and enums without losing raw response fields."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class DakeraSettings:
    """Resolved, non-ambient settings for one Dakera experiment runtime."""

    base_url: str
    memory_type: DakeraMemoryType
    importance: float
    top_k: int
    routing: DakeraRouting
    rerank: bool
    neighborhood: bool
    include_associated: bool
    timeout_seconds: float
    visibility_timeout_seconds: float
    poll_interval_seconds: float
    max_retries: int
    agent_prefix: str
    ttl_seconds: int | None
    expected_server_version: str | None
    server_image: str | None
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls) -> DakeraSettings:
        load_dotenv(override=False)
        api_key = (os.getenv("DAKERA_API_KEY") or "").strip()
        memory_type_text = os.getenv("DAKERA_MEMORY_TYPE", "semantic").strip().lower()
        routing_text = os.getenv("DAKERA_ROUTING", "vector").strip().lower()
        try:
            memory_type = DakeraMemoryType(memory_type_text)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in DakeraMemoryType)
            raise ValueError(f"DAKERA_MEMORY_TYPE must be one of: {allowed}") from exc
        try:
            routing = DakeraRouting(routing_text)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in DakeraRouting)
            raise ValueError(f"DAKERA_ROUTING must be one of: {allowed}") from exc

        settings = cls(
            base_url=os.getenv("DAKERA_URL", "http://127.0.0.1:3300").rstrip("/"),
            api_key=api_key,
            memory_type=memory_type,
            importance=float(os.getenv("DAKERA_IMPORTANCE", "0.5")),
            top_k=int(os.getenv("DAKERA_TOP_K", "5")),
            routing=routing,
            rerank=_parse_bool("DAKERA_RERANK", False),
            neighborhood=_parse_bool("DAKERA_NEIGHBORHOOD", False),
            include_associated=_parse_bool("DAKERA_INCLUDE_ASSOCIATED", False),
            timeout_seconds=float(os.getenv("DAKERA_TIMEOUT_SECONDS", "30")),
            visibility_timeout_seconds=float(
                os.getenv("DAKERA_VISIBILITY_TIMEOUT_SECONDS", "30")
            ),
            poll_interval_seconds=float(
                os.getenv("DAKERA_POLL_INTERVAL_SECONDS", "0.25")
            ),
            max_retries=int(os.getenv("DAKERA_MAX_RETRIES", "3")),
            agent_prefix=os.getenv("DAKERA_AGENT_PREFIX", "membench").strip(),
            ttl_seconds=_parse_optional_positive_int("DAKERA_TTL_SECONDS"),
            expected_server_version=(
                os.getenv("DAKERA_SERVER_VERSION") or ""
            ).strip()
            or None,
            server_image=(os.getenv("DAKERA_SERVER_IMAGE") or "").strip() or None,
        )
        settings.validate_runtime()
        return settings

    @property
    def credential_configured(self) -> bool:
        """Report whether a client credential exists without revealing it."""
        return bool(self.api_key)

    def validate_runtime(self) -> None:
        """Reject ambiguous or mutable runtime configuration."""
        parsed_url = urlsplit(self.base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise DakeraError("DAKERA_URL must be an absolute HTTP(S) URL")
        if parsed_url.username or parsed_url.password:
            raise DakeraError("DAKERA_URL must not contain credentials")
        if parsed_url.query or parsed_url.fragment:
            raise DakeraError("DAKERA_URL must not contain a query or fragment")
        if not self.api_key:
            raise DakeraError(
                "DAKERA_API_KEY is required; do not rely on an unauthenticated fallback"
            )
        if not 0 <= self.importance <= 1:
            raise ValueError("DAKERA_IMPORTANCE must be between zero and one")
        if self.top_k <= 0:
            raise ValueError("DAKERA_TOP_K must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("DAKERA_TIMEOUT_SECONDS must be positive")
        if self.visibility_timeout_seconds <= 0:
            raise ValueError("DAKERA_VISIBILITY_TIMEOUT_SECONDS must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("DAKERA_POLL_INTERVAL_SECONDS must be positive")
        if self.max_retries <= 0:
            raise ValueError("DAKERA_MAX_RETRIES must be positive")
        if not _AGENT_PREFIX_PATTERN.fullmatch(self.agent_prefix):
            raise ValueError(
                "DAKERA_AGENT_PREFIX must start with an alphanumeric character and "
                "contain only letters, numbers, '.', '_' or '-'"
            )
        if self.server_image and not _IMAGE_DIGEST_PATTERN.fullmatch(
            self.server_image
        ):
            raise DakeraError(
                "DAKERA_SERVER_IMAGE must use an immutable @sha256 digest"
            )

    def agent_id_for(self, run_id: str) -> str:
        """Derive the only Dakera agent namespace this runtime may access."""
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(
                "run_id must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_' or '-'"
            )
        return f"{self.agent_prefix}-{run_id}"

    def public_config(self) -> dict[str, Any]:
        """Return reproducibility settings with credentials omitted."""
        return {
            "base_url": self.base_url,
            "credential_configured": self.credential_configured,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "top_k": self.top_k,
            "routing": self.routing.value,
            "rerank": self.rerank,
            "neighborhood": self.neighborhood,
            "include_associated": self.include_associated,
            "timeout_seconds": self.timeout_seconds,
            "visibility_timeout_seconds": self.visibility_timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "max_retries": self.max_retries,
            "agent_prefix": self.agent_prefix,
            "ttl_seconds": self.ttl_seconds,
            "expected_server_version": self.expected_server_version,
            "server_image": self.server_image,
        }


@dataclass(frozen=True)
class DakeraRecord:
    """Canonical view of a Dakera memory plus the untouched SDK response."""

    vendor_id: str
    text: str
    memory_type: DakeraMemoryType
    importance: float
    score: float | None
    metadata: dict[str, Any]
    created_at: str | None
    raw_value: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "vendor_id": self.vendor_id,
            "text": self.text,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "score": self.score,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "raw_value": self.raw_value,
        }


@dataclass(frozen=True)
class DakeraQualification:
    """Non-secret result of one isolated Dakera write/read qualification."""

    run_id: str
    agent_id: str
    sdk_version: str
    server_version: str | None
    write_count: int
    retrieval_count: int
    isolated_agent_empty: bool


def package_version() -> str:
    """Return and enforce the exact Dakera SDK version used by this adapter."""
    try:
        version = importlib.metadata.version("dakera")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DakeraError(
            "Dakera is not installed in this environment; run this command from "
            ".venvs/dakera"
        ) from exc
    if version != DAKERA_SDK_VERSION:
        raise DakeraError(
            f"Dakera SDK version mismatch: expected {DAKERA_SDK_VERSION}, got {version}"
        )
    return version


def create_client(settings: DakeraSettings) -> Any:
    """Create the pinned synchronous SDK client without provider fallback."""
    package_version()
    settings.validate_runtime()
    try:
        from dakera import DakeraClient
    except ImportError as exc:
        raise DakeraError(
            "Dakera is not installed in this environment; run this command from "
            ".venvs/dakera"
        ) from exc
    return DakeraClient(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_seconds,
        connect_timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )

def _normalize_dakera_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DakeraError("Dakera returned an invalid created_at timestamp")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC).isoformat()
    if isinstance(value, str):
        return value
    raise DakeraError("Dakera returned an invalid created_at timestamp")


def _record_from_vendor(value: Any) -> DakeraRecord:
    raw = _jsonable(value)
    if not isinstance(raw, dict):
        raise DakeraError("Dakera returned a memory in an unsupported shape")
    if isinstance(raw.get("memory"), dict):
        envelope = raw
        raw = dict(raw["memory"])
        for key in (
            "score",
            "smart_score",
            "weighted_score",
            "vector_score",
            "text_score",
        ):
            if key in envelope:
                raw[key] = envelope[key]

    vendor_id = raw.get("id", raw.get("memory_id"))
    text = raw.get("content", raw.get("text"))
    if not isinstance(vendor_id, str) or not vendor_id:
        raise DakeraError("Dakera memory response does not contain an ID")
    if not isinstance(text, str) or not text:
        raise DakeraError("Dakera memory response does not contain text")
    try:
        memory_type = DakeraMemoryType(raw.get("memory_type", "episodic"))
    except ValueError as exc:
        raise DakeraError("Dakera returned an unknown memory type") from exc

    score = raw.get("smart_score")
    if score is None:
        score = raw.get("weighted_score")
    if score is None:
        score = raw.get("score")
    metadata = raw.get("metadata")
    return DakeraRecord(
        vendor_id=vendor_id,
        text=text,
        memory_type=memory_type,
        importance=float(raw.get("importance", 0.5)),
        score=float(score) if score is not None else None,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        created_at=_normalize_dakera_timestamp(raw.get("created_at")),
        raw_value=raw,
    )


def _is_namespace_not_found(exc: Exception) -> bool:
    """Identify Dakera's structured missing-agent-namespace response."""
    code = getattr(exc, "code", None)
    code_value = getattr(code, "value", code)
    return (
        exc.__class__.__name__ == "NotFoundError"
        and code_value == "NAMESPACE_NOT_FOUND"
    )


class DakeraRuntime:
    """Deterministic direct-write lifecycle for one run-scoped Dakera agent."""

    def __init__(
        self,
        settings: DakeraSettings,
        run_id: str,
        *,
        agent_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.run_id = run_id
        resolved_agent_id = agent_id or settings.agent_id_for(run_id)
        if not _AGENT_PREFIX_PATTERN.fullmatch(resolved_agent_id):
            raise ValueError(
                "agent_id must start with an alphanumeric character and contain "
                "only letters, numbers, '.', '_' or '-'"
            )
        self.agent_id = resolved_agent_id
        self._client = client
        self._owns_client = client is None
        self._initialized = False
        self._pending_ids: set[str] = set()
        self._health: dict[str, Any] | None = None

    @property
    def health(self) -> dict[str, Any] | None:
        """Return a defensive copy of the non-secret health response."""
        return dict(self._health) if self._health is not None else None

    @property
    def pending_count(self) -> int:
        """Return acknowledged memories still awaiting visibility."""
        return len(self._pending_ids)

    def initialize(self, *, require_empty: bool = True) -> DakeraRuntime:
        """Connect to Dakera and bind this runtime to one empty agent namespace."""
        if self._initialized:
            raise DakeraError("Dakera runtime is already initialized")
        self.settings.validate_runtime()
        if self._client is None:
            self._client = create_client(self.settings)
        try:
            health = self._client.health()
            if not isinstance(health, dict):
                raise DakeraError("Dakera health endpoint returned an invalid response")
            status = str(health.get("status", "")).lower()
            if status and status not in {"healthy", "ok"}:
                raise DakeraError(f"Dakera server is not healthy: {status}")
            server_version = health.get("version")
            if (
                self.settings.expected_server_version
                and server_version != self.settings.expected_server_version
            ):
                raise DakeraError(
                    "Dakera server version mismatch: expected "
                    f"{self.settings.expected_server_version}, got {server_version}"
                )
            self._health = _jsonable(health)
            self._initialized = True
            if require_empty and self.inspect_state(limit=1):
                raise DakeraError(
                    "Dakera agent namespace is not empty for "
                    f"run_id={self.run_id!r}"
                )
        except Exception:
            self.close()
            raise
        return self

    def __enter__(self) -> DakeraRuntime:
        return self.initialize()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_initialized(self) -> Any:
        if not self._initialized or self._client is None:
            raise DakeraError("Dakera runtime has not been initialized")
        return self._client

    def _agent_memories(self, *, limit: int) -> list[Any]:
        client = self._require_initialized()
        try:
            result = client.agent_memories(self.agent_id, limit=limit)
        except Exception as exc:
            if _is_namespace_not_found(exc):
                return []
            raise
        if isinstance(result, dict):
            result = result.get("memories", [])
        if not isinstance(result, list):
            raise DakeraError("Dakera agent memory inspection returned invalid data")
        return result

    def store_memory(
        self,
        content: str,
        *,
        memory_type: DakeraMemoryType | None = None,
        importance: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        source_event_id: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> DakeraRecord:
        """Store one explicit candidate through Dakera's direct memory path."""
        client = self._require_initialized()
        if not content.strip():
            raise ValueError("Dakera memory content must not be empty")
        selected_type = memory_type or self.settings.memory_type
        selected_importance = (
            self.settings.importance if importance is None else importance
        )
        if not 0 <= selected_importance <= 1:
            raise ValueError("Dakera memory importance must be between zero and one")
        resolved_metadata = dict(metadata or {})
        conflict = _RESERVED_METADATA.intersection(resolved_metadata)
        if conflict:
            names = ", ".join(sorted(conflict))
            raise ValueError(f"reserved Dakera metadata cannot be overridden: {names}")
        resolved_metadata["membench_run_id"] = self.run_id
        if source_event_id is not None:
            if not source_event_id.strip():
                raise ValueError("source_event_id must not be empty")
            resolved_metadata["membench_source_event_id"] = source_event_id

        run_tag = f"membench-run:{self.run_id}"
        resolved_tags = list(dict.fromkeys([*(tags or ()), run_tag]))
        response = client.store_memory(
            agent_id=self.agent_id,
            content=content,
            memory_type=selected_type.value,
            importance=selected_importance,
            metadata=resolved_metadata,
            tags=resolved_tags,
            ttl_seconds=self.settings.ttl_seconds,
        )
        record = _record_from_vendor(response)
        self._pending_ids.add(record.vendor_id)
        return record

    def submit_conversation(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        memory_type: DakeraMemoryType | None = None,
        importance: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        source_event_id: str | None = None,
    ) -> DakeraRecord:
        """Normalize one conversation and store it as one direct memory candidate."""
        normalized: list[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError("each message role must be system, user, or assistant")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("each message content must be a non-empty string")
            normalized.append(f"{role}: {content}")
        if not normalized:
            raise ValueError("at least one conversation message is required")
        return self.store_memory(
            "\n".join(normalized),
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
            source_event_id=source_event_id,
        )

    def flush_and_wait(self) -> list[DakeraRecord]:
        """Poll evaluator inspection until every acknowledged write is visible."""
        self._require_initialized()
        expected = set(self._pending_ids)
        if not expected:
            return []
        deadline = time.monotonic() + self.settings.visibility_timeout_seconds
        while True:
            records = self.inspect_state(limit=max(1000, len(expected)))
            visible = {record.vendor_id for record in records}
            if expected.issubset(visible):
                self._pending_ids.difference_update(expected)
                return [record for record in records if record.vendor_id in expected]
            if time.monotonic() >= deadline:
                missing = ", ".join(sorted(expected - visible))
                raise DakeraError(
                    "Dakera write visibility timed out; missing memory IDs: " + missing
                )
            time.sleep(self.settings.poll_interval_seconds)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        memory_type: DakeraMemoryType | None = None,
    ) -> list[DakeraRecord]:
        """Recall ranked memories with every adaptive option explicitly resolved."""
        client = self._require_initialized()
        if not query.strip():
            raise ValueError("Dakera retrieval query must not be empty")
        resolved_top_k = self.settings.top_k if top_k is None else top_k
        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive")
        try:
            response = client.recall(
                agent_id=self.agent_id,
                query=query,
                top_k=resolved_top_k,
                memory_type=(memory_type.value if memory_type else None),
                include_associated=self.settings.include_associated,
                routing=self.settings.routing.value,
                rerank=self.settings.rerank,
                neighborhood=self.settings.neighborhood,
            )
        except Exception as exc:
            if _is_namespace_not_found(exc):
                return []
            raise

        items = response.memories if hasattr(response, "memories") else response
        if isinstance(items, dict):
            items = items.get("memories", [])
        if not isinstance(items, list):
            raise DakeraError("Dakera recall returned invalid data")
        return [_record_from_vendor(item) for item in items]


    def memory_count(self) -> int:
        """Return the server-reported number of memories in this agent namespace."""
        client = self._require_initialized()

        try:
            stats = client.agent_stats(self.agent_id)
        except Exception as exc:
            if _is_namespace_not_found(exc):
                return 0
            raise

        total = stats.get("total_memories")
        if not isinstance(total, int) or total < 0:
            raise DakeraError(
                "Dakera agent stats returned an invalid total_memories value"
            )

        return total

    def inspect_state(self, *, limit: int = 1000) -> list[DakeraRecord]:
        """Return evaluator-only agent state without semantic retrieval."""
        if limit <= 0:
            raise ValueError("inspection limit must be positive")
        records = [
            _record_from_vendor(item)
            for item in self._agent_memories(limit=limit)
        ]
        return sorted(records, key=lambda record: record.vendor_id)

    def reset_and_verify_empty(self) -> None:
        """Delete only this run's derived agent namespace, then verify emptiness."""
        client = self._require_initialized()
        remaining = self.memory_count()

        while remaining > 0:
            batch = self.inspect_state(limit=min(remaining, 1_000))
            if not batch:
                raise DakeraError("Dakera reports a non-empty namespace but returned: "
                f"no deletable records: agent_id={self.agent_id!r}, "
                f"remaining={remaining}")
            previous_count = remaining
            for record in batch:
                client.forget(self.agent_id, record.vendor_id)
            deadline = time.monotonic() + self.settings.visibility_timeout_seconds

            while True:
                remaining = self.memory_count()
                if remaining < previous_count:
                    break

                if time.monotonic() >= deadline:
                    raise DakeraError("Dakera deletion made no visible progress: "
                                      f"agent_id={self.agent_id!r}"
                                      f"remaining={remaining}")
                time.sleep(self.settings.poll_interval_seconds)
        self._pending_ids.clear()

        if self.inspect_state(limit=1):
            raise DakeraError(
                f"Failed to empty Dakera agent namespace {self.agent_id!r}"
            )
        # for record in self.inspect_state(limit=10_000):
        #     client.forget(self.agent_id, record.vendor_id)
        # self._pending_ids.clear()
        # deadline = time.monotonic() + self.settings.visibility_timeout_seconds
        # while self.inspect_state(limit=1):
        #     if time.monotonic() >= deadline:
        #         raise DakeraError(
        #             f"failed to empty Dakera agent namespace {self.agent_id!r}"
        #         )
        #     time.sleep(self.settings.poll_interval_seconds)

    @staticmethod
    def assemble_context(records: Sequence[DakeraRecord]) -> str:
        """Render retrieved records using the same visible fields each time."""
        return "\n".join(
            f"[{record.memory_type.value}] {record.text}" for record in records
        )

    def export_provenance(self, path: Path) -> None:
        """Export non-secret runtime configuration and evaluator-visible state."""
        payload = {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "sdk_version": package_version(),
            "server_health": self.health,
            "configuration": self.settings.public_config(),
            "records": [record.as_dict() for record in self.inspect_state()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def close(self) -> None:
        client = self._client
        self._initialized = False
        self._pending_ids.clear()
        self._health = None
        self._client = None
        if self._owns_client and client is not None and hasattr(client, "close"):
            client.close()


def qualify(settings: DakeraSettings, run_id: str) -> DakeraQualification:
    """Verify empty state, direct storage, recall, and agent isolation."""
    try:
        with DakeraRuntime(settings, run_id) as runtime:
            runtime.store_memory(
                "My field notebook color is amber.",
                memory_type=DakeraMemoryType.SEMANTIC,
                metadata={"qualification": "direct"},
                source_event_id="dakera-qualification-direct",
            )
            writes = runtime.flush_and_wait()
            records = runtime.retrieve(
                "What field notebook color do I prefer?",
                top_k=settings.top_k,
                memory_type=DakeraMemoryType.SEMANTIC,
            )
            if not any("amber" in record.text.lower() for record in records):
                raise DakeraError("Dakera direct memory was not retrieved")

            isolated_agent = settings.agent_id_for(f"{run_id}-isolated")
            client = runtime._require_initialized()
            try:
                isolated = client.agent_memories(isolated_agent, limit=1)
            except Exception as exc:
                if _is_namespace_not_found(exc):
                    isolated = []
                else:
                    raise
            if isinstance(isolated, dict):
                isolated = isolated.get("memories", [])
            if isolated:
                raise DakeraError("Dakera agent namespace isolation failed")

            health = runtime.health or {}
            return DakeraQualification(
                run_id=run_id,
                agent_id=runtime.agent_id,
                sdk_version=package_version(),
                server_version=(
                    str(health["version"])
                    if health.get("version") is not None
                    else None
                ),
                write_count=len(writes),
                retrieval_count=len(records),
                isolated_agent_empty=True,
            )
    except DakeraError:
        raise
    except Exception as exc:
        raise DakeraError(f"Dakera qualification failed: {exc}") from exc
