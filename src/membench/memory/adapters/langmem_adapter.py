"""Canonical adapter for LangMem's selective LangGraph Store writer."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from membench.langmem import (
    LangMemRecord,
    LangMemRuntime,
    LangMemSettings,
)
from membench.langmem import (
    MemoryType as LangMemMemoryType,
)
from membench.langmem import (
    package_versions as langmem_package_versions,
)
from membench.memory.capabilities import AdapterCapabilities
from membench.memory.errors import (
    AdapterConfigurationError,
    AdapterUnavailableError,
    NamespaceNotEmptyError,
    ResetVerificationError,
    RetrievalTransportError,
    StateInspectionError,
    WriteTransportError,
    WriteVisibilityTimeout,
)
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
from membench.memory.normalization import (
    canonical_memory_id,
    canonical_memory_type,
    normalize_memory_state,
    parse_timestamp,
    passes_threshold,
    source_event_ids,
)
from membench.memory.provenance import build_adapter_provenance

_ADAPTER_METADATA_KEYS = frozenset(
    {
        "membench_experiment_id",
        "membench_memory_type",
        "membench_message_metadata",
        "membench_namespace",
        "membench_operation_id",
        "membench_run_id",
        "membench_source_event_id",
        "membench_user_id",
    }
)
_NATIVE_CONVERSATION_OPTIONS = frozenset({"memory_type", "memory_types", "prompt"})
_INSPECTION_LIMIT = 10_000
_EMPTY_STATE_QUERY = "membench empty-state isolation canary"

def normalize_langmem_timestamp(
    value: str | datetime | None,
    *,
    storage_backend: str,
) -> str | datetime | None:
    """Annotate SQLite's timezone-free UTC timestamps before normalization."""
    if value is None:
        return None

    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if storage_backend == "sqlite" and parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


class LangMemAdapter:
    """Translate the common memory lifecycle into LangMem operations."""

    def __init__(self, settings: LangMemSettings) -> None:
        self.settings = settings
        self._scope: MemoryScope | None = None
        self._runtime: LangMemRuntime | None = None
        self._pending_operation_ids: list[str] = []
        self._pending_record_locators: list[tuple[tuple[str, ...], str]] = []

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Return features available under the resolved LangMem settings."""
        enabled = set(self.settings.memory_types)
        return AdapterCapabilities(
            direct_write=True,
            native_conversation_write=True,
            semantic_memory=LangMemMemoryType.SEMANTIC in enabled,
            episodic_memory=LangMemMemoryType.EPISODIC in enabled,
            procedural_memory=LangMemMemoryType.PROCEDURAL in enabled,
            working_memory=False,
            updates=True,
            deletion=self.settings.enable_deletes,
            ttl=False,
            metadata_filters=False,
            threshold=True,
            reranking=False,
            hybrid_search=False,
            associations=False,
        )

    def initialize(self, scope: MemoryScope) -> None:
        """Bind LangMem storage and managers to one canonical scope."""
        if self._runtime is not None:
            raise AdapterConfigurationError("adapter already initialized")

        runtime = LangMemRuntime(
            settings=self.settings,
            run_id=scope.run_id,
            namespace_prefix=scope.namespace,
        )
        try:
            runtime.initialize(require_empty=False)
        except Exception as exc:
            runtime.close()
            raise AdapterUnavailableError("LangMem initialization failed") from exc

        self._scope = scope
        self._runtime = runtime
        self._pending_operation_ids.clear()
        self._pending_record_locators.clear()

    def _require_runtime(
        self, request_scope: MemoryScope | None = None
    ) -> LangMemRuntime:
        if self._runtime is None or self._scope is None:
            raise AdapterConfigurationError("LangMem adapter is not initialized")
        if request_scope is not None and request_scope != self._scope:
            raise AdapterConfigurationError(
                "request scope does not match the initialized LangMem scope"
            )
        return self._runtime

    def _adapter_metadata(
        self,
        operation_id: str,
        memory_types: Sequence[LangMemMemoryType],
        supplied: Mapping[str, Any] | None = None,
        *,
        message_metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._scope is None:
            raise AdapterConfigurationError("LangMem adapter is not initialized")
        metadata = dict(supplied or {})
        conflicts = _ADAPTER_METADATA_KEYS.intersection(metadata)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise AdapterConfigurationError(
                f"reserved adapter metadata cannot be overridden: {names}"
            )
        metadata.update(
            {
                "membench_experiment_id": self._scope.experiment_id,
                "membench_memory_type": ",".join(
                    memory_type.value for memory_type in memory_types
                ),
                "membench_namespace": list(self._scope.namespace),
                "membench_operation_id": operation_id,
                "membench_run_id": self._scope.run_id,
                "membench_user_id": self._scope.user_id,
            }
        )
        if message_metadata and any(message_metadata):
            metadata["membench_message_metadata"] = message_metadata
        return metadata

    @staticmethod
    def _canonical_id(scope: MemoryScope, record: LangMemRecord) -> str:
        return canonical_memory_id(
            "langmem",
            scope,
            record.key,
            qualifiers=(record.memory_type.value,),
        )

    @staticmethod
    def _direct_key(
        scope: MemoryScope,
        operation_id: str,
        memory_type: LangMemMemoryType,
    ) -> str:
        identity = "\x1f".join((*scope.namespace, operation_id, memory_type.value))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _native_memory_type(self, memory_type: MemoryType) -> LangMemMemoryType:
        if memory_type == MemoryType.UNSPECIFIED:
            if len(self.settings.memory_types) == 1:
                return self.settings.memory_types[0]
            raise AdapterConfigurationError(
                "LangMem direct writes must select a memory_type when more than "
                "one family is enabled"
            )
        try:
            selected = LangMemMemoryType(memory_type.value)
        except ValueError as exc:
            raise AdapterConfigurationError(
                f"LangMem does not support {memory_type.value!r} memory"
            ) from exc
        if selected not in self.settings.memory_types:
            raise AdapterConfigurationError(
                f"LangMem memory type is not enabled: {selected.value}"
            )
        return selected

    @staticmethod
    def _parse_native_memory_type(value: Any) -> LangMemMemoryType:
        if not isinstance(value, str) or not value.strip():
            raise AdapterConfigurationError(
                "LangMem memory types must be non-blank strings"
            )
        normalized = value.strip().lower()
        if normalized.endswith("_memory"):
            normalized = normalized.removesuffix("_memory")
        try:
            return LangMemMemoryType(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in LangMemMemoryType)
            raise AdapterConfigurationError(
                f"LangMem memory types must be one of: {allowed}"
            ) from exc

    def _conversation_memory_types(
        self, native_options: Mapping[str, Any]
    ) -> tuple[LangMemMemoryType, ...]:
        unknown = set(native_options) - _NATIVE_CONVERSATION_OPTIONS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise AdapterConfigurationError(
                f"unsupported LangMem conversation options: {names}"
            )

        if native_options.get("prompt") is not None:
            raise AdapterConfigurationError(
                "LangMem extraction instructions are fixed during initialize; "
                "per-request prompt overrides are not supported"
            )

        singular = native_options.get("memory_type")
        plural = native_options.get("memory_types")
        if singular is not None and plural is not None:
            raise AdapterConfigurationError(
                "use either memory_type or memory_types for LangMem, not both"
            )
        if singular is not None:
            selected = (self._parse_native_memory_type(singular),)
        elif plural is not None:
            if (
                not isinstance(plural, (list, tuple))
                or isinstance(plural, (str, bytes))
                or not plural
            ):
                raise AdapterConfigurationError(
                    "LangMem memory_types must be a non-empty list"
                )
            selected = tuple(self._parse_native_memory_type(item) for item in plural)
        else:
            selected = tuple(self.settings.memory_types)

        if len(selected) != len(set(selected)):
            raise AdapterConfigurationError(
                "LangMem memory_types must not contain duplicates"
            )
        disabled = set(selected) - set(self.settings.memory_types)
        if disabled:
            names = ", ".join(sorted(item.value for item in disabled))
            raise AdapterConfigurationError(
                f"LangMem memory types are not enabled: {names}"
            )
        return selected

    def _canonical_record(self, record: LangMemRecord) -> MemoryRecord:
        if self._scope is None:
            raise AdapterConfigurationError("LangMem adapter is not initialized")
        expected_namespace = (*self._scope.namespace, record.memory_type.value)
        if record.namespace != expected_namespace:
            raise ValueError("LangMem returned a record from a different namespace")

        metadata = record.raw_value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        return MemoryRecord(
            canonical_id=self._canonical_id(self._scope, record),
            vendor_id=record.key,
            scope=self._scope,
            text=record.text,
            memory_type=canonical_memory_type(
                record.memory_type,
                mechanism="LangMem",
            ),
            state=normalize_memory_state(event=record.event),
            source_event_ids=source_event_ids(metadata, mechanism="LangMem"),
            created_at=parse_timestamp(
                normalize_langmem_timestamp(
                    record.created_at,
                    storage_backend=self.settings.storage_backend,
                ),
                mechanism="LangMem",
                field_name="created_at",
            ),
            updated_at=parse_timestamp(
                normalize_langmem_timestamp(
                    record.updated_at,
                    storage_backend=self.settings.storage_backend,
                ),
                mechanism="LangMem",
                field_name="updated_at",
            ),
            attributes={
                "metadata": dict(metadata),
                "native_event": record.event,
                "native_namespace": list(record.namespace),
            },
            raw_vendor=record.raw_value,
        )

    def _track_pending(
        self,
        operation_id: str,
        locator: tuple[tuple[str, ...], str] | None = None,
    ) -> None:
        if operation_id not in self._pending_operation_ids:
            self._pending_operation_ids.append(operation_id)
        if locator is not None and locator not in self._pending_record_locators:
            self._pending_record_locators.append(locator)

    def write_direct(self, request: DirectWriteRequest) -> WriteResult:
        """Write an approved memory directly to the scoped LangGraph Store."""
        runtime = self._require_runtime(request.scope)
        native_type = self._native_memory_type(request.memory_type)
        metadata = self._adapter_metadata(
            request.operation_id,
            (native_type,),
            request.metadata,
        )
        key = self._direct_key(request.scope, request.operation_id, native_type)
        started = time.perf_counter()
        try:
            record = runtime.store_memory(
                request.text,
                memory_type=native_type,
                key=key,
                metadata=metadata,
                source_event_id=request.source_event_id,
            )
        except Exception as exc:
            raise WriteTransportError(
                f"LangMem direct write failed for {request.operation_id!r}"
            ) from exc

        canonical_id = self._canonical_id(request.scope, record)
        action = (
            WriteAction.UPDATED
            if (record.event or "ADD").upper() == "UPDATE"
            else WriteAction.CREATED
        )
        self._track_pending(
            request.operation_id,
            (record.namespace, record.key),
        )
        return WriteResult(
            operation_id=request.operation_id,
            status=WriteStatus.SUCCEEDED,
            outcomes=(
                WriteOutcome(
                    action=action,
                    canonical_id=canonical_id,
                    vendor_id=record.key,
                    text=record.text,
                ),
            ),
            latency_seconds=time.perf_counter() - started,
            raw_vendor={"result": record.raw_value},
        )

    def write_conversation(self, request: ConversationWriteRequest) -> WriteResult:
        """Queue a conversation for LangMem extraction at the flush barrier."""
        runtime = self._require_runtime(request.scope)
        memory_types = self._conversation_memory_types(request.native_options)
        messages = [
            {"role": message.role.value, "content": message.content}
            for message in request.messages
        ]
        message_metadata = [dict(message.metadata) for message in request.messages]
        metadata = self._adapter_metadata(
            request.operation_id,
            memory_types,
            message_metadata=message_metadata,
        )

        started = time.perf_counter()
        try:
            runtime.submit_conversation(
                messages,
                memory_types=memory_types,
                metadata=metadata,
                source_event_id=request.source_event_id,
            )
        except Exception as exc:
            raise WriteTransportError(
                f"LangMem conversation write failed for {request.operation_id!r}"
            ) from exc

        self._track_pending(request.operation_id)
        return WriteResult(
            operation_id=request.operation_id,
            status=WriteStatus.QUEUED,
            outcomes=(),
            latency_seconds=time.perf_counter() - started,
            raw_vendor={
                "queued": True,
                "memory_types": [item.value for item in memory_types],
                "message_count": len(messages),
            },
        )

    @staticmethod
    def _write_locators(
        writes: Mapping[str, list[dict[str, Any]]],
    ) -> list[tuple[tuple[str, ...], str]]:
        locators: list[tuple[tuple[str, ...], str]] = []
        for family_writes in writes.values():
            for write in family_writes:
                namespace = write.get("namespace")
                key = write.get("key")
                if (
                    not isinstance(namespace, (list, tuple))
                    or not namespace
                    or not isinstance(key, str)
                    or not key
                ):
                    raise WriteTransportError(
                        "LangMem returned an invalid write descriptor"
                    )
                locator = (tuple(str(part) for part in namespace), key)
                if locator not in locators:
                    locators.append(locator)
        return locators

    def flush_and_wait(self) -> FlushResult:
        """Run queued extraction and verify every reported write is visible."""
        runtime = self._require_runtime()
        pending_ids = tuple(self._pending_operation_ids)
        started = time.perf_counter()
        try:
            native_writes = runtime.flush_and_wait()
            locators = list(self._pending_record_locators)
            for locator in self._write_locators(native_writes):
                if locator not in locators:
                    locators.append(locator)

            state = (
                runtime.inspect_state(limit_per_type=_INSPECTION_LIMIT)
                if locators
                else []
            )
            records_by_locator = {
                (record.namespace, record.key): record for record in state
            }
            missing = [
                locator for locator in locators if locator not in records_by_locator
            ]
            if missing:
                raise WriteVisibilityTimeout(
                    f"LangMem flush left {len(missing)} write(s) invisible"
                )
            native_visible = tuple(
                records_by_locator[locator] for locator in locators
            )
            visible = tuple(
                self._canonical_record(record) for record in native_visible
            )
        except WriteVisibilityTimeout:
            raise
        except Exception as exc:
            if "timed out" in str(exc).casefold():
                raise WriteVisibilityTimeout(
                    "LangMem write visibility timed out"
                ) from exc
            raise WriteTransportError("LangMem write flush failed") from exc

        self._pending_operation_ids.clear()
        self._pending_record_locators.clear()
        return FlushResult(
            completed=True,
            visible_records=visible,
            pending_ids=(),
            latency_seconds=time.perf_counter() - started,
            raw_vendor={
                "operation_ids": list(pending_ids),
                "writes": native_writes,
                "results": [record.raw_value for record in native_visible],
            },
        )

    def _retrieval_memory_types(
        self, requested: Sequence[MemoryType]
    ) -> tuple[LangMemMemoryType, ...]:
        if not requested:
            return tuple(self.settings.memory_types)

        selected: list[LangMemMemoryType] = []
        for memory_type in requested:
            if memory_type in {MemoryType.UNSPECIFIED, MemoryType.WORKING}:
                raise AdapterConfigurationError(
                    f"LangMem retrieval does not support {memory_type.value!r} "
                    "memory"
                )
            native_type = LangMemMemoryType(memory_type.value)
            if native_type not in self.settings.memory_types:
                raise AdapterConfigurationError(
                    f"LangMem memory type is not enabled: {native_type.value}"
                )
            selected.append(native_type)
        return tuple(selected)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Search enabled memory families inside the initialized scope."""
        runtime = self._require_runtime(request.scope)
        if request.filters:
            raise AdapterConfigurationError(
                "LangMem adapter does not support per-request metadata filters"
            )
        if request.native_options:
            raise AdapterConfigurationError(
                "LangMem retrieval native_options are fixed by its settings"
            )
        memory_types = self._retrieval_memory_types(request.memory_types)

        started = time.perf_counter()
        try:
            native_records = runtime.retrieve(
                request.query,
                top_k=request.top_k,
                memory_types=memory_types,
            )
            records = tuple(
                self._canonical_record(record) for record in native_records
            )
        except Exception as exc:
            raise RetrievalTransportError(
                f"LangMem retrieval failed for {request.operation_id!r}"
            ) from exc

        hits = tuple(
            RetrievalHit(
                record=record,
                rank=rank,
                raw_score=native.score,
                score_kind=(
                    ScoreKind.SIMILARITY
                    if native.score is not None
                    else ScoreKind.UNKNOWN
                ),
                passed_threshold=passes_threshold(
                    native.score,
                    request.threshold,
                ),
            )
            for rank, (record, native) in enumerate(
                zip(records, native_records, strict=True),
                start=1,
            )
        )
        return RetrievalResult(
            operation_id=request.operation_id,
            query_sha256=request.query_sha256,
            request_top_k=request.top_k,
            threshold=request.threshold,
            hits=hits,
            latency_seconds=time.perf_counter() - started,
            raw_vendor={
                "memory_types": [item.value for item in memory_types],
                "results": [record.raw_value for record in native_records],
            },
        )

    def inspect_state(self) -> MemorySnapshot:
        """Enumerate visible scoped records without a semantic query."""
        runtime = self._require_runtime()
        if self._scope is None:
            raise AdapterConfigurationError("LangMem adapter is not initialized")
        try:
            native_records = runtime.inspect_state(
                limit_per_type=_INSPECTION_LIMIT
            )
            records = tuple(
                self._canonical_record(record) for record in native_records
            )
        except Exception as exc:
            raise StateInspectionError("LangMem state inspection failed") from exc

        counts = {
            memory_type.value: sum(
                record.memory_type == memory_type for record in native_records
            )
            for memory_type in self.settings.memory_types
        }
        complete = all(count < _INSPECTION_LIMIT for count in counts.values())
        return MemorySnapshot(
            scope=self._scope,
            records=records,
            complete=complete,
            raw_vendor={
                "counts_by_memory_type": counts,
                "limit_per_type": _INSPECTION_LIMIT,
                "pending_ids": list(self._pending_operation_ids),
                "results": [record.raw_value for record in native_records],
            },
        )

    def verify_empty(self) -> EmptyStateResult:
        """Prove all bound memory-family namespaces and queues are empty."""
        runtime = self._require_runtime()
        if self._scope is None:
            raise AdapterConfigurationError("LangMem adapter is not initialized")
        try:
            inspected = runtime.inspect_state(limit_per_type=1)
            retrieved = runtime.retrieve(_EMPTY_STATE_QUERY, top_k=1)
            pending_count = max(
                len(self._pending_operation_ids),
                runtime.pending_count,
            )
        except Exception as exc:
            raise StateInspectionError(
                "LangMem empty-state verification failed"
            ) from exc

        result = EmptyStateResult(
            scope=self._scope,
            empty=not (inspected or retrieved or pending_count),
            inspected_count=len(inspected),
            retrieved_count=len(retrieved),
            pending_count=pending_count,
            raw_vendor={
                "inspection_results": [record.raw_value for record in inspected],
                "retrieval_results": [record.raw_value for record in retrieved],
                "pending_operation_ids": list(self._pending_operation_ids),
                "native_pending_count": runtime.pending_count,
            },
        )
        if not result.empty:
            raise NamespaceNotEmptyError(
                "LangMem scope is not empty: "
                f"inspected={result.inspected_count}, "
                f"retrieved={result.retrieved_count}, "
                f"pending={result.pending_count}"
            )
        return result

    def reset_and_verify_empty(self) -> EmptyStateResult:
        """Clear only the bound scope and prove no memory remains visible."""
        runtime = self._require_runtime()
        try:
            runtime.reset_and_verify_empty()
            self._pending_operation_ids.clear()
            self._pending_record_locators.clear()
            return self.verify_empty()
        except NamespaceNotEmptyError as exc:
            raise ResetVerificationError(
                "LangMem scope is not empty after reset"
            ) from exc
        except Exception as exc:
            raise ResetVerificationError(
                "LangMem reset or empty-state verification failed"
            ) from exc

    def export_provenance(self) -> AdapterProvenance:
        """Return non-secret versions and resolved LangMem configuration."""
        runtime = self._require_runtime()
        if self._scope is None:
            raise AdapterConfigurationError("LangMem adapter is not initialized")
        versions = langmem_package_versions()
        return build_adapter_provenance(
            mechanism="langmem",
            mechanism_version=versions["langmem"],
            storage_backend=self.settings.storage_backend,
            scope=self._scope,
            model_identifiers={
                "writer": self.settings.writer_model,
                "embedding": self.settings.embedding_model,
            },
            package_versions=versions,
            native_configuration={
                "settings": self.settings.public_config(),
                "storage_path": runtime.storage_path,
                "capabilities": self.capabilities.as_dict(),
            },
        )

    def close(self) -> None:
        """Release the runtime and make repeated close calls harmless."""
        runtime = self._runtime
        self._runtime = None
        self._scope = None
        self._pending_operation_ids.clear()
        self._pending_record_locators.clear()
        if runtime is not None:
            runtime.close()
