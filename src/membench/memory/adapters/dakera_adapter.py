"""Canonical adapter for the run-scoped Dakera memory runtime."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

from membench.dakera import (
    DakeraMemoryType,
    DakeraRecord,
    DakeraRouting,
    DakeraRuntime,
    DakeraSettings,
)
from membench.dakera import (
    package_version as dakera_package_version,
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
_NATIVE_CONVERSATION_OPTIONS = frozenset({"memory_type", "importance"})
_EMPTY_STATE_QUERY = "membench empty-state isolation canary"
_INSPECTION_LIMIT = 10_000


class DakeraAdapter:
    """Translate the common memory lifecycle into Dakera operations."""

    def __init__(self, settings: DakeraSettings) -> None:
        self.settings = settings
        self._scope: MemoryScope | None = None
        self._runtime: DakeraRuntime | None = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Return features available under the resolved Dakera settings."""
        return AdapterCapabilities(
            direct_write=True,
            native_conversation_write=False,
            semantic_memory=True,
            episodic_memory=True,
            procedural_memory=True,
            working_memory=True,
            updates=False,
            deletion=True,
            ttl=self.settings.ttl_seconds is not None,
            metadata_filters=False,
            threshold=True,
            reranking=self.settings.rerank,
            hybrid_search=self.settings.routing == DakeraRouting.HYBRID,
            associations=self.settings.include_associated,
        )

    @staticmethod
    def _agent_id_for_scope(
        settings: DakeraSettings,
        scope: MemoryScope,
    ) -> str:
        identity = "\x1f".join(
            (
                "dakera",
                scope.experiment_id,
                scope.run_id,
                scope.user_id,
                *scope.namespace,
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        return f"{settings.agent_prefix}-{digest}"

    def _require_runtime(
        self,
        request_scope: MemoryScope | None = None,
    ) -> DakeraRuntime:
        if self._runtime is None or self._scope is None:
            raise AdapterConfigurationError("Dakera adapter is not initialized")
        if request_scope is not None and request_scope != self._scope:
            raise AdapterConfigurationError(
                "request scope does not match the initialized Dakera scope"
            )
        return self._runtime

    def _adapter_metadata(
        self,
        operation_id: str,
        memory_type: DakeraMemoryType,
        supplied: Mapping[str, Any] | None = None,
        *,
        message_metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._scope is None:
            raise AdapterConfigurationError("Dakera adapter is not initialized")
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
                "membench_user_id": self._scope.user_id,
                "membench_namespace": list(self._scope.namespace),
                "membench_operation_id": operation_id,
                "membench_memory_type": memory_type.value,
            }
        )
        if message_metadata and any(message_metadata):
            metadata["membench_message_metadata"] = message_metadata
        return metadata

    def _memory_type(self, memory_type: MemoryType) -> DakeraMemoryType:
        if memory_type == MemoryType.UNSPECIFIED:
            return self.settings.memory_type
        try:
            return DakeraMemoryType(memory_type.value)
        except ValueError as exc:
            raise AdapterConfigurationError(
                f"Dakera does not support {memory_type.value!r} memory"
            ) from exc

    @staticmethod
    def _canonical_id(scope: MemoryScope, vendor_id: str) -> str:
        return canonical_memory_id("dakera", scope, vendor_id)

    def _canonical_record(self, record: DakeraRecord) -> MemoryRecord:
        if self._scope is None:
            raise AdapterConfigurationError("Dakera adapter is not initialized")

        run_id = record.metadata.get("membench_run_id")
        if run_id is not None and run_id != self._scope.run_id:
            raise ValueError("Dakera returned a record from a different run")
        user_id = record.metadata.get("membench_user_id")
        if user_id is not None and user_id != self._scope.user_id:
            raise ValueError("Dakera returned a record from a different user")

        return MemoryRecord(
            canonical_id=self._canonical_id(self._scope, record.vendor_id),
            vendor_id=record.vendor_id,
            scope=self._scope,
            text=record.text,
            memory_type=canonical_memory_type(
                record.memory_type,
                mechanism="Dakera",
            ),
            state=normalize_memory_state(
                status=str(record.raw_value.get("status", "")),
                expired=record.raw_value.get("is_expired") is True,
            ),
            source_event_ids=source_event_ids(
                record.metadata,
                mechanism="Dakera",
            ),
            created_at=parse_timestamp(
                record.created_at,
                mechanism="Dakera",
                field_name="created_at",
            ),
            attributes={
                "importance": record.importance,
                "metadata": record.metadata,
            },
            raw_vendor=record.raw_value,
        )

    def _write_result(
        self,
        operation_id: str,
        record: DakeraRecord,
        latency_seconds: float,
    ) -> WriteResult:
        if self._scope is None:
            raise AdapterConfigurationError("Dakera adapter is not initialized")
        return WriteResult(
            operation_id=operation_id,
            status=WriteStatus.SUCCEEDED,
            outcomes=(
                WriteOutcome(
                    action=WriteAction.CREATED,
                    canonical_id=self._canonical_id(
                        self._scope,
                        record.vendor_id,
                    ),
                    vendor_id=record.vendor_id,
                    text=record.text,
                ),
            ),
            latency_seconds=latency_seconds,
            raw_vendor={"result": record.raw_value},
        )

    def initialize(self, scope: MemoryScope) -> None:
        """Bind a scope-derived Dakera agent without clearing its state."""
        if self._runtime is not None:
            raise AdapterConfigurationError("adapter already initialized")

        runtime = DakeraRuntime(
            settings=self.settings,
            run_id=scope.run_id,
            agent_id=self._agent_id_for_scope(self.settings, scope),
        )
        try:
            runtime.initialize(require_empty=False)
        except Exception as exc:
            runtime.close()
            raise AdapterUnavailableError("Dakera initialization failed") from exc
        self._scope = scope
        self._runtime = runtime

    def write_direct(self, request: DirectWriteRequest) -> WriteResult:
        """Store one approved memory through Dakera's direct write endpoint."""
        runtime = self._require_runtime(request.scope)
        memory_type = self._memory_type(request.memory_type)
        metadata = self._adapter_metadata(
            request.operation_id,
            memory_type,
            request.metadata,
        )
        started = time.perf_counter()
        try:
            record = runtime.store_memory(
                request.text,
                memory_type=memory_type,
                importance=self.settings.importance,
                metadata=metadata,
                source_event_id=request.source_event_id,
            )
        except Exception as exc:
            raise WriteTransportError(
                f"Dakera direct write failed for {request.operation_id!r}"
            ) from exc
        return self._write_result(
            request.operation_id,
            record,
            time.perf_counter() - started,
        )

    def write_conversation(self, request: ConversationWriteRequest) -> WriteResult:
        """Serialize a conversation into one explicitly typed Dakera memory."""
        runtime = self._require_runtime(request.scope)
        unknown = set(request.native_options) - _NATIVE_CONVERSATION_OPTIONS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise AdapterConfigurationError(
                f"unsupported Dakera conversation options: {names}"
            )

        memory_type_value = request.native_options.get("memory_type")
        if memory_type_value is None:
            memory_type = self.settings.memory_type
        else:
            try:
                memory_type = DakeraMemoryType(memory_type_value)
            except (TypeError, ValueError) as exc:
                allowed = ", ".join(item.value for item in DakeraMemoryType)
                raise AdapterConfigurationError(
                    f"Dakera memory_type must be one of: {allowed}"
                ) from exc

        importance = request.native_options.get(
            "importance",
            self.settings.importance,
        )
        if (
            isinstance(importance, bool)
            or not isinstance(importance, (int, float))
            or not 0 <= importance <= 1
        ):
            raise AdapterConfigurationError(
                "Dakera importance must be between zero and one"
            )

        messages = [
            {"role": message.role.value, "content": message.content}
            for message in request.messages
        ]
        message_metadata = [dict(message.metadata) for message in request.messages]
        metadata = self._adapter_metadata(
            request.operation_id,
            memory_type,
            message_metadata=message_metadata,
        )
        started = time.perf_counter()
        try:
            record = runtime.submit_conversation(
                messages,
                memory_type=memory_type,
                importance=float(importance),
                metadata=metadata,
                source_event_id=request.source_event_id,
            )
        except Exception as exc:
            raise WriteTransportError(
                f"Dakera conversation write failed for {request.operation_id!r}"
            ) from exc
        return self._write_result(
            request.operation_id,
            record,
            time.perf_counter() - started,
        )

    def flush_and_wait(self) -> FlushResult:
        """Wait until every Dakera-acknowledged write is inspectable."""
        runtime = self._require_runtime()
        started = time.perf_counter()
        try:
            native_records = runtime.flush_and_wait()
            records = tuple(
                self._canonical_record(record) for record in native_records
            )
        except Exception as exc:
            if "timed out" in str(exc).casefold():
                raise WriteVisibilityTimeout(
                    "Dakera write visibility timed out"
                ) from exc
            raise WriteTransportError("Dakera write flush failed") from exc
        return FlushResult(
            completed=True,
            visible_records=records,
            pending_ids=(),
            latency_seconds=time.perf_counter() - started,
            raw_vendor={
                "results": [record.raw_value for record in native_records]
            },
        )

    def _retrieval_memory_types(
        self,
        requested: Sequence[MemoryType],
    ) -> tuple[DakeraMemoryType, ...]:
        selected: list[DakeraMemoryType] = []
        for memory_type in requested:
            if memory_type == MemoryType.UNSPECIFIED:
                raise AdapterConfigurationError(
                    "explicit unspecified memory is not valid for Dakera retrieval"
                )
            selected.append(DakeraMemoryType(memory_type.value))
        return tuple(selected)

    def _score_kind(self) -> ScoreKind:
        if self.settings.rerank:
            return ScoreKind.RERANK
        if self.settings.routing == DakeraRouting.VECTOR:
            return ScoreKind.SIMILARITY
        return ScoreKind.RELEVANCE

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Recall ranked records from only the bound Dakera agent."""
        runtime = self._require_runtime(request.scope)
        if request.filters:
            raise AdapterConfigurationError(
                "Dakera adapter does not support per-request metadata filters"
            )
        if request.native_options:
            raise AdapterConfigurationError(
                "Dakera retrieval native_options are fixed by its settings"
            )
        memory_types = self._retrieval_memory_types(request.memory_types)

        started = time.perf_counter()
        try:
            if memory_types:
                candidates = [
                    record
                    for memory_type in memory_types
                    for record in runtime.retrieve(
                        request.query,
                        top_k=request.top_k,
                        memory_type=memory_type,
                    )
                ]
                unique = {record.vendor_id: record for record in candidates}
                native_records = sorted(
                    unique.values(),
                    key=lambda record: (
                        record.score is not None,
                        record.score if record.score is not None else float("-inf"),
                    ),
                    reverse=True,
                )[: request.top_k]
            else:
                native_records = runtime.retrieve(
                    request.query,
                    top_k=request.top_k,
                )
            records = tuple(
                self._canonical_record(record) for record in native_records
            )
        except Exception as exc:
            raise RetrievalTransportError(
                f"Dakera retrieval failed for {request.operation_id!r}"
            ) from exc

        score_kind = self._score_kind()
        hits = tuple(
            RetrievalHit(
                record=record,
                rank=rank,
                raw_score=native.score,
                score_kind=(
                    score_kind if native.score is not None else ScoreKind.UNKNOWN
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
        """Enumerate the bound Dakera agent without a semantic query."""
        runtime = self._require_runtime()
        if self._scope is None:
            raise AdapterConfigurationError("Dakera adapter is not initialized")
        try:
            native_records = runtime.inspect_state(limit=_INSPECTION_LIMIT)
            total_memories = runtime.memory_count()
            records = tuple(
                self._canonical_record(record) for record in native_records
            )
            
        except Exception as exc:
            raise StateInspectionError("Dakera state inspection failed") from exc


        return MemorySnapshot(
            scope=self._scope,
            records=records,
            complete=len(native_records) >= total_memories,
            reported_total_count=total_memories,
            inspection_limit=_INSPECTION_LIMIT,
            raw_vendor={
                "agent_id": runtime.agent_id,
                "requested_limit": _INSPECTION_LIMIT,
                "returned_count": len(records),
                "total_memories": total_memories,
                "unobserved_count": max(0, total_memories - len(records)),
                "coverage_ratio": (
                    1.0
                    if total_memories == 0
                    else len(records) / total_memories
                ),
                "results": [record.raw_value for record in native_records],
            },
        )

    def verify_empty(self) -> EmptyStateResult:
        """Prove the bound Dakera agent has no visible or pending memories."""
        runtime = self._require_runtime()
        if self._scope is None:
            raise AdapterConfigurationError("Dakera adapter is not initialized")
        try:
            inspected = runtime.inspect_state(limit=1)
            retrieved = runtime.retrieve(_EMPTY_STATE_QUERY, top_k=1)
            pending_count = runtime.pending_count
        except Exception as exc:
            raise StateInspectionError(
                "Dakera empty-state verification failed"
            ) from exc

        result = EmptyStateResult(
            scope=self._scope,
            empty=not (inspected or retrieved or pending_count),
            inspected_count=len(inspected),
            retrieved_count=len(retrieved),
            pending_count=pending_count,
            raw_vendor={
                "agent_id": runtime.agent_id,
                "inspection_results": [record.raw_value for record in inspected],
                "retrieval_results": [record.raw_value for record in retrieved],
                "pending_count": pending_count,
            },
        )
        if not result.empty:
            raise NamespaceNotEmptyError(
                "Dakera scope is not empty: "
                f"inspected={result.inspected_count}, "
                f"retrieved={result.retrieved_count}, "
                f"pending={result.pending_count}"
            )
        return result

    def reset_and_verify_empty(self) -> EmptyStateResult:
        """Reset only the bound Dakera agent and prove it is empty."""
        runtime = self._require_runtime()
        try:
            runtime.reset_and_verify_empty()
            return self.verify_empty()
        except NamespaceNotEmptyError as exc:
            raise ResetVerificationError(
                "Dakera scope is not empty after reset"
            ) from exc
        except Exception as exc:
            raise ResetVerificationError(
                "Dakera reset or empty-state verification failed"
            ) from exc

    def export_provenance(self) -> AdapterProvenance:
        """Return non-secret SDK, server, and adapter configuration details."""
        runtime = self._require_runtime()
        if self._scope is None:
            raise AdapterConfigurationError("Dakera adapter is not initialized")
        sdk_version = dakera_package_version()
        health = runtime.health or {}
        server_version = (
            health.get("version")
            or self.settings.expected_server_version
            or "unknown"
        )
        return build_adapter_provenance(
            mechanism="dakera",
            mechanism_version=str(server_version),
            storage_backend="dakera-server",
            scope=self._scope,
            package_versions={"dakera": sdk_version},
            native_configuration={
                "settings": self.settings.public_config(),
                "agent_id": runtime.agent_id,
                "server_health": health,
                "capabilities": self.capabilities.as_dict(),
            },
        )

    def close(self) -> None:
        """Close the owned Dakera runtime and make repeated calls harmless."""
        runtime = self._runtime
        self._runtime = None
        self._scope = None
        if runtime is not None:
            runtime.close()
