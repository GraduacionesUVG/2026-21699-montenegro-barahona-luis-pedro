from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

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
from membench.telemem import (
    TeleMemRecord,
    TeleMemRuntime,
    TeleMemSettings,
)
from membench.telemem import (
    package_versions as telemem_package_versions,
)

_ADAPTER_METADATA_KEYS = frozenset(
    {
        "membench_experiment_id",
        "membench_memory_type",
        "membench_message_metadata",
        "membench_namespace",
        "membench_operation_id",
    }
)
_NATIVE_CONVERSATION_OPTIONS = frozenset({"memory_type", "prompt"})
_EMPTY_STATE_QUERY = "membench empty-state isolation canary"
_INSPECTION_LIMIT = 10_000


class TeleMemAdapter:
    def __init__(self, settings: TeleMemSettings) -> None:
        self.settings = settings
        self._scope: MemoryScope | None = None
        self._runtime: TeleMemRuntime | None = None

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Return features available under the resolved TeleMem settings."""
        return AdapterCapabilities(
            direct_write=True,
            native_conversation_write=True,
            semantic_memory=True,
            episodic_memory=False,
            procedural_memory=True,
            working_memory=False,
            updates=True,
            deletion=True,
            ttl=False,
            metadata_filters=False,
            threshold=self.settings.retrieval_threshold is not None,
            reranking=self.settings.rerank,
            hybrid_search=False,
            associations=False,
        )

    def initialize(self, scope: MemoryScope) -> None:
        if self._runtime is not None:
            raise AdapterConfigurationError("adapter already initialized")

        runtime = TeleMemRuntime(
            settings=self.settings,
            run_id=scope.run_id,
            user_id=scope.user_id,
        )
        try:
            runtime.initialize(require_empty=False)
        except Exception as exc:
            runtime.close()
            raise AdapterUnavailableError(
                "TeleMem initialization failed"
            ) from exc

        self._scope = scope
        self._runtime = runtime

    def _require_runtime(
        self, request_scope: MemoryScope | None = None
    ) -> TeleMemRuntime:
        if self._runtime is None or self._scope is None:
            raise AdapterConfigurationError("TeleMem adapter is not initialized")
        if request_scope is not None and request_scope != self._scope:
            raise AdapterConfigurationError(
                "request scope does not match the initialized TeleMem scope"
            )
        return self._runtime

    def _adapter_metadata(
        self,
        operation_id: str,
        memory_type: str,
        supplied: Mapping[str, Any] | None = None,
        *,
        message_metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._scope is None:
            raise AdapterConfigurationError("TeleMem adapter is not initialized")
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
                "membench_memory_type": memory_type,
                "membench_namespace": list(self._scope.namespace),
                "membench_operation_id": operation_id,
            }
        )
        if message_metadata and any(message_metadata):
            metadata["membench_message_metadata"] = message_metadata
        return metadata

    @staticmethod
    def _canonical_id(scope: MemoryScope, vendor_id: str) -> str:
        return canonical_memory_id("telemem", scope, vendor_id)

    @staticmethod
    def _write_action(event: str | None) -> WriteAction:
        normalized = (event or "ADD").strip().upper()
        actions = {
            "ADD": WriteAction.CREATED,
            "CREATE": WriteAction.CREATED,
            "CREATED": WriteAction.CREATED,
            "UPDATE": WriteAction.UPDATED,
            "UPDATED": WriteAction.UPDATED,
            "DELETE": WriteAction.DELETED,
            "DELETED": WriteAction.DELETED,
            "NONE": WriteAction.SKIPPED,
            "NOOP": WriteAction.SKIPPED,
            "NO_OP": WriteAction.SKIPPED,
            "SKIP": WriteAction.SKIPPED,
            "SKIPPED": WriteAction.SKIPPED,
        }
        try:
            return actions[normalized]
        except KeyError as exc:
            raise WriteTransportError(
                f"TeleMem returned an unknown write event: {normalized!r}"
            ) from exc

    def _write_result(
        self,
        operation_id: str,
        records: list[TeleMemRecord],
        latency_seconds: float,
    ) -> WriteResult:
        if self._scope is None:
            raise AdapterConfigurationError("TeleMem adapter is not initialized")
        outcomes = tuple(
            WriteOutcome(
                action=self._write_action(record.event),
                canonical_id=self._canonical_id(self._scope, record.vendor_id),
                vendor_id=record.vendor_id,
                text=record.text,
            )
            for record in records
        )
        effective = [
            outcome for outcome in outcomes if outcome.action != WriteAction.SKIPPED
        ]
        if not effective:
            status = WriteStatus.NO_OP
        elif len(effective) != len(outcomes):
            status = WriteStatus.PARTIAL
        else:
            status = WriteStatus.SUCCEEDED
        return WriteResult(
            operation_id=operation_id,
            status=status,
            outcomes=outcomes,
            latency_seconds=latency_seconds,
            raw_vendor={"results": [record.raw_value for record in records]},
        )

    @staticmethod
    def _memory_type(record: TeleMemRecord) -> MemoryType:
        value = record.metadata.get("membench_memory_type", "unspecified")
        return canonical_memory_type(
            value,
            mechanism="TeleMem",
            aliases={"procedural_memory": "procedural"},
        )

    def _canonical_record(self, record: TeleMemRecord) -> MemoryRecord:
        if self._scope is None:
            raise AdapterConfigurationError("TeleMem adapter is not initialized")
        if record.user_id is not None and record.user_id != self._scope.user_id:
            raise ValueError("TeleMem returned a record from a different user")
        if record.run_id is not None and record.run_id != self._scope.run_id:
            raise ValueError("TeleMem returned a record from a different run")

        return MemoryRecord(
            canonical_id=self._canonical_id(self._scope, record.vendor_id),
            vendor_id=record.vendor_id,
            scope=self._scope,
            text=record.text,
            memory_type=self._memory_type(record),
            state=normalize_memory_state(
                event=record.event,
                status=str(record.raw_value.get("status", "")),
                expired=(
                    record.raw_value.get(
                        "is_expired", record.raw_value.get("expired", False)
                    )
                    is True
                ),
            ),
            source_event_ids=source_event_ids(
                record.metadata,
                mechanism="TeleMem",
            ),
            created_at=parse_timestamp(
                record.created_at,
                mechanism="TeleMem",
                field_name="created_at",
            ),
            updated_at=parse_timestamp(
                record.updated_at,
                mechanism="TeleMem",
                field_name="updated_at",
            ),
            attributes={
                "metadata": record.metadata,
                "native_event": record.event,
                "role": record.role,
            },
            raw_vendor=record.raw_value,
        )

    def write_direct(self, request: DirectWriteRequest) -> WriteResult:
        """Submit one approved string through TeleMem's ``infer=False`` path."""

        runtime = self._require_runtime(request.scope)
        metadata = self._adapter_metadata(
            request.operation_id,
            request.memory_type.value,
            request.metadata,
        )
        started = time.perf_counter()
        try:
            records = runtime.store_memory(
                request.text,
                metadata=metadata,
                source_event_id=request.source_event_id,
            )
        except Exception as exc:
            raise WriteTransportError(
                f"TeleMem direct write failed for {request.operation_id!r}"
            ) from exc
        return self._write_result(
            request.operation_id,
            records,
            time.perf_counter() - started,
        )

    def write_conversation(self, request: ConversationWriteRequest) -> WriteResult:
        """Submit messages through TeleMem's native ``infer=True`` path."""

        runtime = self._require_runtime(request.scope)
        unknown_options = set(request.native_options) - _NATIVE_CONVERSATION_OPTIONS
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise AdapterConfigurationError(
                f"unsupported TeleMem conversation options: {names}"
            )
        memory_type = request.native_options.get("memory_type")
        if memory_type not in {None, "procedural_memory"}:
            raise AdapterConfigurationError(
                "TeleMem memory_type must be null or 'procedural_memory'"
            )
        prompt = request.native_options.get("prompt")
        if prompt is not None and (
            not isinstance(prompt, str) or not prompt.strip()
        ):
            raise AdapterConfigurationError(
                "TeleMem extraction prompt must be null or a non-blank string"
            )

        messages = [
            {"role": message.role.value, "content": message.content}
            for message in request.messages
        ]
        message_metadata = [dict(message.metadata) for message in request.messages]
        canonical_type = "procedural" if memory_type else "unspecified"
        metadata = self._adapter_metadata(
            request.operation_id,
            canonical_type,
            message_metadata=message_metadata,
        )
        started = time.perf_counter()
        try:
            records = runtime.submit_conversation(
                messages,
                infer=True,
                metadata=metadata,
                source_event_id=request.source_event_id,
                memory_type=memory_type,
                prompt=prompt,
            )
        except Exception as exc:
            raise WriteTransportError(
                f"TeleMem conversation write failed for {request.operation_id!r}"
            ) from exc
        return self._write_result(
            request.operation_id,
            records,
            time.perf_counter() - started,
        )

    def flush_and_wait(self) -> FlushResult:
        """Wait for pending TeleMem writes and normalize every visible record."""

        runtime = self._require_runtime()
        started = time.perf_counter()
        try:
            native_records = runtime.flush_and_wait()
            records = tuple(self._canonical_record(item) for item in native_records)
        except Exception as exc:
            if "timed out" in str(exc).casefold():
                raise WriteVisibilityTimeout(
                    "TeleMem write visibility timed out"
                ) from exc
            raise WriteTransportError("TeleMem write flush failed") from exc
        return FlushResult(
            completed=True,
            visible_records=records,
            pending_ids=(),
            latency_seconds=time.perf_counter() - started,
            raw_vendor={
                "results": [record.raw_value for record in native_records]
            },
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve ranked TeleMem records within the initialized scope."""

        runtime = self._require_runtime(request.scope)
        if request.filters:
            raise AdapterConfigurationError(
                "TeleMem adapter does not support per-request metadata filters"
            )
        if request.native_options:
            raise AdapterConfigurationError(
                "TeleMem retrieval native_options are fixed by its settings"
            )
        if request.memory_types:
            raise AdapterConfigurationError(
                "TeleMem adapter does not support retrieval by canonical memory type"
            )

        configured_threshold = self.settings.retrieval_threshold
        if (
            request.threshold is not None
            and request.threshold != configured_threshold
        ):
            raise AdapterConfigurationError(
                "retrieval threshold does not match the initialized TeleMem settings"
            )
        effective_threshold = configured_threshold
        started = time.perf_counter()
        try:
            native_records = runtime.retrieve(request.query, top_k=request.top_k)
            records = tuple(self._canonical_record(item) for item in native_records)
        except Exception as exc:
            raise RetrievalTransportError(
                f"TeleMem retrieval failed for {request.operation_id!r}"
            ) from exc

        score_kind = (
            ScoreKind.RERANK if self.settings.rerank else ScoreKind.SIMILARITY
        )
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
                    effective_threshold,
                ),
            )
            for rank, (record, native) in enumerate(
                zip(records, native_records, strict=True), start=1
            )
        )
        return RetrievalResult(
            operation_id=request.operation_id,
            query_sha256=request.query_sha256,
            request_top_k=request.top_k,
            threshold=effective_threshold,
            hits=hits,
            latency_seconds=time.perf_counter() - started,
            raw_vendor={
                "results": [record.raw_value for record in native_records]
            },
        )

    def inspect_state(self) -> MemorySnapshot:
        """Enumerate TeleMem state without issuing a semantic search query."""

        runtime = self._require_runtime()
        if self._scope is None:
            raise AdapterConfigurationError("TeleMem adapter is not initialized")
        try:
            native_records = runtime.inspect_state(limit=_INSPECTION_LIMIT)
            records = tuple(self._canonical_record(item) for item in native_records)
        except Exception as exc:
            raise StateInspectionError("TeleMem state inspection failed") from exc
        return MemorySnapshot(
            scope=self._scope,
            records=records,
            complete=len(native_records) < _INSPECTION_LIMIT,
            raw_vendor={
                "results": [record.raw_value for record in native_records]
            },
        )

    def verify_empty(self) -> EmptyStateResult:
        """Prove the bound scope has no visible, retrievable, or pending state."""
        runtime = self._require_runtime()
        if self._scope is None:
            raise AdapterConfigurationError("TeleMem adapter is not initialized")
        try:
            inspected = runtime.inspect_state(limit=1)
            retrieved = runtime.retrieve(_EMPTY_STATE_QUERY, top_k=1)
            pending_count = runtime.pending_count
        except Exception as exc:
            raise StateInspectionError(
                "TeleMem empty-state verification failed"
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
                "pending_count": pending_count,
            },
        )
        if not result.empty:
            raise NamespaceNotEmptyError(
                "TeleMem scope is not empty: "
                f"inspected={result.inspected_count}, "
                f"retrieved={result.retrieved_count}, "
                f"pending={result.pending_count}"
            )
        return result

    def reset_and_verify_empty(self) -> EmptyStateResult:
        """Reset only the bound TeleMem user/run and return empty-state evidence."""

        runtime = self._require_runtime()
        try:
            runtime.reset_and_verify_empty()
            return self.verify_empty()
        except NamespaceNotEmptyError as exc:
            raise ResetVerificationError(
                "TeleMem scope is not empty after reset"
            ) from exc
        except Exception as exc:
            raise ResetVerificationError(
                "TeleMem reset or empty-state verification failed"
            ) from exc

    def export_provenance(self) -> AdapterProvenance:
        """Return non-secret versions and resolved TeleMem configuration."""
        runtime = self._require_runtime()
        if self._scope is None:
            raise AdapterConfigurationError("TeleMem adapter is not initialized")
        versions = telemem_package_versions()
        return build_adapter_provenance(
            mechanism="telemem",
            mechanism_version=versions["telemem"],
            storage_backend="faiss+sqlite-history",
            scope=self._scope,
            model_identifiers={
                "writer": self.settings.llm_model,
                "embedding": self.settings.embedding_model,
            },
            package_versions=versions,
            native_configuration={
                "settings": self.settings.public_config(),
                "paths": {
                    "run_dir": runtime.paths.run_dir,
                    "mem0_dir": runtime.paths.mem0_dir,
                    "vector_dir": runtime.paths.vector_dir,
                    "history_db": runtime.paths.history_db,
                    "collection_name": runtime.paths.collection_name,
                },
                "capabilities": self.capabilities.as_dict(),
            },
        )

    def close(self) -> None:
        """Close the native runtime and make repeated close calls harmless."""

        runtime = self._runtime
        self._runtime = None
        self._scope = None
        if runtime is not None:
            runtime.close()
