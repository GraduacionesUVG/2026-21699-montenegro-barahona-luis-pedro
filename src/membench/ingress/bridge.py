"""Shared external-content to memory-policy bridge."""

from __future__ import annotations

from copy import deepcopy

from membench.ingress.models import (
    ExternalDelivery,
    ExternalIngestionRequest,
    ExternalIngestionResult,
    IngressMode,
)
from membench.ingress.normalization import normalize_external_delivery
from membench.memory.models import MemoryMessage, MemoryScope, MessageRole
from membench.memory.policy import PolicyDispatcher, PolicyWriteRequest


class ExternalIngestionBridge:
    """Submit normalized external content through an already-bound write policy."""

    def __init__(self, dispatcher: PolicyDispatcher) -> None:
        self._dispatcher = dispatcher

    def ingest(
        self,
        delivery: ExternalDelivery,
        *,
        scope: MemoryScope,
        operation_id: str,
        mode: IngressMode,
        native_options: dict[str, object] | None = None,
    ) -> ExternalIngestionResult:
        """Normalize then dispatch one exact delivery without implicit flushing."""

        normalized = normalize_external_delivery(delivery)
        request = ExternalIngestionRequest(
            operation_id=operation_id,
            scope=scope,
            mode=mode,
            delivery=delivery,
            normalized=normalized,
            source_event_id=f"external-delivery:{delivery.delivery_id}",
            native_options=deepcopy(native_options or {}),
        )
        dispatch = self._dispatcher.dispatch(
            PolicyWriteRequest(
                operation_id=request.operation_id,
                scope=request.scope,
                messages=(
                    MemoryMessage(role=MessageRole.USER, content=normalized.text),
                ),
                source_event_id=request.source_event_id,
                metadata={
                    "ingress_mode": request.mode.value,
                    "content_trust": "untrusted_external",
                    "source_kind": delivery.source_kind.value,
                    "source_id": delivery.source_id,
                    "source_content_sha256": delivery.content_sha256,
                    "normalized_content_sha256": normalized.text_sha256,
                    "normalizer": normalized.normalizer,
                },
                native_options=request.native_options,
            )
        )
        return ExternalIngestionResult(
            request=request,
            dispatch=dispatch,
            selected_by_agent=mode == IngressMode.AGENT_SELECTED,
        )
