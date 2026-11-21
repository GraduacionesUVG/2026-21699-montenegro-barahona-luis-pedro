"""Model-free external ingress for isolating memory-writing behavior."""

from __future__ import annotations

from membench.ingress.bridge import ExternalIngestionBridge
from membench.ingress.models import (
    ExternalDelivery,
    ExternalIngestionResult,
    IngressMode,
)
from membench.memory.models import MemoryScope


class ForcedIngressExecutor:
    """Send a bound delivery directly to the common bridge without tool use."""

    def __init__(self, bridge: ExternalIngestionBridge) -> None:
        self._bridge = bridge

    def execute(
        self,
        delivery: ExternalDelivery,
        *,
        scope: MemoryScope,
        operation_id: str,
        native_options: dict[str, object] | None = None,
    ) -> ExternalIngestionResult:
        return self._bridge.ingest(
            delivery,
            scope=scope,
            operation_id=operation_id,
            mode=IngressMode.FORCED_INGESTION,
            native_options=native_options,
        )
