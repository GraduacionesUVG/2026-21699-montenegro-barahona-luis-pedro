"""Canonical external-content ingress for controlled memory-poisoning trials."""

from membench.ingress.agent_selected import (
    AgentSelectedIngestionAttempt,
    AgentSelectedIngressExecutor,
    AgentSelectedIngressOutcome,
)
from membench.ingress.bridge import ExternalIngestionBridge
from membench.ingress.forced import ForcedIngressExecutor
from membench.ingress.models import (
    ExternalDelivery,
    ExternalIngestionRequest,
    ExternalIngestionResult,
    ExternalSourceKind,
    IngressMode,
    NormalizedExternalContent,
    SourceSelectionTask,
)
from membench.ingress.normalization import (
    EXTERNAL_SOURCE_NORMALIZER,
    normalize_external_delivery,
)

__all__ = [
    "EXTERNAL_SOURCE_NORMALIZER",
    "AgentSelectedIngressExecutor",
    "AgentSelectedIngestionAttempt",
    "AgentSelectedIngressOutcome",
    "ExternalDelivery",
    "ExternalIngestionBridge",
    "ExternalIngestionRequest",
    "ExternalIngestionResult",
    "ExternalSourceKind",
    "ForcedIngressExecutor",
    "IngressMode",
    "NormalizedExternalContent",
    "SourceSelectionTask",
    "normalize_external_delivery",
]
