"""Manifest-to-capability requirements and preflight validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from membench.contract import (
    DakeraOptions,
    ExperimentManifest,
    LangMemOptions,
    Mechanism,
    TeleMemOptions,
    WritePolicy,
)
from membench.memory.capabilities import AdapterCapabilities
from membench.memory.errors import (
    AdapterConfigurationError,
    UnsupportedCapabilityError,
)
from membench.memory.protocol import MemoryAdapter

_MEMORY_TYPE_CAPABILITY = {
    "semantic": "semantic_memory",
    "episodic": "episodic_memory",
    "procedural": "procedural_memory",
    "working": "working_memory",
}


@dataclass(frozen=True, slots=True)
class CapabilityValidation:
    """JSON-friendly result of comparing a manifest with one adapter."""

    mechanism: str
    write_policy: str
    required: tuple[str, ...]
    available: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def supported(self) -> bool:
        """Return whether every required capability is available."""
        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        """Return a stable representation suitable for campaign logs."""
        return {
            "mechanism": self.mechanism,
            "write_policy": self.write_policy,
            "required": list(self.required),
            "available": list(self.available),
            "missing": list(self.missing),
            "supported": self.supported,
        }


def requirements_for(manifest: ExperimentManifest) -> tuple[str, ...]:
    """Derive the adapter features required by a resolved manifest."""
    condition = manifest.condition
    options = condition.mechanism_options
    required: set[str] = set()

    if condition.write_policy == WritePolicy.NATIVE_SELECTIVE:
        required.add("native_conversation_write")
    else:
        # Shared guarding happens before the adapter, so the accepted candidate
        # still enters the mechanism through its direct-write operation.
        required.add("direct_write")

    if manifest.retrieval.threshold is not None:
        required.add("threshold")

    if condition.mechanism == Mechanism.TELEMEM:
        telemem_options = _expect_options(options, TeleMemOptions, "TeleMem")
        if telemem_options.rerank:
            required.add("reranking")

    elif condition.mechanism == Mechanism.LANGMEM:
        langmem_options = _expect_options(options, LangMemOptions, "LangMem")
        required.update(
            _MEMORY_TYPE_CAPABILITY[memory_type]
            for memory_type in langmem_options.memory_types
        )
        if langmem_options.enable_deletes:
            required.add("deletion")

    elif condition.mechanism == Mechanism.DAKERA:
        dakera_options = _expect_options(options, DakeraOptions, "Dakera")
        required.add(_MEMORY_TYPE_CAPABILITY[dakera_options.memory_type])
        if dakera_options.rerank:
            required.add("reranking")
        if dakera_options.routing == "hybrid":
            required.add("hybrid_search")
        if dakera_options.include_associated:
            required.add("associations")
        if dakera_options.ttl_seconds is not None:
            required.add("ttl")

    return tuple(sorted(required))


def evaluate_capabilities(
    manifest: ExperimentManifest,
    capabilities: AdapterCapabilities,
) -> CapabilityValidation:
    """Compare requirements with capabilities without raising on mismatch."""
    required = requirements_for(manifest)
    available = tuple(sorted(name for name, enabled in capabilities.items() if enabled))
    missing = tuple(name for name in required if not capabilities[name])
    return CapabilityValidation(
        mechanism=manifest.condition.mechanism.value,
        write_policy=manifest.condition.write_policy.value,
        required=required,
        available=available,
        missing=missing,
    )


def validate_capabilities(
    manifest: ExperimentManifest,
    capabilities: AdapterCapabilities,
) -> CapabilityValidation:
    """Fail fast when an adapter cannot honor the resolved manifest."""
    result = evaluate_capabilities(manifest, capabilities)
    if not result.supported:
        missing = ", ".join(result.missing)
        raise UnsupportedCapabilityError(
            f"{result.mechanism} adapter cannot satisfy "
            f"{result.write_policy} manifest; missing capabilities: {missing}",
            missing=result.missing,
        )
    return result


def validate_adapter_capabilities(
    manifest: ExperimentManifest,
    adapter: MemoryAdapter,
) -> CapabilityValidation:
    """Validate a constructed adapter before its initialization has side effects."""
    return validate_capabilities(manifest, adapter.capabilities)


def _expect_options[OptionsT: (TeleMemOptions, LangMemOptions, DakeraOptions)](
    options: object,
    expected_type: type[OptionsT],
    mechanism_name: str,
) -> OptionsT:
    if not isinstance(options, expected_type):
        raise AdapterConfigurationError(
            f"{mechanism_name} manifest has unresolved mechanism_options"
        )
    return options
