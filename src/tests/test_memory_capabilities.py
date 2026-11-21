from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from membench.dakera import DakeraMemoryType, DakeraRouting
from membench.langmem import MemoryType as LangMemMemoryType
from membench.memory.adapters.dakera_adapter import DakeraAdapter
from membench.memory.adapters.langmem_adapter import LangMemAdapter
from membench.memory.adapters.telemem_adapter import TeleMemAdapter
from membench.memory.capabilities import AdapterCapabilities


def _all_capabilities(**overrides: bool) -> dict[str, bool]:
    values = {
        "direct_write": False,
        "native_conversation_write": False,
        "semantic_memory": False,
        "episodic_memory": False,
        "procedural_memory": False,
        "working_memory": False,
        "updates": False,
        "deletion": False,
        "ttl": False,
        "metadata_filters": False,
        "threshold": False,
        "reranking": False,
        "hybrid_search": False,
        "associations": False,
    }
    values.update(overrides)
    return values


def test_capabilities_are_immutable_mapping_and_validate_requirements() -> None:
    capabilities = AdapterCapabilities.from_mapping(
        _all_capabilities(direct_write=True, semantic_memory=True)
    )

    assert capabilities["direct_write"]
    assert dict(capabilities)["semantic_memory"]
    assert len(capabilities) == 14
    capabilities.require("direct_write", "semantic_memory")
    with pytest.raises(ValueError, match="does not provide"):
        capabilities.require("ttl")
    with pytest.raises(ValueError, match="unknown"):
        capabilities.require("imaginary")
    with pytest.raises(FrozenInstanceError):
        capabilities.ttl = True


def test_capability_mapping_rejects_missing_extra_and_non_boolean_values() -> None:
    missing = _all_capabilities()
    missing.pop("ttl")
    with pytest.raises(ValueError, match="missing: ttl"):
        AdapterCapabilities.from_mapping(missing)

    extra = _all_capabilities()
    extra["imaginary"] = False
    with pytest.raises(ValueError, match="unknown: imaginary"):
        AdapterCapabilities.from_mapping(extra)

    values = _all_capabilities()
    values["ttl"] = 1  # type: ignore[assignment]
    with pytest.raises(TypeError, match="ttl"):
        AdapterCapabilities.from_mapping(values)


def test_adapter_capabilities_reflect_resolved_settings() -> None:
    telemem = TeleMemAdapter(
        SimpleNamespace(retrieval_threshold=0.4, rerank=True)  # type: ignore[arg-type]
    )
    langmem = LangMemAdapter(
        SimpleNamespace(  # type: ignore[arg-type]
            memory_types=(
                LangMemMemoryType.SEMANTIC,
                LangMemMemoryType.PROCEDURAL,
            ),
            enable_deletes=False,
        )
    )
    dakera = DakeraAdapter(
        SimpleNamespace(  # type: ignore[arg-type]
            agent_prefix="membench",
            memory_type=DakeraMemoryType.SEMANTIC,
            importance=0.5,
            ttl_seconds=60,
            rerank=False,
            routing=DakeraRouting.HYBRID,
            include_associated=True,
        )
    )

    assert telemem.capabilities.native_conversation_write
    assert telemem.capabilities.reranking
    assert not telemem.capabilities.episodic_memory

    assert langmem.capabilities.semantic_memory
    assert not langmem.capabilities.episodic_memory
    assert langmem.capabilities.procedural_memory
    assert not langmem.capabilities.deletion

    assert not dakera.capabilities.native_conversation_write
    assert dakera.capabilities.working_memory
    assert dakera.capabilities.ttl
    assert dakera.capabilities.hybrid_search
    assert dakera.capabilities.associations
