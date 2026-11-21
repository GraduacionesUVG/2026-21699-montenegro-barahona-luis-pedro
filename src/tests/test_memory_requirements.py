from pathlib import Path

import pytest

from membench.contract import (
    DakeraOptions,
    LangMemOptions,
    load_manifest,
)
from membench.memory.capabilities import AdapterCapabilities
from membench.memory.errors import UnsupportedCapabilityError
from membench.memory.requirements import (
    evaluate_capabilities,
    requirements_for,
    validate_capabilities,
)

CONFIG_ROOT = Path("configs/experiments/blackbox")


def _capabilities(**enabled: bool) -> AdapterCapabilities:
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
    values.update(enabled)
    return AdapterCapabilities.from_mapping(values)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("example.yaml", ("native_conversation_write",)),
        (
            "langmem_semantic.example.yaml",
            ("native_conversation_write", "semantic_memory"),
        ),
        ("dakera.example.yaml", ("direct_write", "semantic_memory")),
    ],
)
def test_requirements_are_derived_from_example_manifests(
    filename: str,
    expected: tuple[str, ...],
) -> None:
    manifest = load_manifest(CONFIG_ROOT / filename)

    assert requirements_for(manifest) == expected


def test_langmem_requirements_include_selected_memory_families_and_deletion() -> None:
    manifest = load_manifest(CONFIG_ROOT / "langmem_semantic.example.yaml")
    options = manifest.condition.mechanism_options
    assert isinstance(options, LangMemOptions)
    condition = manifest.condition.model_copy(
        update={
            "mechanism_options": options.model_copy(
                update={
                    "memory_types": ("semantic", "episodic", "procedural"),
                    "enable_deletes": True,
                }
            )
        }
    )
    manifest = manifest.model_copy(update={"condition": condition})

    assert requirements_for(manifest) == (
        "deletion",
        "episodic_memory",
        "native_conversation_write",
        "procedural_memory",
        "semantic_memory",
    )


def test_dakera_requirements_include_enabled_native_features() -> None:
    manifest = load_manifest(CONFIG_ROOT / "dakera.example.yaml")
    options = manifest.condition.mechanism_options
    assert isinstance(options, DakeraOptions)
    condition = manifest.condition.model_copy(
        update={
            "mechanism_options": options.model_copy(
                update={
                    "routing": "hybrid",
                    "rerank": True,
                    "include_associated": True,
                    "ttl_seconds": 60,
                }
            )
        }
    )
    manifest = manifest.model_copy(update={"condition": condition})

    assert requirements_for(manifest) == (
        "associations",
        "direct_write",
        "hybrid_search",
        "reranking",
        "semantic_memory",
        "ttl",
    )


def test_retrieval_threshold_is_an_explicit_requirement() -> None:
    manifest = load_manifest(CONFIG_ROOT / "example.yaml")
    retrieval = manifest.retrieval.model_copy(update={"threshold": 0.7})
    manifest = manifest.model_copy(update={"retrieval": retrieval})

    assert requirements_for(manifest) == (
        "native_conversation_write",
        "threshold",
    )


def test_evaluate_capabilities_returns_structured_failure() -> None:
    manifest = load_manifest(CONFIG_ROOT / "dakera.example.yaml")
    result = evaluate_capabilities(
        manifest,
        _capabilities(direct_write=True),
    )

    assert result.supported is False
    assert result.missing == ("semantic_memory",)
    assert result.as_dict() == {
        "mechanism": "dakera",
        "write_policy": "direct",
        "required": ["direct_write", "semantic_memory"],
        "available": ["direct_write"],
        "missing": ["semantic_memory"],
        "supported": False,
    }


def test_validate_capabilities_returns_report_when_supported() -> None:
    manifest = load_manifest(CONFIG_ROOT / "dakera.example.yaml")

    result = validate_capabilities(
        manifest,
        _capabilities(direct_write=True, semantic_memory=True),
    )

    assert result.supported is True
    assert result.missing == ()


def test_validate_capabilities_fails_fast_with_missing_features() -> None:
    manifest = load_manifest(CONFIG_ROOT / "dakera.example.yaml")

    with pytest.raises(UnsupportedCapabilityError) as error:
        validate_capabilities(manifest, _capabilities(direct_write=True))

    assert error.value.missing == ("semantic_memory",)
    assert "dakera adapter" in str(error.value)
