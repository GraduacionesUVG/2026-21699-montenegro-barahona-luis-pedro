from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from membench.contract import (
    DakeraOptions,
    HuggingFaceResponderTokenizerRef,
    LangMemOptions,
    Mechanism,
    TeleMemOptions,
    load_manifest,
)

EXAMPLE = Path("configs/experiments/blackbox/example.yaml")
LANGMEM_EXAMPLE = Path(
    "configs/experiments/blackbox/langmem_semantic.example.yaml"
)
DAKERA_EXAMPLE = Path("configs/experiments/blackbox/dakera.example.yaml")


def _load_yaml(path: Path = EXAMPLE) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(tmp_path: Path, name: str, value: dict[str, Any]) -> Path:
    candidate = tmp_path / name
    candidate.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return candidate


def _dakera_manifest() -> dict[str, Any]:
    return deepcopy(_load_yaml(DAKERA_EXAMPLE))


def test_telemem_example_is_typed_and_hashable() -> None:
    manifest = load_manifest(EXAMPLE)

    assert manifest.schema_version == 1
    assert manifest.condition.mechanism == Mechanism.TELEMEM
    assert isinstance(manifest.condition.mechanism_options, TeleMemOptions)
    assert isinstance(
        manifest.models.responder.tokenizer,
        HuggingFaceResponderTokenizerRef,
    )
    assert manifest.models.responder.tokenizer.transformers_version == "4.57.6"
    assert len(manifest.models.responder.tokenizer.revision) == 40
    assert manifest.condition.mechanism_options.infer
    assert len(manifest.sha256) == 64
    assert manifest.sha256 == load_manifest(EXAMPLE).sha256


def test_telemem_infer_must_agree_with_write_policy(tmp_path: Path) -> None:
    value = _load_yaml()
    value["condition"]["write_policy"] = "direct"

    with pytest.raises(ValidationError, match="infer must be true only"):
        load_manifest(_write_yaml(tmp_path, "telemem-direct.yaml", value))


def test_langmem_example_has_canonical_identity_and_typed_options() -> None:
    manifest = load_manifest(LANGMEM_EXAMPLE)

    assert manifest.condition.mechanism == Mechanism.LANGMEM
    assert manifest.condition.mechanism_version == "0.0.30"
    assert isinstance(manifest.condition.mechanism_options, LangMemOptions)
    assert manifest.condition.mechanism_options.memory_types == ("semantic",)
    assert manifest.condition.mechanism_options.storage_backend == "sqlite"


def test_langmem_rejects_unpinned_writer_version(tmp_path: Path) -> None:
    value = _load_yaml(LANGMEM_EXAMPLE)
    value["condition"]["mechanism_options"]["writer_version"] = "latest"

    with pytest.raises(ValidationError, match="writer_version"):
        load_manifest(_write_yaml(tmp_path, "langmem-latest.yaml", value))


def test_langmem_rejects_unknown_or_duplicate_memory_families(
    tmp_path: Path,
) -> None:
    unknown = _load_yaml(LANGMEM_EXAMPLE)
    unknown["condition"]["mechanism_options"]["memory_types"] = [
        "semantic",
        "imaginary",
    ]
    with pytest.raises(ValidationError, match="memory_types"):
        load_manifest(_write_yaml(tmp_path, "unknown-memory.yaml", unknown))

    duplicate = _load_yaml(LANGMEM_EXAMPLE)
    duplicate["condition"]["mechanism_options"]["memory_types"] = [
        "semantic",
        "semantic",
    ]
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        load_manifest(_write_yaml(tmp_path, "duplicate-memory.yaml", duplicate))


def test_dakera_direct_condition_has_typed_reproducible_options(
) -> None:
    manifest = load_manifest(DAKERA_EXAMPLE)

    assert manifest.condition.mechanism == Mechanism.DAKERA
    assert isinstance(manifest.condition.mechanism_options, DakeraOptions)
    assert manifest.condition.mechanism_options.routing == "vector"
    assert manifest.condition.mechanism_options.ttl_seconds is None
    assert manifest.models.writer is None
    assert manifest.models.embedding is None


def test_dakera_native_selective_is_not_mislabelled(tmp_path: Path) -> None:
    value = _dakera_manifest()
    value["condition"]["write_policy"] = "native_selective"

    with pytest.raises(ValidationError, match="does not support native_selective"):
        load_manifest(_write_yaml(tmp_path, "dakera-native.yaml", value))


def test_dakera_mutable_image_tag_is_rejected(tmp_path: Path) -> None:
    value = _dakera_manifest()
    value["condition"]["mechanism_options"]["server_image"] = (
        "ghcr.io/dakera-ai/dakera:latest"
    )

    with pytest.raises(ValidationError, match="server_image"):
        load_manifest(_write_yaml(tmp_path, "dakera-latest.yaml", value))


def test_manifest_rejects_mutable_dataset_and_invalid_model_metadata(
    tmp_path: Path,
) -> None:
    mutable_dataset = _load_yaml()
    mutable_dataset["dataset"]["revision"] = "main"
    with pytest.raises(ValidationError, match="revision"):
        load_manifest(_write_yaml(tmp_path, "mutable-dataset.yaml", mutable_dataset))

    invalid_model = _load_yaml()
    invalid_model["models"]["writer"]["digest"] = "latest"
    invalid_model["models"]["embedding"]["dimensions"] = 0
    with pytest.raises(ValidationError) as captured:
        load_manifest(_write_yaml(tmp_path, "invalid-model.yaml", invalid_model))
    assert "digest" in str(captured.value)
    assert "dimensions" in str(captured.value)


def test_manifest_rejects_mutable_responder_tokenizer_revision(
    tmp_path: Path,
) -> None:
    value = _load_yaml()
    value["models"]["responder"]["tokenizer"]["revision"] = "main"

    with pytest.raises(ValidationError, match="revision"):
        load_manifest(_write_yaml(tmp_path, "mutable-tokenizer.yaml", value))
