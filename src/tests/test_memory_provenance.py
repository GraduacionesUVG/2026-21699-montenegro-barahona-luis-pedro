from dataclasses import dataclass
from pathlib import Path

import pytest

from membench.memory.models import MemoryScope
from membench.memory.provenance import (
    build_adapter_provenance,
    sanitize_native_configuration,
)


@dataclass
class NativeSettings:
    api_key: str
    storage_path: Path
    credential_configured: bool


def _scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="experiment-001",
        run_id="run-001",
        user_id="persona-001",
        namespace=("membench", "example", "run-001", "persona-001"),
    )


def test_sanitize_native_configuration_recurses_and_redacts_secrets() -> None:
    result = sanitize_native_configuration(
        {
            "settings": NativeSettings(
                api_key="do-not-export",
                storage_path=Path("artifacts/memory"),
                credential_configured=True,
            ),
            "nested": {"postgres_dsn": "postgresql://secret"},
        }
    )

    assert result == {
        "settings": {
            "api_key": "<redacted>",
            "storage_path": str(Path("artifacts/memory")),
            "credential_configured": True,
        },
        "nested": {"postgres_dsn": "<redacted>"},
    }


def test_build_adapter_provenance_returns_the_canonical_model() -> None:
    result = build_adapter_provenance(
        mechanism="example",
        mechanism_version="2.0.0",
        adapter_version="1.0.0",
        storage_backend="sqlite",
        scope=_scope(),
        model_identifiers={"writer": "model:1", "optional": None},
        package_versions={"example-sdk": "2.0.0"},
        native_configuration={"token": "secret", "limit": 5},
    )

    assert result.mechanism == "example"
    assert result.adapter_version == "1.0.0"
    assert result.namespace == _scope().namespace
    assert result.model_identifiers == {"writer": "model:1"}
    assert result.native_configuration == {
        "token": "<redacted>",
        "limit": 5,
    }


def test_provenance_rejects_unsupported_or_blank_values() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        sanitize_native_configuration({"client": object()})
    with pytest.raises(ValueError, match="package version"):
        build_adapter_provenance(
            mechanism="example",
            mechanism_version="2.0.0",
            storage_backend="sqlite",
            scope=_scope(),
            package_versions={"example-sdk": " "},
        )
