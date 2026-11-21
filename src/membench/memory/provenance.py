"""Construction and secret-scrubbing for canonical adapter provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from membench.memory.models import AdapterProvenance, MemoryScope

_PROJECT_DISTRIBUTION = "memory-poisoning-analysis"
_LOCAL_VERSION = "0.1.0+local"
_REDACTED = "<redacted>"
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "credential",
        "credentials",
        "dsn",
        "password",
        "secret",
        "token",
    }
)


def adapter_package_version() -> str:
    """Return the installed harness version, with a source-tree fallback."""
    try:
        return version(_PROJECT_DISTRIBUTION)
    except PackageNotFoundError:
        return _LOCAL_VERSION


def sanitize_native_configuration(value: Any) -> Any:
    """Convert configuration to JSON-compatible data and redact secret fields."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            sanitized[name] = (
                _REDACTED
                if _is_sensitive_key(name)
                else sanitize_native_configuration(item)
            )
        return sanitized
    if isinstance(value, Enum):
        return sanitize_native_configuration(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [sanitize_native_configuration(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "native provenance configuration contains an unsupported value of type "
        f"{type(value).__name__}"
    )


def build_adapter_provenance(
    *,
    mechanism: str,
    mechanism_version: str,
    storage_backend: str,
    scope: MemoryScope,
    model_identifiers: Mapping[str, str | None] | None = None,
    package_versions: Mapping[str, str] | None = None,
    native_configuration: Mapping[str, Any] | None = None,
    adapter_version: str | None = None,
) -> AdapterProvenance:
    """Build the canonical, immutable and non-secret provenance snapshot."""
    models = _non_blank_mapping(model_identifiers or {}, "model identifier")
    packages = _non_blank_mapping(package_versions or {}, "package version")
    sanitized = sanitize_native_configuration(native_configuration or {})
    if not isinstance(sanitized, dict):  # Defensive: input is typed as a mapping.
        raise TypeError("native_configuration must normalize to an object")
    return AdapterProvenance(
        mechanism=mechanism,
        mechanism_version=mechanism_version,
        adapter_version=adapter_version or adapter_package_version(),
        storage_backend=storage_backend,
        model_identifiers=models,
        package_versions=packages,
        namespace=scope.namespace,
        native_configuration=sanitized,
    )


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized.endswith("_configured"):
        return False
    parts = set(normalized.split("_"))
    return (
        normalized == "api_key"
        or normalized.endswith("_api_key")
        or normalized.endswith("_dsn")
        or bool(parts.intersection(_SENSITIVE_KEYS))
    )


def _non_blank_mapping(
    values: Mapping[str, str | None],
    description: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{description} names must be non-blank strings")
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{description} values must be non-blank strings")
        result[key.strip()] = value.strip()
    return result
