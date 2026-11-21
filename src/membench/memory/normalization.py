"""Pure helpers for projecting native memory data into canonical models.

The functions in this module deliberately know nothing about a vendor SDK.
Adapters retain responsibility for reading native response shapes and use these
helpers only for behavior that must be identical across mechanisms.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import Enum
from typing import Any

from membench.memory.models import MemoryScope, MemoryState, MemoryType


def canonical_memory_id(
    mechanism: str,
    scope: MemoryScope,
    vendor_id: str,
    *,
    qualifiers: Iterable[str] = (),
) -> str:
    """Create a stable opaque ID inside a mechanism and canonical namespace.

    ``qualifiers`` supports mechanisms such as LangMem whose vendor keys are
    unique only inside a memory-family namespace. Its ordering is significant.
    """
    mechanism_name = _non_blank(mechanism, "mechanism").lower()
    native_id = _non_blank(vendor_id, "vendor_id")
    normalized_qualifiers = tuple(
        _non_blank(value, "canonical ID qualifier") for value in qualifiers
    )
    identity = "\x1f".join(
        (mechanism_name, *scope.namespace, *normalized_qualifiers, native_id)
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"{mechanism_name}-{digest}"


def parse_timestamp(
    value: str | datetime | None,
    *,
    mechanism: str,
    field_name: str,
) -> datetime | None:
    """Parse a native ISO timestamp and reject ambiguous timezone-free values."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"{mechanism} returned an invalid {field_name} timestamp"
            ) from exc
    else:
        raise ValueError(
            f"{mechanism} returned a non-string {field_name} timestamp"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"{mechanism} returned a timezone-free {field_name} timestamp"
        )
    return parsed


def source_event_ids(
    metadata: Mapping[str, Any],
    *,
    mechanism: str,
    singular_key: str = "membench_source_event_id",
    plural_key: str = "membench_source_event_ids",
) -> tuple[str, ...]:
    """Normalize singular/plural source-event metadata with stable deduping."""
    values: list[Any] = []
    if singular_key in metadata:
        values.append(metadata[singular_key])
    if plural_key in metadata:
        plural = metadata[plural_key]
        if isinstance(plural, str) or not isinstance(plural, (list, tuple)):
            raise ValueError(f"{mechanism} returned invalid source event IDs")
        values.extend(plural)

    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{mechanism} returned an invalid source event ID")
        if value not in result:
            result.append(value)
    return tuple(result)


def canonical_memory_type(
    value: str | Enum | None,
    *,
    mechanism: str,
    aliases: Mapping[str, str] | None = None,
) -> MemoryType:
    """Map a native memory-family label to the canonical enum."""
    if value is None:
        normalized = MemoryType.UNSPECIFIED.value
    else:
        native_value = value.value if isinstance(value, Enum) else value
        if not isinstance(native_value, str) or not native_value.strip():
            raise ValueError(f"{mechanism} returned an invalid memory type")
        normalized = native_value.strip().lower()
    if aliases is not None:
        normalized = aliases.get(normalized, normalized)
    try:
        return MemoryType(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{mechanism} returned an unknown canonical memory type"
        ) from exc


def normalize_memory_state(
    *,
    event: str | None = None,
    status: str | None = None,
    expired: bool = False,
) -> MemoryState:
    """Project common lifecycle signals to one canonical state."""
    normalized_event = (event or "").strip().lower()
    normalized_status = (status or "").strip().lower()
    if normalized_event in {"delete", "deleted"} or normalized_status == "deleted":
        return MemoryState.DELETED
    if expired or normalized_status == "expired":
        return MemoryState.EXPIRED
    return MemoryState.ACTIVE


def passes_threshold(
    score: float | None,
    threshold: float | None,
    *,
    higher_is_better: bool = True,
) -> bool:
    """Apply a threshold while preserving unscored native results."""
    if score is None or threshold is None:
        return True
    if higher_is_better:
        return score >= threshold
    return score <= threshold


def _non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()
