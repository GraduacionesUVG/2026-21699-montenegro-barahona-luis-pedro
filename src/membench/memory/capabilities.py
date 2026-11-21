"""Immutable feature declarations for configured memory adapters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class AdapterCapabilities(Mapping[str, bool]):
    """Feature surface available on one resolved adapter instance.

    These flags describe operations the configured adapter can honor without
    probing a live service. They do not claim that every native backend feature
    is exposed through the common protocol.
    """
    direct_write: bool
    native_conversation_write: bool
    semantic_memory: bool
    episodic_memory: bool
    procedural_memory: bool
    working_memory: bool
    updates: bool
    deletion: bool
    ttl: bool
    metadata_filters: bool
    threshold: bool
    reranking: bool
    hybrid_search: bool
    associations: bool

    def __post_init__(self) -> None:
        invalid = [
            item.name
            for item in fields(self)
            if not isinstance(getattr(self, item.name), bool)
        ]
        if invalid:
            names = ", ".join(invalid)
            raise TypeError(f"adapter capabilities must be booleans: {names}")

    def __getitem__(self, key: str) -> bool:
        if key not in self:
            raise KeyError(key)
        return bool(getattr(self, key))

    def __iter__(self) -> Iterator[str]:
        return (item.name for item in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and any(
            item.name == key for item in fields(self)
        )

    def as_dict(self) -> dict[str, bool]:
        """Return a JSON-serializable copy for logs and provenance."""
        return dict(self)

    def require(self, *names: str) -> None:
        """Raise when a campaign requires an unknown or unavailable feature."""
        unknown = sorted(name for name in names if name not in self)
        if unknown:
            raise ValueError(
                "unknown adapter capabilities: " + ", ".join(unknown)
            )
        unavailable = sorted(name for name in names if not self[name])
        if unavailable:
            raise ValueError(
                "adapter does not provide required capabilities: "
                + ", ".join(unavailable)
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AdapterCapabilities:
        """Build a strict declaration while rejecting missing or extra flags."""
        expected = {item.name for item in fields(cls)}
        supplied = set(values)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unknown: " + ", ".join(extra))
            message = "invalid adapter capabilities (" + "; ".join(details) + ")"
            raise ValueError(message)
        return cls(**dict(values))
