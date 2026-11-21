"""Construction registry for configured memory adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from membench.memory.errors import AdapterConfigurationError

AdapterFactory = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class AdapterRegistration:
    """One canonical mechanism-to-adapter construction rule."""

    mechanism: str
    settings_type: type[Any]
    factory: AdapterFactory


def _mechanism_key(mechanism: str | Enum) -> str:
    value = mechanism.value if isinstance(mechanism, Enum) else mechanism
    if not isinstance(value, str) or not value.strip():
        raise AdapterConfigurationError(
            "memory mechanism must be a non-blank string or string enum"
        )
    return value.strip().lower()


class AdapterRegistry:
    """Register and construct adapters without initializing native services."""

    def __init__(self) -> None:
        self._registrations: dict[str, AdapterRegistration] = {}
        self._canonical_names: set[str] = set()

    def register(
        self,
        mechanism: str | Enum,
        settings_type: type[Any],
        factory: AdapterFactory,
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Register a factory and optional aliases, rejecting name collisions."""
        canonical = _mechanism_key(mechanism)
        if not isinstance(settings_type, type):
            raise TypeError("settings_type must be a class")
        if not callable(factory):
            raise TypeError("adapter factory must be callable")

        names = (canonical, *(_mechanism_key(alias) for alias in aliases))
        if len(names) != len(set(names)):
            raise AdapterConfigurationError(
                f"duplicate aliases supplied for memory mechanism {canonical!r}"
            )
        conflicts = sorted(name for name in names if name in self._registrations)
        if conflicts:
            raise AdapterConfigurationError(
                "memory adapter names are already registered: "
                + ", ".join(conflicts)
            )

        registration = AdapterRegistration(
            mechanism=canonical,
            settings_type=settings_type,
            factory=factory,
        )
        for name in names:
            self._registrations[name] = registration
        self._canonical_names.add(canonical)

    def create(self, mechanism: str | Enum, settings: Any) -> Any:
        """Construct an uninitialized adapter from validated resolved settings."""
        key = _mechanism_key(mechanism)
        registration = self._registrations.get(key)
        if registration is None:
            available = ", ".join(self.available_mechanisms()) or "none"
            raise AdapterConfigurationError(
                f"no memory adapter is registered for {key!r}; available: {available}"
            )
        if not isinstance(settings, registration.settings_type):
            raise AdapterConfigurationError(
                f"{registration.mechanism!r} requires settings of type "
                f"{registration.settings_type.__name__}; got "
                f"{type(settings).__name__}"
            )
        return registration.factory(settings)

    def available_mechanisms(self, *, include_aliases: bool = False) -> tuple[str, ...]:
        """List deterministic canonical names, optionally including aliases."""
        names = self._registrations if include_aliases else self._canonical_names
        return tuple(sorted(names))

    def registration_for(self, mechanism: str | Enum) -> AdapterRegistration:
        """Return immutable registration metadata without creating an adapter."""
        key = _mechanism_key(mechanism)
        try:
            return self._registrations[key]
        except KeyError as exc:
            raise AdapterConfigurationError(
                f"no memory adapter is registered for {key!r}"
            ) from exc


def _build_default_registry() -> AdapterRegistry:
    # Centralizing imports here keeps registration explicit. The runtime modules
    # themselves defer optional third-party SDK imports until initialization.
    from membench.dakera import DakeraSettings
    from membench.langmem import LangMemSettings
    from membench.memory.adapters.dakera_adapter import DakeraAdapter
    from membench.memory.adapters.langmem_adapter import LangMemAdapter
    from membench.memory.adapters.telemem_adapter import TeleMemAdapter
    from membench.telemem import TeleMemSettings

    result = AdapterRegistry()
    result.register("telemem", TeleMemSettings, TeleMemAdapter)
    result.register("langmem", LangMemSettings, LangMemAdapter, aliases=("langgraph",))
    result.register("dakera", DakeraSettings, DakeraAdapter)
    return result


DEFAULT_REGISTRY = _build_default_registry()


def create(mechanism: str | Enum, settings: Any) -> Any:
    """Construct an adapter through the process-wide default registry."""
    return DEFAULT_REGISTRY.create(mechanism, settings)


def available_mechanisms(*, include_aliases: bool = False) -> tuple[str, ...]:
    """List mechanisms exposed by the process-wide default registry."""
    return DEFAULT_REGISTRY.available_mechanisms(include_aliases=include_aliases)
