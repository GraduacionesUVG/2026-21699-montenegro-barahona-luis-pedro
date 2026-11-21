"""Event-sink boundaries used by the qualification runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from membench.qualification.models import QualificationEvent


class QualificationEventSink(Protocol):
    """Accept one already validated qualification event."""

    def emit(self, event: QualificationEvent) -> None: ...


class NullQualificationEventSink:
    """Discard events when artifact logging has not yet been configured."""

    def emit(self, event: QualificationEvent) -> None:
        del event


@dataclass(slots=True)
class InMemoryQualificationEventSink:
    """Retain events for unit tests and interactive smoke runs."""

    events: list[QualificationEvent] = field(default_factory=list)

    def emit(self, event: QualificationEvent) -> None:
        self.events.append(event)
