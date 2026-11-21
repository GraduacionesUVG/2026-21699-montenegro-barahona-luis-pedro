"""Deterministic registry for trusted bounded-agent tool implementations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from membench.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolStatus,
)
from membench.tools.protocol import ToolArgumentValidator, ToolImplementation

if TYPE_CHECKING:
    from membench.agent.models import AgentToolExecutionContext


class ToolRegistryError(RuntimeError):
    """Base error for invalid registry configuration or lookup."""


class DuplicateToolError(ToolRegistryError):
    """Raised when a trusted tool name is registered more than once."""


class ToolNotRegisteredError(ToolRegistryError):
    """Raised when execution is requested for an unknown tool name."""


class FrozenToolRegistryError(ToolRegistryError):
    """Raised when registration is attempted after the catalog is frozen."""


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """One trusted definition-to-implementation association."""

    definition: ToolDefinition
    implementation: ToolImplementation
    argument_validator: ToolArgumentValidator | None = None


class ToolRegistry:
    """Resolve and invoke registered tools without owning their domain data."""

    def __init__(self) -> None:
        self._registrations: dict[str, ToolRegistration] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        """Report whether the trusted catalog can still be modified."""

        return self._frozen

    def register(
        self,
        definition: ToolDefinition,
        implementation: ToolImplementation,
        *,
        argument_validator: ToolArgumentValidator | None = None,
    ) -> None:
        """Associate one unique definition with executable business logic."""

        if self._frozen:
            raise FrozenToolRegistryError(
                "tool registry is frozen and cannot accept registrations"
            )
        if definition.name in self._registrations:
            raise DuplicateToolError(
                f"tool {definition.name!r} is already registered"
            )
        if not callable(getattr(implementation, "invoke", None)):
            raise TypeError("tool implementation must define callable invoke()")
        if argument_validator is not None and not callable(argument_validator):
            raise TypeError("argument_validator must be callable or None")
        self._registrations[definition.name] = ToolRegistration(
            definition=definition,
            implementation=implementation,
            argument_validator=argument_validator,
        )

    def freeze(self) -> None:
        """Prevent catalog changes for a reproducible agent run."""

        self._frozen = True

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return definitions sorted by canonical name for stable prompts."""

        return tuple(
            self._registrations[name].definition
            for name in sorted(self._registrations)
        )

    def resolve(self, name: str) -> ToolDefinition | None:
        """Resolve trusted metadata without exposing the implementation."""

        registration = self._registrations.get(name)
        return None if registration is None else registration.definition

    def registration_for(self, name: str) -> ToolRegistration:
        """Return immutable registration metadata for evaluator inspection."""

        try:
            return self._registrations[name]
        except KeyError as exc:
            raise ToolNotRegisteredError(
                f"tool {name!r} is not registered"
            ) from exc

    def execute(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> ToolResult:
        """Validate, invoke, and normalize one already-authorized tool call."""

        registration = self._registrations.get(call.tool_name)
        if registration is None:
            raise ToolNotRegisteredError(
                f"tool {call.tool_name!r} is not registered"
            )

        started = time.perf_counter()
        try:
            normalized_call = self._validated_call(call, registration)
            output = registration.implementation.invoke(normalized_call, context)
            if not isinstance(output, dict):
                raise TypeError("tool implementation output must be a JSON object")
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolStatus.SUCCEEDED,
                output=dict(output),
                latency_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolStatus.FAILED,
                latency_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
                error_message=str(exc) or "tool invocation failed without a message",
            )

    @staticmethod
    def _validated_call(
        call: ToolCall,
        registration: ToolRegistration,
    ) -> ToolCall:
        validator = registration.argument_validator
        if validator is None:
            return call
        validated = validator(dict(call.arguments))
        if not isinstance(validated, dict):
            raise TypeError("tool argument validator must return a JSON object")
        return call.model_copy(update={"arguments": dict(validated)})

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._registrations

    def __len__(self) -> int:
        return len(self._registrations)
