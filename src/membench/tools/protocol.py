"""Structural protocol shared by mechanism-independent agent toolboxes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from membench.memory.models import JsonObject
from membench.tools.models import ToolCall, ToolDefinition, ToolResult

if TYPE_CHECKING:
    from membench.agent.models import AgentToolExecutionContext


@runtime_checkable
class ToolImplementation(Protocol):
    """Business logic registered under one trusted tool definition."""

    def invoke(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        """Perform the synthetic operation and return its JSON observation."""

        ...


@runtime_checkable
class ToolArgumentValidator(Protocol):
    """Validate and optionally normalize model-proposed tool arguments."""

    def __call__(self, arguments: JsonObject) -> JsonObject:
        """Return validated arguments or raise a descriptive exception."""

        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Resolve trusted definitions and execute policy-approved tool calls."""

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return all registered tool definitions in deterministic order."""

        ...

    def resolve(self, name: str) -> ToolDefinition | None:
        """Resolve a registered definition without invoking its implementation."""

        ...

    def execute(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> ToolResult:
        """Execute one already-authorized call and return canonical evidence."""

        ...
