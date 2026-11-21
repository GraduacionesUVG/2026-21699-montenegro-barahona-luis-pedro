"""Structural boundaries used by the mechanism-independent agent runner."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from membench.agent.models import (
    AgentDecision,
    AgentDecisionContext,
    AgentMemoryContext,
    AgentRunConfiguration,
    AgentStep,
    AgentTask,
    ApprovalDecision,
    PolicyDecision,
    ToolActionDecision,
)
from membench.memory.models import RetrievalResult
from membench.tools.models import ToolDefinition


@runtime_checkable
class ActionDecider(Protocol):
    """Produce one structured decision from an exact, logged prompt context."""

    def verify(self) -> None:
        """Validate model/provider configuration without performing a task."""

        ...

    def decide(self, context: AgentDecisionContext) -> AgentDecision:
        """Return exactly one final, refusal, or proposed tool-call decision."""

        ...


@runtime_checkable
class AgentContextBuilder(Protocol):
    """Select frozen memory once and construct every decision prompt."""

    def select_memory(
        self,
        task: AgentTask,
        retrieval: RetrievalResult,
        tools: tuple[ToolDefinition, ...],
        *,
        token_budget: int,
    ) -> AgentMemoryContext:
        """Select a ranked prefix within the first-step input budget."""

        ...

    def assemble(
        self,
        task: AgentTask,
        memory_context: AgentMemoryContext,
        tools: tuple[ToolDefinition, ...],
        prior_steps: tuple[AgentStep, ...],
        *,
        step_index: int,
        token_budget: int,
    ) -> AgentDecisionContext:
        """Build and hash the exact prompt for the next decision."""

        ...


@runtime_checkable
class ActionPolicy(Protocol):
    """Authorize a proposed tool action without executing it."""

    @property
    def configuration(self) -> AgentRunConfiguration:
        """Return the immutable trusted configuration enforced by the policy."""

        ...

    def evaluate(
        self,
        action: ToolActionDecision,
        tool: ToolDefinition | None,
        *,
        step_index: int,
        tool_calls_so_far: int,
        approval: ApprovalDecision | None = None,
    ) -> PolicyDecision:
        """Return an auditable allow, deny, or approval-required decision."""

        ...


@runtime_checkable
class ApprovalProvider(Protocol):
    """Obtain a trusted, call-bound approval outside model control."""

    def request_approval(
        self,
        task: AgentTask,
        action: ToolActionDecision,
        policy_decision: PolicyDecision,
    ) -> ApprovalDecision | None:
        """Return a decision, or ``None`` when approval is unavailable."""

        ...
