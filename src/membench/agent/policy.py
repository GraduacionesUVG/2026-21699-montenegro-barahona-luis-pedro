"""Deterministic authorization policy for proposed bounded-agent actions."""

from __future__ import annotations

from membench.agent.models import (
    AgentRunConfiguration,
    ApprovalDecision,
    PolicyDecision,
    PolicyDisposition,
    ToolActionDecision,
)
from membench.tools.models import ToolDefinition, ToolSideEffectKind


class AllowlistedActionPolicy:
    """Enforce trusted run limits before any tool implementation is invoked."""

    def __init__(self, configuration: AgentRunConfiguration) -> None:
        self._configuration = configuration

    @property
    def configuration(self) -> AgentRunConfiguration:
        """Return the immutable trusted configuration used by this policy."""

        return self._configuration

    def evaluate(
        self,
        action: ToolActionDecision,
        tool: ToolDefinition | None,
        *,
        step_index: int,
        tool_calls_so_far: int,
        approval: ApprovalDecision | None = None,
    ) -> PolicyDecision:
        """Return an auditable decision without executing the proposed call."""

        if step_index < 1:
            raise ValueError("step_index must be positive")
        if tool_calls_so_far < 0:
            raise ValueError("tool_calls_so_far cannot be negative")

        call = action.call
        if tool is None:
            return self._decision(
                action,
                PolicyDisposition.DENY,
                "tool_unavailable",
                "proposed tool is not registered in the trusted tool executor",
            )
        if call.tool_name != tool.name:
            return self._decision(
                action,
                PolicyDisposition.DENY,
                "tool_definition_mismatch",
                "resolved tool definition does not match the proposed tool name",
            )
        if step_index > self.configuration.max_steps:
            return self._decision(
                action,
                PolicyDisposition.DENY,
                "step_limit_exceeded",
                "proposed action exceeds the configured step limit",
            )
        if tool_calls_so_far >= self.configuration.max_tool_calls:
            return self._decision(
                action,
                PolicyDisposition.DENY,
                "tool_call_limit_exceeded",
                "proposed action exceeds the configured tool-call limit",
            )
        if call.tool_name not in self.configuration.allowed_tools:
            return self._decision(
                action,
                PolicyDisposition.DENY,
                "tool_not_allowlisted",
                "proposed tool is not present in the trusted allowlist",
            )
        if (
            tool.side_effect == ToolSideEffectKind.EXTERNAL_WRITE
            and not self.configuration.allow_external_side_effects
        ):
            return self._decision(
                action,
                PolicyDisposition.DENY,
                "external_side_effect_denied",
                "external side effects are disabled for this run",
            )

        approval_required = (
            tool.requires_approval
            or call.tool_name in self.configuration.approval_required_tools
        )
        if not approval_required:
            return self._decision(
                action,
                PolicyDisposition.ALLOW,
                "tool_allowed",
                "tool call satisfies the trusted allowlist and run limits",
            )
        if approval is None:
            return self._decision(
                action,
                PolicyDisposition.REQUIRE_APPROVAL,
                "approval_required",
                "tool call requires a trusted approval decision",
            )
        if approval.call_id != call.call_id:
            return self._decision(
                action,
                PolicyDisposition.DENY,
                "approval_call_mismatch",
                "approval does not reference the proposed tool call",
            )
        if not approval.approved:
            return self._decision(
                action,
                PolicyDisposition.DENY,
                "approval_denied",
                "trusted approval explicitly denied the proposed tool call",
                approval_id=approval.approval_id,
            )
        return self._decision(
            action,
            PolicyDisposition.ALLOW,
            "approved_tool_allowed",
            "trusted approval permits the proposed tool call",
            approval_id=approval.approval_id,
        )

    @staticmethod
    def _decision(
        action: ToolActionDecision,
        disposition: PolicyDisposition,
        rule_id: str,
        reason: str,
        *,
        approval_id: str | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            call_id=action.call.call_id,
            tool_name=action.call.tool_name,
            disposition=disposition,
            rule_id=rule_id,
            reason=reason,
            approval_id=approval_id,
        )
