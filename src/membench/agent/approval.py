from __future__ import annotations

import hashlib
import json

from membench.agent.models import (
    AgentTask,
    ApprovalDecision,
    PolicyDecision,
    PolicyDisposition,
    ToolActionDecision,
)
from membench.memory.models import MemoryScope
from membench.tools.mock_contacts import normalize_synthetic_recipient
from membench.tools.mock_messaging import (
    SIMULATE_SEND_DEFINITION,
    validate_simulate_send_arguments,
)
from membench.tools.models import ToolDefinition, ToolSideEffectKind


class SyntheticSendApprovalProvider:
    """Approve only side-effect-free synthetic messaging calls."""

    policy_id = "synthetic_send_auto_approval_v1"

    def __init__(
        self,
        expected_scope: MemoryScope,
        tool_definition: ToolDefinition = SIMULATE_SEND_DEFINITION,
    ) -> None:
        self._expected_scope = expected_scope
        self._tool_definition = tool_definition.model_copy(deep=True)

    def request_approval(
        self,
        task: AgentTask,
        action: ToolActionDecision,
        policy_decision: PolicyDecision,
    ) -> ApprovalDecision | None:
        call = action.call

        if policy_decision.disposition != PolicyDisposition.REQUIRE_APPROVAL:
            return None

        if (
            policy_decision.call_id != call.call_id
            or policy_decision.tool_name != call.tool_name
        ):
            return self._decision(
                call.call_id,
                approved=False,
                reason="Approval request does not match the proposed tool call.",
                action=action,
            )

        if task.scope != self._expected_scope:
            return self._decision(
                call.call_id,
                approved=False,
                reason="Task scope does not match the isolated trial scope.",
                action=action,
            )

        if call.tool_name != "simulate_send":
            return self._decision(
                call.call_id,
                approved=False,
                reason="Only simulated messaging can be automatically approved.",
                action=action,
            )

        if (
            self._tool_definition.name != "simulate_send"
            or self._tool_definition.side_effect
            != ToolSideEffectKind.SIMULATED_WRITE
        ):
            return self._decision(
                call.call_id,
                approved=False,
                reason=(
                    "Synthetic approval requires the trusted simulate_send "
                    "SIMULATED_WRITE definition."
                ),
                action=action,
            )

        try:
            recipient = call.arguments.get("recipient")
            if not isinstance(recipient, str):
                raise ValueError("simulate_send recipient must be a string")
            normalize_synthetic_recipient(recipient)
            validate_simulate_send_arguments(dict(call.arguments))
        except ValueError as exc:
            return self._decision(
                call.call_id,
                approved=False,
                reason=f"Invalid synthetic send arguments: {exc}",
                action=action,
            )

        return self._decision(
            call.call_id,
            approved=True,
            reason=(
                "Approved by synthetic_send_auto_approval_v1; "
                "the operation records evidence without network access."
            ),
            action=action,
        )

    def _decision(
        self,
        call_id: str,
        *,
        approved: bool,
        reason: str,
        action: ToolActionDecision,
    ) -> ApprovalDecision:
        identity = {
            "policy_id": self.policy_id,
            "scope": self._expected_scope.model_dump(mode="json"),
            "call_id": call_id,
            "tool_name": action.call.tool_name,
            "arguments": action.call.arguments,
        }
        digest = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:24]

        return ApprovalDecision(
            approval_id=f"synthetic-send-{digest}",
            call_id=call_id,
            approved=approved,
            reason=reason,
        )
