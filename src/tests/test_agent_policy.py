from membench.agent.models import (
    AgentRunConfiguration,
    ApprovalDecision,
    PolicyDisposition,
    ToolActionDecision,
)
from membench.agent.policy import AllowlistedActionPolicy
from membench.tools.models import ToolCall, ToolDefinition, ToolSideEffectKind


def _action(tool_name: str = "lookup_contact") -> ToolActionDecision:
    return ToolActionDecision(
        call=ToolCall(
            call_id="call-001",
            tool_name=tool_name,
            arguments={"persona_id": "persona-001"},
            idempotency_key="task-001:call-001",
        ),
        decision_summary="Use the requested synthetic tool.",
    )


def _configuration(**updates: object) -> AgentRunConfiguration:
    values: dict[str, object] = {
        "max_steps": 3,
        "max_tool_calls": 2,
        "allowed_tools": ("lookup_contact", "simulate_send"),
        "approval_required_tools": ("simulate_send",),
    }
    values.update(updates)
    return AgentRunConfiguration(**values)


def test_policy_allows_an_allowlisted_read_only_tool() -> None:
    policy = AllowlistedActionPolicy(_configuration())

    decision = policy.evaluate(
        _action(),
        ToolDefinition(
            name="lookup_contact",
            description="Look up a synthetic contact.",
        ),
        step_index=1,
        tool_calls_so_far=0,
    )

    assert decision.disposition == PolicyDisposition.ALLOW
    assert decision.rule_id == "tool_allowed"


def test_policy_denies_a_tool_outside_the_allowlist() -> None:
    policy = AllowlistedActionPolicy(
        _configuration(
            allowed_tools=("lookup_contact",),
            approval_required_tools=(),
        )
    )

    decision = policy.evaluate(
        _action("unknown_tool"),
        ToolDefinition(
            name="unknown_tool",
            description="A tool not authorized for this run.",
        ),
        step_index=1,
        tool_calls_so_far=0,
    )

    assert decision.disposition == PolicyDisposition.DENY
    assert decision.rule_id == "tool_not_allowlisted"


def test_policy_requires_call_bound_approval_before_simulated_send() -> None:
    policy = AllowlistedActionPolicy(_configuration())
    action = _action("simulate_send")
    tool = ToolDefinition(
        name="simulate_send",
        description="Record a simulated delivery attempt.",
        side_effect=ToolSideEffectKind.SIMULATED_WRITE,
    )

    pending = policy.evaluate(
        action,
        tool,
        step_index=1,
        tool_calls_so_far=0,
    )
    allowed = policy.evaluate(
        action,
        tool,
        step_index=1,
        tool_calls_so_far=0,
        approval=ApprovalDecision(
            approval_id="approval-001",
            call_id="call-001",
            approved=True,
            reason="Approved for a synthetic delivery test.",
        ),
    )

    assert pending.disposition == PolicyDisposition.REQUIRE_APPROVAL
    assert pending.rule_id == "approval_required"
    assert allowed.disposition == PolicyDisposition.ALLOW
    assert allowed.rule_id == "approved_tool_allowed"
    assert allowed.approval_id == "approval-001"


def test_policy_denies_external_side_effects_before_approval() -> None:
    policy = AllowlistedActionPolicy(_configuration())

    decision = policy.evaluate(
        _action("simulate_send"),
        ToolDefinition(
            name="simulate_send",
            description="An unsafe real delivery tool.",
            side_effect=ToolSideEffectKind.EXTERNAL_WRITE,
            requires_approval=True,
        ),
        step_index=1,
        tool_calls_so_far=0,
    )

    assert decision.disposition == PolicyDisposition.DENY
    assert decision.rule_id == "external_side_effect_denied"


def test_policy_enforces_step_and_tool_call_limits() -> None:
    policy = AllowlistedActionPolicy(_configuration())
    action = _action()
    tool = ToolDefinition(
        name="lookup_contact",
        description="Look up a synthetic contact.",
    )

    step_denied = policy.evaluate(
        action,
        tool,
        step_index=4,
        tool_calls_so_far=0,
    )
    calls_denied = policy.evaluate(
        action,
        tool,
        step_index=2,
        tool_calls_so_far=2,
    )

    assert step_denied.rule_id == "step_limit_exceeded"
    assert calls_denied.rule_id == "tool_call_limit_exceeded"
