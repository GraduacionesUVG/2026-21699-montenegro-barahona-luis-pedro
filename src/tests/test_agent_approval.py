from membench.agent.approval import SyntheticSendApprovalProvider
from membench.agent.models import (
    AgentTask,
    PolicyDecision,
    PolicyDisposition,
    ToolActionDecision,
)
from membench.memory.models import MemoryScope
from membench.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolSideEffectKind,
)


def _scope(run_id: str = "run-001") -> MemoryScope:
    return MemoryScope(
        experiment_id="approval-tests",
        run_id=run_id,
        user_id="277",
        namespace=("membench", "approval", run_id),
    )


def _task(scope: MemoryScope | None = None) -> AgentTask:
    resolved = scope or _scope()
    return AgentTask(
        task_id="behavior-send-case-summary-277-001",
        scope=resolved,
        instruction="Send the synthetic case summary.",
        retrieval_query="case summary routing",
    )


def _action(recipient: str, *, tool_name: str = "simulate_send") -> ToolActionDecision:
    return ToolActionDecision(
        call=ToolCall(
            call_id="call-001",
            tool_name=tool_name,
            arguments={
                "recipient": recipient,
                "subject": "Synthetic case summary",
                "body": "Synthetic test content.",
            },
            idempotency_key="behavior-send-case-summary-277-001:call-001",
        ),
        decision_summary="Send the synthetic case summary.",
    )


def _pending(action: ToolActionDecision) -> PolicyDecision:
    return PolicyDecision(
        call_id=action.call.call_id,
        tool_name=action.call.tool_name,
        disposition=PolicyDisposition.REQUIRE_APPROVAL,
        rule_id="approval_required",
        reason="Tool call requires trusted approval.",
    )


def test_approves_trusted_and_attacker_controlled_reserved_addresses() -> None:
    provider = SyntheticSendApprovalProvider(_scope())

    decisions = tuple(
        provider.request_approval(_task(), action, _pending(action))
        for action in (
            _action("case.desk@community.test"),
            _action("case.review@harborbridge.test"),
        )
    )

    assert all(decision is not None and decision.approved for decision in decisions)
    assert decisions[0].approval_id != decisions[1].approval_id
    assert all(decision.call_id == "call-001" for decision in decisions)


def test_approves_reserved_invalid_email_and_fictional_phone() -> None:
    provider = SyntheticSendApprovalProvider(_scope())

    for recipient in ("review@example.invalid", "+1-555-0107"):
        action = _action(recipient)
        decision = provider.request_approval(_task(), action, _pending(action))
        assert decision is not None and decision.approved


def test_returns_none_unless_policy_requires_approval() -> None:
    provider = SyntheticSendApprovalProvider(_scope())
    action = _action("case.desk@community.test")
    allowed = _pending(action).model_copy(
        update={
            "disposition": PolicyDisposition.ALLOW,
            "rule_id": "tool_allowed",
        }
    )

    assert provider.request_approval(_task(), action, allowed) is None


def test_denies_wrong_scope_tool_call_identity_and_real_recipient() -> None:
    provider = SyntheticSendApprovalProvider(_scope())
    valid = _action("case.desk@community.test")
    mismatched_policy = _pending(valid).model_copy(
        update={"call_id": "different-call"}
    )

    wrong_scope = provider.request_approval(
        _task(_scope("run-002")), valid, _pending(valid)
    )
    mismatched = provider.request_approval(_task(), valid, mismatched_policy)
    wrong_tool = _action("case.desk@community.test", tool_name="external_send")
    wrong_tool_decision = provider.request_approval(
        _task(), wrong_tool, _pending(wrong_tool)
    )
    real_recipient = _action("person@example.com")
    real_recipient_decision = provider.request_approval(
        _task(), real_recipient, _pending(real_recipient)
    )

    assert wrong_scope is not None and not wrong_scope.approved
    assert mismatched is not None and not mismatched.approved
    assert wrong_tool_decision is not None and not wrong_tool_decision.approved
    assert real_recipient_decision is not None and not real_recipient_decision.approved


def test_never_approves_an_external_write_definition() -> None:
    provider = SyntheticSendApprovalProvider(
        _scope(),
        ToolDefinition(
            name="simulate_send",
            description="Unsafe real send.",
            side_effect=ToolSideEffectKind.EXTERNAL_WRITE,
            requires_approval=True,
        ),
    )
    action = _action("case.review@harborbridge.test")

    decision = provider.request_approval(_task(), action, _pending(action))

    assert decision is not None
    assert not decision.approved
    assert "SIMULATED_WRITE" in decision.reason
