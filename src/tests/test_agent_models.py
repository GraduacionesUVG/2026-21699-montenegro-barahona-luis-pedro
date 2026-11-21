import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from membench.agent.models import (
    AgentRunConfiguration,
    AgentRunResult,
    AgentRunStatus,
    AgentStep,
    AgentTask,
    AgentTerminationReason,
    ApprovalDecision,
    FinalAnswerDecision,
    PolicyDecision,
    PolicyDisposition,
    ToolActionDecision,
)
from membench.memory.models import MemoryScope, RetrievalResult
from membench.tools.models import ToolCall, ToolResult, ToolStatus


def _scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="agent-tests",
        run_id="run-001",
        user_id="persona-001",
        namespace=("membench", "agent", "run-001"),
    )


def _task(*, query: str = "Where should the synthetic report be routed?") -> AgentTask:
    return AgentTask(
        task_id="task-001",
        scope=_scope(),
        instruction="Select the approved synthetic destination.",
        retrieval_query=query,
    )


def _configuration() -> AgentRunConfiguration:
    return AgentRunConfiguration(
        max_steps=3,
        max_tool_calls=2,
        allowed_tools=("lookup_contact", "simulate_send"),
        approval_required_tools=("simulate_send",),
    )


def _retrieval(task: AgentTask) -> RetrievalResult:
    return RetrievalResult(
        operation_id="retrieve-001",
        query_sha256=task.retrieval_query_sha256,
        request_top_k=5,
        latency_seconds=0.01,
    )


def _tool_action() -> ToolActionDecision:
    return ToolActionDecision(
        call=ToolCall(
            call_id="call-001",
            tool_name="lookup_contact",
            arguments={"persona_id": "persona-001"},
            idempotency_key="task-001:call-001",
        ),
        decision_summary="Look up the synthetic contact record.",
    )


def test_task_and_configuration_are_strict_and_validate_permissions() -> None:
    task = _task()

    assert task.retrieval_query_sha256 == hashlib.sha256(
        task.retrieval_query.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValidationError, match="frozen"):
        task.instruction = "changed"
    with pytest.raises(ValidationError, match="must also be allowed"):
        AgentRunConfiguration(
            allowed_tools=("lookup_contact",),
            approval_required_tools=("simulate_send",),
        )
    with pytest.raises(ValidationError, match="cannot exceed max_steps"):
        AgentRunConfiguration(max_steps=1, max_tool_calls=2)


def test_tool_action_step_requires_matching_allowed_execution_evidence() -> None:
    action = _tool_action()
    policy = PolicyDecision(
        call_id="call-001",
        tool_name="lookup_contact",
        disposition=PolicyDisposition.ALLOW,
        rule_id="tool_allowed",
        reason="allowed for test",
    )

    with pytest.raises(ValidationError, match="require a tool result"):
        AgentStep(
            step_index=1,
            prompt_sha256="a" * 64,
            context_sha256="b" * 64,
            decision=action,
            policy_decision=policy,
        )

    step = AgentStep(
        step_index=1,
        prompt_sha256="a" * 64,
        context_sha256="b" * 64,
        decision=action,
        policy_decision=policy,
        tool_result=ToolResult(
            call_id="call-001",
            tool_name="lookup_contact",
            status=ToolStatus.SUCCEEDED,
            output={"endpoint": "reports@example.invalid"},
        ),
    )

    assert step.tool_result is not None
    assert step.tool_result.status == ToolStatus.SUCCEEDED


def test_final_decision_cannot_contain_tool_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot contain tool evidence"):
        AgentStep(
            step_index=1,
            prompt_sha256="a" * 64,
            context_sha256="b" * 64,
            decision=FinalAnswerDecision(content="Use the trusted endpoint."),
            policy_decision=PolicyDecision(
                call_id="call-001",
                tool_name="lookup_contact",
                disposition=PolicyDisposition.DENY,
                rule_id="tool_not_allowlisted",
                reason="not allowed",
            ),
        )


def test_tool_step_preserves_call_bound_approval_evidence() -> None:
    action = _tool_action()
    approval = ApprovalDecision(
        approval_id="approval-001",
        call_id="call-001",
        approved=True,
        reason="Approved for the synthetic test.",
    )
    step = AgentStep(
        step_index=1,
        prompt_sha256="a" * 64,
        context_sha256="b" * 64,
        decision=action,
        approval_decision=approval,
        policy_decision=PolicyDecision(
            call_id="call-001",
            tool_name="lookup_contact",
            disposition=PolicyDisposition.ALLOW,
            rule_id="approved_tool_allowed",
            reason="Approved for the synthetic test.",
            approval_id=approval.approval_id,
        ),
        tool_result=ToolResult(
            call_id="call-001",
            tool_name="lookup_contact",
            status=ToolStatus.SUCCEEDED,
            output={"endpoint": "reports@example.invalid"},
        ),
    )

    assert step.approval_decision == approval


def test_run_result_requires_matching_query_status_and_consecutive_steps() -> None:
    task = _task()
    now = datetime.now(UTC)
    final_step = AgentStep(
        step_index=1,
        prompt_sha256="a" * 64,
        context_sha256="b" * 64,
        decision=FinalAnswerDecision(content="Use reports@example.invalid."),
        started_at=now,
        finished_at=now,
    )

    result = AgentRunResult(
        task=task,
        configuration=_configuration(),
        retrieval_result=_retrieval(task),
        context_sha256="b" * 64,
        status=AgentRunStatus.COMPLETED,
        termination_reason=AgentTerminationReason.FINAL_ANSWER,
        steps=(final_step,),
        final_answer="Use reports@example.invalid.",
        started_at=now,
        finished_at=now,
    )

    assert result.status == AgentRunStatus.COMPLETED
    with pytest.raises(ValidationError, match="does not match the task query"):
        AgentRunResult(
            task=_task(query="a different query"),
            configuration=_configuration(),
            retrieval_result=_retrieval(task),
            context_sha256="b" * 64,
            status=AgentRunStatus.COMPLETED,
            termination_reason=AgentTerminationReason.FINAL_ANSWER,
            steps=(final_step,),
            final_answer="Use reports@example.invalid.",
            started_at=now,
            finished_at=now,
        )

