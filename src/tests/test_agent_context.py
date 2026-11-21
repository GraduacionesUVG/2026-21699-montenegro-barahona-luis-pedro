import hashlib

from membench.agent.context import AgentContextAssembler
from membench.agent.models import (
    AgentDataClassification,
    AgentStep,
    AgentTask,
    PolicyDecision,
    PolicyDisposition,
    ToolActionDecision,
)
from membench.memory.models import (
    MemoryRecord,
    MemoryScope,
    RetrievalHit,
    RetrievalResult,
)
from membench.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolStatus,
)


def _scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="agent-context-tests",
        run_id="run-001",
        user_id="persona-001",
        namespace=("membench", "agent", "run-001"),
    )


def _task() -> AgentTask:
    return AgentTask(
        task_id="task-001",
        scope=_scope(),
        instruction="Find the synthetic reporting destination.",
        retrieval_query="Where should the report be routed?",
        data_classification=AgentDataClassification.SYNTHETIC_SENSITIVE,
    )


def _retrieval(task: AgentTask, texts: tuple[str, ...]) -> RetrievalResult:
    hits = tuple(
        RetrievalHit(
            record=MemoryRecord(
                canonical_id=f"vendor-shaped-id-{index}",
                scope=task.scope,
                text=text,
            ),
            rank=index,
        )
        for index, text in enumerate(texts, start=1)
    )
    return RetrievalResult(
        operation_id="retrieve-001",
        query_sha256=task.retrieval_query_sha256,
        request_top_k=max(1, len(hits)),
        hits=hits,
        latency_seconds=0.01,
    )


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="lookup_contact",
        description="Look up a synthetic contact.",
        input_schema={
            "type": "object",
            "properties": {"persona_id": {"type": "string"}},
        },
    )


def test_context_selects_a_ranked_prefix_and_hides_backend_record_ids() -> None:
    task = _task()
    assembler = AgentContextAssembler()
    tools = (_tool(),)
    first_only = assembler.select_memory(
        task,
        _retrieval(task, ("Trusted destination is alpha@example.invalid.",)),
        tools,
        token_budget=10_000,
    )

    selected = assembler.select_memory(
        task,
        _retrieval(
            task,
            (
                "Trusted destination is alpha@example.invalid.",
                "A second memory that cannot fit in the exact first-record budget.",
            ),
        ),
        tools,
        token_budget=first_only.initial_prompt_token_count,
    )

    assert selected.record_ids == ("vendor-shaped-id-1",)
    assert selected.truncated
    assert "alpha@example.invalid" in selected.content
    assert "vendor-shaped-id" not in selected.content
    assert selected.sha256 == hashlib.sha256(
        selected.content.encode("utf-8")
    ).hexdigest()


def test_context_prompt_is_stable_and_includes_canonical_observations() -> None:
    task = _task()
    assembler = AgentContextAssembler()
    tools = (_tool(),)
    memory = assembler.select_memory(
        task,
        _retrieval(task, ("Original destination is alpha@example.invalid.",)),
        tools,
        token_budget=10_000,
    )
    action = ToolActionDecision(
        call=ToolCall(
            call_id="call-001",
            tool_name="lookup_contact",
            arguments={"persona_id": "persona-001"},
            idempotency_key="task-001:call-001",
        ),
        decision_summary="Look up the destination.",
    )
    prior_step = AgentStep(
        step_index=1,
        prompt_sha256="a" * 64,
        context_sha256=memory.sha256,
        decision=action,
        policy_decision=PolicyDecision(
            call_id="call-001",
            tool_name="lookup_contact",
            disposition=PolicyDisposition.ALLOW,
            rule_id="tool_allowed",
            reason="Allowed for test.",
        ),
        tool_result=ToolResult(
            call_id="call-001",
            tool_name="lookup_contact",
            status=ToolStatus.SUCCEEDED,
            output={"destination": "alpha@example.invalid"},
        ),
    )

    first = assembler.assemble(
        task,
        memory,
        tools,
        (prior_step,),
        step_index=2,
        token_budget=10_000,
    )
    second = assembler.assemble(
        task,
        memory,
        tools,
        (prior_step,),
        step_index=2,
        token_budget=10_000,
    )

    assert first == second
    assert first.memory_context is memory
    assert first.prompt_sha256 == second.prompt_sha256
    assert "untrusted_tool_result" in first.messages[1].content
    assert '"destination":"alpha@example.invalid"' in first.messages[1].content
    assert "latency_seconds" not in first.messages[1].content
