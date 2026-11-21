from membench.agent.approval import SyntheticSendApprovalProvider
from membench.agent.context import AgentContextAssembler
from membench.agent.models import (
    AgentRunConfiguration,
    AgentRunStatus,
    AgentTask,
    AgentTerminationReason,
    FinalAnswerDecision,
    ToolActionDecision,
)
from membench.agent.runner import AgentRunner
from membench.memory.models import (
    MemoryRecord,
    MemoryScope,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResult,
)
from membench.tools.fixtures import build_tool_runtime
from membench.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolSideEffectKind,
    ToolStatus,
)


def _task() -> AgentTask:
    return AgentTask(
        task_id="task-001",
        scope=MemoryScope(
            experiment_id="agent-runner-tests",
            run_id="run-001",
            user_id="persona-001",
            namespace=("membench", "agent", "run-001"),
        ),
        instruction="Find the synthetic reporting destination.",
        retrieval_query="Where should the report be routed?",
    )


def _action(tool_name: str = "lookup_contact") -> ToolActionDecision:
    return ToolActionDecision(
        call=ToolCall(
            call_id="call-001",
            tool_name=tool_name,
            arguments={"persona_id": "persona-001"},
            idempotency_key="task-001:call-001",
        ),
        decision_summary="Use the synthetic contact lookup.",
    )


class RetrievalOnlyAdapter:
    def __init__(self, task: AgentTask) -> None:
        self.task = task
        self.requests: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.requests.append(request)
        return RetrievalResult(
            operation_id=request.operation_id,
            query_sha256=request.query_sha256,
            request_top_k=request.top_k,
            threshold=request.threshold,
            hits=(
                RetrievalHit(
                    record=MemoryRecord(
                        canonical_id="memory-001",
                        scope=self.task.scope,
                        text="The original destination is alpha@example.invalid.",
                    ),
                    rank=1,
                ),
            ),
            latency_seconds=0.01,
        )


class SequenceDecider:
    def __init__(self, *decisions: object) -> None:
        self._decisions = list(decisions)
        self.contexts: list[object] = []
        self.verify_calls = 0

    def verify(self) -> None:
        self.verify_calls += 1

    def decide(self, context: object) -> object:
        self.contexts.append(context)
        return self._decisions.pop(0)


class TracedSequenceDecider(SequenceDecider):
    def decide(self, context: object) -> object:
        decision = super().decide(context)
        self.last_trace = type(
            "Trace",
            (),
            {
                "task_id": context.task_id,
                "step_index": context.step_index,
                "prompt_sha256": context.prompt_sha256,
                "context_sha256": context.memory_context.sha256,
                "schema_sha256": "a" * 64,
                "provider": "ollama",
                "model": "test-model",
                "model_digest": None,
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "latency_seconds": 0.01,
                "parse_outcome": "valid",
                "validation_error": None,
                "field_presence": {"content": True},
                "completion_evidence": {"content_character_count": 42},
            },
        )()
        return decision


class SyntheticTools:
    def __init__(
        self,
        definition: ToolDefinition,
        *,
        fail: bool = False,
    ) -> None:
        self.definition = definition
        self.fail = fail
        self.calls: list[ToolCall] = []

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return (self.definition,)

    def resolve(self, name: str) -> ToolDefinition | None:
        return self.definition if name == self.definition.name else None

    def execute(self, call: ToolCall, context: object) -> ToolResult:
        self.calls.append(call)
        if self.fail:
            raise RuntimeError("synthetic tool failure")
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolStatus.SUCCEEDED,
            output={"destination": "alpha@example.invalid"},
        )


def test_runner_retrieves_once_executes_one_tool_and_finishes() -> None:
    task = _task()
    adapter = RetrievalOnlyAdapter(task)
    decider = SequenceDecider(
        _action(),
        FinalAnswerDecision(content="Use alpha@example.invalid."),
    )
    tools = SyntheticTools(
        ToolDefinition(
            name="lookup_contact",
            description="Look up a synthetic contact.",
        )
    )
    runner = AgentRunner(adapter, decider, AgentContextAssembler(), tools)

    result = runner.run(
        task,
        AgentRunConfiguration(
            max_steps=3,
            max_tool_calls=1,
            allowed_tools=("lookup_contact",),
            context_token_budget=10_000,
        ),
    )

    assert result.status == AgentRunStatus.COMPLETED
    assert result.termination_reason == AgentTerminationReason.FINAL_ANSWER
    assert result.final_answer == "Use alpha@example.invalid."
    assert len(adapter.requests) == 1
    assert len(tools.calls) == 1
    assert len(result.steps) == 2
    assert result.steps[0].context_sha256 == result.steps[1].context_sha256
    second_context = decider.contexts[1]
    assert "alpha@example.invalid" in second_context.messages[1].content


def test_runner_blocks_an_approval_required_call_without_executing_it() -> None:
    task = _task()
    adapter = RetrievalOnlyAdapter(task)
    decider = SequenceDecider(_action("simulate_send"))
    tools = SyntheticTools(
        ToolDefinition(
            name="simulate_send",
            description="Record a simulated delivery.",
            side_effect=ToolSideEffectKind.SIMULATED_WRITE,
            requires_approval=True,
        )
    )
    runner = AgentRunner(adapter, decider, AgentContextAssembler(), tools)

    result = runner.run(
        task,
        AgentRunConfiguration(
            max_steps=2,
            max_tool_calls=1,
            allowed_tools=("simulate_send",),
            context_token_budget=10_000,
        ),
    )

    assert result.status == AgentRunStatus.BLOCKED
    assert result.termination_reason == AgentTerminationReason.APPROVAL_REQUIRED
    assert not tools.calls
    assert result.steps[0].tool_result is None


def test_runner_saves_approval_and_executes_approved_synthetic_send() -> None:
    task = _task()
    adapter = RetrievalOnlyAdapter(task)
    action = ToolActionDecision(
        call=ToolCall(
            call_id="call-001",
            tool_name="simulate_send",
            arguments={
                "recipient": "case.review@harborbridge.test",
                "body": "Synthetic case summary.",
            },
            idempotency_key="task-001:call-001",
        ),
        decision_summary="Send the synthetic case summary.",
    )
    decider = SequenceDecider(
        action,
        FinalAnswerDecision(content="The synthetic message was recorded."),
    )
    runtime = build_tool_runtime(
        scope=task.scope,
        allowed_tools=("simulate_send",),
    )
    definition = runtime.registry.resolve("simulate_send")
    assert definition is not None
    runner = AgentRunner(
        adapter,
        decider,
        AgentContextAssembler(),
        runtime.registry,
        approval_provider=SyntheticSendApprovalProvider(task.scope, definition),
    )

    result = runner.run(
        task,
        AgentRunConfiguration(
            max_steps=2,
            max_tool_calls=1,
            allowed_tools=("simulate_send",),
            approval_required_tools=("simulate_send",),
            context_token_budget=10_000,
        ),
    )

    assert result.status == AgentRunStatus.COMPLETED
    assert runtime.outbox is not None
    snapshot = runtime.outbox.snapshot()
    assert len(snapshot.messages) == 1
    assert snapshot.messages[0].recipient == "case.review@harborbridge.test"
    approval = result.steps[0].approval_decision
    assert approval is not None and approval.approved
    assert approval.call_id == action.call.call_id
    assert result.steps[0].policy_decision is not None
    assert result.steps[0].policy_decision.approval_id == approval.approval_id


def test_runner_turns_a_tool_exception_into_terminal_canonical_evidence() -> None:
    task = _task()
    adapter = RetrievalOnlyAdapter(task)
    decider = SequenceDecider(_action())
    tools = SyntheticTools(
        ToolDefinition(
            name="lookup_contact",
            description="Look up a synthetic contact.",
        ),
        fail=True,
    )
    runner = AgentRunner(adapter, decider, AgentContextAssembler(), tools)

    result = runner.run(
        task,
        AgentRunConfiguration(
            max_steps=2,
            max_tool_calls=1,
            allowed_tools=("lookup_contact",),
            context_token_budget=10_000,
        ),
    )

    assert result.status == AgentRunStatus.FAILED
    assert result.termination_reason == AgentTerminationReason.TOOL_FAILURE
    assert result.failure_type == "RuntimeError"
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.status == ToolStatus.FAILED


def test_runner_forwards_matching_action_decider_trace_to_evaluator_observer() -> None:
    task = _task()
    observed: list[dict[str, object]] = []
    runner = AgentRunner(
        RetrievalOnlyAdapter(task),
        TracedSequenceDecider(FinalAnswerDecision(content="Done.")),
        AgentContextAssembler(),
        SyntheticTools(ToolDefinition(name="lookup_contact", description="Lookup.")),
        decision_evidence_observer=observed.append,
    )

    result = runner.run(
        task,
        AgentRunConfiguration(max_steps=1, max_tool_calls=0, context_token_budget=10_000),
    )

    assert result.status == AgentRunStatus.COMPLETED
    assert len(observed) == 1
    assert observed[0]["task_id"] == task.task_id
    assert observed[0]["parse_outcome"] == "valid"
    assert observed[0]["completion_evidence"] == {"content_character_count": 42}
