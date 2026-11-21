"""Bounded mechanism-independent observe-decide-act execution loop."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any
from datetime import UTC, datetime

from membench.agent.models import (
    AgentDecisionContext,
    AgentRunConfiguration,
    AgentRunResult,
    AgentRunStatus,
    AgentStep,
    AgentTask,
    AgentTerminationReason,
    AgentToolExecutionContext,
    FinalAnswerDecision,
    PolicyDisposition,
    RefusalDecision,
    ToolActionDecision,
)
from membench.agent.policy import AllowlistedActionPolicy
from membench.agent.protocol import (
    ActionDecider,
    ActionPolicy,
    AgentContextBuilder,
    ApprovalProvider,
)
from membench.memory.models import RetrievalRequest, RetrievalResult
from membench.memory.protocol import MemoryAdapter
from membench.tools.models import ToolDefinition, ToolResult, ToolStatus
from membench.tools.protocol import ToolExecutor

PolicyFactory = Callable[[AgentRunConfiguration], ActionPolicy]
DecisionEvidenceObserver = Callable[[dict[str, Any]], None]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AgentRunner:
    """Run one bounded task without writing to the bound memory adapter."""

    def __init__(
        self,
        adapter: MemoryAdapter,
        decider: ActionDecider,
        context_builder: AgentContextBuilder,
        tools: ToolExecutor,
        *,
        approval_provider: ApprovalProvider | None = None,
        policy_factory: PolicyFactory = AllowlistedActionPolicy,
        retrieval_observer: Callable[[RetrievalResult], None] | None = None,
        context_observer: Callable[[AgentDecisionContext], None] | None = None,
        decision_evidence_observer: DecisionEvidenceObserver | None = None,
    ) -> None:
        self._adapter = adapter
        self._decider = decider
        self._context_builder = context_builder
        self._tools = tools
        self._approval_provider = approval_provider
        self._policy_factory = policy_factory
        self._retrieval_observer = retrieval_observer
        self._context_observer = context_observer
        self._decision_evidence_observer = decision_evidence_observer

    def run(
        self,
        task: AgentTask,
        configuration: AgentRunConfiguration,
    ) -> AgentRunResult:
        """Retrieve once, then execute only bounded and policy-approved actions."""

        started_at = _utc_now()
        self._decider.verify()
        available_tools = self._available_tools(configuration)
        policy = self._policy_factory(configuration)
        if policy.configuration != configuration:
            raise ValueError("action policy is not bound to the run configuration")

        retrieval = self._adapter.retrieve(
            RetrievalRequest(
                operation_id=f"{task.task_id}:agent_retrieve",
                scope=task.scope,
                query=task.retrieval_query,
                top_k=configuration.retrieval_top_k,
                threshold=configuration.retrieval_threshold,
            )
        )
        if self._retrieval_observer is not None:
            self._retrieval_observer(retrieval.model_copy(deep=True))
        memory_context = self._context_builder.select_memory(
            task,
            retrieval,
            available_tools,
            token_budget=configuration.context_token_budget,
        )

        steps: list[AgentStep] = []
        tool_calls_so_far = 0
        for step_index in range(1, configuration.max_steps + 1):
            try:
                decision_context = self._context_builder.assemble(
                    task,
                    memory_context,
                    available_tools,
                    tuple(steps),
                    step_index=step_index,
                    token_budget=configuration.context_token_budget,
                )
            except Exception as exc:
                return self._failed_result(
                    task,
                    configuration,
                    retrieval,
                    memory_context.sha256,
                    steps,
                    started_at,
                    AgentTerminationReason.CONTEXT_FAILURE,
                    exc,
                )

            # Instrumentation failures must propagate to the lifecycle owner.
            if self._context_observer is not None:
                self._context_observer(decision_context.model_copy(deep=True))
            step_started_at = _utc_now()
            try:
                decision = self._decider.decide(decision_context)
                if not isinstance(
                    decision,
                    (FinalAnswerDecision, RefusalDecision, ToolActionDecision),
                ):
                    raise TypeError("action decider returned an unsupported decision")
                self._observe_decision_evidence(task.task_id, step_index)
            except Exception as exc:
                self._observe_decision_evidence(task.task_id, step_index)
                return self._failed_result(
                    task,
                    configuration,
                    retrieval,
                    memory_context.sha256,
                    steps,
                    started_at,
                    AgentTerminationReason.MODEL_FAILURE,
                    exc,
                )

            if isinstance(decision, FinalAnswerDecision):
                steps.append(
                    AgentStep(
                        step_index=step_index,
                        prompt_sha256=decision_context.prompt_sha256,
                        context_sha256=memory_context.sha256,
                        decision=decision,
                        started_at=step_started_at,
                        finished_at=_utc_now(),
                    )
                )
                return self._result(
                    task,
                    configuration,
                    retrieval,
                    memory_context.sha256,
                    steps,
                    started_at,
                    status=AgentRunStatus.COMPLETED,
                    termination_reason=AgentTerminationReason.FINAL_ANSWER,
                    final_answer=decision.content,
                )

            if isinstance(decision, RefusalDecision):
                steps.append(
                    AgentStep(
                        step_index=step_index,
                        prompt_sha256=decision_context.prompt_sha256,
                        context_sha256=memory_context.sha256,
                        decision=decision,
                        started_at=step_started_at,
                        finished_at=_utc_now(),
                    )
                )
                return self._result(
                    task,
                    configuration,
                    retrieval,
                    memory_context.sha256,
                    steps,
                    started_at,
                    status=AgentRunStatus.REFUSED,
                    termination_reason=AgentTerminationReason.REFUSED,
                )

            tool = self._tools.resolve(decision.call.tool_name)
            policy_decision = policy.evaluate(
                decision,
                tool,
                step_index=step_index,
                tool_calls_so_far=tool_calls_so_far,
            )
            approval = None
            if policy_decision.disposition == PolicyDisposition.REQUIRE_APPROVAL:
                if self._approval_provider is not None:
                    approval = self._approval_provider.request_approval(
                        task,
                        decision,
                        policy_decision,
                    )
                if approval is not None:
                    policy_decision = policy.evaluate(
                        decision,
                        tool,
                        step_index=step_index,
                        tool_calls_so_far=tool_calls_so_far,
                        approval=approval,
                    )

            if policy_decision.disposition != PolicyDisposition.ALLOW:
                steps.append(
                    AgentStep(
                        step_index=step_index,
                        prompt_sha256=decision_context.prompt_sha256,
                        context_sha256=memory_context.sha256,
                        decision=decision,
                        policy_decision=policy_decision,
                        approval_decision=approval,
                        started_at=step_started_at,
                        finished_at=_utc_now(),
                    )
                )
                return self._blocked_result(
                    task,
                    configuration,
                    retrieval,
                    memory_context.sha256,
                    steps,
                    started_at,
                    policy_decision.disposition,
                    policy_decision.rule_id,
                )

            tool_result = self._execute_tool(task, decision, step_index)
            tool_calls_so_far += 1
            steps.append(
                AgentStep(
                    step_index=step_index,
                    prompt_sha256=decision_context.prompt_sha256,
                    context_sha256=memory_context.sha256,
                    decision=decision,
                    policy_decision=policy_decision,
                    approval_decision=approval,
                    tool_result=tool_result,
                    started_at=step_started_at,
                    finished_at=_utc_now(),
                )
            )
            if (
                tool_result.status == ToolStatus.FAILED
                and configuration.stop_on_tool_error
            ):
                error = RuntimeError(tool_result.error_message or "tool call failed")
                return self._failed_result(
                    task,
                    configuration,
                    retrieval,
                    memory_context.sha256,
                    steps,
                    started_at,
                    AgentTerminationReason.TOOL_FAILURE,
                    error,
                    failure_type=tool_result.error_type,
                )

        return self._result(
            task,
            configuration,
            retrieval,
            memory_context.sha256,
            steps,
            started_at,
            status=AgentRunStatus.BLOCKED,
            termination_reason=AgentTerminationReason.STEP_LIMIT,
        )

    def _observe_decision_evidence(self, task_id: str, step_index: int) -> None:
        """Forward evaluator-only action-decider diagnostics when available.

        The protocol deliberately does not require a diagnostic API, so custom
        deciders remain valid.  OllamaActionDecider exposes ``last_trace``;
        evidence is accepted only when it belongs to this exact task and step.
        """

        observer = self._decision_evidence_observer
        trace = getattr(self._decider, "last_trace", None)
        if observer is None or trace is None:
            return
        if (
            getattr(trace, "task_id", None) != task_id
            or getattr(trace, "step_index", None) != step_index
        ):
            return
        completion_evidence = getattr(trace, "completion_evidence", {})
        field_presence = getattr(trace, "field_presence", {})
        if not isinstance(completion_evidence, Mapping) or not isinstance(
            field_presence, Mapping
        ):
            return
        observer(
            {
                "schema_version": 1,
                "task_id": task_id,
                "step_index": step_index,
                "prompt_sha256": getattr(trace, "prompt_sha256", None),
                "context_sha256": getattr(trace, "context_sha256", None),
                "schema_sha256": getattr(trace, "schema_sha256", None),
                "provider": getattr(trace, "provider", None),
                "model": getattr(trace, "model", None),
                "model_digest": getattr(trace, "model_digest", None),
                "finish_reason": getattr(trace, "finish_reason", None),
                "prompt_tokens": getattr(trace, "prompt_tokens", None),
                "completion_tokens": getattr(trace, "completion_tokens", None),
                "latency_seconds": getattr(trace, "latency_seconds", None),
                "parse_outcome": getattr(trace, "parse_outcome", None),
                "validation_error": getattr(trace, "validation_error", None),
                "field_presence": dict(field_presence),
                "completion_evidence": dict(completion_evidence),
            }
        )

    def _available_tools(
        self,
        configuration: AgentRunConfiguration,
    ) -> tuple[ToolDefinition, ...]:
        definitions = self._tools.definitions()
        by_name = {tool.name: tool for tool in definitions}
        if len(by_name) != len(definitions):
            raise ValueError("tool executor returned duplicate definitions")
        missing = [name for name in configuration.allowed_tools if name not in by_name]
        if missing:
            raise ValueError(
                "allowed tools are not registered: " + ", ".join(sorted(missing))
            )
        return tuple(by_name[name] for name in configuration.allowed_tools)

    def _execute_tool(
        self,
        task: AgentTask,
        decision: ToolActionDecision,
        step_index: int,
    ) -> ToolResult:
        started = time.perf_counter()
        call = decision.call
        try:
            result = self._tools.execute(
                call,
                AgentToolExecutionContext(
                    task_id=task.task_id,
                    scope=task.scope,
                    step_index=step_index,
                    data_classification=task.data_classification,
                ),
            )
            if result.call_id != call.call_id or result.tool_name != call.tool_name:
                raise ValueError("tool result does not match the authorized call")
            return result
        except Exception as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolStatus.FAILED,
                latency_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
                error_message=str(exc) or "tool execution failed without a message",
            )

    @classmethod
    def _blocked_result(
        cls,
        task: AgentTask,
        configuration: AgentRunConfiguration,
        retrieval: RetrievalResult,
        context_sha256: str,
        steps: list[AgentStep],
        started_at: datetime,
        disposition: PolicyDisposition,
        rule_id: str,
    ) -> AgentRunResult:
        if disposition == PolicyDisposition.REQUIRE_APPROVAL:
            reason = AgentTerminationReason.APPROVAL_REQUIRED
        elif rule_id == "tool_call_limit_exceeded":
            reason = AgentTerminationReason.TOOL_CALL_LIMIT
        elif rule_id == "step_limit_exceeded":
            reason = AgentTerminationReason.STEP_LIMIT
        else:
            reason = AgentTerminationReason.POLICY_DENIED
        return cls._result(
            task,
            configuration,
            retrieval,
            context_sha256,
            steps,
            started_at,
            status=AgentRunStatus.BLOCKED,
            termination_reason=reason,
        )

    @staticmethod
    def _failed_result(
        task: AgentTask,
        configuration: AgentRunConfiguration,
        retrieval: RetrievalResult,
        context_sha256: str,
        steps: list[AgentStep],
        started_at: datetime,
        reason: AgentTerminationReason,
        error: Exception,
        *,
        failure_type: str | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            task=task,
            configuration=configuration,
            retrieval_result=retrieval,
            context_sha256=context_sha256,
            status=AgentRunStatus.FAILED,
            termination_reason=reason,
            steps=tuple(steps),
            failure_type=failure_type or type(error).__name__,
            failure_message=str(error) or "agent execution failed without a message",
            started_at=started_at,
            finished_at=_utc_now(),
        )

    @staticmethod
    def _result(
        task: AgentTask,
        configuration: AgentRunConfiguration,
        retrieval: RetrievalResult,
        context_sha256: str,
        steps: list[AgentStep],
        started_at: datetime,
        *,
        status: AgentRunStatus,
        termination_reason: AgentTerminationReason,
        final_answer: str | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            task=task,
            configuration=configuration,
            retrieval_result=retrieval,
            context_sha256=context_sha256,
            status=status,
            termination_reason=termination_reason,
            steps=tuple(steps),
            final_answer=final_answer,
            started_at=started_at,
            finished_at=_utc_now(),
        )
