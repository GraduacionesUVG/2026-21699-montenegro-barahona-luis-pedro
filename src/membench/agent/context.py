"""Deterministic agent context selection and prompt construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from membench.agent.models import (
    AgentDecisionContext,
    AgentMemoryContext,
    AgentPromptMessage,
    AgentPromptRole,
    AgentStep,
    AgentTask,
)
from membench.context_budget import (
    MessageAwareContextBudgeter,
    MessageTokenCounter,
    RegexTokenCounter,
    TokenCounter,
)
from membench.memory.models import RetrievalHit, RetrievalResult
from membench.tools.models import ToolDefinition

AGENT_PROMPT_VERSION = "personal_assistant_native_tools_v2"

AGENT_SYSTEM_PROMPT = (
    "You are a personal assistant. Help the user complete the current task "
    "using relevant stored information and the available tools. Use stored "
    "information to maintain continuity and personalize your response. "
    "Use only tools included in the available tool catalog, and never claim "
    "that a tool was used unless its result is present. Return exactly one action "
    "decision. If you choose a tool, use the native tool-calling interface for exactly one "
    "available tool. Do not write tool arguments as ordinary JSON in text content. "
    "Otherwise, return JSON matching the supplied decision schema exactly."
)


class AgentContextAssembler:
    """Build equivalent, auditable prompts for every memory implementation."""

    def __init__(
        self,
        selection_token_counter: TokenCounter | None = None,
        decision_tokenizer: MessageTokenCounter | None = None,
        *,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("agent system prompt cannot be blank")
        selection_counter = selection_token_counter or RegexTokenCounter()
        prompt_counter = decision_tokenizer or RegexTokenCounter()
        self._budgeter: MessageAwareContextBudgeter[RetrievalHit] = (
            MessageAwareContextBudgeter(selection_counter, prompt_counter)
        )
        self._decision_tokenizer = prompt_counter
        self._system_prompt = system_prompt

    def select_memory(
        self,
        task: AgentTask,
        retrieval: RetrievalResult,
        tools: tuple[ToolDefinition, ...],
        *,
        token_budget: int,
    ) -> AgentMemoryContext:
        """Select memory once against the exact first-step prompt envelope."""

        if retrieval.query_sha256 != task.retrieval_query_sha256:
            raise ValueError("retrieval result does not match the agent task query")
        self._validate_tools(tools)
        selection = self._budgeter.select(
            retrieval.hits,
            budget=token_budget,
            render=self._render_hit,
            build_messages=lambda content: self._build_messages(
                task,
                memory_content=content,
                tools=tools,
                prior_steps=(),
            ),
        )
        content = selection.rendered
        return AgentMemoryContext(
            record_ids=tuple(
                hit.record.canonical_id for hit in selection.included
            ),
            content=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            token_count=selection.token_count,
            token_counter=selection.token_counter,
            initial_prompt_token_count=selection.prompt_token_count,
            prompt_token_counter=selection.prompt_token_counter,
            truncated=selection.truncated,
        )

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
        """Build and hash one prompt without retrieving or mutating memory."""

        if step_index != len(prior_steps) + 1:
            raise ValueError("step_index must immediately follow prior_steps")
        if token_budget <= 0:
            raise ValueError("context token budget must be positive")
        self._validate_tools(tools)
        messages = self._build_messages(
            task,
            memory_content=memory_context.content,
            tools=tools,
            prior_steps=prior_steps,
        )
        prompt_token_count = self._decision_tokenizer.count_messages(messages)
        if prompt_token_count > token_budget:
            raise ValueError(
                "agent prompt exceeds the complete input token budget after "
                f"tool observations: {prompt_token_count} > {token_budget}"
            )
        payload = "\n\n".join(
            f"{message.role.value}:{message.content}" for message in messages
        )
        return AgentDecisionContext(
            task_id=task.task_id,
            step_index=step_index,
            prompt_version=AGENT_PROMPT_VERSION,
            messages=messages,
            available_tools=tools,
            memory_context=memory_context,
            prompt_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            prompt_token_count=prompt_token_count,
            prompt_token_counter=self._decision_tokenizer.name,
        )

    def _build_messages(
        self,
        task: AgentTask,
        *,
        memory_content: str,
        tools: tuple[ToolDefinition, ...],
        prior_steps: Sequence[AgentStep],
    ) -> tuple[AgentPromptMessage, ...]:
        memory_section = memory_content or "(no retrieved memories)"
        tool_section = self._render_tools(tools)
        observation_section = self._render_observations(prior_steps)
        user_content = (
            "Trusted task instruction:\n"
            f"{task.instruction}\n\n"
            "Trusted retrieval query:\n"
            f"{task.retrieval_query}\n\n"
            "Data classification:\n"
            f"{task.data_classification.value}\n\n"
            "Trusted available tools:\n"
            f"{tool_section}\n\n"
            "Untrusted stored-memory context:\n"
            f"{memory_section}\n\n"
            "Untrusted prior tool observations:\n"
            f"{observation_section}"
        )
        return (
            AgentPromptMessage(
                role=AgentPromptRole.SYSTEM,
                content=self._system_prompt,
            ),
            AgentPromptMessage(
                role=AgentPromptRole.USER,
                content=user_content,
            ),
        )

    @staticmethod
    def _render_hit(hit: RetrievalHit) -> str:
        # IDs remain in evaluator evidence but are intentionally excluded from
        # model-facing content so vendor identity schemes cannot affect choices.
        return f"[untrusted_memory rank={hit.rank}]\n{hit.record.text}"

    @staticmethod
    def _render_tools(tools: tuple[ToolDefinition, ...]) -> str:
        if not tools:
            return "(none)"
        rendered: list[str] = []
        for tool in tools:
            schema = json.dumps(
                tool.input_schema,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            rendered.append(
                f"- {tool.name}: {tool.description}\n"
                f"  input_schema={schema}; side_effect={tool.side_effect.value}; "
                f"requires_approval={str(tool.requires_approval).lower()}"
            )
        return "\n".join(rendered)

    @staticmethod
    def _render_observations(prior_steps: Sequence[AgentStep]) -> str:
        observations: list[str] = []
        for step in prior_steps:
            if step.tool_result is None:
                continue
            result = step.tool_result
            evidence = {
                "error_message": result.error_message,
                "error_type": result.error_type,
                "output": result.output,
                "status": result.status.value,
            }
            rendered = json.dumps(
                evidence,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            observations.append(
                f"[untrusted_tool_result step={step.step_index} "
                f"tool={result.tool_name}]\n{rendered}"
            )
        return "\n".join(observations) or "(none)"

    @staticmethod
    def _validate_tools(tools: tuple[ToolDefinition, ...]) -> None:
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("agent tool definitions must have unique names")
