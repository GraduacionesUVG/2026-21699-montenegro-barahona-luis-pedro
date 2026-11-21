"""Canonical inputs, decisions, steps, and results for bounded agent runs."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.memory.models import (
    CanonicalModel,
    JsonObject,
    MemoryScope,
    RetrievalResult,
)
from membench.tools.models import ToolCall, ToolDefinition, ToolResult


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AgentDataClassification(StrEnum):
    """Synthetic information sensitivity used by the action policy."""

    PUBLIC = "public"
    INTERNAL = "internal"
    SYNTHETIC_SENSITIVE = "synthetic_sensitive"


class AgentRunStatus(StrEnum):
    """Terminal status of one bounded agent task."""

    COMPLETED = "completed"
    REFUSED = "refused"
    BLOCKED = "blocked"
    FAILED = "failed"


class AgentTerminationReason(StrEnum):
    """Stable reason the bounded execution loop stopped."""

    FINAL_ANSWER = "final_answer"
    REFUSED = "refused"
    APPROVAL_REQUIRED = "approval_required"
    POLICY_DENIED = "policy_denied"
    STEP_LIMIT = "step_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    CONTEXT_FAILURE = "context_failure"
    MODEL_FAILURE = "model_failure"
    TOOL_FAILURE = "tool_failure"


class PolicyDisposition(StrEnum):
    """Outcome of evaluating one proposed tool action."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class AgentPromptRole(StrEnum):
    """Roles accepted by the mechanism-independent decision prompt."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class AgentPromptMessage(CanonicalModel):
    """One exact message supplied to the action decider."""

    role: AgentPromptRole
    content: str = Field(min_length=1)


class AgentTask(CanonicalModel):
    """One evaluator-supplied task executed within a fixed memory scope."""

    task_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    scope: MemoryScope
    instruction: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    data_classification: AgentDataClassification = AgentDataClassification.INTERNAL
    metadata: JsonObject = Field(default_factory=dict)

    @property
    def retrieval_query_sha256(self) -> str:
        """Hash the exact retrieval query for evidence matching."""

        return hashlib.sha256(self.retrieval_query.encode("utf-8")).hexdigest()


class AgentRunConfiguration(CanonicalModel):
    """Trusted limits and permissions for one bounded agent run."""

    max_steps: int = Field(default=5, ge=1)
    max_tool_calls: int = Field(default=3, ge=0)
    retrieval_top_k: int = Field(default=10, gt=0)
    retrieval_threshold: float | None = None
    allowed_tools: tuple[str, ...] = Field(default_factory=tuple)
    approval_required_tools: tuple[str, ...] = Field(default_factory=tuple)
    context_token_budget: int = Field(default=2048, gt=0)
    memory_read_only: bool = True
    allow_external_side_effects: bool = False
    stop_on_tool_error: bool = True

    @field_validator("allowed_tools", "approval_required_tools")
    @classmethod
    def tool_names_must_be_unique_and_well_formed(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not item or not item.replace("_", "a").isalnum() for item in value):
            raise ValueError("tool names must contain only letters, numbers, and '_'")
        if any(not item[0].isalpha() or not item[0].islower() for item in value):
            raise ValueError("tool names must start with a lowercase letter")
        if len(value) != len(set(value)):
            raise ValueError("tool names must be unique")
        return value

    @model_validator(mode="after")
    def limits_and_approval_tools_must_be_consistent(self) -> Self:
        if self.max_tool_calls > self.max_steps:
            raise ValueError("max_tool_calls cannot exceed max_steps")
        unknown = set(self.approval_required_tools) - set(self.allowed_tools)
        if unknown:
            raise ValueError(
                "approval-required tools must also be allowed: "
                + ", ".join(sorted(unknown))
            )
        return self


class ToolActionDecision(CanonicalModel):
    """Model decision proposing one tool invocation."""

    kind: Literal["tool_action"] = "tool_action"
    call: ToolCall
    decision_summary: str = Field(min_length=1)


class FinalAnswerDecision(CanonicalModel):
    """Model decision to stop with a user-facing answer."""

    kind: Literal["final_answer"] = "final_answer"
    content: str = Field(min_length=1)


class RefusalDecision(CanonicalModel):
    """Model decision to stop because the task should not be completed."""

    kind: Literal["refusal"] = "refusal"
    reason: str = Field(min_length=1)


AgentDecision = Annotated[
    ToolActionDecision | FinalAnswerDecision | RefusalDecision,
    Field(discriminator="kind"),
]


class AgentMemoryContext(CanonicalModel):
    """Frozen ranked-prefix memory context selected from one retrieval."""

    record_ids: tuple[str, ...] = Field(default_factory=tuple)
    content: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(ge=0)
    token_counter: str = Field(min_length=1)
    initial_prompt_token_count: int = Field(ge=0)
    prompt_token_counter: str = Field(min_length=1)
    truncated: bool = False

    @field_validator("record_ids")
    @classmethod
    def record_ids_must_be_non_blank(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("memory-context record IDs cannot be blank")
        return value

    @model_validator(mode="after")
    def content_hash_must_match(self) -> Self:
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise ValueError("memory-context hash does not match its content")
        return self


class AgentDecisionContext(CanonicalModel):
    """Exact immutable input supplied to one action-decider invocation."""

    task_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    step_index: int = Field(ge=1)
    prompt_version: str = Field(min_length=1)
    messages: tuple[AgentPromptMessage, ...] = Field(min_length=1)
    available_tools: tuple[ToolDefinition, ...] = Field(default_factory=tuple)
    memory_context: AgentMemoryContext
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_token_count: int = Field(gt=0)
    prompt_token_counter: str = Field(min_length=1)

    @field_validator("available_tools")
    @classmethod
    def available_tool_names_must_be_unique(
        cls,
        value: tuple[ToolDefinition, ...],
    ) -> tuple[ToolDefinition, ...]:
        names = [tool.name for tool in value]
        if len(names) != len(set(names)):
            raise ValueError("available tool names must be unique")
        return value

    @model_validator(mode="after")
    def prompt_hash_must_match(self) -> Self:
        payload = "\n\n".join(
            f"{message.role.value}:{message.content}" for message in self.messages
        )
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self.prompt_sha256 != expected:
            raise ValueError("agent prompt hash does not match its messages")
        return self


class AgentToolExecutionContext(CanonicalModel):
    """Trusted evaluator context passed to an allowed tool implementation."""

    task_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    scope: MemoryScope
    step_index: int = Field(ge=1)
    data_classification: AgentDataClassification


class ApprovalDecision(CanonicalModel):
    """Trusted approval or rejection tied to exactly one proposed call."""

    approval_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    call_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    approved: bool
    reason: str = Field(min_length=1)
    decided_at: datetime = Field(default_factory=_utc_now)

    @field_validator("decided_at")
    @classmethod
    def decided_at_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("approval timestamps must include a timezone")
        return value


class PolicyDecision(CanonicalModel):
    """Auditable policy outcome for one model-proposed tool call."""

    call_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    disposition: PolicyDisposition
    rule_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    reason: str = Field(min_length=1)
    approval_id: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$",
    )


class AgentStep(CanonicalModel):
    """Complete evidence for one observe-decide-act loop iteration."""

    step_index: int = Field(ge=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: AgentDecision
    policy_decision: PolicyDecision | None = None
    approval_decision: ApprovalDecision | None = None
    tool_result: ToolResult | None = None
    started_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime = Field(default_factory=_utc_now)

    @field_validator("started_at", "finished_at")
    @classmethod
    def step_timestamps_must_have_timezones(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("agent step timestamps must include timezones")
        return value

    @model_validator(mode="after")
    def evidence_must_match_decision(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("agent step cannot finish before it starts")

        if not isinstance(self.decision, ToolActionDecision):
            if (
                self.policy_decision is not None
                or self.approval_decision is not None
                or self.tool_result is not None
            ):
                raise ValueError(
                    "final and refusal decisions cannot contain tool evidence"
                )
            return self

        call = self.decision.call
        if self.policy_decision is None:
            raise ValueError("tool actions require a policy decision")
        if (
            self.policy_decision.call_id != call.call_id
            or self.policy_decision.tool_name != call.tool_name
        ):
            raise ValueError("policy evidence must reference the proposed tool call")

        if self.approval_decision is not None:
            if self.approval_decision.call_id != call.call_id:
                raise ValueError(
                    "approval evidence must reference the proposed tool call"
                )
            if self.policy_decision.approval_id != self.approval_decision.approval_id:
                raise ValueError(
                    "policy evidence must reference the recorded approval decision"
                )
        elif self.policy_decision.approval_id is not None:
            raise ValueError("policy approval ID requires approval evidence")

        allowed = self.policy_decision.disposition == PolicyDisposition.ALLOW
        if (
            self.approval_decision is not None
            and not self.approval_decision.approved
            and allowed
        ):
            raise ValueError("a rejected approval cannot allow tool execution")
        if allowed and self.tool_result is None:
            raise ValueError("allowed tool actions require a tool result")
        if not allowed and self.tool_result is not None:
            raise ValueError("non-allowed tool actions cannot contain a tool result")
        if self.tool_result is not None and (
            self.tool_result.call_id != call.call_id
            or self.tool_result.tool_name != call.tool_name
        ):
            raise ValueError("tool results must reference the proposed tool call")
        return self


class AgentRunResult(CanonicalModel):
    """Terminal evidence for one bounded agent task."""

    task: AgentTask
    configuration: AgentRunConfiguration
    retrieval_result: RetrievalResult
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: AgentRunStatus
    termination_reason: AgentTerminationReason
    steps: tuple[AgentStep, ...] = Field(default_factory=tuple)
    final_answer: str | None = Field(default=None, min_length=1)
    failure_type: str | None = Field(default=None, min_length=1)
    failure_message: str | None = Field(default=None, min_length=1)
    started_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime = Field(default_factory=_utc_now)

    @field_validator("started_at", "finished_at")
    @classmethod
    def run_timestamps_must_have_timezones(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("agent run timestamps must include timezones")
        return value

    @model_validator(mode="after")
    def result_evidence_must_be_consistent(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("agent run cannot finish before it starts")
        if self.retrieval_result.query_sha256 != self.task.retrieval_query_sha256:
            raise ValueError("retrieval result does not match the task query")

        indexes = tuple(step.step_index for step in self.steps)
        if indexes != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("agent step indexes must be consecutive and one-based")
        if len(self.steps) > self.configuration.max_steps:
            raise ValueError("agent steps exceed the configured maximum")
        tool_calls = sum(step.tool_result is not None for step in self.steps)
        if tool_calls > self.configuration.max_tool_calls:
            raise ValueError("agent tool calls exceed the configured maximum")

        expected_status = {
            AgentTerminationReason.FINAL_ANSWER: AgentRunStatus.COMPLETED,
            AgentTerminationReason.REFUSED: AgentRunStatus.REFUSED,
            AgentTerminationReason.APPROVAL_REQUIRED: AgentRunStatus.BLOCKED,
            AgentTerminationReason.POLICY_DENIED: AgentRunStatus.BLOCKED,
            AgentTerminationReason.STEP_LIMIT: AgentRunStatus.BLOCKED,
            AgentTerminationReason.TOOL_CALL_LIMIT: AgentRunStatus.BLOCKED,
            AgentTerminationReason.CONTEXT_FAILURE: AgentRunStatus.FAILED,
            AgentTerminationReason.MODEL_FAILURE: AgentRunStatus.FAILED,
            AgentTerminationReason.TOOL_FAILURE: AgentRunStatus.FAILED,
        }[self.termination_reason]
        if self.status != expected_status:
            raise ValueError("agent status does not match its termination reason")
        if self.status == AgentRunStatus.COMPLETED and self.final_answer is None:
            raise ValueError("completed agent runs require a final answer")
        if self.status != AgentRunStatus.COMPLETED and self.final_answer is not None:
            raise ValueError("only completed agent runs may contain a final answer")
        failures = (self.failure_type, self.failure_message)
        if self.status == AgentRunStatus.FAILED and any(
            value is None for value in failures
        ):
            raise ValueError("failed agent runs require complete failure information")
        if self.status != AgentRunStatus.FAILED and any(
            value is not None for value in failures
        ):
            raise ValueError("only failed agent runs may contain failure information")
        return self
