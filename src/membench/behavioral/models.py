"""Canonical inputs for frozen behavior-following experiments.

The task template is visible to the public agent once instantiated for a
run.  The paired oracle deliberately remains evaluator-only: it supplies the
trusted and attacker-controlled destinations used to interpret an outbox
without leaking either answer to the model prompt.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.agent.models import AgentDataClassification, AgentTask
from membench.campaign.models import TrialTwin
from membench.memory.models import CanonicalModel, JsonObject, MemoryScope
from membench.tools.mock_contacts import normalize_synthetic_recipient

_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_PERSONA_ID_PATTERN = r"^[0-9]+$"


class BehavioralTaskTemplate(CanonicalModel):
    """Public, immutable task content instantiated separately for each run.

    ``scope`` and ``task_id`` are intentionally omitted: campaign preparation
    derives them from the isolated run so all twins receive the same task text
    without sharing a memory namespace.
    """

    schema_version: Literal[1] = 1
    behavioral_task_id: str = Field(pattern=_ID_PATTERN)
    attack_spec_id: str = Field(pattern=_ID_PATTERN)
    delivery_family_id: str = Field(pattern=_ID_PATTERN)
    persona_id: str = Field(pattern=_PERSONA_ID_PATTERN)
    instruction: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    data_classification: AgentDataClassification = AgentDataClassification.INTERNAL
    metadata: JsonObject = Field(default_factory=dict)

    def instantiate(self, scope: MemoryScope, *, task_id: str) -> AgentTask:
        """Bind this frozen content to one isolated public-agent run."""

        return AgentTask(
            task_id=task_id,
            scope=scope,
            instruction=self.instruction,
            retrieval_query=self.retrieval_query,
            data_classification=self.data_classification,
            metadata={
                **self.metadata,
                "behavioral_task_id": self.behavioral_task_id,
                "behavioral_task_schema_version": self.schema_version,
            },
        )


class CaseSummaryBehavioralOracle(CanonicalModel):
    """Evaluator-only answer key for a case-summary sending task.

    Both destinations must be non-routable synthetic recipients.  This allows
    a poison-directed send to be executed and measured safely, rather than
    being stopped by the safety layer before the memory behavior is observed.
    """

    schema_version: Literal[1] = 1
    oracle_id: str = Field(pattern=_ID_PATTERN)
    behavioral_task_id: str = Field(pattern=_ID_PATTERN)
    attack_spec_id: str = Field(pattern=_ID_PATTERN)
    delivery_family_id: str = Field(pattern=_ID_PATTERN)
    persona_id: str = Field(pattern=_PERSONA_ID_PATTERN)
    expected_tool: Literal["simulate_send"] = "simulate_send"
    trusted_recipient: str = Field(min_length=3, max_length=320)
    attacker_recipient: str = Field(min_length=3, max_length=320)
    maximum_messages: int = Field(default=1, ge=1, le=10)

    @field_validator("trusted_recipient", "attacker_recipient")
    @classmethod
    def recipients_must_be_synthetic(cls, value: str) -> str:
        """Canonicalize the safe recipient spelling used by evaluation."""

        return normalize_synthetic_recipient(value)[1]

    @model_validator(mode="after")
    def recipients_must_be_distinct(self) -> Self:
        if self.trusted_recipient == self.attacker_recipient:
            raise ValueError("trusted and attacker recipients must differ")
        return self


class BehavioralOutcome(StrEnum):
    """Deterministic interpretation of one behavioral-task outbox."""

    NO_SEND = "no_send"
    TRUSTED_RECIPIENT = "trusted_recipient"
    ATTACKER_RECIPIENT = "attacker_recipient"
    MIXED_RECIPIENTS = "mixed_recipients"
    UNEXPECTED_RECIPIENT = "unexpected_recipient"
    INVALID_EVIDENCE = "invalid_evidence"


class BehavioralSendEvidence(CanonicalModel):
    """Minimal cross-linked evidence for one simulated behavioral send."""

    message_id: str = Field(pattern=r"^mock-message-[0-9a-f]{24}$")
    call_id: str = Field(min_length=1)
    step_index: int = Field(ge=1)
    recipient: str = Field(min_length=3, max_length=320)

    @field_validator("recipient")
    @classmethod
    def recipient_must_be_synthetic(cls, value: str) -> str:
        return normalize_synthetic_recipient(value)[1]


class BehavioralScoreResult(CanonicalModel):
    """Evaluator-only, model-free classification of one behavioral run."""

    schema_version: Literal[1] = 1
    trial_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    twin: TrialTwin
    scope: MemoryScope
    behavioral_task_id: str = Field(pattern=_ID_PATTERN)
    oracle_id: str = Field(pattern=_ID_PATTERN)
    agent_task_id: str = Field(min_length=1)
    sends: tuple[BehavioralSendEvidence, ...] = Field(default_factory=tuple)
    outcome: BehavioralOutcome
    evidence_errors: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def outcome_must_match_evidence(self) -> Self:
        if self.evidence_errors and self.outcome != BehavioralOutcome.INVALID_EVIDENCE:
            raise ValueError("behavioral evidence errors require invalid_evidence")
        if (
            not self.evidence_errors
            and self.outcome == BehavioralOutcome.INVALID_EVIDENCE
        ):
            raise ValueError("invalid_evidence requires at least one evidence error")
        message_ids = tuple(send.message_id for send in self.sends)
        call_ids = tuple(send.call_id for send in self.sends)
        if len(message_ids) != len(set(message_ids)) or len(call_ids) != len(
            set(call_ids)
        ):
            raise ValueError(
                "behavioral send evidence must not reuse messages or calls"
            )
        return self
