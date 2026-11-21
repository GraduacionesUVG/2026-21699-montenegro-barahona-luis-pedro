"""Safe bridge from agent inbox reads to external-memory ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from membench.ingress.bridge import ExternalIngestionBridge
from membench.ingress.models import (
    ExternalDelivery,
    ExternalIngestionResult,
    IngressMode,
)
from membench.memory.models import CanonicalModel, MemoryScope
from membench.tools.models import ToolResult, ToolStatus


class AgentSelectedIngressOutcome(StrEnum):
    """One safe disposition of an observed ``read_message`` result."""

    INGESTED = "ingested"
    DUPLICATE_READ = "duplicate_read"
    READ_FAILED = "read_failed"
    TARGET_NOT_FOUND = "target_not_found"
    WRONG_MESSAGE = "wrong_message"
    READ_PAYLOAD_MISMATCH = "read_payload_mismatch"


class AgentSelectedIngestionAttempt(CanonicalModel):
    """Auditable decision made after one agent ``read_message`` result."""

    delivery_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    read_result: ToolResult
    outcome: AgentSelectedIngressOutcome
    observed_message_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9_.-]*$"
    )
    observed_content_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    ingestion: ExternalIngestionResult | None = None

    @model_validator(mode="after")
    def outcome_must_match_ingestion(self) -> Self:
        if self.read_result.tool_name != "read_message":
            raise ValueError("agent-selected ingress requires a read_message result")
        if self.outcome == AgentSelectedIngressOutcome.INGESTED:
            if self.ingestion is None:
                raise ValueError("an ingested attempt requires ingestion evidence")
            if not self.ingestion.selected_by_agent:
                raise ValueError("agent-selected ingestion must be selected by agent")
        elif self.ingestion is not None:
            raise ValueError("non-ingested attempts cannot contain ingestion evidence")
        if (
            self.outcome == AgentSelectedIngressOutcome.READ_FAILED
            and self.read_result.status != ToolStatus.FAILED
        ):
            raise ValueError("read_failed requires a failed read_message result")
        if (
            self.outcome != AgentSelectedIngressOutcome.READ_FAILED
            and self.read_result.status != ToolStatus.SUCCEEDED
        ):
            raise ValueError("successful read evidence is required for this outcome")
        return self


class AgentSelectedIngressExecutor:
    """Permit exactly one ingestion only after a verified target-email read.

    This executor is deliberately stateful and scoped to one trial.  It does
    not trust content returned in the agent-visible tool result: the result is
    used solely to prove that the exact frozen email was read, then the frozen
    :class:`ExternalDelivery` is passed to the common ingestion bridge.
    """

    def __init__(
        self,
        bridge: ExternalIngestionBridge,
        delivery: ExternalDelivery,
    ) -> None:
        self._bridge = bridge
        self._delivery = ExternalDelivery.model_validate(
            delivery.model_dump(mode="json")
        )
        self._ingested = False

    @property
    def ingested(self) -> bool:
        """Whether this trial has already submitted its one permitted write."""

        return self._ingested

    def observe_read(
        self,
        read_result: ToolResult,
        *,
        scope: MemoryScope,
        operation_id: str,
        native_options: Mapping[str, object] | None = None,
    ) -> AgentSelectedIngestionAttempt:
        """Validate one read result and, at most once, dispatch frozen content."""

        result = ToolResult.model_validate(read_result.model_dump(mode="json"))
        if result.tool_name != "read_message":
            raise ValueError("agent-selected ingress accepts read_message results only")
        if result.status == ToolStatus.FAILED:
            return AgentSelectedIngestionAttempt(
                delivery_id=self._delivery.delivery_id,
                read_result=result,
                outcome=AgentSelectedIngressOutcome.READ_FAILED,
            )

        message = result.output.get("message")
        if result.output.get("found") is not True or not isinstance(message, Mapping):
            return AgentSelectedIngestionAttempt(
                delivery_id=self._delivery.delivery_id,
                read_result=result,
                outcome=AgentSelectedIngressOutcome.TARGET_NOT_FOUND,
            )
        message_id = message.get("message_id")
        content_sha256 = message.get("content_sha256")
        if not isinstance(message_id, str) or message_id != self._delivery.delivery_id:
            return AgentSelectedIngestionAttempt(
                delivery_id=self._delivery.delivery_id,
                read_result=result,
                outcome=AgentSelectedIngressOutcome.WRONG_MESSAGE,
                observed_message_id=message_id if isinstance(message_id, str) else None,
                observed_content_sha256=(
                    content_sha256 if isinstance(content_sha256, str) else None
                ),
            )
        if (
            not isinstance(content_sha256, str)
            or content_sha256 != self._delivery.content_sha256
            or message.get("content") != self._delivery.content
        ):
            return AgentSelectedIngestionAttempt(
                delivery_id=self._delivery.delivery_id,
                read_result=result,
                outcome=AgentSelectedIngressOutcome.READ_PAYLOAD_MISMATCH,
                observed_message_id=message_id,
                observed_content_sha256=(
                    content_sha256 if isinstance(content_sha256, str) else None
                ),
            )
        if self._ingested:
            return AgentSelectedIngestionAttempt(
                delivery_id=self._delivery.delivery_id,
                read_result=result,
                outcome=AgentSelectedIngressOutcome.DUPLICATE_READ,
                observed_message_id=message_id,
                observed_content_sha256=content_sha256,
            )

        ingestion = self._bridge.ingest(
            self._delivery,
            scope=scope,
            operation_id=operation_id,
            mode=IngressMode.AGENT_SELECTED,
            native_options=deepcopy(dict(native_options or {})),
        )
        self._ingested = True
        return AgentSelectedIngestionAttempt(
            delivery_id=self._delivery.delivery_id,
            read_result=result,
            outcome=AgentSelectedIngressOutcome.INGESTED,
            observed_message_id=message_id,
            observed_content_sha256=content_sha256,
            ingestion=ingestion,
        )
