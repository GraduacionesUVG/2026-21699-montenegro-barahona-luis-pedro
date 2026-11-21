"""Canonical tool definitions, calls, and results for bounded agents."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator, model_validator

from membench.memory.models import CanonicalModel, JsonObject


class ToolSideEffectKind(StrEnum):
    """Maximum side effect a tool may produce."""

    READ_ONLY = "read_only"
    SIMULATED_WRITE = "simulated_write"
    EXTERNAL_WRITE = "external_write"


class ToolStatus(StrEnum):
    """Outcome of an executed tool call."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ToolDefinition(CanonicalModel):
    """Trusted description of one allowlistable agent tool."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    input_schema: JsonObject = Field(default_factory=dict)
    side_effect: ToolSideEffectKind = ToolSideEffectKind.READ_ONLY
    requires_approval: bool = False


class ToolCall(CanonicalModel):
    """One model-proposed invocation before policy evaluation."""

    call_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    arguments: JsonObject = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1)


class ToolResult(CanonicalModel):
    """Canonical observation produced after an allowed tool call."""

    call_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    status: ToolStatus
    output: JsonObject = Field(default_factory=dict)
    latency_seconds: float = Field(default=0.0, ge=0)
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("error_type", "error_message")
    @classmethod
    def optional_error_strings_must_not_be_blank(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("tool error strings must be null or non-blank")
        return value

    @model_validator(mode="after")
    def status_must_match_error_fields(self) -> Self:
        errors = (self.error_type, self.error_message)
        if self.status == ToolStatus.SUCCEEDED and any(
            value is not None for value in errors
        ):
            raise ValueError("successful tool results cannot contain errors")
        if self.status == ToolStatus.FAILED and any(value is None for value in errors):
            raise ValueError("failed tool results require complete error information")
        return self
