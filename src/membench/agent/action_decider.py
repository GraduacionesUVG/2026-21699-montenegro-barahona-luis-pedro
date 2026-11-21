"""Provider-neutral structured action selection for the bounded agent."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pydantic import ValidationError

from membench.agent.models import (
    AgentDecision,
    AgentDecisionContext,
    AgentPromptMessage,
    FinalAnswerDecision,
    RefusalDecision,
    ToolActionDecision,
)
from membench.memory.models import CanonicalModel
from membench.ollama import OllamaError, OllamaService
from membench.tools.models import ToolCall, ToolDefinition


class ActionDeciderError(RuntimeError):
    """Base error for action-decider configuration or generation failures."""


class ActionDecisionGenerationError(ActionDeciderError):
    """Raised when a provider cannot produce a complete decision response."""


class ActionDecisionValidationError(ActionDeciderError):
    """Raised when provider output does not match the canonical decision schema."""


@dataclass(frozen=True)
class ActionDeciderSettings:
    """Resolved Ollama model and generation settings for structured decisions."""

    model: str
    model_digest: str | None
    base_url: str
    temperature: float
    seed: int
    max_output_tokens: int
    timeout_seconds: float
    think: bool = False
    capture_raw_response: bool = False

    @classmethod
    def from_env(cls) -> ActionDeciderSettings:
        """Resolve an Ollama agent model with responder-compatible fallbacks."""

        load_dotenv(override=False)
        explicit_model = os.getenv("AGENT_DECIDER_MODEL", "").strip()
        explicit_digest = os.getenv("AGENT_DECIDER_MODEL_DIGEST", "").strip()
        responder_is_ollama = (
            os.getenv("RESPONDER_PROVIDER", "ollama").strip().lower() == "ollama"
        )
        responder_model = (
            os.getenv("RESPONDER_MODEL", "").strip()
            if responder_is_ollama
            else ""
        )
        fallback_model = (
            responder_model
            or os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b").strip()
        )
        model = explicit_model or fallback_model
        fallback_digest = (
            (
                os.getenv("RESPONDER_MODEL_DIGEST", "").strip()
                if responder_is_ollama
                else ""
            )
            or os.getenv("OLLAMA_CHAT_MODEL_DIGEST", "").strip()
        )
        digest = explicit_digest or (
            fallback_digest if model == fallback_model else ""
        )
        base_url = os.getenv(
            "AGENT_DECIDER_OLLAMA_BASE_URL",
            os.getenv(
                "RESPONDER_OLLAMA_BASE_URL",
                os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ),
        )

        settings = cls(
            model=model.strip(),
            model_digest=digest or None,
            base_url=base_url.rstrip("/"),
            temperature=float(os.getenv("AGENT_DECIDER_TEMPERATURE", "0")),
            seed=int(os.getenv("AGENT_DECIDER_SEED", "42")),
            max_output_tokens=int(
                os.getenv("AGENT_DECIDER_MAX_OUTPUT_TOKENS", "512")
            ),
            timeout_seconds=float(
                os.getenv("AGENT_DECIDER_TIMEOUT_SECONDS", "180")
            ),
            think=_environment_bool("AGENT_DECIDER_THINK", default=False),
            capture_raw_response=_environment_bool(
                "AGENT_DECIDER_CAPTURE_RAW_RESPONSE",
                default=False,
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject incomplete Ollama model or generation configuration."""

        if not self.model.strip():
            raise ActionDeciderError("action-decider model must not be empty")
        if not self.base_url.strip():
            raise ActionDeciderError("action-decider base URL must not be empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError(
                "AGENT_DECIDER_TEMPERATURE must be between zero and two"
            )
        if self.max_output_tokens <= 0:
            raise ValueError("AGENT_DECIDER_MAX_OUTPUT_TOKENS must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("AGENT_DECIDER_TIMEOUT_SECONDS must be positive")

    def public_config(self) -> dict[str, object]:
        """Return non-secret settings suitable for experiment provenance."""

        return {
            "provider": "ollama",
            "model": self.model,
            "model_digest": self.model_digest,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "think": self.think,
            "capture_raw_response": self.capture_raw_response,
        }


class _ActionDecisionEnvelope(CanonicalModel):
    """Root object used because structured-output schemas require an object."""

    decision: AgentDecision


def action_decision_json_schema() -> dict[str, Any]:
    """Return the exact provider schema, requiring the union discriminator."""

    schema = _ActionDecisionEnvelope.model_json_schema()
    definitions = schema.get("$defs")
    if isinstance(definitions, dict):
        for name in (
            "ToolActionDecision",
            "FinalAnswerDecision",
            "RefusalDecision",
        ):
            definition = definitions.get(name)
            if not isinstance(definition, dict):
                continue
            required = definition.setdefault("required", [])
            if isinstance(required, list) and "kind" not in required:
                required.insert(0, "kind")
    return schema


@dataclass(frozen=True)
class ActionDecisionCompletion:
    """Provider response before canonical parsing and trace attachment."""

    content: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    raw_response: dict[str, Any]
    response_keys: tuple[str, ...] = ()
    message_keys: tuple[str, ...] = ()
    field_presence: dict[str, bool] = field(default_factory=dict)
    thinking_character_count: int = 0
    thinking_sha256: str | None = None
    tool_call_count: int = 0

    def diagnostic_evidence(self, *, include_raw_response: bool) -> dict[str, Any]:
        """Return evaluator-only provider-shape evidence.

        Deliberately never retains raw ``thinking``.  It is recorded only as
        presence, length, and digest so it cannot be mistaken for a decision.
        """

        evidence: dict[str, Any] = {
            "response_keys": list(self.response_keys),
            "message_keys": list(self.message_keys),
            "field_presence": dict(self.field_presence),
            "content_character_count": len(self.content or ""),
            "content_sha256": _text_sha256(self.content),
            "thinking_character_count": self.thinking_character_count,
            "thinking_sha256": self.thinking_sha256,
            "tool_call_count": self.tool_call_count,
        }
        if include_raw_response:
            evidence["raw_response"] = self.raw_response
        return evidence


@dataclass(frozen=True)
class ActionDecisionTrace:
    """Non-secret execution evidence for the most recent model decision."""

    task_id: str
    step_index: int
    prompt_sha256: str
    context_sha256: str
    schema_sha256: str
    provider: str
    model: str
    model_digest: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    latency_seconds: float
    parse_outcome: Literal["valid", "empty_content", "invalid_json", "schema_error"] = "valid"
    validation_error: str | None = None
    field_presence: dict[str, bool] = field(default_factory=dict)
    completion_evidence: dict[str, Any] = field(default_factory=dict)


class ActionDecisionBackend(Protocol):
    """Provider boundary for one structured decision completion."""

    def complete(
        self,
        messages: Sequence[AgentPromptMessage],
        schema: Mapping[str, Any],
        settings: ActionDeciderSettings,
        tools: Sequence[ToolDefinition],
    ) -> ActionDecisionCompletion:
        """Generate one JSON decision envelope constrained by ``schema``."""

        ...


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None,
    timeout: float,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=request_headers,
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ActionDecisionGenerationError(
            f"action-decider provider returned HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ActionDecisionGenerationError(
            f"action-decider provider is unavailable at {url}: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise ActionDecisionGenerationError(
            "action-decider provider returned a non-object JSON response"
        )
    return result


def _message_payload(
    messages: Sequence[AgentPromptMessage],
) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("at least one action-decider message is required")
    return [
        {"role": message.role.value, "content": message.content}
        for message in messages
    ]


class OllamaActionDecisionBackend:
    """Request schema-constrained decisions from Ollama's native chat API."""

    def complete(
        self,
        messages: Sequence[AgentPromptMessage],
        schema: Mapping[str, Any],
        settings: ActionDeciderSettings,
        tools: Sequence[ToolDefinition],
    ) -> ActionDecisionCompletion:
        response = _post_json(
            f"{settings.base_url}/api/chat",
            {
                "model": settings.model,
                "messages": _message_payload(messages),
                "stream": False,
                "think": settings.think,
                "format": dict(schema),
                "tools": _ollama_tool_payload(tools),
                "options": {
                    "temperature": settings.temperature,
                    "seed": settings.seed,
                    "num_predict": settings.max_output_tokens,
                },
            },
            headers=None,
            timeout=settings.timeout_seconds,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        thinking = message.get("thinking") if isinstance(message, Mapping) else None
        tool_calls = message.get("tool_calls") if isinstance(message, Mapping) else None

        if content is not None and not isinstance(content, str):
            raise ActionDecisionGenerationError(
                "Ollama returned non-text decision content"
            )
        if thinking is not None and not isinstance(thinking, str):
            raise ActionDecisionGenerationError(
                "Ollama returned non-text thinking content"
            )
        if tool_calls is not None and not isinstance(tool_calls, Sequence):
            raise ActionDecisionGenerationError(
                "Ollama returned non-sequence tool_calls"
            )
        return ActionDecisionCompletion(
            content=content,
            prompt_tokens=_optional_int(response.get("prompt_eval_count")),
            completion_tokens=_optional_int(response.get("eval_count")),
            finish_reason=_optional_string(response.get("done_reason")),
            raw_response=response,
            response_keys=tuple(sorted(str(key) for key in response)),
            message_keys=(
                tuple(sorted(str(key) for key in message))
                if isinstance(message, Mapping)
                else ()
            ),
            field_presence={
                "message": isinstance(message, Mapping),
                "content": isinstance(content, str) and bool(content.strip()),
                "thinking": isinstance(thinking, str) and bool(thinking.strip()),
                "tool_calls": isinstance(tool_calls, Sequence)
                and not isinstance(tool_calls, (str, bytes))
                and bool(tool_calls),
            },
            thinking_character_count=len(thinking or ""),
            thinking_sha256=_text_sha256(thinking),
            tool_call_count=(
                len(tool_calls)
                if isinstance(tool_calls, Sequence)
                and not isinstance(tool_calls, (str, bytes))
                else 0
            ),
        )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _environment_bool(name: str, *, default: bool) -> bool:
    """Parse an explicit boolean environment setting without silent coercion."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ActionDeciderError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0"
    )


def _text_sha256(value: str | None) -> str | None:
    return (
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        if isinstance(value, str)
        else None
    )


def _ollama_tool_payload(
    tools: Sequence[ToolDefinition],
) -> list[dict[str, Any]]:
    """Render trusted tool definitions in Ollama's native chat format."""

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            },
        }
        for tool in tools
    ]


class OllamaActionDecider:
    """Generate and strictly parse one canonical Ollama decision per step."""

    def __init__(
        self,
        settings: ActionDeciderSettings,
        *,
        backend: ActionDecisionBackend | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self._backend = backend or OllamaActionDecisionBackend()
        self._schema = action_decision_json_schema()
        serialized_schema = json.dumps(
            self._schema,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.schema_sha256 = hashlib.sha256(
            serialized_schema.encode("utf-8")
        ).hexdigest()
        self.last_trace: ActionDecisionTrace | None = None

    def verify(self) -> None:
        """Validate settings and verify Ollama model identity locally."""

        self.settings.validate()
        service = OllamaService(
            self.settings.base_url,
            timeout=self.settings.timeout_seconds,
        )
        try:
            if not service.has_model(self.settings.model):
                raise ActionDeciderError(
                    f"Ollama action-decider model is not installed: "
                    f"{self.settings.model}"
                )
            service.verify_digest(self.settings.model, self.settings.model_digest)
        except (OllamaError, OSError) as exc:
            raise ActionDeciderError(
                f"failed to verify Ollama action-decider model: {exc}"
            ) from exc

    def decide(self, context: AgentDecisionContext) -> AgentDecision:
        """Generate, trace, and validate exactly one canonical model decision."""

        started = time.perf_counter()
        try:
            completion = self._backend.complete(
                context.messages,
                self._schema,
                self.settings,
                context.available_tools,
            )
        except ActionDeciderError:
            raise
        except Exception as exc:
            raise ActionDecisionGenerationError(
                f"action-decision generation failed: {exc}"
            ) from exc
        latency = time.perf_counter() - started
        self.last_trace = ActionDecisionTrace(
            task_id=context.task_id,
            step_index=context.step_index,
            prompt_sha256=context.prompt_sha256,
            context_sha256=context.memory_context.sha256,
            schema_sha256=self.schema_sha256,
            provider="ollama",
            model=self.settings.model,
            model_digest=self.settings.model_digest,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            finish_reason=completion.finish_reason,
            latency_seconds=latency,
            field_presence=dict(completion.field_presence),
            completion_evidence=completion.diagnostic_evidence(
                include_raw_response=self.settings.capture_raw_response,
            ),
        )

        if completion.finish_reason == "length":
            self._mark_last_trace(
                "invalid_json",
                f"finish_reason={completion.finish_reason}",
            )
            raise ActionDecisionGenerationError(
                "action-decision response was incomplete: "
                f"finish_reason={completion.finish_reason}"
            )
        native_tool_decision = self._normalize_native_tool_call(completion, context)
        if native_tool_decision is not None:
            return native_tool_decision
        if completion.content is None or not completion.content.strip():
            self._mark_last_trace("empty_content", "provider returned no decision content")
            raise ActionDecisionGenerationError(
                "action-decider provider returned no decision content"
            )
        try:
            envelope = _ActionDecisionEnvelope.model_validate_json(
                completion.content
            )
        except ValidationError as exc:
            validation_error = (
                "provider output did not match the canonical agent-decision "
                f"schema ({exc.error_count()} validation errors)"
            )
            self._mark_last_trace("schema_error", validation_error)
            raise ActionDecisionValidationError(
                validation_error
            ) from exc

        decision = envelope.decision
        if not isinstance(
            decision,
            (ToolActionDecision, FinalAnswerDecision, RefusalDecision),
        ):
            self._mark_last_trace(
                "schema_error",
                "provider output contained an unsupported agent decision",
            )
            raise ActionDecisionValidationError(
                "provider output contained an unsupported agent decision"
            )
        return decision

    def _normalize_native_tool_call(
        self,
        completion: ActionDecisionCompletion,
        context: AgentDecisionContext,
    ) -> ToolActionDecision | None:
        """Convert one unambiguous Ollama native call into canonical evidence.

        This is normalization only: the returned decision still goes through the
        ordinary runner policy, approval provider, and tool argument validation.
        Text plus a native call, multiple calls, or malformed call fields are
        rejected rather than guessed.
        """

        message = completion.raw_response.get("message")
        raw_calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        if raw_calls is None:
            return None
        if completion.content is not None and completion.content.strip():
            self._mark_last_trace(
                "schema_error",
                "provider returned both decision content and native tool calls",
            )
            raise ActionDecisionValidationError(
                "provider returned both decision content and native tool calls"
            )
        if (
            not isinstance(raw_calls, Sequence)
            or isinstance(raw_calls, (str, bytes))
            or len(raw_calls) != 1
        ):
            self._mark_last_trace(
                "schema_error",
                "provider must return exactly one native tool call",
            )
            raise ActionDecisionValidationError(
                "provider must return exactly one native tool call"
            )
        raw_call = raw_calls[0]
        function = raw_call.get("function") if isinstance(raw_call, Mapping) else None
        if not isinstance(function, Mapping):
            self._mark_last_trace("schema_error", "native tool call has no function")
            raise ActionDecisionValidationError("native tool call has no function")
        call_id = raw_call.get("id")
        tool_name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(arguments, Mapping):
            self._mark_last_trace(
                "schema_error", "native tool-call arguments must be an object"
            )
            raise ActionDecisionValidationError(
                "native tool-call arguments must be an object"
            )
        try:
            call = ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                idempotency_key=f"{context.task_id}:{call_id}",
            )
        except ValidationError as exc:
            self._mark_last_trace(
                "schema_error", "native tool call does not meet canonical constraints"
            )
            raise ActionDecisionValidationError(
                "native tool call does not meet canonical constraints"
            ) from exc
        return ToolActionDecision(
            call=call,
            decision_summary=f"Provider-native tool proposal: {call.tool_name}",
        )

    def _mark_last_trace(
        self,
        outcome: Literal["empty_content", "invalid_json", "schema_error"],
        error: str,
    ) -> None:
        """Preserve completion evidence even when canonical parsing fails."""

        if self.last_trace is not None:
            self.last_trace = replace(
                self.last_trace,
                parse_outcome=outcome,
                validation_error=error,
            )

    def public_config(self) -> dict[str, object]:
        """Return reproducible non-secret model and schema configuration."""

        return {
            **self.settings.public_config(),
            "decision_schema_sha256": self.schema_sha256,
            "native_tool_call_normalization": True,
        }
