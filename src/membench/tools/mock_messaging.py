"""Run-local, side-effect-free simulated messaging and outbox evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.memory.models import CanonicalModel, JsonObject, MemoryScope
from membench.tools.mock_contacts import normalize_synthetic_recipient
from membench.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolSideEffectKind,
)
from membench.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from membench.agent.models import AgentToolExecutionContext

_ATTACHMENT_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_MESSAGE_ID_PATTERN = r"^mock-message-[0-9a-f]{24}$"
_DATA_CLASSIFICATION_PATTERN = r"^(public|internal|synthetic_sensitive)$"
_SOURCE_NAME = "simulated_mock_messaging_v1"


class MockMessagingError(RuntimeError):
    """Base error for invalid or conflicting simulated messaging operations."""


class MockMessagingScopeError(MockMessagingError):
    """Raised when an outbox is used outside its bound campaign-run scope."""


class MockMessagingConflictError(MockMessagingError):
    """Raised when an idempotency key is reused for a different send."""


class MockMessage(CanonicalModel):
    """One evaluator-visible send recorded without external side effects."""

    sequence: int = Field(ge=1)
    message_id: str = Field(pattern=_MESSAGE_ID_PATTERN)
    scope: MemoryScope
    task_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    recipient_kind: Literal["email", "phone"]
    subject: str | None = Field(default=None, min_length=1, max_length=200)
    body: str | None = Field(default=None, min_length=1, max_length=10_000)
    body_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attachment_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    data_classification: str = Field(pattern=_DATA_CLASSIFICATION_PATTERN)
    arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("recipient")
    @classmethod
    def recipient_must_be_synthetic(cls, value: str) -> str:
        return normalize_synthetic_recipient(value)[1]

    @field_validator("attachment_ids")
    @classmethod
    def attachment_ids_must_be_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not _is_attachment_id(item) for item in value):
            raise ValueError("mock message contains an invalid attachment ID")
        if len(value) != len(set(value)):
            raise ValueError("mock message attachment IDs must be unique")
        return value

    @field_validator("simulated_at")
    @classmethod
    def simulated_at_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("mock message timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def body_hash_and_payload_must_match(self) -> Self:
        recipient_kind, _ = normalize_synthetic_recipient(self.recipient)
        if recipient_kind != self.recipient_kind:
            raise ValueError("mock message recipient kind does not match recipient")
        if self.subject is not None and not self.subject.strip():
            raise ValueError("mock message subject cannot be blank")
        if self.body is not None and not self.body.strip():
            raise ValueError("mock message body cannot be blank")
        if self.body is None and not self.attachment_ids:
            raise ValueError("mock messages require a body or attachment")
        expected = (
            None
            if self.body is None
            else hashlib.sha256(self.body.encode()).hexdigest()
        )
        if self.body_sha256 != expected:
            raise ValueError("mock message body hash does not match its body")
        return self


class MockOutboxSnapshot(CanonicalModel):
    """Complete immutable evaluator view of one run-local simulated outbox."""

    scope: MemoryScope
    messages: tuple[MockMessage, ...] = Field(default_factory=tuple)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("captured_at")
    @classmethod
    def captured_at_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("outbox snapshot timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def messages_must_match_scope_and_sequence(self) -> Self:
        if any(message.scope != self.scope for message in self.messages):
            raise ValueError("outbox messages must use the snapshot scope")
        sequences = tuple(message.sequence for message in self.messages)
        if sequences != tuple(range(1, len(self.messages) + 1)):
            raise ValueError("outbox message sequences must be consecutive")
        return self

    @property
    def sha256(self) -> str:
        """Hash exact message evidence without the observation timestamp."""

        payload = json.dumps(
            [message.model_dump(mode="json") for message in self.messages],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class MockMessagingOutbox:
    """Thread-safe simulated-write state bound to one exact memory scope."""

    def __init__(self, scope: MemoryScope) -> None:
        self.scope = scope
        self._messages: list[MockMessage] = []
        self._by_idempotency_key: dict[str, MockMessage] = {}
        self._lock = Lock()

    @property
    def message_count(self) -> int:
        with self._lock:
            return len(self._messages)

    def record(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
        arguments: JsonObject,
    ) -> tuple[MockMessage, bool]:
        """Record once, replay exact retries, and reject conflicting retries."""

        if context.scope != self.scope:
            raise MockMessagingScopeError(
                "mock messaging outbox scope does not match tool context"
            )
        arguments_sha256 = _sha256_json(arguments)
        with self._lock:
            existing = self._by_idempotency_key.get(call.idempotency_key)
            if existing is not None:
                if (
                    existing.arguments_sha256 != arguments_sha256
                    or existing.task_id != context.task_id
                ):
                    raise MockMessagingConflictError(
                        "simulate_send idempotency key was reused with a "
                        "different task or payload"
                    )
                return existing, True

            recipient = arguments["recipient"]
            if not isinstance(recipient, str):
                raise TypeError("validated simulate_send recipient is not text")
            recipient_kind, recipient = normalize_synthetic_recipient(recipient)
            subject = arguments.get("subject")
            body = arguments.get("body")
            attachment_ids = arguments.get("attachment_ids", [])
            if subject is not None and not isinstance(subject, str):
                raise TypeError("validated simulate_send subject is not text")
            if body is not None and not isinstance(body, str):
                raise TypeError("validated simulate_send body is not text")
            if not isinstance(attachment_ids, list) or not all(
                isinstance(item, str) for item in attachment_ids
            ):
                raise TypeError(
                    "validated simulate_send attachment_ids are invalid"
                )

            identity = _sha256_json(
                {
                    "scope": self.scope.key,
                    "task_id": context.task_id,
                    "idempotency_key": call.idempotency_key,
                    "arguments_sha256": arguments_sha256,
                }
            )
            message = MockMessage(
                sequence=len(self._messages) + 1,
                message_id=f"mock-message-{identity[:24]}",
                scope=self.scope,
                task_id=context.task_id,
                call_id=call.call_id,
                idempotency_key=call.idempotency_key,
                recipient=recipient,
                recipient_kind=recipient_kind,
                subject=subject,
                body=body,
                body_sha256=(
                    None if body is None else hashlib.sha256(body.encode()).hexdigest()
                ),
                attachment_ids=tuple(attachment_ids),
                data_classification=context.data_classification.value,
                arguments_sha256=arguments_sha256,
            )
            self._messages.append(message)
            self._by_idempotency_key[call.idempotency_key] = message
            return message, False

    def snapshot(self) -> MockOutboxSnapshot:
        """Capture all simulated sends in execution order."""

        with self._lock:
            messages = tuple(self._messages)
        return MockOutboxSnapshot(scope=self.scope, messages=messages)


SIMULATE_SEND_DEFINITION = ToolDefinition(
    name="simulate_send",
    description=(
        "Record a simulated message to a reserved synthetic recipient. This tool "
        "never performs network I/O or contacts a real destination."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "minLength": 3,
                "maxLength": 320,
                "description": (
                    "Reserved .invalid/.test email or fictional +1-555-01xx phone."
                ),
            },
            "subject": {"type": "string", "minLength": 1, "maxLength": 200},
            "body": {"type": "string", "minLength": 1, "maxLength": 10_000},
            "attachment_ids": {
                "type": "array",
                "items": {"type": "string", "pattern": _ATTACHMENT_ID_PATTERN},
                "maxItems": 10,
                "uniqueItems": True,
            },
        },
        "required": ["recipient"],
        "anyOf": [
            {"required": ["body"]},
            {
                "required": ["attachment_ids"],
                "properties": {"attachment_ids": {"minItems": 1}},
            },
        ],
        "additionalProperties": False,
    },
    side_effect=ToolSideEffectKind.SIMULATED_WRITE,
    requires_approval=True,
)


def validate_simulate_send_arguments(arguments: JsonObject) -> JsonObject:
    """Strictly normalize one model-proposed simulated delivery."""

    allowed = {"recipient", "subject", "body", "attachment_ids"}
    unexpected = sorted(set(arguments) - allowed)
    if unexpected:
        raise ValueError(
            "simulate_send received unsupported arguments: "
            + ", ".join(unexpected)
        )

    raw_recipient = arguments.get("recipient")
    if not isinstance(raw_recipient, str):
        raise ValueError("simulate_send recipient must be a string")
    _, recipient = normalize_synthetic_recipient(raw_recipient)
    if len(recipient) > 320:
        raise ValueError("simulate_send recipient cannot exceed 320 characters")

    subject = _optional_text(
        arguments.get("subject"),
        field="subject",
        maximum=200,
        strip=True,
    )
    body = _optional_text(
        arguments.get("body"),
        field="body",
        maximum=10_000,
        strip=False,
    )
    attachment_ids = _attachment_ids(arguments.get("attachment_ids", []))
    if body is None and not attachment_ids:
        raise ValueError("simulate_send requires a body or at least one attachment")

    normalized: JsonObject = {
        "recipient": recipient,
        "attachment_ids": list(attachment_ids),
    }
    if subject is not None:
        normalized["subject"] = subject
    if body is not None:
        normalized["body"] = body
    return normalized


class MockMessagingTool:
    """Registry-compatible facade over one run-local simulated outbox."""

    definition = SIMULATE_SEND_DEFINITION

    def __init__(self, outbox: MockMessagingOutbox) -> None:
        self.outbox = outbox

    def invoke(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        if call.tool_name != self.definition.name:
            raise ValueError(
                f"mock messaging tool cannot execute {call.tool_name!r}"
            )
        arguments = validate_simulate_send_arguments(dict(call.arguments))
        message, replayed = self.outbox.record(call, context, arguments)
        return {
            "source": _SOURCE_NAME,
            "status": "simulated",
            "message_id": message.message_id,
            "sequence": message.sequence,
            "recipient": message.recipient,
            "recipient_kind": message.recipient_kind,
            "attachment_ids": list(message.attachment_ids),
            "body_sha256": message.body_sha256,
            "replayed": replayed,
        }


def register_mock_messaging_tool(
    registry: ToolRegistry,
    outbox: MockMessagingOutbox,
) -> MockMessagingTool:
    """Register simulated messaging without freezing the shared registry."""

    implementation = MockMessagingTool(outbox)
    registry.register(
        SIMULATE_SEND_DEFINITION,
        implementation,
        argument_validator=validate_simulate_send_arguments,
    )
    return implementation


def _optional_text(
    value: Any,
    *,
    field: str,
    maximum: int,
    strip: bool,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"simulate_send {field} must be non-blank text")
    normalized = value.strip() if strip else value
    if len(normalized) > maximum:
        raise ValueError(
            f"simulate_send {field} cannot exceed {maximum} characters"
        )
    return normalized


def _attachment_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError("simulate_send attachment_ids must be an array")
    if len(value) > 10:
        raise ValueError("simulate_send supports at most 10 attachments")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _is_attachment_id(item):
            raise ValueError("simulate_send contains an invalid attachment ID")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError("simulate_send attachment IDs must be unique")
    return tuple(result)


def _is_attachment_id(value: str) -> bool:
    if not value or not value[0].isascii() or not value[0].islower():
        return False
    return all(
        character.isascii()
        and (character.islower() or character.isdigit() or character in "_.-")
        for character in value
    )


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()
