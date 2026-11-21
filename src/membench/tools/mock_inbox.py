"""Deterministic, read-only inbox for frozen external email deliveries.

The inbox contains only the exact deliveries assigned to one reconstructed
trial.  It never creates messages or performs network access.  A later ingress
orchestrator can take the ``delivery_id`` returned by ``read_message`` and pass
that same frozen :class:`ExternalDelivery` to the policy-dispatch bridge.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from membench.ingress.models import ExternalDelivery, ExternalSourceKind
from membench.memory.models import JsonObject
from membench.tools.models import ToolCall, ToolDefinition, ToolSideEffectKind
from membench.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from membench.agent.models import AgentToolExecutionContext

_IDENTIFIER_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_SOURCE_NAME = "frozen_mock_inbox_v1"


class MockInboxConfigurationError(ValueError):
    """Raised when one frozen inbox cannot be safely constructed."""


class MockInbox:
    """Frozen email inventory with replay-index-gated visibility.

    Deliveries are immutable, but are not necessarily visible to the agent when
    a trial begins.  The lifecycle runner advances the source-message cursor
    only after a replay batch has completed successfully.  Search and read
    therefore cannot disclose a delivery before its frozen availability point.
    """

    def __init__(self, deliveries: Iterable[ExternalDelivery]) -> None:
        ordered = tuple(sorted(deliveries, key=lambda item: item.delivery_id))
        if not ordered:
            raise MockInboxConfigurationError("mock inbox requires at least one email")
        if any(item.source_kind != ExternalSourceKind.EMAIL for item in ordered):
            raise MockInboxConfigurationError(
                "mock inbox accepts email deliveries only"
            )
        ids = tuple(item.delivery_id for item in ordered)
        if len(ids) != len(set(ids)):
            raise MockInboxConfigurationError("mock inbox delivery IDs must be unique")
        self._deliveries = ordered
        self._by_id = {delivery.delivery_id: delivery for delivery in ordered}
        self._source_message_index = -1
        self.revision_sha256 = _sha256_json(
            [delivery.model_dump(mode="json") for delivery in ordered]
        )

    @property
    def deliveries(self) -> tuple[ExternalDelivery, ...]:
        """Return the frozen evaluator-visible delivery inventory."""

        return self._deliveries

    @property
    def source_message_index(self) -> int:
        """Return the greatest successfully replayed source-message index."""

        return self._source_message_index

    def advance_to_source_message_index(self, source_message_index: int) -> None:
        """Expose deliveries available after a completed source-message index.

        The cursor is deliberately monotonic: replay cannot move backward and
        a later agent step cannot make a future email visible early.
        """

        if isinstance(source_message_index, bool) or not isinstance(
            source_message_index, int
        ):
            raise TypeError("source message index must be an integer")
        if source_message_index < 0:
            raise ValueError("source message index must be nonnegative")
        if source_message_index < self._source_message_index:
            raise MockInboxConfigurationError(
                "mock inbox source-message index cannot move backward"
            )
        self._source_message_index = source_message_index

    def available_deliveries(self) -> tuple[ExternalDelivery, ...]:
        """Return only emails whose frozen replay boundary has been reached."""

        return tuple(
            delivery
            for delivery in self._deliveries
            if delivery.available_after_source_message_index
            <= self._source_message_index
        )

    def get(self, delivery_id: str) -> ExternalDelivery | None:
        """Return one exact email, without exposing whether another ID exists."""

        delivery = self._by_id.get(delivery_id)
        if (
            delivery is None
            or delivery.available_after_source_message_index
            > self._source_message_index
        ):
            return None
        return delivery

    def search(
        self, query: str, *, limit: int
    ) -> tuple[tuple[ExternalDelivery, ...], int]:
        """Search stable headers and bodies, returning metadata-only matches.

        Search deliberately normalizes only presentation differences: Unicode
        compatibility forms, letter case, punctuation, and runs of whitespace.
        It remains a deterministic lexical search; it does not use fuzzy,
        semantic, or model-assisted matching.
        """

        normalized = _normalize_search_text(query)
        if not normalized:
            return (), 0
        ranked: list[tuple[int, str, ExternalDelivery]] = []
        for delivery in self.available_deliveries():
            metadata = (
                _normalize_search_text(delivery.delivery_id),
                _normalize_search_text(delivery.source_id),
                _normalize_search_text(delivery.title),
                _normalize_search_text(delivery.sender or ""),
            )
            if normalized in metadata:
                rank = 0
            elif any(field.startswith(normalized) for field in metadata):
                rank = 1
            elif any(normalized in field for field in metadata):
                rank = 2
            elif normalized in _normalize_search_text(delivery.content):
                rank = 3
            else:
                continue
            ranked.append((rank, delivery.delivery_id, delivery))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in ranked[:limit]), len(ranked)


SEARCH_MESSAGES_DEFINITION = ToolDefinition(
    name="search_messages",
    description=(
        "Search the frozen synthetic inbox available for this trial. Returns "
        "email metadata and content hashes, not message bodies."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "description": "Text to match against email headers or content.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    side_effect=ToolSideEffectKind.READ_ONLY,
    requires_approval=False,
)

READ_MESSAGE_DEFINITION = ToolDefinition(
    name="read_message",
    description="Read one exact frozen synthetic email by message_id.",
    input_schema={
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "pattern": _IDENTIFIER_PATTERN,
                "description": "Exact frozen email message ID.",
            }
        },
        "required": ["message_id"],
        "additionalProperties": False,
    },
    side_effect=ToolSideEffectKind.READ_ONLY,
    requires_approval=False,
)


def validate_search_messages_arguments(arguments: JsonObject) -> JsonObject:
    """Strictly validate and normalize a model-proposed inbox search."""

    unexpected = sorted(set(arguments) - {"query", "limit"})
    if unexpected:
        raise ValueError(
            "search_messages received unsupported arguments: " + ", ".join(unexpected)
        )
    query = _required_text(arguments.get("query"), field="query", maximum=256)
    limit = arguments.get("limit", 5)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("search_messages limit must be an integer from 1 to 10")
    return {"query": query, "limit": limit}


def validate_read_message_arguments(arguments: JsonObject) -> JsonObject:
    """Strictly validate and normalize a model-proposed exact email read."""

    unexpected = sorted(set(arguments) - {"message_id"})
    if unexpected:
        raise ValueError(
            "read_message received unsupported arguments: " + ", ".join(unexpected)
        )
    message_id = _required_text(
        arguments.get("message_id"), field="message_id", maximum=256
    )
    if not _is_identifier(message_id):
        raise ValueError("read_message message_id has an invalid format")
    return {"message_id": message_id}


class MockInboxTool:
    """Registry-compatible facade for metadata search and exact email reads."""

    def __init__(self, inbox: MockInbox) -> None:
        self.inbox = inbox

    def invoke(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        """Dispatch one bounded inbox operation; context is retained in its trace."""

        del context
        if call.tool_name == SEARCH_MESSAGES_DEFINITION.name:
            return self._search(call)
        if call.tool_name == READ_MESSAGE_DEFINITION.name:
            return self._read(call)
        raise ValueError(f"mock inbox cannot execute {call.tool_name!r}")

    def _search(self, call: ToolCall) -> JsonObject:
        arguments = validate_search_messages_arguments(dict(call.arguments))
        query = arguments["query"]
        limit = arguments["limit"]
        if not isinstance(query, str) or not isinstance(limit, int):
            raise TypeError("validated search_messages arguments have invalid types")
        matches, total_matches = self.inbox.search(query, limit=limit)
        return {
            "source": _SOURCE_NAME,
            "inbox_revision_sha256": self.inbox.revision_sha256,
            "available_message_count": len(self.inbox.available_deliveries()),
            "returned_count": len(matches),
            "total_matches": total_matches,
            "truncated": total_matches > len(matches),
            "messages": [_metadata_record(item) for item in matches],
        }

    def _read(self, call: ToolCall) -> JsonObject:
        arguments = validate_read_message_arguments(dict(call.arguments))
        message_id = arguments["message_id"]
        if not isinstance(message_id, str):
            raise TypeError("validated read_message arguments have invalid types")
        delivery = self.inbox.get(message_id)
        return {
            "source": _SOURCE_NAME,
            "inbox_revision_sha256": self.inbox.revision_sha256,
            "available_message_count": len(self.inbox.available_deliveries()),
            "found": delivery is not None,
            "message": None if delivery is None else _content_record(delivery),
        }


def register_mock_inbox_tools(
    registry: ToolRegistry,
    inbox: MockInbox,
    *,
    tool_names: Iterable[str] | None = None,
) -> MockInboxTool:
    """Register a deliberately bounded inbox search/read surface."""

    supported = {SEARCH_MESSAGES_DEFINITION.name, READ_MESSAGE_DEFINITION.name}
    selected = frozenset(supported if tool_names is None else tool_names)
    if not selected:
        raise ValueError("at least one mock inbox tool must be selected")
    if unknown := selected - supported:
        raise ValueError(
            "mock inbox tool selection contains unsupported names: "
            + ", ".join(sorted(unknown))
        )
    implementation = MockInboxTool(inbox)
    if SEARCH_MESSAGES_DEFINITION.name in selected:
        registry.register(
            SEARCH_MESSAGES_DEFINITION,
            implementation,
            argument_validator=validate_search_messages_arguments,
        )
    if READ_MESSAGE_DEFINITION.name in selected:
        registry.register(
            READ_MESSAGE_DEFINITION,
            implementation,
            argument_validator=validate_read_message_arguments,
        )
    return implementation


def _metadata_record(delivery: ExternalDelivery) -> JsonObject:
    return {
        "message_id": delivery.delivery_id,
        "source_id": delivery.source_id,
        "sender": delivery.sender,
        "title": delivery.title,
        "content_sha256": delivery.content_sha256,
        "content_length": len(delivery.content),
    }


def _content_record(delivery: ExternalDelivery) -> JsonObject:
    return {**_metadata_record(delivery), "content": delivery.content}


def _required_text(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"inbox tool {field} must be non-blank text")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"inbox tool {field} cannot exceed {maximum} characters")
    return normalized


def _is_identifier(value: str) -> bool:
    if not value or not value[0].isascii() or not value[0].islower():
        return False
    return all(
        character.isascii()
        and (character.islower() or character.isdigit() or character in "_.-")
        for character in value
    )


def _normalize_search_text(value: str) -> str:
    """Normalize presentation-only lexical differences for inbox discovery.

    NFKC makes compatibility spellings comparable; punctuation and every form
    of whitespace become a single separator.  Letters and digits, including
    non-ASCII ones, are retained so this function does not transliterate or
    otherwise change a message's meaning.
    """

    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = [
        character if character.isalnum() else " "
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
