"""Deterministic, persona-scoped synthetic contact lookup tool."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from string import ascii_lowercase, digits
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field, field_validator, model_validator

from membench.memory.models import CanonicalModel, JsonObject
from membench.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolSideEffectKind,
)
from membench.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from membench.agent.models import AgentToolExecutionContext

_CONTACT_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_SYNTHETIC_PHONE_PATTERN = r"^\+1-555-01[0-9]{2}$"
_SOURCE_NAME = "trusted_mock_contacts_v1"


class MockContactConfigurationError(ValueError):
    """Raised when the trusted synthetic directory is ambiguous or unsafe."""


class MockContact(CanonicalModel):
    """One non-routable contact owned by a single synthetic persona."""

    contact_id: str = Field(pattern=_CONTACT_ID_PATTERN)
    owner_user_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    email: str | None = Field(default=None, min_length=3)
    phone: str | None = Field(default=None, pattern=_SYNTHETIC_PHONE_PATTERN)
    organization: str | None = Field(default=None, min_length=1)
    relationship: str | None = Field(default=None, min_length=1)
    aliases: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("email")
    @classmethod
    def email_must_be_non_routable(cls, value: str | None) -> str | None:
        if value is None:
            return None
        kind, normalized = normalize_synthetic_recipient(value)
        if kind != "email":
            raise ValueError("mock contact email must be a synthetic email")
        return normalized

    @field_validator("aliases")
    @classmethod
    def aliases_must_be_unique_and_deterministic(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not alias.strip() for alias in value):
            raise ValueError("mock contact aliases cannot be blank")
        normalized = [alias.casefold() for alias in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("mock contact aliases must be case-insensitively unique")
        return tuple(sorted(value, key=str.casefold))

    @model_validator(mode="after")
    def at_least_one_destination_is_required(self) -> Self:
        if self.email is None and self.phone is None:
            raise ValueError("mock contacts require an email or fictional phone")
        return self

    def public_record(self) -> JsonObject:
        """Return the trusted fields visible to the bounded agent."""

        return {
            "contact_id": self.contact_id,
            "display_name": self.display_name,
            "email": self.email,
            "phone": self.phone,
            "organization": self.organization,
            "relationship": self.relationship,
            "aliases": list(self.aliases),
        }


class MockContactDirectory:
    """Immutable contact fixture with deterministic persona-scoped search."""

    def __init__(self, contacts: Iterable[MockContact]) -> None:
        ordered = tuple(
            sorted(
                contacts,
                key=lambda item: (item.owner_user_id, item.contact_id),
            )
        )
        if not ordered:
            raise MockContactConfigurationError(
                "mock contact directory requires at least one contact"
            )

        keys = [(item.owner_user_id, item.contact_id) for item in ordered]
        if len(keys) != len(set(keys)):
            raise MockContactConfigurationError(
                "mock contact IDs must be unique within each owner_user_id"
            )
        destinations: list[tuple[str, str, str]] = []
        for item in ordered:
            if item.email is not None:
                destinations.append(
                    (item.owner_user_id, "email", item.email.casefold())
                )
            if item.phone is not None:
                destinations.append((item.owner_user_id, "phone", item.phone))
        if len(destinations) != len(set(destinations)):
            raise MockContactConfigurationError(
                "mock contact destinations must be unique within each owner_user_id"
            )

        self._contacts = ordered
        self._by_owner_and_id = {
            (item.owner_user_id, item.contact_id): item for item in ordered
        }
        payload = json.dumps(
            [item.model_dump(mode="json") for item in ordered],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        self.revision_sha256 = hashlib.sha256(payload).hexdigest()

    @property
    def contacts(self) -> tuple[MockContact, ...]:
        """Return the deterministic trusted fixture for evaluator inspection."""

        return self._contacts

    def get(self, owner_user_id: str, contact_id: str) -> MockContact | None:
        """Resolve one exact ID without crossing the persona boundary."""

        return self._by_owner_and_id.get((owner_user_id, contact_id))

    def search(
        self,
        owner_user_id: str,
        query: str,
        *,
        limit: int,
    ) -> tuple[MockContact, ...]:
        """Return exact, prefix, then substring matches in stable order."""

        normalized = query.casefold()
        ranked: list[tuple[int, str, MockContact]] = []
        for contact in self._contacts:
            if contact.owner_user_id != owner_user_id:
                continue
            fields = tuple(
                value.casefold()
                for value in (
                    contact.contact_id,
                    contact.display_name,
                    contact.email,
                    contact.phone,
                    contact.organization,
                    contact.relationship,
                    *contact.aliases,
                )
                if value is not None
            )
            if normalized in fields:
                rank = 0
            elif any(value.startswith(normalized) for value in fields):
                rank = 1
            elif any(normalized in value for value in fields):
                rank = 2
            else:
                continue
            ranked.append((rank, contact.contact_id, contact))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in ranked[:limit])


LOOKUP_CONTACT_DEFINITION = ToolDefinition(
    name="lookup_contact",
    description=(
        "Look up trusted synthetic contact destinations for the current persona. "
        "Use either an exact contact_id or a name, alias, email, phone, "
        "organization, or relationship query."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "contact_id": {
                "type": "string",
                "pattern": _CONTACT_ID_PATTERN,
                "description": "Exact trusted synthetic contact ID.",
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "description": "Trusted-directory search text.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
        },
        "oneOf": [
            {"required": ["contact_id"]},
            {"required": ["query"]},
        ],
        "additionalProperties": False,
    },
    side_effect=ToolSideEffectKind.READ_ONLY,
    requires_approval=False,
)


def validate_lookup_contact_arguments(arguments: JsonObject) -> JsonObject:
    """Strictly validate and normalize a model-proposed lookup request."""

    allowed = {"contact_id", "query", "limit"}
    unexpected = sorted(set(arguments) - allowed)
    if unexpected:
        raise ValueError(
            "lookup_contact received unsupported arguments: "
            + ", ".join(unexpected)
        )

    contact_id = _optional_non_blank_string(arguments.get("contact_id"))
    query = _optional_non_blank_string(arguments.get("query"))
    if (contact_id is None) == (query is None):
        raise ValueError(
            "lookup_contact requires exactly one of contact_id or query"
        )
    limit = arguments.get("limit", 5)
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 10
    ):
        raise ValueError("lookup_contact limit must be an integer from 1 to 10")

    if contact_id is not None:
        if not _is_contact_id(contact_id):
            raise ValueError("lookup_contact contact_id has an invalid format")
        if "limit" in arguments:
            raise ValueError("lookup_contact limit is valid only with query")
        return {"contact_id": contact_id}
    return {"query": query, "limit": limit}


class MockContactsTool:
    """Read-only implementation backed by one trusted fixture directory."""

    definition = LOOKUP_CONTACT_DEFINITION

    def __init__(self, directory: MockContactDirectory) -> None:
        self.directory = directory

    def invoke(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        """Resolve only contacts owned by the active memory-scope user."""

        if call.tool_name != self.definition.name:
            raise ValueError(
                f"mock contacts tool cannot execute {call.tool_name!r}"
            )
        arguments = validate_lookup_contact_arguments(dict(call.arguments))
        owner_user_id = context.scope.user_id
        contact_id = arguments.get("contact_id")
        if isinstance(contact_id, str):
            contact = self.directory.get(owner_user_id, contact_id)
            matches = () if contact is None else (contact,)
            mode = "contact_id"
            limit = 1
        else:
            query = arguments["query"]
            limit = arguments["limit"]
            if not isinstance(query, str) or not isinstance(limit, int):
                raise TypeError("validated lookup arguments have invalid types")
            matches = self.directory.search(
                owner_user_id,
                query,
                limit=limit,
            )
            mode = "query"

        return {
            "source": _SOURCE_NAME,
            "directory_revision_sha256": self.directory.revision_sha256,
            "lookup_mode": mode,
            "found": bool(matches),
            "returned_count": len(matches),
            "limit": limit,
            "contacts": [item.public_record() for item in matches],
        }


def register_mock_contacts_tool(
    registry: ToolRegistry,
    directory: MockContactDirectory,
) -> MockContactsTool:
    """Register one trusted contacts implementation without freezing the registry."""

    implementation = MockContactsTool(directory)
    registry.register(
        LOOKUP_CONTACT_DEFINITION,
        implementation,
        argument_validator=validate_lookup_contact_arguments,
    )
    return implementation


def normalize_synthetic_recipient(value: str) -> tuple[str, str]:
    """Classify and normalize a reserved email or fictional phone recipient."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("synthetic recipient must be a non-blank string")
    normalized = value.strip().casefold()
    if re.fullmatch(_SYNTHETIC_PHONE_PATTERN, normalized):
        return "phone", normalized
    if (
        normalized.count("@") != 1
        or normalized.startswith("@")
        or normalized.endswith("@")
        or any(character.isspace() for character in normalized)
    ):
        raise ValueError(
            "synthetic recipient must be an email or fictional +1-555-01xx phone"
        )
    domain = normalized.rsplit("@", maxsplit=1)[1]
    if not (domain.endswith(".invalid") or domain.endswith(".test")):
        raise ValueError(
            "synthetic recipient email must use a reserved .invalid or .test domain"
        )
    return "email", normalized


def _optional_non_blank_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("lookup_contact string arguments must be non-blank")
    if len(value.strip()) > 256:
        raise ValueError(
            "lookup_contact string arguments cannot exceed 256 characters"
        )
    return value.strip()


def _is_contact_id(value: str) -> bool:
    if not value or value[0] not in ascii_lowercase:
        return False
    allowed = frozenset(ascii_lowercase + digits + "_.-")
    return all(character in allowed for character in value)
