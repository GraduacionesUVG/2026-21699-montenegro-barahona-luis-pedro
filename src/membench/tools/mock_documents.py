"""Deterministic, persona-scoped synthetic document search and reading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from membench.memory.models import CanonicalModel, JsonObject
from membench.tools.models import ToolCall, ToolDefinition, ToolSideEffectKind
from membench.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from membench.agent.models import AgentToolExecutionContext

_IDENTIFIER_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_SOURCE_NAME = "trusted_mock_documents_v1"


class DocumentClassification(StrEnum):
    """Sensitivity label attached to a trusted synthetic document."""

    PUBLIC = "public"
    INTERNAL = "internal"
    SYNTHETIC_SENSITIVE = "synthetic_sensitive"


class MockDocumentConfigurationError(ValueError):
    """Raised when a synthetic document corpus is ambiguous or invalid."""


class MockDocument(CanonicalModel):
    """One immutable synthetic document visible to explicitly allowed personas."""

    document_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    authorized_user_ids: tuple[str, ...] = Field(min_length=1)
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=100_000)
    classification: DocumentClassification = DocumentClassification.INTERNAL
    version: int = Field(default=1, ge=1)
    approved: bool = False
    tags: tuple[str, ...] = Field(default_factory=tuple)
    approved_recipient_contact_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("authorized_user_ids")
    @classmethod
    def authorized_users_must_be_unique_and_deterministic(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("authorized document user IDs cannot be blank")
        if len(value) != len(set(value)):
            raise ValueError("authorized document user IDs must be unique")
        return tuple(sorted(value))

    @field_validator("tags", "approved_recipient_contact_ids")
    @classmethod
    def identifiers_must_be_unique_and_deterministic(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not _is_identifier(item) for item in value):
            raise ValueError(
                "document tags and approved recipient IDs must use lowercase "
                "identifier syntax"
            )
        if len(value) != len(set(value)):
            raise ValueError(
                "document tags and approved recipient IDs must be unique"
            )
        return tuple(sorted(value))

    @property
    def content_sha256(self) -> str:
        """Hash the exact content returned by the read tool."""

        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def metadata_record(self) -> JsonObject:
        """Return searchable metadata without exposing document content."""

        return {
            "document_id": self.document_id,
            "title": self.title,
            "classification": self.classification.value,
            "version": self.version,
            "approved": self.approved,
            "tags": list(self.tags),
            "approved_recipient_contact_ids": list(
                self.approved_recipient_contact_ids
            ),
            "content_sha256": self.content_sha256,
            "content_length": len(self.content),
        }

    def content_record(self) -> JsonObject:
        """Return metadata plus the exact synthetic document content."""

        return {**self.metadata_record(), "content": self.content}


class MockDocumentDirectory:
    """Immutable corpus with deterministic, persona-authorized retrieval."""

    def __init__(self, documents: Iterable[MockDocument]) -> None:
        ordered = tuple(sorted(documents, key=lambda item: item.document_id))
        if not ordered:
            raise MockDocumentConfigurationError(
                "mock document directory requires at least one document"
            )

        document_ids = [item.document_id for item in ordered]
        if len(document_ids) != len(set(document_ids)):
            raise MockDocumentConfigurationError(
                "mock document IDs must be globally unique"
            )

        self._documents = ordered
        self._by_id = {item.document_id: item for item in ordered}
        self.revision_sha256 = _sha256_json(
            [item.model_dump(mode="json") for item in ordered]
        )

    @property
    def documents(self) -> tuple[MockDocument, ...]:
        """Return the deterministic fixture for evaluator inspection."""

        return self._documents

    def get(self, user_id: str, document_id: str) -> MockDocument | None:
        """Return an exact document only when the persona is authorized."""

        document = self._by_id.get(document_id)
        if document is None or user_id not in document.authorized_user_ids:
            return None
        return document

    def search(
        self,
        user_id: str,
        query: str,
        *,
        limit: int,
        classifications: tuple[DocumentClassification, ...] = (),
    ) -> tuple[tuple[MockDocument, ...], int]:
        """Search authorized content and return stable matches plus total count."""

        normalized = query.casefold()
        allowed_classifications = set(classifications)
        ranked: list[tuple[int, str, MockDocument]] = []
        for document in self._documents:
            if user_id not in document.authorized_user_ids:
                continue
            if (
                allowed_classifications
                and document.classification not in allowed_classifications
            ):
                continue

            metadata_fields = (
                document.document_id.casefold(),
                document.title.casefold(),
                *(tag.casefold() for tag in document.tags),
            )
            if normalized in metadata_fields:
                rank = 0
            elif any(field.startswith(normalized) for field in metadata_fields):
                rank = 1
            elif any(normalized in field for field in metadata_fields):
                rank = 2
            elif normalized in document.content.casefold():
                rank = 3
            else:
                continue
            ranked.append((rank, document.document_id, document))

        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in ranked[:limit]), len(ranked)


SEARCH_DOCUMENTS_DEFINITION = ToolDefinition(
    name="search_documents",
    description=(
        "Search the trusted synthetic document corpus available to the current "
        "persona. Returns metadata and content hashes, not document content."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "description": "Text to match against document metadata or content.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
            "classifications": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [item.value for item in DocumentClassification],
                },
                "uniqueItems": True,
                "default": [],
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    side_effect=ToolSideEffectKind.READ_ONLY,
    requires_approval=False,
)

READ_DOCUMENT_DEFINITION = ToolDefinition(
    name="read_document",
    description=(
        "Read one exact trusted synthetic document available to the current "
        "persona by document_id."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "pattern": _IDENTIFIER_PATTERN,
                "description": "Exact trusted synthetic document ID.",
            }
        },
        "required": ["document_id"],
        "additionalProperties": False,
    },
    side_effect=ToolSideEffectKind.READ_ONLY,
    requires_approval=False,
)


def validate_search_documents_arguments(arguments: JsonObject) -> JsonObject:
    """Strictly validate and normalize a model-proposed document search."""

    unexpected = sorted(set(arguments) - {"query", "limit", "classifications"})
    if unexpected:
        raise ValueError(
            "search_documents received unsupported arguments: "
            + ", ".join(unexpected)
        )

    query = _required_text(arguments.get("query"), field="query", maximum=256)
    limit = arguments.get("limit", 5)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise ValueError("search_documents limit must be an integer from 1 to 10")

    raw_classifications = arguments.get("classifications", [])
    if not isinstance(raw_classifications, (list, tuple)) or isinstance(
        raw_classifications, (str, bytes)
    ):
        raise ValueError("search_documents classifications must be an array")
    classifications: list[str] = []
    for item in raw_classifications:
        if not isinstance(item, str):
            raise ValueError(
                "search_documents classifications must contain only strings"
            )
        try:
            classification = DocumentClassification(item)
        except ValueError as exc:
            raise ValueError(
                f"search_documents has an unknown classification: {item!r}"
            ) from exc
        classifications.append(classification.value)
    if len(classifications) != len(set(classifications)):
        raise ValueError("search_documents classifications must be unique")

    return {
        "query": query,
        "limit": limit,
        "classifications": classifications,
    }


def validate_read_document_arguments(arguments: JsonObject) -> JsonObject:
    """Strictly validate and normalize a model-proposed exact document read."""

    unexpected = sorted(set(arguments) - {"document_id"})
    if unexpected:
        raise ValueError(
            "read_document received unsupported arguments: "
            + ", ".join(unexpected)
        )
    document_id = _required_text(
        arguments.get("document_id"),
        field="document_id",
        maximum=256,
    )
    if not _is_identifier(document_id):
        raise ValueError("read_document document_id has an invalid format")
    return {"document_id": document_id}


class MockDocumentsTool:
    """Registry-compatible search and read facade over one trusted corpus."""

    def __init__(self, directory: MockDocumentDirectory) -> None:
        self.directory = directory

    def invoke(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        """Dispatch a persona-scoped document search or exact read."""

        if call.tool_name == SEARCH_DOCUMENTS_DEFINITION.name:
            return self._search(call, context)
        if call.tool_name == READ_DOCUMENT_DEFINITION.name:
            return self._read(call, context)
        raise ValueError(f"mock documents tool cannot execute {call.tool_name!r}")

    def _search(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        arguments = validate_search_documents_arguments(dict(call.arguments))
        query = arguments["query"]
        limit = arguments["limit"]
        classification_values = arguments["classifications"]
        if (
            not isinstance(query, str)
            or not isinstance(limit, int)
            or not isinstance(classification_values, list)
        ):
            raise TypeError("validated search_documents arguments have invalid types")
        classifications = tuple(
            DocumentClassification(item) for item in classification_values
        )
        matches, total_matches = self.directory.search(
            context.scope.user_id,
            query,
            limit=limit,
            classifications=classifications,
        )
        return {
            "source": _SOURCE_NAME,
            "directory_revision_sha256": self.directory.revision_sha256,
            "returned_count": len(matches),
            "total_matches": total_matches,
            "truncated": total_matches > len(matches),
            "documents": [item.metadata_record() for item in matches],
        }

    def _read(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        arguments = validate_read_document_arguments(dict(call.arguments))
        document_id = arguments["document_id"]
        if not isinstance(document_id, str):
            raise TypeError("validated read_document arguments have invalid types")
        document = self.directory.get(context.scope.user_id, document_id)
        return {
            "source": _SOURCE_NAME,
            "directory_revision_sha256": self.directory.revision_sha256,
            "found": document is not None,
            "document": None if document is None else document.content_record(),
        }


def register_mock_document_tools(
    registry: ToolRegistry,
    directory: MockDocumentDirectory,
    *,
    tool_names: Iterable[str] | None = None,
) -> MockDocumentsTool:
    """Register selected document operations without freezing the registry.

    Omitting ``tool_names`` preserves the normal paired search-and-read tool
    catalog.  A bounded campaign may register only one operation when its
    allowlist deliberately exposes a narrower discovery surface.
    """

    selected = (
        frozenset({SEARCH_DOCUMENTS_DEFINITION.name, READ_DOCUMENT_DEFINITION.name})
        if tool_names is None
        else frozenset(tool_names)
    )
    supported = {SEARCH_DOCUMENTS_DEFINITION.name, READ_DOCUMENT_DEFINITION.name}
    if not selected:
        raise ValueError("at least one mock document tool must be selected")
    if unknown := selected - supported:
        raise ValueError(
            "mock document tool selection contains unsupported names: "
            + ", ".join(sorted(unknown))
        )
    implementation = MockDocumentsTool(directory)
    if SEARCH_DOCUMENTS_DEFINITION.name in selected:
        registry.register(
            SEARCH_DOCUMENTS_DEFINITION,
            implementation,
            argument_validator=validate_search_documents_arguments,
        )
    if READ_DOCUMENT_DEFINITION.name in selected:
        registry.register(
            READ_DOCUMENT_DEFINITION,
            implementation,
            argument_validator=validate_read_document_arguments,
        )
    return implementation


def _required_text(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"document tool {field} must be non-blank text")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(
            f"document tool {field} cannot exceed {maximum} characters"
        )
    return normalized


def _is_identifier(value: str) -> bool:
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
