"""Validated, optional fixture bundles for deterministic mock-tool runtimes.

Fixtures describe the stable, evaluator-controlled environment that an agent can
discover through mock contact and document tools.  They intentionally exclude
attack-bearing delivery content: clean, matched-benign, and poisoned variants
belong to frozen trial projections so the shared fixture is identical for every
twin.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from membench.memory.models import CanonicalModel, MemoryScope
from membench.tools.mock_contacts import (
    MockContact,
    MockContactDirectory,
    register_mock_contacts_tool,
)
from membench.tools.mock_documents import (
    MockDocument,
    MockDocumentDirectory,
    register_mock_document_tools,
)
from membench.tools.mock_inbox import MockInbox, register_mock_inbox_tools
from membench.tools.mock_messaging import (
    MockMessagingOutbox,
    register_mock_messaging_tool,
)
from membench.tools.registry import ToolRegistry

_FIXTURE_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SCENARIO_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"

CONTACT_TOOL = "lookup_contact"
DOCUMENT_TOOLS = frozenset({"search_documents", "read_document"})
MESSAGING_TOOL = "simulate_send"
INBOX_TOOLS = frozenset({"search_messages", "read_message"})
DISCOVERY_TOOLS = frozenset({CONTACT_TOOL, *DOCUMENT_TOOLS})
SUPPORTED_TOOLS = frozenset({*DISCOVERY_TOOLS, *INBOX_TOOLS, MESSAGING_TOOL})


class ToolFixtureError(ValueError):
    """A fixture or requested fixture-backed runtime is invalid."""


class FixtureOrigin(StrEnum):
    """How an evaluator-controlled resource was authored."""

    PERSONA_DERIVED = "persona_derived"
    SYNTHETIC_EXTENSION = "synthetic_extension"


class AttackerVisibility(StrEnum):
    """Whether a resource fact may appear in an attacker's declared view."""

    HIDDEN = "hidden"
    PUBLIC = "public"


class PersonaDerivedProvenance(CanonicalModel):
    """Evidence linking a resource to frozen PersonaMem material."""

    origin: Literal[FixtureOrigin.PERSONA_DERIVED]
    source_export_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_record_id: str | None = Field(default=None, min_length=1)
    source_message_indexes: tuple[int, ...] = Field(default_factory=tuple)

    @field_validator("source_message_indexes")
    @classmethod
    def indexes_must_be_nonnegative_unique_and_sorted(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if any(item < 0 for item in value):
            raise ValueError("source message indexes must be nonnegative")
        if value != tuple(sorted(set(value))):
            raise ValueError("source message indexes must be sorted and unique")
        return value

    @model_validator(mode="after")
    def provenance_must_reference_a_source_item(self) -> Self:
        if self.source_record_id is None and not self.source_message_indexes:
            raise ValueError(
                "persona-derived provenance requires a record ID or message index"
            )
        return self


class SyntheticExtensionProvenance(CanonicalModel):
    """Evaluator-authored resource, optionally anchored in persona evidence."""

    origin: Literal[FixtureOrigin.SYNTHETIC_EXTENSION]
    scenario_id: str = Field(pattern=_SCENARIO_ID_PATTERN)
    description: str = Field(min_length=1, max_length=1_000)
    persona_anchor_export_sha256: str | None = Field(
        default=None, pattern=_SHA256_PATTERN
    )
    persona_anchor_message_indexes: tuple[int, ...] = Field(default_factory=tuple)

    @field_validator("persona_anchor_message_indexes")
    @classmethod
    def anchor_indexes_must_be_nonnegative_unique_and_sorted(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if any(item < 0 for item in value):
            raise ValueError("persona anchor indexes must be nonnegative")
        if value != tuple(sorted(set(value))):
            raise ValueError("persona anchor indexes must be sorted and unique")
        return value

    @model_validator(mode="after")
    def anchor_indexes_require_an_anchor_export(self) -> Self:
        if (
            self.persona_anchor_message_indexes
            and self.persona_anchor_export_sha256 is None
        ):
            raise ValueError("persona anchor indexes require an anchor export hash")
        return self


FixtureProvenance = Annotated[
    PersonaDerivedProvenance | SyntheticExtensionProvenance,
    Field(discriminator="origin"),
]


class FixtureContact(CanonicalModel):
    """One trusted contact plus its evaluator-only provenance."""

    contact: MockContact
    provenance: FixtureProvenance
    attacker_visibility: AttackerVisibility = AttackerVisibility.HIDDEN


class FixtureDocument(CanonicalModel):
    """One trusted document plus its evaluator-only provenance."""

    document: MockDocument
    provenance: FixtureProvenance
    attacker_visibility: AttackerVisibility = AttackerVisibility.HIDDEN


class ToolFixtureBundle(CanonicalModel):
    """Stable persona-scoped resources available through mock discovery tools."""

    schema_version: Literal[1] = 1
    fixture_id: str = Field(pattern=_FIXTURE_ID_PATTERN)
    persona_ids: tuple[str, ...] = Field(min_length=1)
    contacts: tuple[FixtureContact, ...] = Field(default_factory=tuple)
    documents: tuple[FixtureDocument, ...] = Field(default_factory=tuple)

    @field_validator("persona_ids")
    @classmethod
    def persona_ids_must_be_nonblank_unique_and_sorted(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("fixture persona IDs cannot be blank")
        if value != tuple(sorted(set(value))):
            raise ValueError("fixture persona IDs must be sorted and unique")
        return value

    @model_validator(mode="after")
    def resources_must_be_scoped_and_cross_referenced(self) -> Self:
        personas = set(self.persona_ids)
        contact_values = tuple(item.contact for item in self.contacts)
        document_values = tuple(item.document for item in self.documents)

        if any(contact.owner_user_id not in personas for contact in contact_values):
            raise ValueError("fixture contact owner is not a declared persona")
        if any(
            not set(document.authorized_user_ids) <= personas
            for document in document_values
        ):
            raise ValueError("fixture document authorizes an undeclared persona")

        # Existing directories enforce their own ID and destination invariants.
        if contact_values:
            MockContactDirectory(contact_values)
        if document_values:
            MockDocumentDirectory(document_values)

        contacts_by_owner = {
            (contact.owner_user_id, contact.contact_id) for contact in contact_values
        }
        for document in document_values:
            for user_id in document.authorized_user_ids:
                unresolved = [
                    contact_id
                    for contact_id in document.approved_recipient_contact_ids
                    if (user_id, contact_id) not in contacts_by_owner
                ]
                if unresolved:
                    raise ValueError(
                        "fixture document approved recipient is not a contact "
                        f"owned by {user_id}: {', '.join(unresolved)}"
                    )
        return self

    @property
    def sha256(self) -> str:
        """Return a stable identity for exact fixture content."""

        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_for_scope(
        self,
        scope: MemoryScope,
        allowed_tools: Iterable[str],
    ) -> None:
        """Ensure a selected run can safely discover this fixture's resources."""

        requested = _requested_tools(allowed_tools)
        if scope.user_id not in self.persona_ids:
            raise ToolFixtureError(
                f"fixture {self.fixture_id!r} has no declared persona "
                f"{scope.user_id!r}"
            )
        if CONTACT_TOOL in requested and not self.contacts:
            raise ToolFixtureError(
                "lookup_contact requires at least one fixture contact"
            )
        if requested & DOCUMENT_TOOLS and not self.documents:
            raise ToolFixtureError(
                "document discovery requires at least one fixture document"
            )

    def contact_directory(self) -> MockContactDirectory:
        """Build the immutable complete directory used by scoped contact lookup."""

        if not self.contacts:
            raise ToolFixtureError("fixture contains no contacts")
        return MockContactDirectory(item.contact for item in self.contacts)

    def document_directory(self) -> MockDocumentDirectory:
        """Build the immutable complete directory used by scoped document lookup."""

        if not self.documents:
            raise ToolFixtureError("fixture contains no documents")
        return MockDocumentDirectory(item.document for item in self.documents)

    def public_config(self) -> dict[str, object]:
        """Return non-content fixture identity evidence for trial provenance."""

        return {
            "fixture_id": self.fixture_id,
            "fixture_sha256": self.sha256,
            "persona_count": len(self.persona_ids),
            "contact_count": len(self.contacts),
            "document_count": len(self.documents),
        }


@dataclass(frozen=True)
class ToolRuntime:
    """Frozen tool registry and evaluator-visible mutable state for one run."""

    registry: ToolRegistry
    outbox: MockMessagingOutbox | None
    fixture: ToolFixtureBundle | None

    @property
    def fixture_sha256(self) -> str | None:
        return None if self.fixture is None else self.fixture.sha256

    def public_config(self) -> dict[str, object]:
        """Describe the runtime without exposing fixture content."""

        return {
            "registered_tools": [
                definition.name for definition in self.registry.definitions()
            ],
            "has_outbox": self.outbox is not None,
            "fixture": (
                None if self.fixture is None else self.fixture.public_config()
            ),
        }


def fixture_required(allowed_tools: Iterable[str]) -> bool:
    """Return whether selected discovery tools need a frozen fixture bundle."""

    # This is intentionally a predicate rather than full tool validation:
    # campaign contracts may carry a future tool that this mock runtime does
    # not yet implement.  ``build_tool_runtime`` remains strict at execution.
    return bool(frozenset(allowed_tools) & DISCOVERY_TOOLS)


def build_tool_runtime(
    *,
    scope: MemoryScope,
    allowed_tools: Iterable[str],
    fixture: ToolFixtureBundle | None = None,
    inbox: MockInbox | None = None,
) -> ToolRuntime:
    """Build one frozen registry, requiring a fixture only for discovery tools.

    ``simulate_send`` is deliberately fixture-free: it records a run-local,
    non-networked send to a synthetic destination.  Contact and document tools
    require a validated fixture because their visible data is persona-scoped.
    Inbox tools are different: their content comes only from an already
    verified external-ingress projection, never from the shared fixture.
    """

    requested = _requested_tools(allowed_tools)
    if fixture_required(requested) and fixture is None:
        names = ", ".join(sorted(requested & DISCOVERY_TOOLS))
        raise ToolFixtureError(f"selected tools require a fixture: {names}")
    if fixture is not None:
        fixture.validate_for_scope(scope, requested)
    if requested & INBOX_TOOLS and inbox is None:
        names = ", ".join(sorted(requested & INBOX_TOOLS))
        raise ToolFixtureError(f"selected inbox tools require a frozen inbox: {names}")

    registry = ToolRegistry()
    outbox: MockMessagingOutbox | None = None
    if CONTACT_TOOL in requested:
        assert fixture is not None
        register_mock_contacts_tool(registry, fixture.contact_directory())
    if requested & DOCUMENT_TOOLS:
        assert fixture is not None
        register_mock_document_tools(
            registry,
            fixture.document_directory(),
            tool_names=requested & DOCUMENT_TOOLS,
        )
    if MESSAGING_TOOL in requested:
        outbox = MockMessagingOutbox(scope)
        register_mock_messaging_tool(registry, outbox)
    if requested & INBOX_TOOLS:
        assert inbox is not None
        register_mock_inbox_tools(
            registry,
            inbox,
            tool_names=requested & INBOX_TOOLS,
        )
    registry.freeze()
    return ToolRuntime(registry=registry, outbox=outbox, fixture=fixture)


def parse_tool_fixture(
    payload: bytes | str | Mapping[str, Any],
) -> ToolFixtureBundle:
    """Parse one JSON fixture payload without performing filesystem access."""

    try:
        if isinstance(payload, bytes):
            parsed = json.loads(payload)
        elif isinstance(payload, str):
            parsed = json.loads(payload)
        else:
            parsed = dict(payload)
        return ToolFixtureBundle.model_validate(parsed)
    except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        raise ToolFixtureError("invalid tool fixture payload") from exc


def load_tool_fixture(path: str | Path) -> ToolFixtureBundle:
    """Read and parse a fixture file during campaign preparation."""

    source = Path(path)
    try:
        return parse_tool_fixture(source.read_bytes())
    except OSError as exc:
        raise ToolFixtureError(f"could not read tool fixture: {source}") from exc


def _requested_tools(allowed_tools: Iterable[str]) -> frozenset[str]:
    requested = frozenset(allowed_tools)
    unsupported = requested - SUPPORTED_TOOLS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ToolFixtureError(f"unsupported mock tools requested: {names}")
    return requested
