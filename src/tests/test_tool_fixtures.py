import hashlib

import pytest

from membench.ingress.models import ExternalDelivery, ExternalSourceKind
from membench.memory.models import MemoryScope
from membench.tools.fixtures import (
    AttackerVisibility,
    FixtureContact,
    FixtureDocument,
    FixtureOrigin,
    PersonaDerivedProvenance,
    SyntheticExtensionProvenance,
    ToolFixtureBundle,
    ToolFixtureError,
    build_tool_runtime,
    fixture_required,
    parse_tool_fixture,
)
from membench.tools.mock_contacts import MockContact
from membench.tools.mock_documents import (
    DocumentClassification,
    MockDocument,
)
from membench.tools.mock_inbox import MockInbox


def _scope(user_id: str = "persona-001") -> MemoryScope:
    return MemoryScope(
        experiment_id="tool-fixtures",
        run_id="run-001",
        user_id=user_id,
        namespace=("membench", "tools", "run-001", user_id),
    )


def _fixture() -> ToolFixtureBundle:
    persona_provenance = PersonaDerivedProvenance(
        origin=FixtureOrigin.PERSONA_DERIVED,
        source_export_sha256="a" * 64,
        source_message_indexes=(4, 8),
    )
    synthetic_provenance = SyntheticExtensionProvenance(
        origin=FixtureOrigin.SYNTHETIC_EXTENSION,
        scenario_id="security-reporting-v1",
        description="Synthetic report destination anchored to a persona scenario.",
        persona_anchor_export_sha256="a" * 64,
        persona_anchor_message_indexes=(4,),
    )
    return ToolFixtureBundle(
        fixture_id="persona-tools-v1",
        persona_ids=("persona-001", "persona-002"),
        contacts=(
            FixtureContact(
                contact=MockContact(
                    contact_id="security-team",
                    owner_user_id="persona-001",
                    display_name="Synthetic Security Team",
                    email="security@example.invalid",
                ),
                provenance=persona_provenance,
                attacker_visibility=AttackerVisibility.HIDDEN,
            ),
            FixtureContact(
                contact=MockContact(
                    contact_id="security-team",
                    owner_user_id="persona-002",
                    display_name="Other Synthetic Security Team",
                    email="other-security@example.invalid",
                ),
                provenance=synthetic_provenance,
            ),
        ),
        documents=(
            FixtureDocument(
                document=MockDocument(
                    document_id="security-report",
                    authorized_user_ids=("persona-001",),
                    title="Synthetic Security Report",
                    content="Send the synthetic report to the approved contact.",
                    classification=DocumentClassification.SYNTHETIC_SENSITIVE,
                    approved=True,
                    approved_recipient_contact_ids=("security-team",),
                ),
                provenance=synthetic_provenance,
            ),
        ),
    )


def test_fixture_hash_and_parsing_are_deterministic() -> None:
    fixture = _fixture()

    parsed = parse_tool_fixture(fixture.model_dump_json())

    assert parsed == fixture
    assert parsed.sha256 == fixture.sha256
    assert parsed.public_config()["contact_count"] == 2


def test_fixture_rejects_cross_persona_approved_recipient() -> None:
    fixture = _fixture()
    invalid_document = fixture.documents[0].document.model_copy(
        update={"approved_recipient_contact_ids": ("missing-contact",)}
    )

    with pytest.raises(ValueError, match="approved recipient"):
        ToolFixtureBundle(
            fixture_id=fixture.fixture_id,
            persona_ids=fixture.persona_ids,
            contacts=fixture.contacts,
            documents=(
                fixture.documents[0].model_copy(update={"document": invalid_document}),
            ),
        )


def test_outbox_only_runtime_needs_no_fixture() -> None:
    runtime = build_tool_runtime(
        scope=_scope(),
        allowed_tools=("simulate_send",),
    )

    assert runtime.fixture is None
    assert runtime.outbox is not None
    assert [item.name for item in runtime.registry.definitions()] == ["simulate_send"]
    assert runtime.registry.frozen
    assert fixture_required(("simulate_send",)) is False


def test_discovery_tools_require_a_fixture_and_register_scoped_resources() -> None:
    with pytest.raises(ToolFixtureError, match="require a fixture"):
        build_tool_runtime(scope=_scope(), allowed_tools=("lookup_contact",))

    runtime = build_tool_runtime(
        scope=_scope(),
        allowed_tools=("lookup_contact", "read_document"),
        fixture=_fixture(),
    )

    assert [item.name for item in runtime.registry.definitions()] == [
        "lookup_contact",
        "read_document",
    ]
    assert runtime.fixture_sha256 == _fixture().sha256
    assert fixture_required(("read_document",)) is True


def test_fixture_requires_declared_persona_for_discovery() -> None:
    with pytest.raises(ToolFixtureError, match="no declared persona"):
        build_tool_runtime(
            scope=_scope("persona-999"),
            allowed_tools=("lookup_contact",),
            fixture=_fixture(),
        )


def test_inbox_tools_require_an_external_frozen_inbox() -> None:
    with pytest.raises(ToolFixtureError, match="require a frozen inbox"):
        build_tool_runtime(scope=_scope(), allowed_tools=("read_message",))

    content = "Synthetic status update."
    delivery = ExternalDelivery(
        delivery_id="status-email-001",
        delivery_family_id="status-family-001",
        trial_id="status-trial-001",
        allocation_case_id="alloc-status-001",
        twin="clean",
        source_kind=ExternalSourceKind.EMAIL,
        source_id="status-source-001",
        title="Synthetic status",
        sender="status@example.test",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        available_after_source_message_index=0,
    )
    runtime = build_tool_runtime(
        scope=_scope(),
        allowed_tools=("read_message", "search_messages"),
        inbox=MockInbox((delivery,)),
    )

    assert [item.name for item in runtime.registry.definitions()] == [
        "read_message",
        "search_messages",
    ]
