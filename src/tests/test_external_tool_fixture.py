"""Regression checks for the initial short-persona-derived tool fixture."""

from __future__ import annotations

from pathlib import Path

from membench.tools.fixtures import FixtureOrigin, load_tool_fixture

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "fixtures"
    / "external-tool-pilot-v1.json"
)


def test_external_tool_pilot_fixture_is_valid_and_persona_scoped() -> None:
    """The initial three-persona fixture remains safe and deterministic."""

    bundle = load_tool_fixture(FIXTURE_PATH)

    assert bundle.fixture_id == "external-tool-pilot-v1"
    assert bundle.persona_ids == ("277", "371", "436")
    assert len(bundle.contacts) == 6
    assert len(bundle.documents) == 3
    assert {contact.contact.contact_id for contact in bundle.contacts} == {
        "advocacy-case-desk",
        "exchange-support-desk",
        "external-case-review",
        "external-quality-review",
        "external-scholarship-review",
        "local-sourcing-desk",
    }
    assert all(
        contact.contact.email is not None
        and contact.contact.email.endswith(".test")
        and contact.provenance.origin is FixtureOrigin.SYNTHETIC_EXTENSION
        for contact in bundle.contacts
    )


def test_external_tool_pilot_documents_approve_only_trusted_contacts() -> None:
    """Attacker-controlled contacts are discoverable but never pre-approved."""

    bundle = load_tool_fixture(FIXTURE_PATH)
    approved_by_persona = {
        document.document.authorized_user_ids[0]: (
            document.document.approved_recipient_contact_ids
        )
        for document in bundle.documents
    }

    assert approved_by_persona == {
        "277": ("advocacy-case-desk",),
        "371": ("exchange-support-desk",),
        "436": ("local-sourcing-desk",),
    }
    assert all(
        "unapproved external contact" == contact.contact.relationship
        for contact in bundle.contacts
        if contact.contact.contact_id.startswith("external-")
    )
