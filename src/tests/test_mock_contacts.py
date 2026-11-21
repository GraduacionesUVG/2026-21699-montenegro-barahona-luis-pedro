import pytest
from pydantic import ValidationError

from membench.agent.models import (
    AgentDataClassification,
    AgentToolExecutionContext,
)
from membench.memory.models import MemoryScope
from membench.tools.mock_contacts import (
    LOOKUP_CONTACT_DEFINITION,
    MockContact,
    MockContactConfigurationError,
    MockContactDirectory,
    MockContactsTool,
    register_mock_contacts_tool,
    validate_lookup_contact_arguments,
)
from membench.tools.models import ToolCall, ToolSideEffectKind, ToolStatus
from membench.tools.registry import ToolRegistry


def _contacts() -> tuple[MockContact, ...]:
    return (
        MockContact(
            contact_id="security-team",
            owner_user_id="persona-001",
            display_name="Synthetic Security Team",
            email="security@example.invalid",
            phone="+1-555-0101",
            organization="Example Research",
            relationship="security contact",
            aliases=("Incident Desk", "Security"),
        ),
        MockContact(
            contact_id="travel-agent",
            owner_user_id="persona-001",
            display_name="Synthetic Travel Agent",
            email="travel@example.test",
            phone="+1-555-0102",
            relationship="travel agent",
        ),
        MockContact(
            contact_id="security-team",
            owner_user_id="persona-002",
            display_name="Other Persona Security Team",
            email="other-security@example.invalid",
            phone="+1-555-0103",
        ),
    )


def _context(user_id: str = "persona-001") -> AgentToolExecutionContext:
    return AgentToolExecutionContext(
        task_id="task-001",
        scope=MemoryScope(
            experiment_id="mock-contacts",
            run_id="run-001",
            user_id=user_id,
            namespace=("membench", "tools", "run-001", user_id),
        ),
        step_index=1,
        data_classification=AgentDataClassification.SYNTHETIC_SENSITIVE,
    )


def _call(arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        call_id="call-001",
        tool_name="lookup_contact",
        arguments=arguments,
        idempotency_key="task-001:call-001",
    )


def test_mock_contacts_require_non_routable_destinations() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        MockContact(
            contact_id="real-contact",
            owner_user_id="persona-001",
            display_name="Unsafe Contact",
            email="person@example.com",
        )
    with pytest.raises(ValidationError, match="fictional phone"):
        MockContact(
            contact_id="missing-destination",
            owner_user_id="persona-001",
            display_name="Missing Destination",
        )


def test_directory_is_deterministic_and_rejects_ambiguous_destinations() -> None:
    contacts = _contacts()
    first = MockContactDirectory(contacts)
    second = MockContactDirectory(reversed(contacts))

    assert first.contacts == second.contacts
    assert first.revision_sha256 == second.revision_sha256

    duplicate = contacts[0].model_copy(update={"contact_id": "security-backup"})
    with pytest.raises(MockContactConfigurationError, match="destinations"):
        MockContactDirectory((*contacts, duplicate))


def test_argument_validation_is_strict_and_normalizes_strings() -> None:
    assert validate_lookup_contact_arguments(
        {"contact_id": " security-team "}
    ) == {"contact_id": "security-team"}
    assert validate_lookup_contact_arguments({"query": " Security "}) == {
        "query": "Security",
        "limit": 5,
    }

    with pytest.raises(ValueError, match="exactly one"):
        validate_lookup_contact_arguments(
            {"contact_id": "security-team", "query": "security"}
        )
    with pytest.raises(ValueError, match="unsupported"):
        validate_lookup_contact_arguments({"query": "security", "owner": "other"})
    with pytest.raises(ValueError, match="integer"):
        validate_lookup_contact_arguments({"query": "security", "limit": True})


def test_registry_lookup_is_read_only_and_scoped_to_current_persona() -> None:
    registry = ToolRegistry()
    directory = MockContactDirectory(_contacts())
    implementation = register_mock_contacts_tool(registry, directory)

    assert isinstance(implementation, MockContactsTool)
    assert LOOKUP_CONTACT_DEFINITION.side_effect == ToolSideEffectKind.READ_ONLY
    assert LOOKUP_CONTACT_DEFINITION.requires_approval is False

    result = registry.execute(_call({"query": "security"}), _context())

    assert result.status == ToolStatus.SUCCEEDED
    assert result.output["source"] == "trusted_mock_contacts_v1"
    assert result.output["returned_count"] == 1
    assert result.output["contacts"][0]["email"] == "security@example.invalid"
    assert "other-security@example.invalid" not in str(result.output)

    other = registry.execute(
        _call({"contact_id": "security-team"}),
        _context("persona-002"),
    )
    assert other.status == ToolStatus.SUCCEEDED
    assert other.output["contacts"][0]["email"] == (
        "other-security@example.invalid"
    )


def test_exact_lookup_does_not_fall_back_to_memory_or_another_owner() -> None:
    tool = MockContactsTool(MockContactDirectory(_contacts()))

    output = tool.invoke(
        _call({"contact_id": "unknown-contact"}),
        _context(),
    )

    assert output["found"] is False
    assert output["returned_count"] == 0
    assert output["contacts"] == []
