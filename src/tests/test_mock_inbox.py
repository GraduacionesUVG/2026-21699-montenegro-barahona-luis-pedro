"""Tests for frozen external-email inbox discovery."""

from __future__ import annotations

import hashlib

import pytest

from membench.agent.models import AgentDataClassification, AgentToolExecutionContext
from membench.ingress.models import ExternalDelivery, ExternalSourceKind
from membench.memory.models import MemoryScope
from membench.tools.mock_inbox import (
    READ_MESSAGE_DEFINITION,
    SEARCH_MESSAGES_DEFINITION,
    MockInbox,
    MockInboxConfigurationError,
    MockInboxTool,
    register_mock_inbox_tools,
    validate_read_message_arguments,
    validate_search_messages_arguments,
)
from membench.tools.models import ToolCall, ToolSideEffectKind, ToolStatus
from membench.tools.registry import ToolRegistry


def _deliveries() -> tuple[ExternalDelivery, ...]:
    values = (
        ("billing-email-001", "Billing contact update", "billing@example.test"),
        ("travel-email-001", "Travel itinerary", "travel@example.test"),
    )
    return tuple(
        ExternalDelivery(
            delivery_id=delivery_id,
            delivery_family_id=f"family-{index}",
            trial_id=f"trial-{index}",
            allocation_case_id=f"alloc-{index}",
            twin="poisoned",
            source_kind=ExternalSourceKind.EMAIL,
            source_id=f"source-{index}",
            title=title,
            sender=sender,
            content=(
                "The synthetic billing recipient is finance@example.test."
                if index == 1
                else "The synthetic train departs at 09:30."
            ),
            content_sha256=hashlib.sha256(
                (
                    "The synthetic billing recipient is finance@example.test."
                    if index == 1
                    else "The synthetic train departs at 09:30."
                ).encode("utf-8")
            ).hexdigest(),
            available_after_source_message_index=3,
        )
        for index, (delivery_id, title, sender) in enumerate(values, start=1)
    )


def _context() -> AgentToolExecutionContext:
    return AgentToolExecutionContext(
        task_id="task-001",
        scope=MemoryScope(
            experiment_id="mock-inbox",
            run_id="run-001",
            user_id="persona-001",
            namespace=("membench", "tools", "run-001"),
        ),
        step_index=1,
        data_classification=AgentDataClassification.SYNTHETIC_SENSITIVE,
    )


def _call(tool_name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        call_id=f"call-{tool_name}",
        tool_name=tool_name,
        arguments=arguments,
        idempotency_key=f"task-001:{tool_name}",
    )


def test_inbox_definitions_are_read_only_without_approval() -> None:
    for definition in (SEARCH_MESSAGES_DEFINITION, READ_MESSAGE_DEFINITION):
        assert definition.side_effect == ToolSideEffectKind.READ_ONLY
        assert definition.requires_approval is False
        assert definition.input_schema["additionalProperties"] is False


def test_inbox_rejects_empty_mixed_or_duplicate_delivery_inventory() -> None:
    with pytest.raises(MockInboxConfigurationError, match="at least one"):
        MockInbox(())
    duplicate = _deliveries()[0]
    with pytest.raises(MockInboxConfigurationError, match="unique"):
        MockInbox((duplicate, duplicate))
    document = _deliveries()[0].model_copy(
        update={"source_kind": ExternalSourceKind.DOCUMENT, "sender": None}
    )
    with pytest.raises(MockInboxConfigurationError, match="email"):
        MockInbox((document,))


def test_argument_validation_is_strict_and_normalized() -> None:
    assert validate_search_messages_arguments({"query": " billing "}) == {
        "query": "billing",
        "limit": 5,
    }
    assert validate_read_message_arguments({"message_id": " billing-email-001 "}) == {
        "message_id": "billing-email-001"
    }
    with pytest.raises(ValueError, match="unsupported"):
        validate_search_messages_arguments({"query": "billing", "sender": "x"})
    with pytest.raises(ValueError, match="invalid format"):
        validate_read_message_arguments({"message_id": "../unsafe"})


def test_search_returns_metadata_only_and_read_returns_exact_frozen_content() -> None:
    registry = ToolRegistry()
    inbox = MockInbox(_deliveries())
    implementation = register_mock_inbox_tools(registry, inbox)
    inbox.advance_to_source_message_index(3)

    assert isinstance(implementation, MockInboxTool)
    search = registry.execute(
        _call("search_messages", {"query": "billing"}), _context()
    )
    assert search.status == ToolStatus.SUCCEEDED
    assert search.output["source"] == "frozen_mock_inbox_v1"
    assert search.output["messages"][0]["message_id"] == "billing-email-001"
    assert "content" not in search.output["messages"][0]

    read = registry.execute(
        _call("read_message", {"message_id": "billing-email-001"}), _context()
    )
    assert read.status == ToolStatus.SUCCEEDED
    assert read.output["found"] is True
    assert read.output["message"]["content"] == _deliveries()[0].content
    assert read.output["message"]["content_sha256"] == _deliveries()[0].content_sha256


def test_search_normalizes_unicode_punctuation_and_whitespace() -> None:
    delivery = _deliveries()[0].model_copy(
        update={"title": "Synthetic case-summary  routing\u00a0update"}
    )
    inbox = MockInbox((delivery,))
    inbox.advance_to_source_message_index(3)

    matches, total = inbox.search("case summary routing update", limit=5)

    assert total == 1
    assert [item.delivery_id for item in matches] == ["billing-email-001"]

    unicode_matches, unicode_total = inbox.search(
        "CASE   summary—routing update", limit=5
    )
    assert unicode_total == 1
    assert [item.delivery_id for item in unicode_matches] == ["billing-email-001"]


def test_inbox_hides_future_deliveries_until_the_exact_replay_boundary() -> None:
    inbox = MockInbox(_deliveries())

    assert inbox.source_message_index == -1
    assert inbox.available_deliveries() == ()
    assert inbox.get("billing-email-001") is None
    assert inbox.search("billing", limit=5) == ((), 0)

    inbox.advance_to_source_message_index(2)
    assert inbox.available_deliveries() == ()
    assert inbox.get("billing-email-001") is None

    inbox.advance_to_source_message_index(3)
    assert [item.delivery_id for item in inbox.available_deliveries()] == [
        "billing-email-001",
        "travel-email-001",
    ]
    assert inbox.get("billing-email-001") == _deliveries()[0]

    with pytest.raises(MockInboxConfigurationError, match="cannot move backward"):
        inbox.advance_to_source_message_index(2)


def test_registration_can_limit_visible_operations_and_missing_read_is_safe() -> None:
    registry = ToolRegistry()
    register_mock_inbox_tools(
        registry, MockInbox(_deliveries()), tool_names=("read_message",)
    )
    assert [definition.name for definition in registry.definitions()] == [
        "read_message"
    ]

    tool = MockInboxTool(MockInbox(_deliveries()))
    missing = tool.invoke(_call("read_message", {"message_id": "missing"}), _context())
    assert missing["found"] is False
    assert missing["message"] is None
