import pytest

from membench.agent.models import (
    AgentDataClassification,
    AgentToolExecutionContext,
)
from membench.memory.models import MemoryScope
from membench.tools.mock_messaging import (
    SIMULATE_SEND_DEFINITION,
    MockMessagingOutbox,
    MockMessagingScopeError,
    MockMessagingTool,
    register_mock_messaging_tool,
    validate_simulate_send_arguments,
)
from membench.tools.models import ToolCall, ToolSideEffectKind, ToolStatus
from membench.tools.registry import ToolRegistry


def _scope(run_id: str = "run-001") -> MemoryScope:
    return MemoryScope(
        experiment_id="mock-messaging",
        run_id=run_id,
        user_id="persona-001",
        namespace=("membench", "tools", run_id, "persona-001"),
    )


def _context(scope: MemoryScope | None = None) -> AgentToolExecutionContext:
    return AgentToolExecutionContext(
        task_id="task-001",
        scope=scope or _scope(),
        step_index=1,
        data_classification=AgentDataClassification.SYNTHETIC_SENSITIVE,
    )


def _call(
    *,
    recipient: str = "security@example.invalid",
    body: str = "Synthetic report attached.",
    idempotency_key: str = "task-001:send-001",
) -> ToolCall:
    return ToolCall(
        call_id="call-001",
        tool_name="simulate_send",
        arguments={
            "recipient": recipient,
            "subject": "Synthetic report",
            "body": body,
            "attachment_ids": ["report-001"],
        },
        idempotency_key=idempotency_key,
    )


def test_simulated_send_definition_is_safe_and_requires_approval() -> None:
    assert SIMULATE_SEND_DEFINITION.side_effect == (
        ToolSideEffectKind.SIMULATED_WRITE
    )
    assert SIMULATE_SEND_DEFINITION.requires_approval is True
    assert SIMULATE_SEND_DEFINITION.input_schema["additionalProperties"] is False


def test_argument_validation_rejects_real_destinations_and_empty_payloads() -> None:
    normalized = validate_simulate_send_arguments(
        {
            "recipient": " SECURITY@EXAMPLE.INVALID ",
            "body": "Synthetic body",
        }
    )
    assert normalized["recipient"] == "security@example.invalid"

    with pytest.raises(ValueError, match="reserved"):
        validate_simulate_send_arguments(
            {"recipient": "real@example.com", "body": "Unsafe"}
        )
    with pytest.raises(ValueError, match="body or at least one attachment"):
        validate_simulate_send_arguments(
            {"recipient": "security@example.invalid"}
        )
    with pytest.raises(ValueError, match="unique"):
        validate_simulate_send_arguments(
            {
                "recipient": "security@example.invalid",
                "attachment_ids": ["report-001", "report-001"],
            }
        )


def test_registry_records_simulated_send_without_network_side_effects() -> None:
    scope = _scope()
    outbox = MockMessagingOutbox(scope)
    registry = ToolRegistry()
    implementation = register_mock_messaging_tool(registry, outbox)

    assert isinstance(implementation, MockMessagingTool)
    result = registry.execute(_call(), _context(scope))

    assert result.status == ToolStatus.SUCCEEDED
    assert result.output["status"] == "simulated"
    assert result.output["recipient"] == "security@example.invalid"
    assert result.output["replayed"] is False
    assert outbox.message_count == 1
    snapshot = outbox.snapshot()
    assert len(snapshot.messages) == 1
    assert snapshot.messages[0].body == "Synthetic report attached."
    assert snapshot.messages[0].attachment_ids == ("report-001",)
    assert snapshot.sha256


def test_idempotent_retry_replays_once_and_conflicting_retry_fails() -> None:
    scope = _scope()
    outbox = MockMessagingOutbox(scope)
    registry = ToolRegistry()
    register_mock_messaging_tool(registry, outbox)
    call = _call()

    first = registry.execute(call, _context(scope))
    replay = registry.execute(call, _context(scope))
    conflict = registry.execute(
        _call(body="Different synthetic content"),
        _context(scope),
    )

    assert first.status == ToolStatus.SUCCEEDED
    assert replay.status == ToolStatus.SUCCEEDED
    assert replay.output["message_id"] == first.output["message_id"]
    assert replay.output["replayed"] is True
    assert conflict.status == ToolStatus.FAILED
    assert conflict.error_type == "MockMessagingConflictError"
    assert outbox.message_count == 1


def test_outbox_rejects_cross_run_contexts() -> None:
    outbox = MockMessagingOutbox(_scope("run-001"))
    tool = MockMessagingTool(outbox)

    with pytest.raises(MockMessagingScopeError, match="scope"):
        tool.invoke(_call(), _context(_scope("run-002")))
