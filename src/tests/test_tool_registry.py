import pytest

from membench.agent.models import (
    AgentDataClassification,
    AgentToolExecutionContext,
)
from membench.memory.models import JsonObject, MemoryScope
from membench.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolStatus,
)
from membench.tools.protocol import ToolExecutor
from membench.tools.registry import (
    DuplicateToolError,
    FrozenToolRegistryError,
    ToolNotRegisteredError,
    ToolRegistry,
)


def _context() -> AgentToolExecutionContext:
    return AgentToolExecutionContext(
        task_id="task-001",
        scope=MemoryScope(
            experiment_id="tool-registry-tests",
            run_id="run-001",
            user_id="persona-001",
            namespace=("membench", "tools", "run-001"),
        ),
        step_index=1,
        data_classification=AgentDataClassification.SYNTHETIC_SENSITIVE,
    )


def _definition(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Synthetic {name} tool.",
        input_schema={"type": "object"},
    )


def _call(name: str = "lookup_contact") -> ToolCall:
    return ToolCall(
        call_id="call-001",
        tool_name=name,
        arguments={"persona_id": " persona-001 "},
        idempotency_key="task-001:call-001",
    )


class RecordingTool:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.contexts: list[AgentToolExecutionContext] = []

    def invoke(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        self.calls.append(call)
        self.contexts.append(context)
        return {
            "persona_id": call.arguments["persona_id"],
            "destination": "alpha@example.invalid",
        }


class FailingTool:
    def invoke(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        raise RuntimeError("synthetic tool failure")


class InvalidOutputTool:
    def invoke(
        self,
        call: ToolCall,
        context: AgentToolExecutionContext,
    ) -> JsonObject:
        return ["not", "an", "object"]  # type: ignore[return-value]


def _normalize_arguments(arguments: JsonObject) -> JsonObject:
    persona_id = arguments.get("persona_id")
    if not isinstance(persona_id, str) or not persona_id.strip():
        raise ValueError("persona_id must be a non-blank string")
    return {"persona_id": persona_id.strip()}


def test_registry_resolves_definitions_in_deterministic_order() -> None:
    registry = ToolRegistry()
    lookup = RecordingTool()
    registry.register(_definition("simulate_send"), RecordingTool())
    registry.register(_definition("lookup_contact"), lookup)

    assert isinstance(registry, ToolExecutor)
    assert len(registry) == 2
    assert "lookup_contact" in registry
    assert tuple(item.name for item in registry.definitions()) == (
        "lookup_contact",
        "simulate_send",
    )
    assert registry.resolve("lookup_contact") == _definition("lookup_contact")
    assert registry.resolve("unknown") is None
    assert registry.registration_for("lookup_contact").implementation is lookup


def test_registry_validates_invokes_and_normalizes_successful_results() -> None:
    registry = ToolRegistry()
    implementation = RecordingTool()
    registry.register(
        _definition("lookup_contact"),
        implementation,
        argument_validator=_normalize_arguments,
    )
    context = _context()

    result = registry.execute(_call(), context)

    assert isinstance(result, ToolResult)
    assert result.status == ToolStatus.SUCCEEDED
    assert result.call_id == "call-001"
    assert result.tool_name == "lookup_contact"
    assert result.output == {
        "persona_id": "persona-001",
        "destination": "alpha@example.invalid",
    }
    assert result.latency_seconds >= 0
    assert implementation.calls[0].arguments == {"persona_id": "persona-001"}
    assert implementation.contexts == [context]


@pytest.mark.parametrize(
    ("implementation", "expected_error"),
    [
        (FailingTool(), "RuntimeError"),
        (InvalidOutputTool(), "TypeError"),
    ],
)
def test_registry_converts_implementation_failures_to_canonical_results(
    implementation: object,
    expected_error: str,
) -> None:
    registry = ToolRegistry()
    registry.register(_definition("lookup_contact"), implementation)  # type: ignore[arg-type]

    result = registry.execute(_call(), _context())

    assert result.status == ToolStatus.FAILED
    assert result.error_type == expected_error
    assert result.error_message
    assert result.output == {}


def test_registry_rejects_duplicates_unknown_tools_and_post_freeze_changes() -> None:
    registry = ToolRegistry()
    registry.register(_definition("lookup_contact"), RecordingTool())

    with pytest.raises(DuplicateToolError, match="already registered"):
        registry.register(_definition("lookup_contact"), RecordingTool())
    with pytest.raises(ToolNotRegisteredError, match="not registered"):
        registry.registration_for("unknown")
    with pytest.raises(ToolNotRegisteredError, match="not registered"):
        registry.execute(_call("unknown"), _context())

    registry.freeze()

    assert registry.frozen
    with pytest.raises(FrozenToolRegistryError, match="frozen"):
        registry.register(_definition("simulate_send"), RecordingTool())


def test_argument_validation_failure_does_not_invoke_the_tool() -> None:
    registry = ToolRegistry()
    implementation = RecordingTool()
    registry.register(
        _definition("lookup_contact"),
        implementation,
        argument_validator=_normalize_arguments,
    )
    invalid = _call().model_copy(update={"arguments": {"persona_id": ""}})

    result = registry.execute(invalid, _context())

    assert result.status == ToolStatus.FAILED
    assert result.error_type == "ValueError"
    assert not implementation.calls
