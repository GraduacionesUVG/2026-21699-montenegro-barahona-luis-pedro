import pytest
from pydantic import ValidationError

from membench.agent.models import (
    AgentDataClassification,
    AgentToolExecutionContext,
)
from membench.memory.models import MemoryScope
from membench.tools.mock_documents import (
    READ_DOCUMENT_DEFINITION,
    SEARCH_DOCUMENTS_DEFINITION,
    DocumentClassification,
    MockDocument,
    MockDocumentConfigurationError,
    MockDocumentDirectory,
    MockDocumentsTool,
    register_mock_document_tools,
    validate_read_document_arguments,
    validate_search_documents_arguments,
)
from membench.tools.models import ToolCall, ToolSideEffectKind, ToolStatus
from membench.tools.registry import ToolRegistry


def _documents() -> tuple[MockDocument, ...]:
    return (
        MockDocument(
            document_id="security-runbook",
            authorized_user_ids=("persona-001",),
            title="Synthetic Security Runbook",
            content="Report synthetic incidents to the approved security contact.",
            classification=DocumentClassification.SYNTHETIC_SENSITIVE,
            version=2,
            approved=True,
            tags=("runbook", "security"),
            approved_recipient_contact_ids=("security-team",),
        ),
        MockDocument(
            document_id="travel-itinerary",
            authorized_user_ids=("persona-001",),
            title="Synthetic Travel Itinerary",
            content="The synthetic train departs at 09:30.",
            classification=DocumentClassification.INTERNAL,
            tags=("travel",),
        ),
        MockDocument(
            document_id="other-private-note",
            authorized_user_ids=("persona-002",),
            title="Other Persona Private Note",
            content="This synthetic content must not cross persona boundaries.",
            classification=DocumentClassification.SYNTHETIC_SENSITIVE,
            tags=("security",),
        ),
    )


def _context(user_id: str = "persona-001") -> AgentToolExecutionContext:
    return AgentToolExecutionContext(
        task_id="task-001",
        scope=MemoryScope(
            experiment_id="mock-documents",
            run_id="run-001",
            user_id=user_id,
            namespace=("membench", "tools", "run-001", user_id),
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


def test_document_definitions_are_read_only_without_approval() -> None:
    for definition in (SEARCH_DOCUMENTS_DEFINITION, READ_DOCUMENT_DEFINITION):
        assert definition.side_effect == ToolSideEffectKind.READ_ONLY
        assert definition.requires_approval is False
        assert definition.input_schema["additionalProperties"] is False


def test_registration_can_limit_the_visible_document_operations() -> None:
    registry = ToolRegistry()

    register_mock_document_tools(
        registry,
        MockDocumentDirectory(_documents()),
        tool_names=("read_document",),
    )

    assert [item.name for item in registry.definitions()] == ["read_document"]


def test_document_model_and_directory_are_deterministic() -> None:
    documents = _documents()
    first = MockDocumentDirectory(documents)
    second = MockDocumentDirectory(reversed(documents))

    assert first.documents == second.documents
    assert first.revision_sha256 == second.revision_sha256
    assert len(documents[0].content_sha256) == 64

    with pytest.raises(MockDocumentConfigurationError, match="globally unique"):
        MockDocumentDirectory((*documents, documents[0]))
    with pytest.raises(ValidationError, match="must be unique"):
        MockDocument(
            document_id="duplicate-users",
            authorized_user_ids=("persona-001", "persona-001"),
            title="Invalid fixture",
            content="Synthetic invalid fixture.",
        )


def test_document_argument_validation_is_strict_and_normalized() -> None:
    assert validate_search_documents_arguments(
        {"query": " security ", "classifications": ["internal"]}
    ) == {
        "query": "security",
        "limit": 5,
        "classifications": ["internal"],
    }
    assert validate_read_document_arguments(
        {"document_id": " security-runbook "}
    ) == {"document_id": "security-runbook"}

    with pytest.raises(ValueError, match="unsupported"):
        validate_search_documents_arguments({"query": "x", "owner": "other"})
    with pytest.raises(ValueError, match="unknown classification"):
        validate_search_documents_arguments(
            {"query": "x", "classifications": ["secret"]}
        )
    with pytest.raises(ValueError, match="invalid format"):
        validate_read_document_arguments({"document_id": "../unsafe"})


def test_registry_search_returns_metadata_only_and_is_persona_scoped() -> None:
    registry = ToolRegistry()
    directory = MockDocumentDirectory(_documents())
    implementation = register_mock_document_tools(registry, directory)

    assert isinstance(implementation, MockDocumentsTool)
    assert len(registry) == 2

    result = registry.execute(
        _call("search_documents", {"query": "security"}),
        _context(),
    )

    assert result.status == ToolStatus.SUCCEEDED
    assert result.output["source"] == "trusted_mock_documents_v1"
    assert result.output["returned_count"] == 1
    assert result.output["total_matches"] == 1
    assert result.output["documents"][0]["document_id"] == "security-runbook"
    assert "content" not in result.output["documents"][0]
    assert "other-private-note" not in str(result.output)


def test_search_ranking_filtering_and_truncation_are_deterministic() -> None:
    directory = MockDocumentDirectory(_documents())

    matches, total = directory.search("persona-001", "synthetic", limit=1)
    filtered, filtered_total = directory.search(
        "persona-001",
        "synthetic",
        limit=5,
        classifications=(DocumentClassification.INTERNAL,),
    )

    assert total == 2
    assert len(matches) == 1
    assert [item.document_id for item in filtered] == ["travel-itinerary"]
    assert filtered_total == 1


def test_exact_read_returns_content_and_hash() -> None:
    registry = ToolRegistry()
    directory = MockDocumentDirectory(_documents())
    register_mock_document_tools(registry, directory)

    result = registry.execute(
        _call("read_document", {"document_id": "security-runbook"}),
        _context(),
    )

    assert result.status == ToolStatus.SUCCEEDED
    assert result.output["found"] is True
    record = result.output["document"]
    assert record["content"] == _documents()[0].content
    assert record["content_sha256"] == _documents()[0].content_sha256


def test_missing_and_unauthorized_reads_are_indistinguishable() -> None:
    tool = MockDocumentsTool(MockDocumentDirectory(_documents()))

    unauthorized = tool.invoke(
        _call("read_document", {"document_id": "other-private-note"}),
        _context(),
    )
    missing = tool.invoke(
        _call("read_document", {"document_id": "missing-note"}),
        _context(),
    )

    assert unauthorized == missing
    assert unauthorized["found"] is False
    assert unauthorized["document"] is None
