from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from membench.memory.models import (
    AdapterProvenance,
    DirectWriteRequest,
    EmptyStateResult,
    MemoryMessage,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResult,
    ScoreKind,
)
from membench.memory.protocol import MemoryAdapter


def _scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="experiment-001",
        run_id="run-001",
        user_id="user-001",
        namespace=("membench", "run-001", "user-001"),
    )


def _record() -> MemoryRecord:
    created = datetime(2026, 9, 13, tzinfo=UTC)
    return MemoryRecord(
        canonical_id="memory-001",
        vendor_id="vendor-001",
        scope=_scope(),
        text="The user prefers dark mode.",
        memory_type=MemoryType.SEMANTIC,
        source_event_ids=("event-001",),
        created_at=created,
        updated_at=created + timedelta(seconds=1),
    )


def test_scope_and_requests_are_strict_and_immutable() -> None:
    scope = _scope()
    request = DirectWriteRequest(
        operation_id="write-001",
        scope=scope,
        text="The user prefers dark mode.",
        memory_type=MemoryType.SEMANTIC,
        source_event_id="event-001",
    )

    assert scope.key == "membench/run-001/user-001"
    assert request.metadata == {}
    with pytest.raises(ValidationError):
        request.text = "changed"
    with pytest.raises(ValidationError):
        MemoryScope(
            experiment_id="experiment-001",
            run_id="run-001",
            user_id="user-001",
            namespace=("membench", " "),
        )


def test_retrieval_request_hashes_query_and_result_enforces_ranking() -> None:
    request = RetrievalRequest(
        operation_id="retrieve-001",
        scope=_scope(),
        query="What display theme does the user prefer?",
        top_k=2,
        memory_types=(MemoryType.SEMANTIC,),
    )
    result = RetrievalResult(
        operation_id=request.operation_id,
        query_sha256=request.query_sha256,
        request_top_k=request.top_k,
        hits=(
            RetrievalHit(
                record=_record(),
                rank=1,
                raw_score=0.91,
                score_kind=ScoreKind.SIMILARITY,
            ),
        ),
        latency_seconds=0.01,
    )

    assert len(result.query_sha256) == 64
    assert result.model_dump(mode="json")["hits"][0]["record"]["memory_type"] == (
        "semantic"
    )
    with pytest.raises(ValidationError, match="consecutive"):
        RetrievalResult(
            operation_id=request.operation_id,
            query_sha256=request.query_sha256,
            request_top_k=2,
            hits=(RetrievalHit(record=_record(), rank=2),),
            latency_seconds=0,
        )


def test_records_require_consistent_aware_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        MemoryRecord(
            canonical_id="memory-001",
            scope=_scope(),
            text="A memory",
            created_at=datetime(2026, 9, 13),
        )

    created = datetime(2026, 9, 13, tzinfo=UTC)
    with pytest.raises(ValidationError, match="precede"):
        MemoryRecord(
            canonical_id="memory-001",
            scope=_scope(),
            text="A memory",
            created_at=created,
            updated_at=created - timedelta(seconds=1),
        )


def test_empty_state_requires_counts_to_match_flag() -> None:
    result = EmptyStateResult(
        scope=_scope(),
        empty=True,
        inspected_count=0,
        retrieved_count=0,
    )
    assert result.empty

    with pytest.raises(ValidationError, match="must agree"):
        EmptyStateResult(
            scope=_scope(),
            empty=True,
            inspected_count=1,
            retrieved_count=0,
        )


def test_message_and_provenance_have_minimal_serializable_shapes() -> None:
    message = MemoryMessage(role="user", content="Remember this preference.")
    provenance = AdapterProvenance(
        mechanism="langmem",
        mechanism_version="0.0.30",
        adapter_version="1",
        storage_backend="sqlite",
        model_identifiers={"writer": "qwen2.5:7b"},
        package_versions={"langmem": "0.0.30"},
        namespace=_scope().namespace,
    )

    assert message.model_dump(mode="json")["role"] == "user"
    assert provenance.model_dump(mode="json")["captured_at"].endswith("Z")


def test_memory_adapter_is_a_runtime_checkable_structural_protocol() -> None:
    class IncompleteAdapter:
        pass

    assert not isinstance(IncompleteAdapter(), MemoryAdapter)
