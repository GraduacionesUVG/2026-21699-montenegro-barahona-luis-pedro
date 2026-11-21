import hashlib
from datetime import UTC, datetime

import pytest

from membench.memory.models import MemoryScope, MemoryState, MemoryType
from membench.memory.normalization import (
    canonical_memory_id,
    canonical_memory_type,
    normalize_memory_state,
    parse_timestamp,
    passes_threshold,
    source_event_ids,
)


@pytest.fixture
def scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="experiment-001",
        run_id="run-001",
        user_id="persona-001",
        namespace=("membench", "telemem", "run-001", "persona-001"),
    )


def test_canonical_memory_id_is_stable_and_supports_qualifiers(
    scope: MemoryScope,
) -> None:
    identity = "\x1f".join(("telemem", *scope.namespace, "native-001"))
    expected = "telemem-" + hashlib.sha256(identity.encode()).hexdigest()

    assert canonical_memory_id("TeleMem", scope, "native-001") == expected
    assert canonical_memory_id(
        "telemem",
        scope,
        "native-001",
        qualifiers=("semantic",),
    ) != expected


def test_timestamp_normalization_requires_a_timezone() -> None:
    expected = datetime(2026, 9, 15, tzinfo=UTC)

    assert parse_timestamp(
        "2026-09-15T00:00:00Z",
        mechanism="Example",
        field_name="created_at",
    ) == expected
    assert parse_timestamp(
        expected,
        mechanism="Example",
        field_name="created_at",
    ) is expected
    with pytest.raises(ValueError, match="timezone-free"):
        parse_timestamp(
            "2026-09-15T00:00:00",
            mechanism="Example",
            field_name="created_at",
        )


def test_source_event_ids_merge_and_deduplicate_in_stable_order() -> None:
    assert source_event_ids(
        {
            "membench_source_event_id": "event-001",
            "membench_source_event_ids": ["event-001", "event-002"],
        },
        mechanism="Example",
    ) == ("event-001", "event-002")

    with pytest.raises(ValueError, match="source event"):
        source_event_ids(
            {"membench_source_event_ids": "event-001"},
            mechanism="Example",
        )


def test_memory_type_state_and_threshold_normalization() -> None:
    assert canonical_memory_type(
        "procedural_memory",
        mechanism="Example",
        aliases={"procedural_memory": "procedural"},
    ) == MemoryType.PROCEDURAL
    assert normalize_memory_state(event="DELETE") == MemoryState.DELETED
    assert normalize_memory_state(status="expired") == MemoryState.EXPIRED
    assert normalize_memory_state() == MemoryState.ACTIVE
    assert passes_threshold(0.8, 0.8)
    assert not passes_threshold(0.7, 0.8)
    assert passes_threshold(0.2, 0.3, higher_is_better=False)
    assert passes_threshold(None, 0.8)

