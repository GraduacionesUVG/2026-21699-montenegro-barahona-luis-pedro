import pytest

from membench.canary_data import CanarySuite
from membench.contract import Mechanism, WritePolicy
from membench.memory.models import (
    AdapterProvenance,
    MemoryRecord,
    MemoryScope,
    MemorySnapshot,
    MemoryType,
    RetrievalResult,
)
from membench.memory.policy import PolicyDispatchAction
from membench.qualification.assertions import (
    assert_dispatch_action,
    assert_provenance_mechanism,
    assert_quiescence_reached,
    assert_retrieval_fragments,
    assert_snapshot_content_unchanged,
    assert_snapshot_full_state_unchanged,
    compare_snapshot_state,
    snapshot_canary_assertions,
    snapshot_content_fingerprint,
    snapshot_full_state_fingerprint,
)
from membench.qualification.models import (
    QualificationStage,
    QuiescenceConfiguration,
)
from membench.qualification.quiescence import wait_for_quiescence


def _scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="assertion-tests",
        run_id="run-001",
        user_id="user-001",
        namespace=("membench", "qualification", "run-001"),
    )


def test_guard_rejection_requires_a_nonwriting_dispatch_action() -> None:
    canary = CanarySuite.load().canary("guard_injection_v1")

    accepted = assert_dispatch_action(
        WritePolicy.SHARED_GUARDED,
        canary,
        PolicyDispatchAction.QUARANTINE,
    )
    incorrectly_written = assert_dispatch_action(
        WritePolicy.SHARED_GUARDED,
        canary,
        PolicyDispatchAction.GUARDED_WRITE,
    )

    assert accepted.passed
    assert not incorrectly_written.passed


def test_retrieval_assertion_reports_missing_required_fragments() -> None:
    canary = CanarySuite.load().canary("semantic_preference_v1")
    result = RetrievalResult(
        operation_id="retrieve-001",
        query_sha256="0" * 64,
        request_top_k=5,
        hits=(),
        latency_seconds=0,
    )

    assertion = assert_retrieval_fragments(canary, 0, result)

    assert not assertion.passed
    assert assertion.observed["missing_fragments"]


def test_rejected_canary_must_be_absent_from_snapshot_and_provenance() -> None:
    canary = CanarySuite.load().canary("guard_injection_v1")
    assertions = snapshot_canary_assertions(
        WritePolicy.SHARED_GUARDED,
        (canary,),
        MemorySnapshot(scope=_scope()),
    )

    assert all(item.passed for item in assertions)


def test_provenance_assertion_detects_wrong_mechanism() -> None:
    provenance = AdapterProvenance(
        mechanism="langmem",
        mechanism_version="test",
        adapter_version="test",
        storage_backend="memory",
        namespace=_scope().namespace,
    )

    assertion = assert_provenance_mechanism(provenance, Mechanism.TELEMEM)

    assert not assertion.passed


def _memory_record(
    *,
    text: str = "The notebook marker is HELIOTROPE-7Q2M.",
    importance: float = 0.5,
    metadata: dict[str, str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        canonical_id="canonical-001",
        vendor_id="vendor-001",
        scope=_scope(),
        text=text,
        memory_type=MemoryType.SEMANTIC,
        source_event_ids=("event-001",),
        attributes={
            "importance": importance,
            "metadata": metadata or {"source": "qualification"},
        },
        raw_vendor={"access_count": 0},
    )


def test_content_comparison_allows_operational_importance_changes() -> None:
    before = MemorySnapshot(scope=_scope(), records=(_memory_record(),))
    after = MemorySnapshot(
        scope=_scope(),
        records=(_memory_record(importance=0.575),),
    )

    content = assert_snapshot_content_unchanged(
        before,
        after,
        assertion_id="retrieval_content_unchanged:test",
        stage=QualificationStage.VERIFY_RETRIEVAL_UNCHANGED,
    )
    full_state = assert_snapshot_full_state_unchanged(
        before,
        after,
        assertion_id="retrieval_operational_unchanged:test",
        stage=QualificationStage.VERIFY_RETRIEVAL_UNCHANGED,
        required=False,
    )

    assert content.passed
    assert not full_state.passed
    assert content.required
    assert not full_state.required
    assert snapshot_content_fingerprint(before) == snapshot_content_fingerprint(
        after
    )
    assert snapshot_full_state_fingerprint(
        before
    ) != snapshot_full_state_fingerprint(after)
    delta = compare_snapshot_state(before, after)
    assert delta.content_unchanged
    assert not delta.full_state_unchanged
    assert delta.created_record_ids == ()
    assert delta.deleted_record_ids == ()
    assert delta.content_changes == ()
    assert delta.operational_changes[0].changed_fields == (
        "attributes.importance",
    )
    assert delta.importance_changes[0].before == 0.5
    assert delta.importance_changes[0].after == 0.575
    assert delta.importance_changes[0].delta == pytest.approx(0.075)


def test_content_comparison_rejects_text_and_record_set_changes() -> None:
    before = MemorySnapshot(scope=_scope(), records=(_memory_record(),))
    changed_text = MemorySnapshot(
        scope=_scope(),
        records=(_memory_record(text="The notebook marker was replaced."),),
    )
    added_record = _memory_record(text="A probe was written back.").model_copy(
        update={
            "canonical_id": "canonical-002",
            "vendor_id": "vendor-002",
        }
    )
    changed_records = MemorySnapshot(
        scope=_scope(),
        records=(*before.records, added_record),
    )

    for after in (changed_text, changed_records):
        assertion = assert_snapshot_content_unchanged(
            before,
            after,
            assertion_id="retrieval_content_unchanged:test",
            stage=QualificationStage.VERIFY_RETRIEVAL_UNCHANGED,
        )
        assert not assertion.passed

    text_delta = compare_snapshot_state(before, changed_text)
    assert not text_delta.content_unchanged
    assert text_delta.content_changes[0].changed_fields == ("text",)
    record_delta = compare_snapshot_state(before, changed_records)
    assert not record_delta.content_unchanged
    assert record_delta.created_record_ids == ("canonical-002",)


def test_content_comparison_preserves_source_metadata() -> None:
    before = MemorySnapshot(scope=_scope(), records=(_memory_record(),))
    after = MemorySnapshot(
        scope=_scope(),
        records=(
            _memory_record(metadata={"source": "different-qualification"}),
        ),
    )

    assertion = assert_snapshot_content_unchanged(
        before,
        after,
        assertion_id="retrieval_content_unchanged:test",
        stage=QualificationStage.VERIFY_RETRIEVAL_UNCHANGED,
    )

    assert not assertion.passed
    delta = compare_snapshot_state(before, after)
    assert delta.content_changes[0].changed_fields == ("metadata.source",)


def test_bounded_comparison_reports_observed_state_without_claiming_exactness(
) -> None:
    before = MemorySnapshot(
        scope=_scope(),
        records=(_memory_record(),),
        complete=False,
        reported_total_count=2,
        inspection_limit=1,
    )
    after = before.model_copy(deep=True)

    delta = compare_snapshot_state(before, after)

    assert delta.comparison_level == "bounded"
    assert delta.observed_content_unchanged
    assert delta.observed_full_state_unchanged
    assert not delta.content_unchanged
    assert not delta.full_state_unchanged
    assert delta.before_reported_total_count == 2
    assert delta.after_reported_total_count == 2


def test_bounded_comparison_detects_a_reported_total_change() -> None:
    before = MemorySnapshot(
        scope=_scope(),
        records=(_memory_record(),),
        complete=False,
        reported_total_count=2,
        inspection_limit=1,
    )
    after = before.model_copy(update={"reported_total_count": 3})

    delta = compare_snapshot_state(before, after)

    assert delta.comparison_level == "bounded"
    assert not delta.observed_content_unchanged
    assert not delta.content_unchanged


def test_bounded_quiescence_records_coverage_without_requiring_completeness(
) -> None:
    current_time = 0.0

    def clock() -> float:
        return current_time

    def sleep(seconds: float) -> None:
        nonlocal current_time
        current_time += seconds

    snapshot = MemorySnapshot(
        scope=_scope(),
        records=(_memory_record(),),
        complete=False,
        reported_total_count=2,
        inspection_limit=1,
    )
    evidence = wait_for_quiescence(
        lambda: snapshot,
        QuiescenceConfiguration(
            timeout_seconds=1,
            poll_interval_seconds=0.1,
            stable_window_seconds=0.1,
            minimum_stable_samples=2,
            enumeration_policy="bounded",
        ),
        phase="bounded_test",
        expected_scope=_scope(),
        clock=clock,
        sleep=sleep,
    )
    assertion = assert_quiescence_reached(
        evidence,
        assertion_id="quiescence_reached:bounded_test",
    )

    assert evidence.stabilized
    assert evidence.enumeration_policy == "bounded"
    assert not evidence.final_snapshot.complete
    assert evidence.observations[-1].reported_total_count == 2
    assert evidence.observations[-1].unobserved_count == 1
    assert evidence.observations[-1].coverage_ratio == pytest.approx(0.5)
    assert assertion.passed
    assert assertion.expected["enumeration_policy"] == "bounded"
