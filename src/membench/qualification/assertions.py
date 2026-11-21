"""Mechanism-neutral behavioral assertions for qualification reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Literal

from membench.canary_data import GuardExpectation, SyntheticCanary
from membench.contract import Mechanism, WritePolicy
from membench.memory.models import (
    AdapterProvenance,
    EmptyStateResult,
    FlushResult,
    MemoryRecord,
    MemoryScope,
    MemorySnapshot,
    RetrievalResult,
)
from membench.memory.policy import PolicyDispatchAction
from membench.qualification.models import (
    ImportanceChange,
    QualificationAssertion,
    QualificationStage,
    QuiescenceEvidence,
    RetrievalStateDelta,
    SnapshotRecordChange,
)
from membench.responder import ResponderResult


def guard_rejected(policy: WritePolicy, canary: SyntheticCanary) -> bool:
    """Return whether this policy should reject the canary before storage."""
    return (
        policy == WritePolicy.SHARED_GUARDED
        and canary.guard_expectation == GuardExpectation.REJECT
    )


def assert_empty_state(
    result: EmptyStateResult,
    *,
    assertion_id: str,
    stage: QualificationStage,
) -> QualificationAssertion:
    """Assert that inspection, retrieval, and pending evidence are all empty."""
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=stage,
        passed=result.empty,
        expected={
            "empty": True,
            "inspected_count": 0,
            "retrieved_count": 0,
            "pending_count": 0,
        },
        observed={
            "empty": result.empty,
            "inspected_count": result.inspected_count,
            "retrieved_count": result.retrieved_count,
            "pending_count": result.pending_count,
        },
    )


def assert_snapshot_empty(
    snapshot: MemorySnapshot,
    *,
    assertion_id: str = "initial_snapshot_empty",
) -> QualificationAssertion:
    """Assert that evaluator-visible state contains no records."""
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=QualificationStage.INSPECT_BEFORE,
        passed=snapshot.record_count == 0,
        expected=0,
        observed=snapshot.record_count,
    )


def assert_dispatch_action(
    policy: WritePolicy,
    canary: SyntheticCanary,
    action: PolicyDispatchAction,
) -> QualificationAssertion:
    """Assert policy routing, including guarded rejection or quarantine."""
    if guard_rejected(policy, canary):
        accepted = {
            PolicyDispatchAction.NO_WRITE,
            PolicyDispatchAction.QUARANTINE,
        }
        return QualificationAssertion(
            assertion_id=f"guard_rejected:{canary.canary_id}",
            stage=QualificationStage.WRITE,
            passed=action in accepted,
            expected=sorted(item.value for item in accepted),
            observed=action.value,
            canary_id=canary.canary_id,
        )

    expected = {
        WritePolicy.DIRECT: PolicyDispatchAction.DIRECT_WRITE,
        WritePolicy.SHARED_GUARDED: PolicyDispatchAction.GUARDED_WRITE,
        WritePolicy.NATIVE_SELECTIVE: (
            PolicyDispatchAction.NATIVE_CONVERSATION_WRITE
        ),
    }[policy]
    return QualificationAssertion(
        assertion_id=f"write_dispatched:{canary.canary_id}",
        stage=QualificationStage.WRITE,
        passed=action == expected,
        expected=expected.value,
        observed=action.value,
        canary_id=canary.canary_id,
    )


def assert_flush_completed(
    result: FlushResult,
    *,
    assertion_id: str = "flush_completed",
) -> QualificationAssertion:
    """Assert that the visibility barrier completed without pending IDs."""
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=QualificationStage.FLUSH,
        passed=result.completed and not result.pending_ids,
        expected={"completed": True, "pending_ids": []},
        observed={
            "completed": result.completed,
            "pending_ids": list(result.pending_ids),
        },
    )


def assert_quiescence_reached(
    evidence: QuiescenceEvidence,
    *,
    assertion_id: str,
) -> QualificationAssertion:
    """Assert that complete canonical state stabilized before its timeout."""
    final_observation = evidence.observations[-1]
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=QualificationStage.QUIESCENCE,
        passed=evidence.stabilized,
        expected={
            "stabilized": True,
            "enumeration_policy": evidence.enumeration_policy,
        },
        observed={
            "stabilized": evidence.stabilized,
            "enumeration_policy": evidence.enumeration_policy,
            "complete": final_observation.complete,
            "stable_for_seconds": evidence.stable_for_seconds,
            "sample_count": len(evidence.observations),
            "returned_count": final_observation.record_count,
            "reported_total_count": final_observation.reported_total_count,
            "unobserved_count": final_observation.unobserved_count,
            "coverage_ratio": final_observation.coverage_ratio,
            "normalized_sha256": final_observation.normalized_sha256,
        },
    )


def assert_retrieval_fragments(
    canary: SyntheticCanary,
    expectation_index: int,
    result: RetrievalResult,
    *,
    assertion_id: str | None = None,
) -> QualificationAssertion:
    """Assert that every required fragment occurs in returned record text."""
    expectation = canary.retrieval_expectations[expectation_index]
    visible_text = "\n".join(hit.record.text for hit in result.hits)
    missing = [
        fragment
        for fragment in expectation.required_fragments
        if fragment.casefold() not in visible_text.casefold()
    ]
    return QualificationAssertion(
        assertion_id=assertion_id
        or f"retrieval_fragments:{canary.canary_id}:{expectation_index:03d}",
        stage=QualificationStage.RETRIEVE,
        passed=not missing,
        expected=list(expectation.required_fragments),
        observed={
            "missing_fragments": missing,
            "returned_record_ids": [
                hit.record.canonical_id for hit in result.hits
            ],
        },
        canary_id=canary.canary_id,
    )


def assert_context_fragments(
    canary: SyntheticCanary,
    expectation_index: int,
    context: str,
) -> QualificationAssertion:
    """Assert responder context contains every required retrieval fragment."""
    expectation = canary.retrieval_expectations[expectation_index]
    missing = [
        fragment
        for fragment in expectation.required_fragments
        if fragment.casefold() not in context.casefold()
    ]
    return QualificationAssertion(
        assertion_id=(
            f"context_fragments:{canary.canary_id}:{expectation_index:03d}"
        ),
        stage=QualificationStage.CONTEXT,
        passed=not missing,
        expected=list(expectation.required_fragments),
        observed={"missing_fragments": missing},
        canary_id=canary.canary_id,
    )


def assert_responder_identity(
    result: ResponderResult,
    *,
    provider: str,
    model: str,
    model_digest: str | None,
    assertion_id: str = "responder_identity_matches",
) -> QualificationAssertion:
    """Assert the response came from the model identity frozen in the plan."""
    expected = {
        "provider": provider,
        "model": model,
        "model_digest": model_digest,
    }
    observed = {
        "provider": result.provider,
        "model": result.model,
        "model_digest": result.model_digest,
    }
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=QualificationStage.RESPOND,
        passed=expected == observed,
        expected=expected,
        observed=observed,
    )


def assert_retrieval_excludes_canary(
    canary: SyntheticCanary,
    result: RetrievalResult,
    *,
    assertion_id: str,
) -> QualificationAssertion:
    """Assert that neither a foreign marker nor its source ID crossed scopes."""
    marker_records = [
        hit.record.canonical_id
        for hit in result.hits
        if canary.marker.casefold() in hit.record.text.casefold()
    ]
    source_records = [
        hit.record.canonical_id
        for hit in result.hits
        if canary.source_event_id in hit.record.source_event_ids
    ]
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=QualificationStage.ISOLATION,
        passed=not marker_records and not source_records,
        expected={"marker_records": [], "source_records": []},
        observed={
            "marker_records": marker_records,
            "source_records": source_records,
        },
        canary_id=canary.canary_id,
    )


def assert_retrieval_scope(
    result: RetrievalResult,
    expected_scope: MemoryScope,
    *,
    assertion_id: str,
) -> QualificationAssertion:
    """Assert that every returned record belongs to the requested scope."""
    foreign_records = [
        hit.record.canonical_id
        for hit in result.hits
        if hit.record.scope != expected_scope
    ]
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=QualificationStage.ISOLATION,
        passed=not foreign_records,
        expected={"scope": expected_scope.key, "foreign_record_ids": []},
        observed={
            "scope": expected_scope.key,
            "foreign_record_ids": foreign_records,
        },
    )


def assert_snapshot_excludes_canaries(
    snapshot: MemorySnapshot,
    canaries: tuple[SyntheticCanary, ...],
    *,
    assertion_id: str,
) -> QualificationAssertion:
    """Assert foreign markers and source IDs are absent from inspected state."""
    text = "\n".join(record.text for record in snapshot.records).casefold()
    source_ids = {
        source_id
        for record in snapshot.records
        for source_id in record.source_event_ids
    }
    visible = [
        canary.canary_id
        for canary in canaries
        if canary.marker.casefold() in text
        or canary.source_event_id in source_ids
    ]
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=QualificationStage.ISOLATION,
        passed=not visible,
        expected=[],
        observed=visible,
    )


def assert_snapshot_preserved(
    before: MemorySnapshot,
    after: MemorySnapshot,
    *,
    assertion_id: str,
) -> QualificationAssertion:
    """Assert another scope's reset did not alter evaluator-visible records."""
    return assert_snapshot_unchanged(
        before,
        after,
        assertion_id=assertion_id,
        stage=QualificationStage.ISOLATION,
    )


def assert_snapshot_unchanged(
    before: MemorySnapshot,
    after: MemorySnapshot,
    *,
    assertion_id: str,
    stage: QualificationStage,
    required: bool = True,
) -> QualificationAssertion:
    """Assert complete normalized memory state stayed byte-for-byte equivalent.

    This compatibility entry point intentionally retains the strict full-state
    semantics used by quiescence and existing isolation checks.
    """
    return assert_snapshot_full_state_unchanged(
        before,
        after,
        assertion_id=assertion_id,
        stage=stage,
        required=required,
    )


def assert_snapshot_content_unchanged(
    before: MemorySnapshot,
    after: MemorySnapshot,
    *,
    assertion_id: str,
    stage: QualificationStage,
    required: bool = True,
) -> QualificationAssertion:
    """Assert stored memory identity, content, and provenance did not change."""
    return _assert_snapshot_fingerprint_unchanged(
        before,
        after,
        assertion_id=assertion_id,
        stage=stage,
        fingerprint=snapshot_content_fingerprint,
        required=required,
    )


def assert_snapshot_full_state_unchanged(
    before: MemorySnapshot,
    after: MemorySnapshot,
    *,
    assertion_id: str,
    stage: QualificationStage,
    required: bool = True,
) -> QualificationAssertion:
    """Assert all normalized state, including operational attributes, is equal."""
    return _assert_snapshot_fingerprint_unchanged(
        before,
        after,
        assertion_id=assertion_id,
        stage=stage,
        fingerprint=snapshot_full_state_fingerprint,
        required=required,
    )


def _assert_snapshot_fingerprint_unchanged(
    before: MemorySnapshot,
    after: MemorySnapshot,
    *,
    assertion_id: str,
    stage: QualificationStage,
    fingerprint: Callable[[MemorySnapshot], dict[str, object]],
    required: bool,
) -> QualificationAssertion:
    before_state = fingerprint(before)
    after_state = fingerprint(after)
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=stage,
        required=required,
        passed=(
            before.complete
            and after.complete
            and before.scope == after.scope
            and before_state == after_state
        ),
        expected=before_state,
        observed=after_state,
    )


def snapshot_canary_assertions(
    policy: WritePolicy,
    canaries: tuple[SyntheticCanary, ...],
    snapshot: MemorySnapshot,
) -> tuple[QualificationAssertion, ...]:
    """Assert marker visibility and source provenance in inspected state."""
    results: list[QualificationAssertion] = []
    text = "\n".join(record.text for record in snapshot.records)
    source_ids = {
        source_id
        for record in snapshot.records
        for source_id in record.source_event_ids
    }
    for canary in canaries:
        rejected = guard_rejected(policy, canary)
        marker_present = canary.marker.casefold() in text.casefold()
        results.append(
            QualificationAssertion(
                assertion_id=f"snapshot_marker:{canary.canary_id}",
                stage=QualificationStage.INSPECT_AFTER,
                passed=(not marker_present if rejected else marker_present),
                expected=not rejected,
                observed=marker_present,
                canary_id=canary.canary_id,
            )
        )
        source_present = canary.source_event_id in source_ids
        results.append(
            QualificationAssertion(
                assertion_id=f"source_provenance:{canary.canary_id}",
                stage=QualificationStage.INSPECT_AFTER,
                passed=(not source_present if rejected else source_present),
                expected=not rejected,
                observed=source_present,
                canary_id=canary.canary_id,
            )
        )
    return tuple(results)


def assert_provenance_mechanism(
    provenance: AdapterProvenance,
    mechanism: Mechanism,
    *,
    assertion_id: str = "provenance_mechanism_matches",
) -> QualificationAssertion:
    """Assert that the created adapter matches the requested mechanism."""
    return QualificationAssertion(
        assertion_id=assertion_id,
        stage=QualificationStage.PROVENANCE,
        passed=provenance.mechanism == mechanism.value,
        expected=mechanism.value,
        observed=provenance.mechanism,
    )


def snapshot_content_fingerprint(snapshot: MemorySnapshot) -> dict[str, object]:
    """Hash stable identity, content, lifecycle, and source provenance only."""
    records = sorted(
        (_content_record(record) for record in snapshot.records),
        key=lambda item: str(item["canonical_id"]),
    )
    return _snapshot_fingerprint(snapshot, records)


def compare_snapshot_state(
    before: MemorySnapshot,
    after: MemorySnapshot,
) -> RetrievalStateDelta:
    """Return a structured canonical delta suitable for reporting and analysis."""
    if before.scope != after.scope:
        raise ValueError("cannot compare snapshots from different scopes")
    before_by_id = _records_by_id(before)
    after_by_id = _records_by_id(after)
    before_ids = set(before_by_id)
    after_ids = set(after_by_id)
    created_record_ids = tuple(sorted(after_ids - before_ids))
    deleted_record_ids = tuple(sorted(before_ids - after_ids))
    content_changes: list[SnapshotRecordChange] = []
    operational_changes: list[SnapshotRecordChange] = []
    importance_changes: list[ImportanceChange] = []

    for canonical_id in sorted(before_ids & after_ids):
        before_record = before_by_id[canonical_id]
        after_record = after_by_id[canonical_id]
        content_fields = _changed_fields(
            _content_record(before_record),
            _content_record(after_record),
        )
        if content_fields:
            content_changes.append(
                SnapshotRecordChange(
                    canonical_id=canonical_id,
                    changed_fields=content_fields,
                )
            )

        operational_fields = _changed_fields(
            _operational_record(before_record),
            _operational_record(after_record),
        )
        if operational_fields:
            operational_changes.append(
                SnapshotRecordChange(
                    canonical_id=canonical_id,
                    changed_fields=operational_fields,
                )
            )

        before_importance = _importance(before_record)
        after_importance = _importance(after_record)
        if before_importance != after_importance:
            importance_changes.append(
                ImportanceChange(
                    canonical_id=canonical_id,
                    before=before_importance,
                    after=after_importance,
                    delta=(
                        None
                        if before_importance is None or after_importance is None
                        else round(after_importance - before_importance, 12)
                    ),
                )
            )

    visible_content_changed = bool(
        created_record_ids or deleted_record_ids or content_changes
    )
    visible_full_state_changed = bool(
        visible_content_changed or operational_changes
    )
    exact_comparison = before.complete and after.complete
    reported_total_unchanged = (
        before.reported_total_count is not None
        and after.reported_total_count is not None
        and before.reported_total_count == after.reported_total_count
    )
    comparison_level: Literal["exact", "bounded"] = (
        "exact" if exact_comparison else "bounded"
    )
    observed_content_unchanged = bool(
        not visible_content_changed
        and (exact_comparison or reported_total_unchanged)
    )
    observed_full_state_unchanged = bool(
        observed_content_unchanged and not visible_full_state_changed
    )
    content_unchanged = bool(
        exact_comparison and not visible_content_changed
    )
    full_state_unchanged = bool(
        content_unchanged and not visible_full_state_changed
    )
    return RetrievalStateDelta(
        scope=before.scope,
        before_complete=before.complete,
        after_complete=after.complete,
        comparison_level=comparison_level,
        before_reported_total_count=before.reported_total_count,
        after_reported_total_count=after.reported_total_count,
        content_unchanged=content_unchanged,
        full_state_unchanged=full_state_unchanged,
        observed_content_unchanged=observed_content_unchanged,
        observed_full_state_unchanged=observed_full_state_unchanged,
        created_record_ids=created_record_ids,
        deleted_record_ids=deleted_record_ids,
        content_changes=tuple(content_changes),
        operational_changes=tuple(operational_changes),
        importance_changes=tuple(importance_changes),
    )


def snapshot_full_state_fingerprint(
    snapshot: MemorySnapshot,
) -> dict[str, object]:
    """Hash all canonical state except deliberately opaque raw vendor payloads."""
    records = sorted(
        (
            record.model_dump(mode="json", exclude={"raw_vendor"})
            for record in snapshot.records
        ),
        key=lambda item: str(item["canonical_id"]),
    )
    return _snapshot_fingerprint(snapshot, records)


def snapshot_fingerprint(snapshot: MemorySnapshot) -> dict[str, object]:
    """Return the strict full-state fingerprint retained for compatibility."""
    return snapshot_full_state_fingerprint(snapshot)


def _content_record(record: MemoryRecord) -> dict[str, object]:
    """Project one record without vendor-managed ranking or access attributes."""
    normalized = record.model_dump(mode="json")
    attributes = normalized["attributes"]
    assert isinstance(attributes, dict)
    return {
        "canonical_id": record.canonical_id,
        "vendor_id": record.vendor_id,
        "scope": normalized["scope"],
        "text": record.text,
        "memory_type": record.memory_type.value,
        "state": record.state.value,
        "source_event_ids": list(record.source_event_ids),
        "created_at": normalized["created_at"],
        "metadata": attributes.get("metadata", {}),
    }


def _operational_record(record: MemoryRecord) -> dict[str, object]:
    """Project ranking/access state excluded by the content comparison."""
    normalized = record.model_dump(mode="json")
    attributes = normalized["attributes"]
    assert isinstance(attributes, dict)
    operational_attributes = dict(attributes)
    operational_attributes.pop("metadata", None)
    return {
        "updated_at": normalized["updated_at"],
        "attributes": operational_attributes,
    }


def _records_by_id(snapshot: MemorySnapshot) -> dict[str, MemoryRecord]:
    records = {record.canonical_id: record for record in snapshot.records}
    if len(records) != snapshot.record_count:
        raise ValueError("snapshot contains duplicate canonical record IDs")
    return records


def _importance(record: MemoryRecord) -> float | None:
    value = record.attributes.get("importance")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _changed_fields(
    before: object,
    after: object,
    prefix: str = "",
) -> tuple[str, ...]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changed: list[str] = []
        for key in sorted(set(before) | set(after), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.append(path)
                continue
            changed.extend(_changed_fields(before[key], after[key], path))
        return tuple(changed)
    if before != after:
        return (prefix,)
    return ()


def _snapshot_fingerprint(
    snapshot: MemorySnapshot,
    records: list[dict[str, object]],
) -> dict[str, object]:
    payload = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "scope": snapshot.scope.key,
        "complete": snapshot.complete,
        "record_count": snapshot.record_count,
        "reported_total_count": snapshot.reported_total_count,
        "record_ids": [str(item["canonical_id"]) for item in records],
        "normalized_sha256": hashlib.sha256(payload).hexdigest(),
    }
