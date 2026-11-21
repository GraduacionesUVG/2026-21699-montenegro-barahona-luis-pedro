"""Qualification-level stabilization for eventually consistent memory stores."""

from __future__ import annotations

import time
from collections.abc import Callable

from membench.memory.errors import StateInspectionError
from membench.memory.models import MemoryScope, MemorySnapshot
from membench.qualification.assertions import snapshot_fingerprint
from membench.qualification.models import (
    QuiescenceConfiguration,
    QuiescenceEvidence,
    QuiescenceObservation,
)


class QualificationQuiescenceTimeout(RuntimeError):
    """Raised when canonical memory state does not stabilize before timeout."""


def wait_for_quiescence(
    inspect_state: Callable[[], MemorySnapshot],
    configuration: QuiescenceConfiguration,
    *,
    phase: str,
    expected_scope: MemoryScope,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> QuiescenceEvidence:
    """Poll canonical snapshots until state is stable for the configured window."""
    started_at = clock()
    stable_since = started_at
    stable_sample_count = 0
    previous_fingerprint: dict[str, object] | None = None
    observations: list[QuiescenceObservation] = []

    while True:
        snapshot = inspect_state()
        if snapshot.scope != expected_scope:
            raise StateInspectionError(
                "quiescence inspection returned a snapshot for the wrong scope"
            )

        observed_at = clock()
        elapsed_seconds = max(0.0, observed_at - started_at)
        fingerprint = snapshot_fingerprint(snapshot)
        changed = (
            previous_fingerprint is not None
            and fingerprint != previous_fingerprint
        )
        if previous_fingerprint is None or changed:
            stable_since = observed_at
            stable_sample_count = 1
        else:
            stable_sample_count += 1
        stable_for_seconds = max(0.0, observed_at - stable_since)
        observations.append(
            QuiescenceObservation(
                sample_index=len(observations) + 1,
                elapsed_seconds=elapsed_seconds,
                changed_since_previous=changed,
                complete=snapshot.complete,
                record_count=snapshot.record_count,
                record_ids=tuple(
                    sorted(record.canonical_id for record in snapshot.records)
                ),
                normalized_sha256=str(fingerprint["normalized_sha256"]),
                reported_total_count=snapshot.reported_total_count,
                unobserved_count=snapshot.unobserved_count,
                coverage_ratio=snapshot.coverage_ratio,
            )
        )

        completeness_satisfied = (
            snapshot.complete
            or configuration.enumeration_policy == "bounded"
        )

        stabilized = (
            completeness_satisfied
            and stable_sample_count >= configuration.minimum_stable_samples
            and stable_for_seconds >= configuration.stable_window_seconds
        )
        if stabilized or elapsed_seconds >= configuration.timeout_seconds:
            return QuiescenceEvidence(
                phase=phase,
                scope=expected_scope,
                stabilized=stabilized,
                enumeration_policy=configuration.enumeration_policy,
                timeout_seconds=configuration.timeout_seconds,
                poll_interval_seconds=configuration.poll_interval_seconds,
                stable_window_seconds=configuration.stable_window_seconds,
                minimum_stable_samples=(
                    configuration.minimum_stable_samples
                ),
                stable_for_seconds=stable_for_seconds,
                latency_seconds=elapsed_seconds,
                observations=tuple(observations),
                final_snapshot=snapshot,
            )

        previous_fingerprint = fingerprint
        remaining = configuration.timeout_seconds - elapsed_seconds
        sleep(min(configuration.poll_interval_seconds, remaining))
