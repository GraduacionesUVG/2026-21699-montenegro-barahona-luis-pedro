"""Tests for deriving neutral external-ingress hosts from frozen projections."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import membench.campaign  # noqa: F401
from membench.curation import neutral_hosts
from membench.curation.neutral_hosts import (
    NeutralHostError,
    build_neutral_host_projection_set,
    load_neutral_host_config,
)
from membench.curation.projections import (
    ProjectionKind,
    memory_visible_replay_signature,
)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs" / "neutral_hosts" / "external-hosts-v1.json"
_SOURCE = _ROOT / "data" / "blackbox" / "projections-v1"


def test_builder_derives_three_identical_neutral_host_twin_families() -> None:
    config = load_neutral_host_config(_CONFIG)

    built = build_neutral_host_projection_set(
        config,
        source_projection_directory=_SOURCE,
    )

    assert built.manifest.projection_set_id == config.projection_set_id
    assert len(built.cases) == 3
    assert len(built.trials) == 9
    source_ids = {
        "277": "alloc-abrupt-false-correction-005",
        "371": "alloc-abrupt-false-correction-003",
        "436": "alloc-abrupt-false-correction-002",
    }
    for case in built.cases:
        assert case.projection_kind == ProjectionKind.NEUTRAL_EXTERNAL_HOST
        assert case.source_allocation_case_id == source_ids[case.persona_id]
        trials = tuple(
            trial
            for trial in built.trials
            if trial.allocation_case_id == case.allocation_case_id
        )
        assert {trial.twin for trial in trials} == {
            "clean",
            "matched_benign",
            "poisoned",
        }
        assert len({memory_visible_replay_signature(trial) for trial in trials}) == 1
        assert all(
            event.source_message_index is not None
            for trial in trials
            for event in trial.replay_events
        )


def test_builder_rejects_delivery_availability_absent_from_host_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_neutral_host_config(_CONFIG)
    invalid_delivery_set = SimpleNamespace(
        projection_set_id=config.projection_set_id,
        families=(
            SimpleNamespace(
                persona_id="277",
                delivery_family_id="delivery-test-277",
                deliveries=(
                    SimpleNamespace(
                        delivery_id="delivery-test-277-poisoned",
                        available_after_source_message_index=999_999,
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        neutral_hosts,
        "load_delivery_projection_set",
        lambda _: invalid_delivery_set,
    )

    with pytest.raises(NeutralHostError, match="availability index is absent"):
        build_neutral_host_projection_set(
            config,
            source_projection_directory=_SOURCE,
            delivery_projection_directories=("unused-in-test",),
        )
