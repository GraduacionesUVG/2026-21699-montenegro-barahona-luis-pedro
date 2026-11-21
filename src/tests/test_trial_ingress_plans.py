"""Tests for compact, frozen external ingress plans."""

from __future__ import annotations

from pathlib import Path

import pytest

from membench.agent.models import AgentRunConfiguration
from membench.curation.external_attack_specs import (
    ExternalAttackMethodology,
    ExternalAttackObjective,
    ExternalAttackSpec,
    build_external_attack_spec_set,
)
from membench.curation.external_delivery_recipes import (
    load_external_delivery_recipe_set,
    render_external_delivery_projection_set,
)
from membench.curation.projections import AttackReviewStatus
from membench.ingress.models import ExternalSourceKind, IngressMode
from membench.ingress.plans import (
    TrialIngressPlanError,
    build_trial_ingress_plan_set,
    export_trial_ingress_plan_set,
    load_trial_ingress_plan_set,
)
from membench.tools.fixtures import load_tool_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = load_tool_fixture(ROOT / "configs/fixtures/external-tool-pilot-v1.json")
RECIPES = load_external_delivery_recipe_set(
    ROOT / "configs/external_delivery_recipes" / "external-tool-pilot-v1.example.json"
)


def _attack_set():  # type: ignore[no-untyped-def]
    return build_external_attack_spec_set(
        attack_set_id="external-tool-pilot-v1-attacks",
        projection_set_id="external-tool-pilot-v1",
        specs=(
            ExternalAttackSpec(
                spec_ordinal=1,
                attack_spec_id="external-routing-277-001",
                allocation_case_id="alloc-external-routing-277-001",
                attack_id="bb-ext-001",
                attack_methodology=ExternalAttackMethodology.DESTINATION_REDIRECTION,
                objective=ExternalAttackObjective.DESTINATION_REDIRECTION,
                variant="unapproved_recipient",
                source_kind=ExternalSourceKind.EMAIL,
                insert_after_source_message_index=4,
                target_reference_sha256="a" * 64,
                public_attacker_view_sha256="b" * 64,
                review_status=AttackReviewStatus.APPROVED,
            ),
        ),
    )


def _delivery_set():  # type: ignore[no-untyped-def]
    attack_set = _attack_set()
    return render_external_delivery_projection_set(
        RECIPES,
        fixture=FIXTURE,
        attack_set=attack_set,
    )


def _host_trial_ids(delivery_set):  # type: ignore[no-untyped-def]
    return {
        delivery.trial_id: f"alloc-host-277--{delivery.twin}"
        for family in delivery_set.families
        for delivery in family.deliveries
    }


def _selection_configuration() -> AgentRunConfiguration:
    return AgentRunConfiguration(
        max_steps=4,
        max_tool_calls=2,
        allowed_tools=("read_message", "search_messages"),
    )


def test_agent_selected_plans_keep_only_compact_selection_references() -> None:
    delivery_set = _delivery_set()
    plan_set = build_trial_ingress_plan_set(
        ingress_plan_set_id="external-tool-pilot-v1-agent-selected",
        ingress_mode=IngressMode.AGENT_SELECTED,
        delivery_set=delivery_set,
        attack_set=_attack_set(),
        fixture=FIXTURE,
        host_trial_ids=_host_trial_ids(delivery_set),
        selection_agent_configuration=_selection_configuration(),
    )

    assert len(plan_set.plans) == 3
    assert {plan.twin for plan in plan_set.plans} == {
        "clean",
        "matched_benign",
        "poisoned",
    }
    assert all(plan.fixture_sha256 == FIXTURE.sha256 for plan in plan_set.plans)
    assert all(plan.selection_task_id is not None for plan in plan_set.plans)
    assert all(
        plan.allowed_tools == ("read_message", "search_messages")
        for plan in plan_set.plans
    )
    assert all(not plan.probe_writeback_allowed for plan in plan_set.plans)
    assert all(not plan.response_writeback_allowed for plan in plan_set.plans)
    assert all(
        plan.selection_agent_configuration == _selection_configuration()
        for plan in plan_set.plans
    )


def test_forced_ingestion_plans_have_no_selection_surface() -> None:
    delivery_set = _delivery_set()
    plan_set = build_trial_ingress_plan_set(
        ingress_plan_set_id="external-tool-pilot-v1-forced",
        ingress_mode=IngressMode.FORCED_INGESTION,
        delivery_set=delivery_set,
        attack_set=_attack_set(),
        fixture=FIXTURE,
        host_trial_ids=_host_trial_ids(delivery_set),
    )

    assert all(plan.selection_task_id is None for plan in plan_set.plans)
    assert all(plan.selection_task_sha256 is None for plan in plan_set.plans)
    assert all(plan.allowed_tools == () for plan in plan_set.plans)


def test_ingress_plan_artifact_round_trips_with_manifest_validation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    delivery_set = _delivery_set()
    plan_set = build_trial_ingress_plan_set(
        ingress_plan_set_id="external-tool-pilot-v1-forced",
        ingress_mode=IngressMode.FORCED_INGESTION,
        delivery_set=delivery_set,
        attack_set=_attack_set(),
        fixture=FIXTURE,
        host_trial_ids=_host_trial_ids(delivery_set),
    )

    manifest = export_trial_ingress_plan_set(plan_set, tmp_path)
    loaded = load_trial_ingress_plan_set(tmp_path)

    assert loaded == plan_set
    assert manifest == plan_set.manifest()
    assert (
        loaded.plan_for_trial(loaded.plans[0].trial_id).sha256 == loaded.plans[0].sha256
    )
    assert (
        loaded.plan_for_host_trial(loaded.plans[0].host_trial_id).sha256
        == loaded.plans[0].sha256
    )


def test_plan_builder_rejects_delivery_artifacts_without_fixture_provenance() -> None:
    delivery_set = _delivery_set()
    family = delivery_set.families[0].model_copy(
        update={"persona_id": None, "fixture_id": None, "fixture_sha256": None}
    )
    unbound = delivery_set.model_copy(update={"families": (family,)})

    with pytest.raises(TrialIngressPlanError, match="no frozen fixture binding"):
        build_trial_ingress_plan_set(
            ingress_plan_set_id="external-tool-pilot-v1-forced",
            ingress_mode=IngressMode.FORCED_INGESTION,
            delivery_set=unbound,
            attack_set=_attack_set(),
            fixture=FIXTURE,
            host_trial_ids=_host_trial_ids(unbound),
        )


def test_plan_builder_requires_an_explicit_complete_host_trial_mapping() -> None:
    delivery_set = _delivery_set()
    host_trial_ids = _host_trial_ids(delivery_set)
    host_trial_ids.pop(next(iter(host_trial_ids)))

    with pytest.raises(TrialIngressPlanError, match="cover every external trial"):
        build_trial_ingress_plan_set(
            ingress_plan_set_id="external-tool-pilot-v1-forced",
            ingress_mode=IngressMode.FORCED_INGESTION,
            delivery_set=delivery_set,
            attack_set=_attack_set(),
            fixture=FIXTURE,
            host_trial_ids=host_trial_ids,
        )
