"""Campaign binding validation for frozen external ingress plans."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from membench.campaign.artifacts import canonical_json_bytes
from membench.campaign.bindings import CampaignRunBindings, RunProjectionBinding
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    CampaignConfiguration,
    ChildEnvironmentSpec,
    TrialTwin,
)
from membench.ingress.models import ExternalSourceKind, IngressMode
from membench.ingress.plans import TrialIngressPlan, TrialIngressPlanSet


def _reference(
    *, artifact_id: str, kind: ArtifactKind, payload: bytes
) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        kind=kind,
        relative_path=f"inputs/{artifact_id}.json",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        media_type="application/json",
    )


def _ingress_plan_set(plans) -> TrialIngressPlanSet:  # type: ignore[no-untyped-def]
    return TrialIngressPlanSet(
        ingress_plan_set_id="external-ingress-v1",
        ingress_mode=IngressMode.FORCED_INGESTION,
        delivery_set_id="external-deliveries-v1",
        delivery_manifest_sha256="a" * 64,
        attack_set_id="external-attacks-v1",
        attack_manifest_sha256="b" * 64,
        fixture_id="external-fixture-v1",
        fixture_sha256="c" * 64,
        plans=tuple(
            TrialIngressPlan(
                plan_ordinal=ordinal,
                ingress_plan_id=f"ingress-external-{plan.run.twin.value}",
                trial_id=f"external-trial-{plan.run.twin.value}",
                host_trial_id=plan.projection.trial_id,
                allocation_case_id="alloc-external-001",
                twin=plan.run.twin,
                persona_id=plan.run.persona_id,
                ingress_mode=IngressMode.FORCED_INGESTION,
                delivery_family_id="delivery-external-001",
                delivery_id=f"delivery-external-001.{plan.run.twin.value}",
                delivery_content_sha256=(str(ordinal) * 64)[:64],
                attack_spec_id="external-attack-001",
                attack_spec_sha256="d" * 64,
                source_kind=ExternalSourceKind.EMAIL,
                fixture_id="external-fixture-v1",
                fixture_sha256="c" * 64,
            )
            for ordinal, plan in enumerate(plans, start=1)
        ),
    )


def _campaign_contracts(trial_plan_factory):  # type: ignore[no-untyped-def]
    plans = tuple(trial_plan_factory(twin=twin) for twin in TrialTwin)
    projection = _reference(
        artifact_id="projection-manifest",
        kind=ArtifactKind.DATASET_PROJECTION,
        payload=b"projection-manifest",
    )
    attacks = ArtifactReference(
        artifact_id="attack-specs",
        kind=ArtifactKind.ATTACK_BATTERY,
        relative_path="inputs/attack-specs.jsonl",
        sha256="e" * 64,
        size_bytes=1,
        media_type="application/x-ndjson",
    )
    plan_set = _ingress_plan_set(plans)
    ingress_manifest = _reference(
        artifact_id="trial-ingress-plans-manifest",
        kind=ArtifactKind.TRIAL_INGRESS_PLAN,
        payload=canonical_json_bytes(plan_set.manifest().model_dump(mode="json")),
    )
    configuration = CampaignConfiguration(
        campaign_id="pilot-001",
        seed=20261006,
        runs=tuple(plan.run for plan in plans),
        environments=(
            ChildEnvironmentSpec(
                mechanism="telemem",
                python_executable=".venvs/telemem/Scripts/python.exe",
            ),
        ),
        frozen_inputs=(projection, attacks, ingress_manifest),
        artifact_root="artifacts/campaigns/ingress-binding-test",
    )
    bindings = tuple(
        RunProjectionBinding.from_trial_plan(
            plan,
            condition_id="telemem-native",
            model_profile_id="qwen-agent",
        ).with_ingress_plan(
            plan_set.plan_for_host_trial(plan.projection.trial_id),
            ingress_plan_set_id=plan_set.ingress_plan_set_id,
        )
        for plan in plans
    )
    inventory = CampaignRunBindings(
        campaign_id="pilot-001",
        preparation_sha256="f" * 64,
        projection_set_id="synthetic-projections-v1",
        projection_manifest=projection,
        ingress_plan_manifest=ingress_manifest,
        bindings=bindings,
    )
    return configuration, inventory, plan_set


def _rebind_plan_set(
    configuration: CampaignConfiguration,
    inventory: CampaignRunBindings,
    plan_set: TrialIngressPlanSet,
) -> tuple[CampaignConfiguration, CampaignRunBindings]:
    manifest = _reference(
        artifact_id="trial-ingress-plans-manifest",
        kind=ArtifactKind.TRIAL_INGRESS_PLAN,
        payload=canonical_json_bytes(plan_set.manifest().model_dump(mode="json")),
    )
    bindings = tuple(
        binding.with_ingress_plan(
            plan_set.plan_for_host_trial(binding.trial_id),
            ingress_plan_set_id=plan_set.ingress_plan_set_id,
        )
        for binding in inventory.bindings
    )
    previous_manifest = inventory.ingress_plan_manifest
    inventory = CampaignRunBindings.model_validate(
        {
            **inventory.model_dump(mode="json"),
            "ingress_plan_manifest": manifest,
            "bindings": bindings,
        }
    )
    configuration = CampaignConfiguration.model_validate(
        {
            **configuration.model_dump(mode="json"),
            "frozen_inputs": [
                manifest if ref == previous_manifest else ref
                for ref in configuration.frozen_inputs
            ],
        }
    )
    return configuration, inventory


def test_external_bindings_validate_exact_frozen_ingress_plan_set(
    trial_plan_factory,
) -> None:
    configuration, inventory, plan_set = _campaign_contracts(trial_plan_factory)

    inventory.validate_campaign(configuration, ingress_plan_set=plan_set)
    assert all(binding.has_ingress_plan for binding in inventory.bindings)


def test_ingress_reference_fields_are_all_or_none(trial_plan_factory) -> None:
    _, inventory, _ = _campaign_contracts(trial_plan_factory)
    data = inventory.bindings[0].model_dump(mode="json")
    data["ingress_plan_sha256"] = None

    with pytest.raises(ValidationError, match="must be present together"):
        RunProjectionBinding.model_validate(data)


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing", "require an ingress-plan manifest"),
        ("kind", "trial ingress plan"),
        ("visibility", "evaluator-only"),
        ("media_type", "evaluator-only JSON"),
    ],
)
def test_external_bindings_require_an_evaluator_only_json_manifest(
    trial_plan_factory,
    change: str,
    match: str,
) -> None:
    _, inventory, _ = _campaign_contracts(trial_plan_factory)
    data = inventory.model_dump(mode="json")
    if change == "missing":
        data["ingress_plan_manifest"] = None
    elif change == "kind":
        data["ingress_plan_manifest"]["kind"] = "other"
    elif change == "visibility":
        data["ingress_plan_manifest"]["visibility"] = "public"
    else:
        data["ingress_plan_manifest"]["media_type"] = "application/x-ndjson"

    with pytest.raises(ValidationError, match=match):
        CampaignRunBindings.model_validate(data)


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing_loaded", "requires loaded ingress plans"),
        ("missing_frozen", "not a frozen campaign input"),
        ("wrong_plan_sha", "does not match loaded plan"),
        ("wrong_persona", "persona or twin"),
        ("different_mode", "one ingress mode"),
        ("different_family", "one delivery family"),
        ("duplicate_delivery", "distinct delivery IDs"),
        ("duplicate_content", "distinct delivery-content hashes"),
    ],
)
def test_external_binding_validation_rejects_mismatched_ingress(
    trial_plan_factory,
    change: str,
    match: str,
) -> None:
    configuration, inventory, plan_set = _campaign_contracts(trial_plan_factory)
    if change == "missing_loaded":
        with pytest.raises(ValueError, match=match):
            inventory.validate_campaign(configuration)
        return
    if change == "missing_frozen":
        configuration = configuration.model_copy(
            update={
                "frozen_inputs": tuple(
                    ref
                    for ref in configuration.frozen_inputs
                    if ref != inventory.ingress_plan_manifest
                )
            }
        )
    elif change == "wrong_plan_sha":
        altered = list(inventory.bindings)
        altered[0] = altered[0].model_copy(update={"ingress_plan_sha256": "0" * 64})
        inventory = inventory.model_copy(update={"bindings": tuple(altered)})
    else:
        plans = list(plan_set.plans)
        if change == "wrong_persona":
            plans[0] = plans[0].model_copy(update={"persona_id": "999"})
        elif change == "different_mode":
            plans[1] = plans[1].model_copy(
                update={"ingress_mode": IngressMode.AGENT_SELECTED}
            )
        elif change == "different_family":
            plans[1] = plans[1].model_copy(
                update={"delivery_family_id": "delivery-external-002"}
            )
        elif change == "duplicate_delivery":
            plans[1] = plans[1].model_copy(update={"delivery_id": plans[0].delivery_id})
        else:
            plans[1] = plans[1].model_copy(
                update={"delivery_content_sha256": plans[0].delivery_content_sha256}
            )
        plan_set = plan_set.model_copy(update={"plans": tuple(plans)})
        configuration, inventory = _rebind_plan_set(configuration, inventory, plan_set)

    with pytest.raises(ValueError, match=match):
        inventory.validate_campaign(configuration, ingress_plan_set=plan_set)


def test_conversation_only_campaign_remains_valid_without_ingress(
    trial_plan_factory,
) -> None:
    configuration, inventory, _ = _campaign_contracts(trial_plan_factory)
    bindings = tuple(
        binding.model_copy(
            update={
                "ingress_plan_set_id": None,
                "ingress_plan_id": None,
                "ingress_plan_sha256": None,
                    "external_attack_spec_id": None,
                    "external_attack_spec_sha256": None,
                    "external_delivery_family_id": None,
                    "external_ingress_mode": None,
            }
        )
        for binding in inventory.bindings
    )
    inventory = CampaignRunBindings.model_validate(
        {
            **inventory.model_dump(mode="json"),
            "ingress_plan_manifest": None,
            "bindings": bindings,
        }
    )
    configuration = configuration.model_copy(
        update={
            "frozen_inputs": tuple(
                ref
                for ref in configuration.frozen_inputs
                if ref.kind != ArtifactKind.TRIAL_INGRESS_PLAN
            )
        }
    )

    inventory.validate_campaign(configuration)
