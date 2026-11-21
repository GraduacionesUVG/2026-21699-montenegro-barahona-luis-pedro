"""Resolve one child run's external ingress from frozen campaign evidence.

This module deliberately stops at identity resolution.  It does not schedule a
delivery, register inbox tools, or write to memory.  Keeping those operations
separate means a child can prove it has the exact intended external source
before any mutable runtime component is initialized.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from membench.campaign.artifacts import CampaignArtifactStore, canonical_json_bytes
from membench.campaign.bindings import CampaignRunBindings, RunProjectionBinding
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
    CampaignRunSpec,
)
from membench.curation.delivery_projections import (
    DELIVERY_MANIFEST_FILE,
    DELIVERY_PROJECTIONS_FILE,
    DeliveryProjectionError,
    DeliveryProjectionFamily,
    load_delivery_projection_set,
    validate_delivery_attack_links,
)
from membench.curation.external_attack_specs import (
    EXTERNAL_ATTACK_MANIFEST_FILE,
    EXTERNAL_ATTACK_SPECS_FILE,
    ExternalAttackSpec,
    ExternalAttackSpecError,
    load_external_attack_spec_set,
)
from membench.ingress.models import ExternalDelivery, IngressMode, SourceSelectionTask
from membench.ingress.plans import (
    INGRESS_PLAN_MANIFEST_FILE,
    INGRESS_PLANS_FILE,
    TrialIngressPlan,
    TrialIngressPlanError,
    TrialIngressPlanSet,
    load_trial_ingress_plan_set,
)
from membench.tools.fixtures import ToolFixtureBundle

_DESTINATION = "inputs/external_ingress"


class IngressResolutionError(RuntimeError):
    """Frozen ingress evidence cannot safely be used by the current child."""


@dataclass(frozen=True)
class ResolvedIngress:
    """One verified external source bound to one host trial.

    ``delivery`` retains the source content only after every identifier and
    digest has been checked against the compact run binding and frozen plans.
    A later lifecycle integration may consume this object without re-reading a
    mutable configuration directory.
    """

    plan_set: TrialIngressPlanSet
    plan: TrialIngressPlan
    delivery_family: DeliveryProjectionFamily
    delivery: ExternalDelivery
    attack_spec: ExternalAttackSpec
    selection_task: SourceSelectionTask | None


def resolve_frozen_ingress(
    *,
    store: CampaignArtifactStore,
    frozen_inputs: tuple[ArtifactReference, ...],
    bindings: CampaignRunBindings,
    binding: RunProjectionBinding,
    run: CampaignRunSpec,
    fixture: ToolFixtureBundle | None,
) -> ResolvedIngress | None:
    """Resolve and validate the optional ingress attached to one run binding.

    Every source file is verified through its ``ArtifactReference`` before the
    existing artifact loaders parse it.  This rejects a changed, missing, or
    substituted delivery before an adapter, tool, or model can observe it.
    """

    if not binding.has_ingress_plan:
        return None
    if bindings.ingress_plan_manifest is None:
        raise IngressResolutionError("ingress binding has no plan manifest")
    if fixture is None:
        raise IngressResolutionError("ingress binding requires a frozen fixture")

    manifest_reference = bindings.ingress_plan_manifest
    _require_frozen_reference(
        frozen_inputs,
        expected=manifest_reference,
        artifact_id="inputs:external-ingress:plans-manifest",
        kind=ArtifactKind.TRIAL_INGRESS_PLAN,
        relative_path=f"{_DESTINATION}/{INGRESS_PLAN_MANIFEST_FILE}",
        media_type="application/json",
    )
    plans_reference = _require_frozen_reference(
        frozen_inputs,
        artifact_id="inputs:external-ingress:plans",
        kind=ArtifactKind.TRIAL_INGRESS_PLAN,
        relative_path=f"{_DESTINATION}/{INGRESS_PLANS_FILE}",
        media_type="application/x-ndjson",
    )
    deliveries_reference = _require_frozen_reference(
        frozen_inputs,
        artifact_id="inputs:external-ingress:deliveries",
        kind=ArtifactKind.DATASET_PROJECTION,
        relative_path=f"{_DESTINATION}/{DELIVERY_PROJECTIONS_FILE}",
        media_type="application/x-ndjson",
    )
    _require_frozen_reference(
        frozen_inputs,
        artifact_id="inputs:external-ingress:delivery-manifest",
        kind=ArtifactKind.OTHER,
        relative_path=f"{_DESTINATION}/{DELIVERY_MANIFEST_FILE}",
        media_type="application/json",
    )
    _require_frozen_reference(
        frozen_inputs,
        artifact_id="inputs:external-ingress:attack-specs",
        kind=ArtifactKind.ATTACK_BATTERY,
        relative_path=f"{_DESTINATION}/{EXTERNAL_ATTACK_SPECS_FILE}",
        media_type="application/x-ndjson",
    )
    _require_frozen_reference(
        frozen_inputs,
        artifact_id="inputs:external-ingress:attack-manifest",
        kind=ArtifactKind.OTHER,
        relative_path=f"{_DESTINATION}/{EXTERNAL_ATTACK_MANIFEST_FILE}",
        media_type="application/json",
    )

    for reference in (
        manifest_reference,
        plans_reference,
        deliveries_reference,
    ):
        store.verify(reference)
    # The remaining required references are verified inside this loop too.  It
    # is intentionally separate from parsing so unregistered sibling files are
    # never trusted merely because they share the same directory.
    for artifact_id in (
        "inputs:external-ingress:delivery-manifest",
        "inputs:external-ingress:attack-specs",
        "inputs:external-ingress:attack-manifest",
    ):
        store.verify(_reference_by_id(frozen_inputs, artifact_id))

    source_directory = store.resolve_path(manifest_reference.relative_path).parent
    if (
        store.resolve_path(plans_reference.relative_path).parent != source_directory
        or store.resolve_path(deliveries_reference.relative_path).parent
        != source_directory
    ):
        raise IngressResolutionError("external ingress artifacts have different roots")
    try:
        plan_set = load_trial_ingress_plan_set(source_directory)
        delivery_set = load_delivery_projection_set(source_directory)
        attack_set = load_external_attack_spec_set(source_directory)
        validate_delivery_attack_links(delivery_set, attack_set)
    except (
        TrialIngressPlanError,
        DeliveryProjectionError,
        ExternalAttackSpecError,
        ValueError,
    ) as exc:
        raise IngressResolutionError(
            f"invalid frozen external ingress artifacts: {exc}"
        ) from exc

    _validate_plan_set_binding(plan_set, bindings, binding)
    try:
        plan = plan_set.plan_for_host_trial(binding.trial_id)
    except KeyError as exc:
        raise IngressResolutionError(
            "current host trial has no frozen ingress plan"
        ) from exc
    _validate_plan_for_run(plan, binding, run, fixture)

    try:
        family = next(
            item
            for item in delivery_set.families
            if item.delivery_family_id == plan.delivery_family_id
        )
    except StopIteration as exc:
        raise IngressResolutionError(
            "ingress plan delivery family is unavailable"
        ) from exc
    try:
        delivery = next(
            item for item in family.deliveries if item.delivery_id == plan.delivery_id
        )
    except StopIteration as exc:
        raise IngressResolutionError("ingress plan delivery is unavailable") from exc
    try:
        attack_spec = attack_set.spec_for(plan.attack_spec_id)
    except KeyError as exc:
        raise IngressResolutionError(
            "ingress plan attack specification is unavailable"
        ) from exc

    _validate_delivery(plan, family, delivery)
    if attack_spec.sha256 != plan.attack_spec_sha256:
        raise IngressResolutionError("ingress plan attack specification hash differs")
    selection_task = _validated_selection_task(plan, family.selection_task)
    return ResolvedIngress(
        plan_set=plan_set,
        plan=plan,
        delivery_family=family,
        delivery=delivery,
        attack_spec=attack_spec,
        selection_task=selection_task,
    )


def _require_frozen_reference(
    references: tuple[ArtifactReference, ...],
    *,
    artifact_id: str,
    kind: ArtifactKind,
    relative_path: str,
    media_type: str,
    expected: ArtifactReference | None = None,
) -> ArtifactReference:
    reference = _reference_by_id(references, artifact_id)
    if expected is not None and reference != expected:
        raise IngressResolutionError("ingress plan manifest differs from run bindings")
    if (
        reference.kind != kind
        or reference.relative_path != relative_path
        or reference.media_type != media_type
        or reference.visibility != ArtifactVisibility.EVALUATOR_ONLY
    ):
        raise IngressResolutionError(
            f"frozen ingress artifact has an unexpected identity: {artifact_id}"
        )
    return reference


def _reference_by_id(
    references: tuple[ArtifactReference, ...], artifact_id: str
) -> ArtifactReference:
    matches = tuple(item for item in references if item.artifact_id == artifact_id)
    if len(matches) != 1:
        raise IngressResolutionError(
            f"child inputs require exactly one {artifact_id} artifact"
        )
    return matches[0]


def _validate_plan_set_binding(
    plan_set: TrialIngressPlanSet,
    bindings: CampaignRunBindings,
    binding: RunProjectionBinding,
) -> None:
    manifest = plan_set.manifest()
    if bindings.ingress_plan_manifest is None:
        raise IngressResolutionError("ingress plan manifest is missing")
    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    if (
        bindings.ingress_plan_manifest.sha256
        != hashlib.sha256(manifest_bytes).hexdigest()
        or bindings.ingress_plan_manifest.size_bytes != len(manifest_bytes)
    ):
        raise IngressResolutionError(
            "frozen ingress plan manifest does not match plans"
        )
    if binding.ingress_plan_set_id != plan_set.ingress_plan_set_id:
        raise IngressResolutionError("ingress plan set differs from run binding")


def _validate_plan_for_run(
    plan: TrialIngressPlan,
    binding: RunProjectionBinding,
    run: CampaignRunSpec,
    fixture: ToolFixtureBundle,
) -> None:
    if (
        binding.ingress_plan_id != plan.ingress_plan_id
        or binding.ingress_plan_sha256 != plan.sha256
    ):
        raise IngressResolutionError("ingress plan ID or hash differs from run binding")
    if plan.host_trial_id != binding.trial_id:
        raise IngressResolutionError("ingress plan host trial differs from run binding")
    if plan.persona_id != run.persona_id or plan.twin != run.twin:
        raise IngressResolutionError("ingress plan persona or twin differs from run")
    if plan.fixture_id != fixture.fixture_id or plan.fixture_sha256 != fixture.sha256:
        raise IngressResolutionError("ingress plan fixture provenance differs")


def _validate_delivery(
    plan: TrialIngressPlan,
    family: DeliveryProjectionFamily,
    delivery: ExternalDelivery,
) -> None:
    if (
        family.persona_id != plan.persona_id
        or family.fixture_id != plan.fixture_id
        or family.fixture_sha256 != plan.fixture_sha256
        or family.attack_spec_id != plan.attack_spec_id
        or family.attack_spec_sha256 != plan.attack_spec_sha256
    ):
        raise IngressResolutionError("delivery family differs from ingress plan")
    if (
        delivery.trial_id != plan.trial_id
        or delivery.twin != plan.twin
        or delivery.delivery_family_id != plan.delivery_family_id
        or delivery.content_sha256 != plan.delivery_content_sha256
        or delivery.source_kind != plan.source_kind
        or delivery.allocation_case_id != plan.allocation_case_id
    ):
        raise IngressResolutionError("delivery differs from ingress plan")


def _validated_selection_task(
    plan: TrialIngressPlan,
    task: SourceSelectionTask | None,
) -> SourceSelectionTask | None:
    if plan.ingress_mode == IngressMode.FORCED_INGESTION:
        return None
    if task is None:
        raise IngressResolutionError("agent-selected ingress has no selection task")
    task_sha256 = hashlib.sha256(canonical_json_bytes(task)).hexdigest()
    if (
        plan.selection_task_id != task.selection_task_id
        or plan.selection_task_sha256 != task_sha256
        or plan.allowed_tools != task.allowed_tools
    ):
        raise IngressResolutionError("ingress selection task differs from plan")
    return task
