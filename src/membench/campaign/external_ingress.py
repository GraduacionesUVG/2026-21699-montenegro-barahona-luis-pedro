"""Load, bind, and freeze email-only external ingress for campaign preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from membench.campaign.artifacts import CampaignArtifactStore, canonical_json_bytes
from membench.campaign.inputs import CampaignInputs
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
)
from membench.campaign.preparation_models import CampaignPreparationOptions
from membench.curation.delivery_projections import (
    DELIVERY_MANIFEST_FILE,
    DELIVERY_PROJECTIONS_FILE,
    DeliveryProjectionError,
    DeliveryProjectionSet,
    load_delivery_projection_set,
    validate_delivery_attack_links,
)
from membench.curation.external_attack_specs import (
    EXTERNAL_ATTACK_MANIFEST_FILE,
    EXTERNAL_ATTACK_SPECS_FILE,
    ExternalAttackSpecError,
    ExternalAttackSpecSet,
    load_external_attack_spec_set,
)
from membench.ingress.models import ExternalSourceKind
from membench.ingress.plans import (
    INGRESS_PLAN_MANIFEST_FILE,
    INGRESS_PLANS_FILE,
    TrialIngressPlanError,
    TrialIngressPlanSet,
    build_trial_ingress_plan_set,
)

_DESTINATION = "inputs/external_ingress"
_SOURCE_FILES = (
    EXTERNAL_ATTACK_SPECS_FILE,
    EXTERNAL_ATTACK_MANIFEST_FILE,
    DELIVERY_PROJECTIONS_FILE,
    DELIVERY_MANIFEST_FILE,
)


class ExternalIngressInputError(RuntimeError):
    """The external source artifact cannot safely join this campaign."""


@dataclass(frozen=True)
class ExternalIngressInputs:
    """Verified, byte-preserving source snapshot plus its derived plan set."""

    source_directory: Path
    attack_set: ExternalAttackSpecSet
    delivery_set: DeliveryProjectionSet
    payloads: tuple[tuple[str, bytes], ...]
    plan_set: TrialIngressPlanSet


@dataclass(frozen=True)
class FrozenExternalIngress:
    """Campaign-local references for copied deliveries and generated plans."""

    artifacts: tuple[ArtifactReference, ...]
    ingress_plan_manifest: ArtifactReference
    plan_set: TrialIngressPlanSet


def load_external_ingress_inputs(
    options: CampaignPreparationOptions,
    campaign_inputs: CampaignInputs,
    *,
    project_root: Path,
) -> ExternalIngressInputs | None:
    """Verify a frozen external artifact and bind it to selected host trials.

    This preparation path intentionally supports email only.  Document models
    remain available in the generic ingress package but cannot be scheduled
    until their child-runtime delivery path exists.
    """

    options = CampaignPreparationOptions.model_validate(options.model_dump(mode="json"))
    configured = options.external_ingress
    if configured is None:
        return None
    if campaign_inputs.tool_fixture is None:
        raise ExternalIngressInputError(
            "external ingress preparation requires a frozen tool fixture"
        )
    source = (project_root / configured.external_artifact_directory).resolve()
    payloads = _read_external_snapshot(source)
    try:
        attack_set = load_external_attack_spec_set(source)
        delivery_set = load_delivery_projection_set(source)
        validate_delivery_attack_links(delivery_set, attack_set)
    except (
        DeliveryProjectionError,
        ExternalAttackSpecError,
        ValueError,
    ) as exc:
        raise ExternalIngressInputError(
            f"invalid frozen external ingress artifact: {exc}"
        ) from exc
    _validate_snapshot_bytes(payloads, attack_set, delivery_set)
    if any(
        family.source_kind != ExternalSourceKind.EMAIL
        for family in delivery_set.families
    ):
        raise ExternalIngressInputError(
            "campaign preparation currently supports external email ingress only"
        )
    if options.purpose == "experiment" and any(
        spec.review_status.value != "approved" for spec in attack_set.specs
    ):
        raise ExternalIngressInputError(
            "unreviewed external attacks require pilot campaign purpose"
        )

    host_trials = {
        trial.trial_id: (case.case.persona_id, trial.twin)
        for case in campaign_inputs.cases
        for trial in case.trials
    }
    unknown_hosts = sorted(set(configured.host_trial_ids.values()) - set(host_trials))
    if unknown_hosts:
        raise ExternalIngressInputError(
            "external ingress references unselected host trials: "
            + ", ".join(unknown_hosts)
        )
    try:
        plan_set = build_trial_ingress_plan_set(
            ingress_plan_set_id=configured.ingress_plan_set_id,
            ingress_mode=configured.ingress_mode,
            delivery_set=delivery_set,
            attack_set=attack_set,
            fixture=campaign_inputs.tool_fixture,
            host_trial_ids=configured.host_trial_ids,
            selection_agent_configuration=configured.selection_agent_configuration,
        )
    except (TrialIngressPlanError, ValueError) as exc:
        raise ExternalIngressInputError(
            f"could not construct external ingress plans: {exc}"
        ) from exc
    for plan in plan_set.plans:
        persona_id, twin = host_trials[plan.host_trial_id]
        if plan.persona_id != persona_id or plan.twin != twin:
            raise ExternalIngressInputError(
                "external ingress host trial persona or twin does not match "
                f"delivery plan: {plan.host_trial_id}"
            )
        if plan.ingress_mode.value == "agent_selected":
            task = next(
                family.selection_task
                for family in delivery_set.families
                if family.delivery_family_id == plan.delivery_family_id
            )
            assert task is not None
            assert plan.selection_agent_configuration is not None
            if plan.selection_agent_configuration.allowed_tools != task.allowed_tools:
                raise ExternalIngressInputError(
                    "selection configuration tools differ from the frozen task"
                )
    return ExternalIngressInputs(
        source_directory=source,
        attack_set=attack_set,
        delivery_set=delivery_set,
        payloads=tuple(payloads.items()),
        plan_set=plan_set,
    )


def freeze_external_ingress_inputs(
    inputs: ExternalIngressInputs,
    store: CampaignArtifactStore,
) -> FrozenExternalIngress:
    """Copy verified external evidence and derived ingress plans into a campaign."""

    store.prepare_directory(_DESTINATION)
    payloads = dict(inputs.payloads)
    references = (
        store.write_bytes(
            artifact_id="inputs:external-ingress:attack-specs",
            kind=ArtifactKind.ATTACK_BATTERY,
            relative_path=f"{_DESTINATION}/{EXTERNAL_ATTACK_SPECS_FILE}",
            content=payloads[EXTERNAL_ATTACK_SPECS_FILE],
            media_type="application/x-ndjson",
            visibility=ArtifactVisibility.EVALUATOR_ONLY,
        ),
        store.write_bytes(
            artifact_id="inputs:external-ingress:attack-manifest",
            kind=ArtifactKind.OTHER,
            relative_path=f"{_DESTINATION}/{EXTERNAL_ATTACK_MANIFEST_FILE}",
            content=payloads[EXTERNAL_ATTACK_MANIFEST_FILE],
            media_type="application/json",
            visibility=ArtifactVisibility.EVALUATOR_ONLY,
        ),
        store.write_bytes(
            artifact_id="inputs:external-ingress:deliveries",
            kind=ArtifactKind.DATASET_PROJECTION,
            relative_path=f"{_DESTINATION}/{DELIVERY_PROJECTIONS_FILE}",
            content=payloads[DELIVERY_PROJECTIONS_FILE],
            media_type="application/x-ndjson",
            visibility=ArtifactVisibility.EVALUATOR_ONLY,
        ),
        store.write_bytes(
            artifact_id="inputs:external-ingress:delivery-manifest",
            kind=ArtifactKind.OTHER,
            relative_path=f"{_DESTINATION}/{DELIVERY_MANIFEST_FILE}",
            content=payloads[DELIVERY_MANIFEST_FILE],
            media_type="application/json",
            visibility=ArtifactVisibility.EVALUATOR_ONLY,
        ),
        store.write_bytes(
            artifact_id="inputs:external-ingress:plans",
            kind=ArtifactKind.TRIAL_INGRESS_PLAN,
            relative_path=f"{_DESTINATION}/{INGRESS_PLANS_FILE}",
            content=inputs.plan_set.jsonl_bytes(),
            media_type="application/x-ndjson",
            visibility=ArtifactVisibility.EVALUATOR_ONLY,
        ),
    )
    manifest = store.write_bytes(
        artifact_id="inputs:external-ingress:plans-manifest",
        kind=ArtifactKind.TRIAL_INGRESS_PLAN,
        relative_path=f"{_DESTINATION}/{INGRESS_PLAN_MANIFEST_FILE}",
        content=canonical_json_bytes(
            inputs.plan_set.manifest().model_dump(mode="json")
        ),
        media_type="application/json",
        visibility=ArtifactVisibility.EVALUATOR_ONLY,
    )
    return FrozenExternalIngress(
        artifacts=(*references, manifest),
        ingress_plan_manifest=manifest,
        plan_set=inputs.plan_set,
    )


def _read_external_snapshot(source: Path) -> dict[str, bytes]:
    if not source.is_dir() or source.is_symlink():
        raise ExternalIngressInputError(
            f"external ingress artifact directory does not exist: {source}"
        )
    payloads: dict[str, bytes] = {}
    try:
        for name in _SOURCE_FILES:
            path = (source / name).resolve()
            if (
                not path.is_relative_to(source)
                or path.is_symlink()
                or not path.is_file()
            ):
                raise ExternalIngressInputError(
                    f"missing or unsafe external ingress artifact file: {name}"
                )
            payloads[name] = path.read_bytes()
    except OSError as exc:
        raise ExternalIngressInputError(
            f"could not read external ingress artifact: {exc}"
        ) from exc
    return payloads


def _validate_snapshot_bytes(
    payloads: dict[str, bytes],
    attack_set: ExternalAttackSpecSet,
    delivery_set: DeliveryProjectionSet,
) -> None:
    """Ensure parsed inputs and retained bytes describe exactly the same export."""

    if payloads[EXTERNAL_ATTACK_SPECS_FILE] != attack_set.jsonl_bytes() or (
        payloads[EXTERNAL_ATTACK_MANIFEST_FILE]
        != canonical_json_bytes(attack_set.manifest().model_dump(mode="json"))
    ):
        raise ExternalIngressInputError(
            "external attack artifact changed while preparation was reading it"
        )
    if payloads[DELIVERY_PROJECTIONS_FILE] != delivery_set.jsonl_bytes() or (
        payloads[DELIVERY_MANIFEST_FILE]
        != canonical_json_bytes(delivery_set.manifest().model_dump(mode="json"))
    ):
        raise ExternalIngressInputError(
            "external delivery artifact changed while preparation was reading it"
        )
