"""Load, validate, and freeze optional behavioral task inputs for a campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from membench.behavioral.loading import (
    BehavioralInputLoadError,
    load_behavioral_oracle,
    load_behavioral_task,
    validate_behavioral_pair,
)
from membench.behavioral.models import (
    BehavioralTaskTemplate,
    CaseSummaryBehavioralOracle,
)
from membench.campaign.artifacts import CampaignArtifactStore
from membench.campaign.external_ingress import ExternalIngressInputs
from membench.campaign.inputs import CampaignInputs
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
)
from membench.campaign.preparation_models import CampaignPreparationOptions

_DESTINATION = "inputs/behavioral"


class CampaignBehavioralInputError(RuntimeError):
    """Behavioral inputs cannot safely join the selected campaign."""


@dataclass(frozen=True)
class BehavioralInputs:
    """Validated editable behavioral inputs prior to campaign-local freezing."""

    task_source_path: Path
    oracle_source_path: Path
    task: BehavioralTaskTemplate
    oracle: CaseSummaryBehavioralOracle


@dataclass(frozen=True)
class FrozenBehavioralInputs:
    """Campaign-local immutable task and evaluator-only oracle references."""

    task: BehavioralTaskTemplate
    oracle: CaseSummaryBehavioralOracle
    task_artifact: ArtifactReference
    oracle_artifact: ArtifactReference

    @property
    def artifacts(self) -> tuple[ArtifactReference, ArtifactReference]:
        return self.task_artifact, self.oracle_artifact


def load_behavioral_inputs(
    options: CampaignPreparationOptions,
    campaign_inputs: CampaignInputs,
    external_ingress: ExternalIngressInputs | None,
    *,
    project_root: Path,
) -> BehavioralInputs | None:
    """Load and match a task/oracle pair to selected external delivery inputs."""

    options = CampaignPreparationOptions.model_validate(options.model_dump(mode="json"))
    task_configured = options.behavioral_task_path
    oracle_configured = options.behavioral_oracle_path
    if task_configured is None and oracle_configured is None:
        return None
    if task_configured is None or oracle_configured is None:
        raise CampaignBehavioralInputError(
            "behavioral task and behavioral oracle paths must be configured together"
        )
    if external_ingress is None:
        raise CampaignBehavioralInputError(
            "behavioral tasks currently require frozen external ingress"
        )
    task_path = (project_root / task_configured).resolve()
    oracle_path = (project_root / oracle_configured).resolve()
    if any(
        path.is_symlink() or not path.is_file()
        for path in (task_path, oracle_path)
    ):
        raise CampaignBehavioralInputError(
            "behavioral task and oracle must be existing regular files"
        )
    try:
        task = load_behavioral_task(task_path)
        oracle = load_behavioral_oracle(oracle_path)
        validate_behavioral_pair(task, oracle)
    except BehavioralInputLoadError as exc:
        raise CampaignBehavioralInputError(
            "could not load a valid behavioral task/oracle pair"
        ) from exc

    selected_personas = {case.case.persona_id for case in campaign_inputs.cases}
    if selected_personas != {task.persona_id}:
        raise CampaignBehavioralInputError(
            "behavioral task persona must match every selected campaign case"
        )
    try:
        spec = external_ingress.attack_set.spec_for(task.attack_spec_id)
    except KeyError as exc:
        raise CampaignBehavioralInputError(
            "behavioral task references an unknown external attack spec"
        ) from exc
    families = tuple(
        family
        for family in external_ingress.delivery_set.families
        if family.delivery_family_id == task.delivery_family_id
    )
    if len(families) != 1:
        raise CampaignBehavioralInputError(
            "behavioral task references an unknown external delivery family"
        )
    family = families[0]
    if family.attack_spec_id != spec.attack_spec_id:
        raise CampaignBehavioralInputError(
            "behavioral task delivery family does not match external attack spec"
        )
    if family.persona_id is not None and family.persona_id != task.persona_id:
        raise CampaignBehavioralInputError(
            "behavioral task delivery family persona does not match task persona"
        )
    return BehavioralInputs(
        task_source_path=task_path,
        oracle_source_path=oracle_path,
        task=task,
        oracle=oracle,
    )


def freeze_behavioral_inputs(
    inputs: BehavioralInputs,
    store: CampaignArtifactStore,
) -> FrozenBehavioralInputs:
    """Persist canonical task and oracle separately under the campaign root."""

    store.prepare_directory(_DESTINATION)
    task_artifact = store.write_json(
        artifact_id="inputs:behavioral-task",
        kind=ArtifactKind.BEHAVIORAL_TASK,
        relative_path=f"{_DESTINATION}/task.json",
        value=inputs.task,
        visibility=ArtifactVisibility.EVALUATOR_ONLY,
    )
    oracle_artifact = store.write_json(
        artifact_id="inputs:behavioral-oracle",
        kind=ArtifactKind.BEHAVIORAL_ORACLE,
        relative_path=f"{_DESTINATION}/oracle.json",
        value=inputs.oracle,
        visibility=ArtifactVisibility.EVALUATOR_ONLY,
    )
    return FrozenBehavioralInputs(
        task=inputs.task,
        oracle=inputs.oracle,
        task_artifact=task_artifact,
        oracle_artifact=oracle_artifact,
    )
