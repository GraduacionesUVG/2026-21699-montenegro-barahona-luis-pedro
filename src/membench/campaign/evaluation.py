"""Load and freeze one optional evaluator-only retrieval probe per campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from membench.campaign.artifacts import CampaignArtifactStore
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
)
from membench.campaign.preparation_models import CampaignPreparationOptions
from membench.evaluation.loading import EvaluationProbeLoadError, load_evaluation_probe
from membench.evaluation.models import EvaluationProbe

_DESTINATION = "inputs/evaluation"


class CampaignEvaluationInputError(RuntimeError):
    """An editable evaluator probe cannot safely join a prepared campaign."""


@dataclass(frozen=True)
class EvaluationProbeInput:
    """One validated editable probe, before it becomes campaign evidence."""

    source_path: Path
    probe: EvaluationProbe


@dataclass(frozen=True)
class FrozenEvaluationProbe:
    """Campaign-local immutable reference to the evaluator probe."""

    probe: EvaluationProbe
    artifact: ArtifactReference


def load_evaluation_probe_input(
    options: CampaignPreparationOptions,
    *,
    project_root: Path,
) -> EvaluationProbeInput | None:
    """Load the optional editable source without treating it as frozen yet."""

    options = CampaignPreparationOptions.model_validate(options.model_dump(mode="json"))
    configured = options.evaluation_probe_path
    if configured is None:
        return None
    path = (project_root / configured).resolve()
    if path.is_symlink() or not path.is_file():
        raise CampaignEvaluationInputError(
            "evaluation probe must be an existing regular file"
        )
    try:
        probe = load_evaluation_probe(path)
    except EvaluationProbeLoadError as exc:
        raise CampaignEvaluationInputError(
            "could not load evaluation probe"
        ) from exc
    return EvaluationProbeInput(source_path=path, probe=probe)


def freeze_evaluation_probe_input(
    source: EvaluationProbeInput,
    store: CampaignArtifactStore,
) -> FrozenEvaluationProbe:
    """Persist canonical evaluator input as a sealed campaign dependency."""

    store.prepare_directory(_DESTINATION)
    artifact = store.write_json(
        artifact_id="inputs:evaluation-probe",
        kind=ArtifactKind.EVALUATION_PROBE,
        relative_path=f"{_DESTINATION}/probe.json",
        value=source.probe,
        visibility=ArtifactVisibility.EVALUATOR_ONLY,
    )
    return FrozenEvaluationProbe(probe=source.probe, artifact=artifact)
