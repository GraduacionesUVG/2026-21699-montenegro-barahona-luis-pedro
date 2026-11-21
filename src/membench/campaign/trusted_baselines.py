"""Load and freeze optional trusted-baseline seed manifests for a campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from membench.campaign.artifacts import CampaignArtifactStore
from membench.campaign.inputs import CampaignInputs
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
)
from membench.campaign.preparation_models import CampaignPreparationOptions
from membench.ingress.models import TrustedBaselineManifest, TrustedBaselineSeed

_DESTINATION = "inputs/trusted_baselines"


class CampaignTrustedBaselineInputError(RuntimeError):
    """An editable baseline source cannot safely join a campaign."""


@dataclass(frozen=True)
class TrustedBaselineInputs:
    """Validated editable seed manifest before campaign-local freezing."""

    source_path: Path
    manifest: TrustedBaselineManifest

    def for_persona(self, persona_id: str) -> TrustedBaselineSeed:
        matches = tuple(
            item
            for item in self.manifest.baselines
            if item.persona_id == int(persona_id)
        )
        if len(matches) != 1:
            raise CampaignTrustedBaselineInputError(
                "trusted baseline manifest must contain exactly one seed for "
                f"persona {persona_id}"
            )
        return matches[0]


@dataclass(frozen=True)
class FrozenTrustedBaselines:
    """Immutable trusted-baseline manifest and its sealed artifact reference."""

    manifest: TrustedBaselineManifest
    artifact: ArtifactReference

    def for_persona(self, persona_id: str) -> TrustedBaselineSeed:
        """Return the one frozen seed assigned to a selected persona."""

        matches = tuple(
            item
            for item in self.manifest.baselines
            if item.persona_id == int(persona_id)
        )
        if len(matches) != 1:
            raise CampaignTrustedBaselineInputError(
                "frozen trusted baselines contain no unique seed for persona "
                f"{persona_id}"
            )
        return matches[0]


def load_trusted_baseline_inputs(
    options: CampaignPreparationOptions,
    campaign_inputs: CampaignInputs,
    *,
    project_root: Path,
) -> TrustedBaselineInputs | None:
    """Load an optional manifest and require one seed per selected persona."""

    options = CampaignPreparationOptions.model_validate(options.model_dump(mode="json"))
    configured = options.trusted_baseline_path
    if configured is None:
        return None
    path = (project_root / configured).resolve()
    if (
        not path.is_relative_to(project_root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise CampaignTrustedBaselineInputError(
            "trusted baseline manifest must be an existing regular file below "
            "the project root"
        )
    try:
        manifest = TrustedBaselineManifest.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise CampaignTrustedBaselineInputError(
            "could not load a valid trusted baseline manifest"
        ) from exc
    inputs = TrustedBaselineInputs(source_path=path, manifest=manifest)
    for persona_id in {case.case.persona_id for case in campaign_inputs.cases}:
        inputs.for_persona(persona_id)
    return inputs


def freeze_trusted_baseline_inputs(
    inputs: TrustedBaselineInputs,
    store: CampaignArtifactStore,
) -> FrozenTrustedBaselines:
    """Persist canonical trusted seeds as evaluator-only campaign evidence."""

    store.prepare_directory(_DESTINATION)
    artifact = store.write_json(
        artifact_id="inputs:trusted-baselines",
        kind=ArtifactKind.TRUSTED_BASELINE,
        relative_path=f"{_DESTINATION}/manifest.json",
        value=inputs.manifest,
        visibility=ArtifactVisibility.EVALUATOR_ONLY,
    )
    return FrozenTrustedBaselines(manifest=inputs.manifest, artifact=artifact)
