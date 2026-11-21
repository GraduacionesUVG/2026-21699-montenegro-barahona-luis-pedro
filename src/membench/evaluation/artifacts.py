"""Immutable, evaluator-only persistence for one evaluation execution."""

from __future__ import annotations

import re
from dataclasses import dataclass

from membench.campaign.artifacts import CampaignArtifactStore
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
)
from membench.evaluation.models import EvaluationExecutionResult

_DIRECTORY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class EvaluationArtifactError(RuntimeError):
    """Raised when evaluator evidence cannot be assigned a safe identity."""


@dataclass(frozen=True, slots=True)
class EvaluationArtifactSet:
    """References for every immutable file emitted by one evaluation probe."""

    probe: ArtifactReference
    retrieval: ArtifactReference
    marker_matches: ArtifactReference
    state_before: ArtifactReference
    state_after: ArtifactReference
    state_delta: ArtifactReference
    execution: ArtifactReference

    @property
    def references(self) -> tuple[ArtifactReference, ...]:
        """Return all references in their deterministic evidence order."""

        return (
            self.probe,
            self.retrieval,
            self.marker_matches,
            self.state_before,
            self.state_after,
            self.state_delta,
            self.execution,
        )


def write_evaluation_artifacts(
    store: CampaignArtifactStore,
    execution: EvaluationExecutionResult,
    *,
    directory: str = "evaluation",
) -> EvaluationArtifactSet:
    """Persist complete, immutable evidence for one scored retrieval probe.

    ``store`` owns atomic writes, collision detection, checksums, and final
    sealing.  This function only gives evaluator evidence stable identities and
    paths beneath the current trial artifact root.
    """

    if not _DIRECTORY_PATTERN.fullmatch(directory):
        raise EvaluationArtifactError("evaluation artifact directory is invalid")

    evaluation = execution.evaluation
    probe = evaluation.probe
    trial_id = evaluation.trial_id
    prefix = f"{directory}/{probe.probe_id}"
    identity_prefix = f"evaluation:{trial_id}:{probe.probe_id}"
    visibility = ArtifactVisibility.EVALUATOR_ONLY

    return EvaluationArtifactSet(
        probe=store.write_json(
            artifact_id=f"{identity_prefix}:probe",
            kind=ArtifactKind.EVALUATION_PROBE,
            relative_path=f"{prefix}/probe.json",
            value=probe,
            visibility=visibility,
        ),
        retrieval=store.write_json(
            artifact_id=f"{identity_prefix}:retrieval",
            kind=ArtifactKind.EVALUATION_RESULT,
            relative_path=f"{prefix}/retrieval.json",
            value=evaluation.retrieval,
            visibility=visibility,
        ),
        marker_matches=store.write_json(
            artifact_id=f"{identity_prefix}:marker-matches",
            kind=ArtifactKind.EVALUATION_RESULT,
            relative_path=f"{prefix}/marker-matches.json",
            value=evaluation.marker_matches,
            visibility=visibility,
        ),
        state_before=store.write_json(
            artifact_id=f"{identity_prefix}:state-before",
            kind=ArtifactKind.SNAPSHOT,
            relative_path=f"{prefix}/state-before.json",
            value=execution.before_snapshot,
            visibility=visibility,
        ),
        state_after=store.write_json(
            artifact_id=f"{identity_prefix}:state-after",
            kind=ArtifactKind.SNAPSHOT,
            relative_path=f"{prefix}/state-after.json",
            value=execution.after_snapshot,
            visibility=visibility,
        ),
        state_delta=store.write_json(
            artifact_id=f"{identity_prefix}:state-delta",
            kind=ArtifactKind.EVALUATION_RESULT,
            relative_path=f"{prefix}/state-delta.json",
            value=execution.state_delta,
            visibility=visibility,
        ),
        execution=store.write_json(
            artifact_id=f"{identity_prefix}:execution",
            kind=ArtifactKind.EVALUATION_RESULT,
            relative_path=f"{prefix}/execution.json",
            value=execution,
            visibility=visibility,
        ),
    )
