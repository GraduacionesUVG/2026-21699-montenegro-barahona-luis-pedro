"""Load editable evaluator probes and verify frozen probe artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from membench.campaign.artifacts import CampaignArtifactStore, canonical_json_bytes
from membench.campaign.models import ArtifactKind, ChildRunResult
from membench.evaluation.aggregation import (
    EvaluationAggregationError,
    EvaluationAggregator,
    EvaluationAnalysis,
    EvaluationRunEvidence,
)
from membench.evaluation.models import EvaluationExecutionResult, EvaluationProbe
from membench.memory.models import CanonicalModel

if TYPE_CHECKING:
    from membench.campaign.bindings import CampaignRunBindings

EVALUATION_PROBE_VERSION = "evaluation_probe_v1"
EVALUATION_PROBE_FILE = "evaluation_probe.json"
EVALUATION_PROBE_MANIFEST_FILE = "evaluation_probe_manifest.json"


class EvaluationProbeLoadError(RuntimeError):
    """Raised when editable or frozen probe input cannot be trusted."""


class EvaluationEvidenceLoadError(RuntimeError):
    """Sealed trial artifacts cannot be attributed to an evaluation binding."""


class EvaluationProbeManifest(CanonicalModel):
    """Integrity metadata for one exact, frozen evaluator-probe JSON file."""

    schema_version: Literal[1] = 1
    version: Literal["evaluation_probe_v1"] = EVALUATION_PROBE_VERSION
    probe_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    probe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_evaluation_probe(path: str | Path) -> EvaluationProbe:
    """Load one editable JSON probe without treating it as frozen evidence."""

    source = Path(path)
    try:
        return EvaluationProbe.model_validate_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise EvaluationProbeLoadError(
            f"could not load evaluation probe: {source}"
        ) from exc


def export_frozen_evaluation_probe(
    probe: EvaluationProbe,
    destination: str | Path,
) -> EvaluationProbeManifest:
    """Write canonical probe bytes plus a matching immutable manifest.

    The destination may exist, but a prior probe or manifest is never
    overwritten.  This lets a caller safely choose a fresh artifact directory
    for every approved probe revision.
    """

    output = Path(destination)
    probe_path = output / EVALUATION_PROBE_FILE
    manifest_path = output / EVALUATION_PROBE_MANIFEST_FILE
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationProbeLoadError(
            f"could not create frozen evaluation probe directory: {output}"
        ) from exc
    if probe_path.exists() or manifest_path.exists():
        raise EvaluationProbeLoadError("frozen evaluation-probe output already exists")

    content = canonical_json_bytes(probe)
    manifest = EvaluationProbeManifest(
        probe_id=probe.probe_id,
        probe_sha256=hashlib.sha256(content).hexdigest(),
    )
    try:
        probe_path.write_bytes(content)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
    except OSError as exc:
        raise EvaluationProbeLoadError(
            f"could not write frozen evaluation probe: {output}"
        ) from exc
    return manifest


def load_frozen_evaluation_probe(destination: str | Path) -> EvaluationProbe:
    """Load and integrity-check a canonical probe artifact and its manifest."""

    root = Path(destination)
    probe_path = root / EVALUATION_PROBE_FILE
    manifest_path = root / EVALUATION_PROBE_MANIFEST_FILE
    try:
        content = probe_path.read_bytes()
        manifest = EvaluationProbeManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise EvaluationProbeLoadError(
            f"could not load frozen evaluation probe: {root}"
        ) from exc
    if hashlib.sha256(content).hexdigest() != manifest.probe_sha256:
        raise EvaluationProbeLoadError(
            "evaluation-probe JSON hash does not match manifest"
        )
    try:
        probe = EvaluationProbe.model_validate_json(content)
    except ValueError as exc:
        raise EvaluationProbeLoadError(
            "frozen evaluation-probe JSON is invalid"
        ) from exc
    if canonical_json_bytes(probe) != content:
        raise EvaluationProbeLoadError(
            "frozen evaluation-probe JSON is not canonical"
        )
    expected_manifest = EvaluationProbeManifest(
        probe_id=probe.probe_id,
        probe_sha256=hashlib.sha256(content).hexdigest(),
    )
    if manifest != expected_manifest:
        raise EvaluationProbeLoadError(
            "evaluation-probe manifest does not match decoded content"
        )
    return probe


def load_evaluation_run_evidence(
    store: CampaignArtifactStore,
    bindings: CampaignRunBindings,
    child: ChildRunResult,
) -> EvaluationRunEvidence:
    """Rehydrate one run's evaluator execution from its sealed child artifacts."""

    try:
        binding = bindings.binding_for(child.run.run_id)
    except KeyError as exc:
        raise EvaluationEvidenceLoadError(
            "child has no matching evaluation binding"
        ) from exc
    if not binding.has_evaluation_probe:
        raise EvaluationEvidenceLoadError("child run has no evaluation-probe binding")

    expected_path = (
        f"children/{child.run.run_id}/trial/evaluation/"
        f"{binding.evaluation_probe_id}/execution.json"
    )
    matches = tuple(
        reference
        for reference in child.artifacts
        if reference.kind == ArtifactKind.EVALUATION_RESULT
        and reference.relative_path == expected_path
    )
    if len(matches) != 1:
        raise EvaluationEvidenceLoadError(
            "child must contain exactly one bound evaluation execution artifact"
        )
    try:
        execution = EvaluationExecutionResult.model_validate_json(
            store.verify(matches[0]).read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise EvaluationEvidenceLoadError(
            "could not load sealed evaluation execution"
        ) from exc

    probe = execution.evaluation.probe
    if (
        probe.probe_id != binding.evaluation_probe_id
        or hashlib.sha256(canonical_json_bytes(probe)).hexdigest()
        != binding.evaluation_probe_sha256
        or execution.evaluation.trial_id != binding.trial_id
    ):
        raise EvaluationEvidenceLoadError(
            "sealed evaluation execution differs from its run binding"
        )
    try:
        return EvaluationRunEvidence(
            campaign_id=child.run.campaign_id,
            trial_family_id=child.run.trial_family_id,
            run_id=child.run.run_id,
            response_replicate=child.run.response_replicate,
            twin=child.run.twin,
            persona_id=child.run.persona_id,
            mechanism=child.run.mechanism,
            policy=child.run.policy,
            execution=execution,
        )
    except ValueError as exc:
        raise EvaluationEvidenceLoadError(
            "sealed evaluation execution does not match its campaign run"
        ) from exc


def aggregate_sealed_evaluations(
    store: CampaignArtifactStore,
    bindings: CampaignRunBindings,
    children: tuple[ChildRunResult, ...],
) -> EvaluationAnalysis:
    """Pair all frozen evaluator executions without accessing memory backends."""

    if bindings.evaluation_probe is None:
        raise EvaluationEvidenceLoadError("campaign has no evaluator probe")
    try:
        evidence = tuple(
            load_evaluation_run_evidence(store, bindings, child) for child in children
        )
        return EvaluationAggregator().aggregate(evidence)
    except EvaluationAggregationError as exc:
        raise EvaluationEvidenceLoadError(
            "sealed evaluation evidence cannot form fair twin comparisons"
        ) from exc
