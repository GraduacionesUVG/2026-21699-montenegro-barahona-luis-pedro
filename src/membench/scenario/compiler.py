"""Read-only preflight compiler for human-authored scenario specifications.

This module deliberately stops before rendering legacy campaign artifacts.  It
answers the useful early question: does the editable scenario fit the frozen
neutral host it claims to use, and at which *actual* replay indexes will its
baseline and ingress occur?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from membench.campaign.models import TrialTwin
from membench.curation.projections import (
    ProjectionKind,
    ProjectionSetError,
    load_projection_set,
    memory_visible_replay_signature,
)
from membench.scenario.ids import ScenarioIds
from membench.scenario.models import ScenarioSpec
from membench.scenario.profiles import (
    ScenarioConditionProfile,
    ScenarioConditionProfileRegistry,
    ScenarioProfileError,
    load_condition_profile_registry,
)
from membench.scenario.validation import (
    ScenarioValidationError,
    ScenarioValidationReport,
    validate_scenario_spec,
)


class ScenarioCompilerError(RuntimeError):
    """A scenario cannot be compiled against the requested frozen host."""


@dataclass(frozen=True)
class NeutralHostEvidence:
    """The host evidence used to validate a scenario dry run."""

    projection_directory: Path
    projection_set_id: str
    allocation_case_id: str
    persona_id: int
    source_allocation_case_id: str
    trial_ids: tuple[str, str, str]
    source_message_indexes: tuple[int, ...]


@dataclass(frozen=True)
class ScenarioDryRunPlan:
    """Deterministic pre-render plan; it does not write any files."""

    scenario_key: str
    ids: ScenarioIds
    neutral_host: NeutralHostEvidence
    baseline_after_source_message_index: int | None
    ingress_after_source_message_index: int
    condition_profiles: tuple[str, ...]
    response_replicates: int
    matrix_runs: tuple[ScenarioMatrixRun, ...]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready evidence suitable for a future CLI ``--json`` mode."""

        return {
            "scenario_key": self.scenario_key,
            "ids": {
                "attack_spec_id": self.ids.attack_spec_id,
                "delivery_family_id": self.ids.delivery_family_id,
                "baseline_id": self.ids.baseline_id,
                "evaluation_probe_id": self.ids.evaluation_probe_id,
                "ingress_plan_set_id": self.ids.ingress_plan_set_id,
            },
            "neutral_host": {
                "projection_directory": str(self.neutral_host.projection_directory),
                "projection_set_id": self.neutral_host.projection_set_id,
                "allocation_case_id": self.neutral_host.allocation_case_id,
                "persona_id": self.neutral_host.persona_id,
                "source_allocation_case_id": (
                    self.neutral_host.source_allocation_case_id
                ),
                "trial_ids": self.neutral_host.trial_ids,
                "source_message_indexes": self.neutral_host.source_message_indexes,
            },
            "baseline_after_source_message_index": (
                self.baseline_after_source_message_index
            ),
            "ingress_after_source_message_index": (
                self.ingress_after_source_message_index
            ),
            "condition_profiles": self.condition_profiles,
            "response_replicates": self.response_replicates,
            "matrix_runs": [run.as_dict() for run in self.matrix_runs],
        }


@dataclass(frozen=True)
class ScenarioMatrixRun:
    """One generated mechanism/policy/replicate condition in a dry run."""

    run_key: str
    condition_profile_id: str
    mechanism: str
    write_policy: str
    response_replicate: int

    def as_dict(self) -> dict[str, object]:
        return {
            "run_key": self.run_key,
            "condition_profile_id": self.condition_profile_id,
            "mechanism": self.mechanism,
            "write_policy": self.write_policy,
            "response_replicate": self.response_replicate,
        }


def load_scenario_spec(path: str | Path) -> ScenarioSpec:
    """Load one authored JSON scenario with an actionable error boundary."""

    source = Path(path)
    try:
        return ScenarioSpec.model_validate_json(source.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise ScenarioCompilerError(
            f"could not load scenario specification {source}: {exc}"
        ) from exc


def load_neutral_host(
    projection_directory: str | Path, *, spec: ScenarioSpec
) -> NeutralHostEvidence:
    """Load one verified neutral host and its actual common replay indexes."""

    directory = Path(projection_directory)
    try:
        projections = load_projection_set(directory)
    except ProjectionSetError as exc:
        raise ScenarioCompilerError(
            f"could not load frozen projection set {directory}: {exc}"
        ) from exc

    case_id = spec.subject.neutral_host_case_id
    cases = [case for case in projections.cases if case.allocation_case_id == case_id]
    if len(cases) != 1:
        raise ScenarioCompilerError(
            f"neutral host case {case_id!r} was not found exactly once"
        )
    case = cases[0]
    if case.projection_kind != ProjectionKind.NEUTRAL_EXTERNAL_HOST:
        raise ScenarioCompilerError(f"case {case_id!r} is not a neutral external host")
    if int(case.persona_id) != spec.subject.persona_id:
        raise ScenarioCompilerError(
            f"neutral host persona {case.persona_id} does not match scenario persona "
            f"{spec.subject.persona_id}"
        )
    if case.source_allocation_case_id is None:
        raise ScenarioCompilerError(
            f"neutral host case {case_id!r} has no source provenance"
        )

    host_trials = [
        trial for trial in projections.trials if trial.allocation_case_id == case_id
    ]
    by_twin = {trial.twin: trial for trial in host_trials}
    expected_twins = (TrialTwin.CLEAN, TrialTwin.MATCHED_BENIGN, TrialTwin.POISONED)
    if set(by_twin) != set(expected_twins) or len(host_trials) != len(expected_twins):
        raise ScenarioCompilerError(
            f"neutral host {case_id!r} must contain exactly clean, matched-benign, "
            "and poisoned trials"
        )
    if any(
        trial.projection_kind != ProjectionKind.NEUTRAL_EXTERNAL_HOST
        for trial in host_trials
    ):
        raise ScenarioCompilerError(
            f"neutral host {case_id!r} contains a non-neutral trial"
        )

    signatures = {
        memory_visible_replay_signature(by_twin[twin]) for twin in expected_twins
    }
    if len(signatures) != 1:
        raise ScenarioCompilerError(
            f"neutral host {case_id!r} does not have identical memory-visible replay "
            "across its three twins"
        )
    indexes = tuple(
        event.source_message_index for event in by_twin[TrialTwin.CLEAN].replay_events
    )
    if any(index is None for index in indexes):
        raise ScenarioCompilerError(
            f"neutral host {case_id!r} has an event without source index"
        )

    return NeutralHostEvidence(
        projection_directory=directory,
        projection_set_id=spec.subject.projection_set_id,
        allocation_case_id=case_id,
        persona_id=spec.subject.persona_id,
        source_allocation_case_id=case.source_allocation_case_id,
        trial_ids=tuple(by_twin[twin].trial_id for twin in expected_twins),
        source_message_indexes=tuple(indexes),  # type: ignore[arg-type]
    )


def compile_dry_run(
    spec: ScenarioSpec,
    *,
    projection_directory: str | Path,
    condition_profile_registry: ScenarioConditionProfileRegistry,
) -> ScenarioDryRunPlan:
    """Resolve a scenario against frozen host evidence without rendering artifacts."""

    host = load_neutral_host(projection_directory, spec=spec)
    try:
        report = validate_scenario_spec(
            spec,
            host_source_message_indexes=host.source_message_indexes,
            known_condition_profiles={
                profile.profile_id for profile in condition_profile_registry.profiles
            },
        )
    except ScenarioValidationError as exc:
        raise ScenarioCompilerError(str(exc)) from exc
    profiles = _resolve_profiles(spec, condition_profile_registry)
    return _plan_from_report(spec, host, report, profiles)


def load_and_compile_dry_run(
    scenario_path: str | Path,
    *,
    projection_directory: str | Path,
    condition_profile_registry_path: str | Path,
) -> ScenarioDryRunPlan:
    """Convenience entry point for a future CLI command."""

    return compile_dry_run(
        load_scenario_spec(scenario_path),
        projection_directory=projection_directory,
        condition_profile_registry=load_condition_profile_registry(
            condition_profile_registry_path
        ),
    )


def _plan_from_report(
    spec: ScenarioSpec,
    host: NeutralHostEvidence,
    report: ScenarioValidationReport,
    profiles: tuple[ScenarioConditionProfile, ...],
) -> ScenarioDryRunPlan:
    return ScenarioDryRunPlan(
        scenario_key=report.scenario_key,
        ids=report.ids,
        neutral_host=host,
        baseline_after_source_message_index=(
            None
            if spec.baseline is None
            else spec.baseline.seed_after_source_message_index
        ),
        ingress_after_source_message_index=spec.ingress.available_after_source_message_index,
        condition_profiles=spec.matrix.condition_profiles,
        response_replicates=spec.matrix.response_replicates,
        matrix_runs=_expand_matrix(spec, report.ids, profiles),
    )


def _resolve_profiles(
    spec: ScenarioSpec,
    registry: ScenarioConditionProfileRegistry,
) -> tuple[ScenarioConditionProfile, ...]:
    try:
        return tuple(registry.resolve(name) for name in spec.matrix.condition_profiles)
    except ScenarioProfileError as exc:
        raise ScenarioCompilerError(str(exc)) from exc


def _expand_matrix(
    spec: ScenarioSpec,
    ids: ScenarioIds,
    profiles: tuple[ScenarioConditionProfile, ...],
) -> tuple[ScenarioMatrixRun, ...]:
    return tuple(
        ScenarioMatrixRun(
            run_key=f"{ids.campaign_id(profile.profile_id)}-r{replicate}",
            condition_profile_id=profile.profile_id,
            mechanism=profile.condition.mechanism.value,
            write_policy=profile.condition.write_policy.value,
            response_replicate=replicate,
        )
        for profile in profiles
        for replicate in range(1, spec.matrix.response_replicates + 1)
    )
