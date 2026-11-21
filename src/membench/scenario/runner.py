"""Bridge validated ScenarioSpec plans into the existing campaign preparer."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from membench.campaign.artifacts import canonical_json_bytes
from membench.campaign.models import ChildEnvironmentSpec
from membench.campaign.preparation import PreparedCampaign, prepare_campaign
from membench.campaign.preparation_models import (
    CampaignPreparationOptions,
    CampaignPurpose,
    ExternalIngressPreparation,
)
from membench.evaluation.models import (
    EvaluationProbe,
    FalseFactInsertionSettings,
    InformationReplacementSettings,
    StoredInstructionPersistenceSettings,
)
from membench.ingress.models import (
    IngressMode,
    TrustedBaselineManifest,
    TrustedBaselineSeed,
)
from membench.memory.models import CanonicalModel
from membench.qualification.models import QuiescenceConfiguration
from membench.scenario.compiler import ScenarioDryRunPlan, compile_dry_run
from membench.scenario.external import (
    export_forced_external_artifacts,
    render_forced_external_artifacts,
)
from membench.scenario.models import ScenarioAttackFamily, ScenarioSpec
from membench.scenario.profiles import (
    ScenarioConditionProfile,
    ScenarioConditionProfileRegistry,
)
from membench.tools.fixtures import load_tool_fixture


class ScenarioBuildError(RuntimeError):
    """A validated scenario has insufficient executable campaign inputs."""


class ScenarioCampaignSources(CanonicalModel):
    """Paths to already-authored/frozen sources required by campaign preparation.

    The scenario owns the attack content, host, matrix names, and boundaries.
    These paths retain compatibility with the existing preparation pipeline until
    the scenario renderer can materialize external deliveries itself.
    """

    external_ingress: ExternalIngressPreparation
    tool_fixture_path: Path | None = None
    trusted_baseline_path: Path | None = None
    evaluation_probe_path: Path | None = None
    behavioral_task_path: Path | None = None
    behavioral_oracle_path: Path | None = None
    seed: int = 42
    purpose: CampaignPurpose = "pilot"
    artifact_root: str
    quiescence: QuiescenceConfiguration = QuiescenceConfiguration()
    cleanup_storage: bool = True
    max_parallel_children: int = 1
    fail_fast: bool = False


class ScenarioGeneratedCampaignSources(CanonicalModel):
    """Only filesystem inputs that a scenario cannot safely embed itself."""

    fixture_path: Path
    generated_input_directory: Path
    external_artifact_directory: Path
    artifact_root: str
    seed: int = 42
    purpose: CampaignPurpose = "pilot"
    behavioral_task_path: Path | None = None
    behavioral_oracle_path: Path | None = None
    quiescence: QuiescenceConfiguration = QuiescenceConfiguration()
    cleanup_storage: bool = True
    max_parallel_children: int = 1
    fail_fast: bool = False


@dataclass(frozen=True)
class ScenarioCampaignBuild:
    """The dry plan and executable preparation options derived from it."""

    dry_run: ScenarioDryRunPlan
    options: CampaignPreparationOptions


def build_campaign_options(
    spec: ScenarioSpec,
    *,
    projection_directory: str | Path,
    condition_profile_registry: ScenarioConditionProfileRegistry,
    sources: ScenarioCampaignSources,
) -> ScenarioCampaignBuild:
    """Generate existing preparation options without writing campaign artifacts."""

    dry_run = compile_dry_run(
        spec,
        projection_directory=projection_directory,
        condition_profile_registry=condition_profile_registry,
    )
    ingress = sources.external_ingress
    if ingress.ingress_mode != IngressMode.FORCED_INGESTION:
        raise ScenarioBuildError(
            "scenario runner currently supports forced ingress only"
        )
    if tuple(sorted(ingress.host_trial_ids.values())) != tuple(
        sorted(dry_run.neutral_host.trial_ids)
    ):
        raise ScenarioBuildError(
            "external ingress host trials must exactly match the resolved neutral host"
        )

    selected_profiles = tuple(
        condition_profile_registry.resolve(name)
        for name in spec.matrix.condition_profiles
    )
    missing = [
        profile.profile_id
        for profile in selected_profiles
        if profile.execution_condition is None
    ]
    if missing:
        raise ScenarioBuildError(
            "profile(s) lack executable manifest/agent/environment details: "
            + ", ".join(missing)
        )
    conditions = tuple(
        profile.execution_condition for profile in selected_profiles
    )
    assert all(condition is not None for condition in conditions)
    environments = _unique_environments(selected_profiles)

    try:
        options = CampaignPreparationOptions(
            campaign_id=dry_run.scenario_key,
            projection_directory=Path(projection_directory),
            tool_fixture_path=sources.tool_fixture_path,
            external_ingress=ingress,
            trusted_baseline_path=sources.trusted_baseline_path,
            evaluation_probe_path=sources.evaluation_probe_path,
            behavioral_task_path=sources.behavioral_task_path,
            behavioral_oracle_path=sources.behavioral_oracle_path,
            case_ids=(dry_run.neutral_host.allocation_case_id,),
            conditions=conditions,  # type: ignore[arg-type]
            environments=environments,  # type: ignore[arg-type]
            seed=sources.seed,
            response_replicates=spec.matrix.response_replicates,
            purpose=sources.purpose,
            quiescence=sources.quiescence,
            cleanup_storage=sources.cleanup_storage,
            artifact_root=sources.artifact_root,
            max_parallel_children=sources.max_parallel_children,
            fail_fast=sources.fail_fast,
        )
    except ValueError as exc:
        raise ScenarioBuildError(
            f"generated campaign options are invalid: {exc}"
        ) from exc
    return ScenarioCampaignBuild(dry_run=dry_run, options=options)


def _unique_environments(
    profiles: tuple[ScenarioConditionProfile, ...],
) -> tuple[ChildEnvironmentSpec, ...]:
    """Return one compatible child environment per memory mechanism.

    A scenario matrix can vary native/direct policy while retaining one
    mechanism runtime.  Campaign preparation intentionally permits only one
    child environment per mechanism, so identical profile environments are
    deduplicated here.  Different environments for the same mechanism would
    make child selection ambiguous and are rejected before any artifacts are
    written.
    """

    selected: dict[object, ChildEnvironmentSpec] = {}
    source_profiles: dict[object, str] = {}
    for profile in profiles:
        environment = profile.environment
        assert environment is not None
        mechanism = environment.mechanism
        existing = selected.get(mechanism)
        if existing is None:
            selected[mechanism] = environment
            source_profiles[mechanism] = profile.profile_id
        elif existing != environment:
            raise ScenarioBuildError(
                "profiles "
                f"{source_profiles[mechanism]!r} and {profile.profile_id!r} "
                f"declare incompatible environments for mechanism "
                f"{mechanism.value!r}"
            )
    return tuple(selected.values())


def prepare_scenario_campaign(
    spec: ScenarioSpec,
    *,
    projection_directory: str | Path,
    condition_profile_registry: ScenarioConditionProfileRegistry,
    sources: ScenarioCampaignSources,
    project_root: str | Path,
    source_environment: Mapping[str, str] | None = None,
) -> PreparedCampaign:
    """Build and freeze a campaign through the established preparation pipeline."""

    build = build_campaign_options(
        spec,
        projection_directory=projection_directory,
        condition_profile_registry=condition_profile_registry,
        sources=sources,
    )
    return prepare_campaign(
        build.options,
        project_root=project_root,
        source_environment=source_environment,
    )


def prepare_generated_scenario_campaign(
    spec: ScenarioSpec,
    *,
    projection_directory: str | Path,
    condition_profile_registry: ScenarioConditionProfileRegistry,
    sources: ScenarioGeneratedCampaignSources,
    project_root: str | Path,
    env_file: str | Path | None = None,
) -> PreparedCampaign:
    """Render scenario-owned inputs, then freeze them through campaign preparation.

    All generated destinations must be new.  Existing campaign preparation then
    seals copies of these evaluator-only sources into its final artifact root.
    """

    root = Path(project_root).resolve()
    dry_run = compile_dry_run(
        spec,
        projection_directory=projection_directory,
        condition_profile_registry=condition_profile_registry,
    )
    fixture_path = _resolve_project_path(root, sources.fixture_path)
    fixture = load_tool_fixture(fixture_path)
    external_destination = _resolve_project_path(
        root, sources.external_artifact_directory
    )
    rendered = render_forced_external_artifacts(spec, dry_run=dry_run, fixture=fixture)
    export_forced_external_artifacts(rendered, external_destination)

    generated = _resolve_project_path(root, sources.generated_input_directory)
    generated.mkdir(parents=True, exist_ok=True)
    baseline_path = _write_baseline(spec, dry_run, fixture, generated)
    probe_path = _write_evaluation_probe(spec, dry_run, generated)
    campaign_sources = ScenarioCampaignSources(
        external_ingress=ExternalIngressPreparation(
            external_artifact_directory=sources.external_artifact_directory,
            ingress_plan_set_id=rendered.ingress_plans.ingress_plan_set_id,
            ingress_mode=IngressMode.FORCED_INGESTION,
            host_trial_ids=rendered.host_trial_ids,
        ),
        tool_fixture_path=sources.fixture_path,
        trusted_baseline_path=baseline_path,
        evaluation_probe_path=probe_path,
        behavioral_task_path=sources.behavioral_task_path,
        behavioral_oracle_path=sources.behavioral_oracle_path,
        seed=sources.seed,
        purpose=sources.purpose,
        artifact_root=sources.artifact_root,
        quiescence=sources.quiescence,
        cleanup_storage=sources.cleanup_storage,
        max_parallel_children=sources.max_parallel_children,
        fail_fast=sources.fail_fast,
    )
    return prepare_scenario_campaign(
        spec,
        projection_directory=projection_directory,
        condition_profile_registry=condition_profile_registry,
        sources=campaign_sources,
        project_root=root,
        source_environment=_scenario_source_environment(env_file, project_root=root),
    )


def _write_baseline(
    spec: ScenarioSpec, dry_run: ScenarioDryRunPlan, fixture, directory: Path
) -> Path | None:
    if spec.baseline is None:
        return None
    path = directory / "trusted-baselines.json"
    _require_new(path)
    baseline = TrustedBaselineSeed(
        baseline_id=dry_run.ids.baseline_id,
        persona_id=spec.subject.persona_id,
        content=spec.baseline.content,
        retrieval_query=spec.baseline.retrieval_query,
        required_markers=tuple(sorted(spec.evaluation.original_markers)),
        forbidden_pre_ingress_markers=tuple(sorted(spec.evaluation.replacement_markers)),
        available_after_source_message_index=spec.baseline.seed_after_source_message_index,
        fixture_id=fixture.fixture_id,
        fixture_sha256=fixture.sha256,
        content_sha256=__import__("hashlib").sha256(spec.baseline.content.encode()).hexdigest(),
    )
    path.write_bytes(canonical_json_bytes(TrustedBaselineManifest(
        baseline_set_id=f"baselines-{dry_run.ids.scenario_key}", baselines=(baseline,)
    ).model_dump(mode="json")))
    return path


def _write_evaluation_probe(
    spec: ScenarioSpec, dry_run: ScenarioDryRunPlan, directory: Path
) -> Path:
    path = directory / "evaluation-probe.json"
    _require_new(path)
    original = tuple(sorted(spec.evaluation.original_markers))
    replacement = tuple(sorted(spec.evaluation.replacement_markers))
    if spec.attack_family == ScenarioAttackFamily.FALSE_FACT_INSERTION:
        settings = FalseFactInsertionSettings(
            poison_markers=replacement,
            trusted_ground_truth=spec.external_attack.target_reference,  # type: ignore[union-attr]
        )
    elif spec.attack_family == ScenarioAttackFamily.STORED_INSTRUCTION:
        settings = StoredInstructionPersistenceSettings(instruction_markers=replacement)
    else:
        assert spec.baseline is not None
        settings = InformationReplacementSettings(
            baseline_id=dry_run.ids.baseline_id,
            baseline_markers=original,
            replacement_markers=replacement,
            trusted_ground_truth=spec.baseline.content,
        )
    probe = EvaluationProbe(
        probe_id=dry_run.ids.evaluation_probe_id,
        attack_spec_id=dry_run.ids.attack_spec_id,
        delivery_family_id=dry_run.ids.delivery_family_id,
        persona_id=str(spec.subject.persona_id),
        query=spec.evaluation.query,
        top_k=spec.evaluation.top_k,
        threshold=spec.evaluation.threshold,
        settings=settings,
    )
    path.write_bytes(canonical_json_bytes(probe.model_dump(mode="json")))
    return path


def _resolve_project_path(root: Path, value: Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ScenarioBuildError(
            "generated scenario paths must remain below project root"
        )
    return path


def _require_new(path: Path) -> None:
    if path.exists():
        raise ScenarioBuildError(
            f"refusing to overwrite generated scenario input: {path}"
        )


def _scenario_source_environment(
    env_file: str | Path | None, *, project_root: Path
) -> dict[str, str]:
    """Match the campaign CLI's dotenv semantics for Python scenario entrypoints."""

    source = dict(os.environ)
    if env_file is None:
        return source
    path = Path(env_file)
    resolved = path if path.is_absolute() else project_root / path
    if not resolved.is_file():
        raise ScenarioBuildError(f"environment file does not exist: {resolved}")
    for key, value in dotenv_values(resolved).items():
        if value is not None:
            source.setdefault(key, value)
    return source
