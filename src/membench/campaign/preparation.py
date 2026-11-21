"""Prepare a frozen campaign without initializing memory or launching children.

Preparation owns preparation/*; the launcher owns campaign_configuration.json
and final sealing. Partial directories are retained on failure, never reused.
No resume, CLI, model discovery, or backend health checks are implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from membench.campaign.aggregate import CampaignAggregator
from membench.campaign.artifacts import CampaignArtifactStore
from membench.campaign.behavioral import (
    BehavioralInputs,
    FrozenBehavioralInputs,
    freeze_behavioral_inputs,
    load_behavioral_inputs,
)
from membench.campaign.bindings import CampaignRunBindings
from membench.campaign.environments import ChildEnvironmentRegistry
from membench.campaign.evaluation import (
    EvaluationProbeInput,
    FrozenEvaluationProbe,
    freeze_evaluation_probe_input,
    load_evaluation_probe_input,
)
from membench.campaign.external_ingress import (
    ExternalIngressInputs,
    FrozenExternalIngress,
    freeze_external_ingress_inputs,
    load_external_ingress_inputs,
)
from membench.campaign.inputs import (
    FrozenCampaignInputs,
    freeze_campaign_inputs,
    load_campaign_inputs,
)
from membench.campaign.launcher import CampaignLauncher, ChildProcessExecutor
from membench.campaign.matrix import CampaignMatrix, build_campaign_matrix
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
    CampaignConfiguration,
)
from membench.campaign.preparation_models import (
    CampaignPreparationCompletion,
    CampaignPreparationOptions,
)
from membench.campaign.trusted_baselines import (
    FrozenTrustedBaselines,
    TrustedBaselineInputs,
    freeze_trusted_baseline_inputs,
    load_trusted_baseline_inputs,
)
from membench.ingress.models import TrustedBaselineManifest
from membench.ingress.plans import TrialIngressPlanSet


class CampaignPreparationError(RuntimeError):
    """Stage-local failure, with the original diagnostic in __cause__.

    artifact_root is set only if this invocation created the directory.
    Summaries omit arbitrary exception payloads that could contain secrets.
    """

    def __init__(self, stage: str, *, artifact_root: Path | None = None) -> None:
        self.stage = stage
        self.artifact_root = artifact_root
        super().__init__(f"campaign preparation failed during {stage}")


@dataclass(frozen=True)
class PreparedCampaign:
    """Validated in-process handoff; never serialize the environment registry."""

    configuration: CampaignConfiguration
    bindings: CampaignRunBindings
    completion: ArtifactReference
    artifacts: CampaignArtifactStore = field(repr=False)
    environments: ChildEnvironmentRegistry = field(repr=False)
    ingress_plan_set: TrialIngressPlanSet | None = field(default=None, repr=False)
    trusted_baselines: TrustedBaselineManifest | None = field(
        default=None, repr=False
    )

    def create_launcher(
        self,
        *,
        child_executor: ChildProcessExecutor | None = None,
        aggregator: CampaignAggregator | None = None,
    ) -> CampaignLauncher:
        """Construct, but do not run, a launcher using the same unsealed store."""
        if self.artifacts.sealed:
            raise ValueError("prepared campaign artifact store is already sealed")
        for reference in self.artifacts.references():
            self.artifacts.verify(reference)
        self.artifacts.verify(self.completion)
        references = {ref.artifact_id: ref for ref in self.artifacts.references()}
        persisted_configuration = CampaignConfiguration.model_validate_json(
            self.artifacts.verify(references["preparation:configuration"]).read_bytes()
        )
        persisted_bindings = CampaignRunBindings.model_validate_json(
            self.artifacts.verify(references["preparation:run-bindings"]).read_bytes()
        )
        if (
            self.configuration != persisted_configuration
            or self.bindings != persisted_bindings
        ):
            raise ValueError("launcher inputs differ from frozen preparation")
        self.bindings.validate_campaign(
            self.configuration,
            ingress_plan_set=self.ingress_plan_set,
            trusted_baselines=self.trusted_baselines,
        )
        cleanup = {binding.cleanup_storage for binding in self.bindings.bindings}
        if len(cleanup) != 1:
            raise ValueError("prepared campaign must use a single cleanup policy")
        return CampaignLauncher(
            self.configuration,
            self.environments,
            self.artifacts,
            child_executor=child_executor,
            aggregator=aggregator,
            bindings=self.bindings,
            cleanup_storage=cleanup.pop(),
        )


class CampaignPreparation:
    """Stateless coordinator, not a serialized Pydantic model."""

    def prepare_campaign(
        self,
        options: CampaignPreparationOptions,
        *,
        project_root: str | Path,
        source_environment: Mapping[str, str] | None = None,
    ) -> PreparedCampaign:
        """Validate first, persist second, and never start a live trial.

        Environment preflight checks paths and declared variables only, not
        installed SDK versions, child entrypoint readiness, or server health.
        Explicit model pins are preserved; no .env/default model inference.
        """
        stage = "validate_options"
        created_root: Path | None = None
        try:
            options = CampaignPreparationOptions.model_validate(
                options.model_dump(mode="json")
            )
            root = Path(project_root).resolve()
            stage = "resolve_artifact_root"
            destination = self._resolve_artifact_root(options, project_root=root)
            stage = "load_inputs"
            inputs = load_campaign_inputs(options, project_root=root)
            stage = "load_external_ingress"
            external_ingress = load_external_ingress_inputs(
                options,
                inputs,
                project_root=root,
            )
            stage = "load_trusted_baselines"
            trusted_baselines = load_trusted_baseline_inputs(
                options,
                inputs,
                project_root=root,
            )
            stage = "load_evaluation_probe"
            evaluation_probe = load_evaluation_probe_input(
                options,
                project_root=root,
            )
            stage = "load_behavioral_inputs"
            behavioral_inputs = load_behavioral_inputs(
                options,
                inputs,
                external_ingress,
                project_root=root,
            )
            stage = "build_matrix"
            matrix = build_campaign_matrix(
                options,
                inputs,
                behavioral_task=(
                    None if behavioral_inputs is None else behavioral_inputs.task
                ),
            )
            stage = "preflight_environments"
            environments = self._preflight_environments(
                options,
                project_root=root,
                source_environment=source_environment,
            )
            stage = "create_artifact_root"
            store = CampaignArtifactStore.prepare(destination)
            created_root = store.root
            stage = "freeze_inputs"
            frozen = freeze_campaign_inputs(inputs, store)
            stage = "freeze_external_ingress"
            frozen_external = self._freeze_external_ingress(
                external_ingress,
                store,
            )
            stage = "freeze_trusted_baselines"
            frozen_baselines = self._freeze_trusted_baselines(
                trusted_baselines, store
            )
            stage = "freeze_evaluation_probe"
            frozen_evaluation = self._freeze_evaluation_probe(evaluation_probe, store)
            stage = "freeze_behavioral_inputs"
            frozen_behavioral = self._freeze_behavioral_inputs(
                behavioral_inputs, store
            )
            stage = "build_run_bindings"
            bindings = self._build_run_bindings(
                options,
                matrix,
                frozen,
                frozen_external,
                frozen_baselines,
                frozen_evaluation,
                frozen_behavioral,
            )
            stage = "persist_preparation"
            configuration, configuration_ref = self._persist_preparation(
                options,
                matrix,
                bindings,
                frozen,
                frozen_external,
                frozen_baselines,
                frozen_evaluation,
                frozen_behavioral,
                store,
            )
            stage = "validate_prepared_campaign"
            self._validate_prepared_campaign(
                options,
                matrix,
                bindings,
                configuration,
                configuration_ref,
                environments,
                store,
                frozen_external,
                frozen_baselines,
                frozen_evaluation,
                frozen_behavioral,
            )
            stage = "record_completion"
            completion = store.write_json(
                artifact_id="preparation:complete",
                kind=ArtifactKind.OTHER,
                relative_path="preparation/complete.json",
                value=CampaignPreparationCompletion(
                    campaign_id=options.campaign_id,
                    preparation_sha256=options.sha256,
                    matrix_sha256=matrix.sha256,
                    projection_set_id=matrix.projection_set_id,
                    selected_case_ids=inputs.case_ids,
                    run_count=matrix.run_count,
                    purpose=options.purpose,
                    configuration=configuration_ref,
                    artifacts=store.references(),
                ),
                visibility=ArtifactVisibility.EVALUATOR_ONLY,
            )
            store.verify(completion)
            return PreparedCampaign(
                configuration=configuration,
                bindings=bindings,
                completion=completion,
                artifacts=store,
                environments=environments,
                ingress_plan_set=(
                    None if frozen_external is None else frozen_external.plan_set
                ),
                trusted_baselines=(
                    None if frozen_baselines is None else frozen_baselines.manifest
                ),
            )
        except Exception as exc:
            raise CampaignPreparationError(stage, artifact_root=created_root) from exc

    def _resolve_artifact_root(
        self, options: CampaignPreparationOptions, *, project_root: Path
    ) -> Path:
        """Reject unsafe resolved paths and existing output, including symlinks."""
        if not project_root.is_dir():
            raise ValueError("project root must be an existing directory")
        configured = project_root / options.artifact_root
        destination = configured.resolve()
        if destination == project_root or not destination.is_relative_to(project_root):
            raise ValueError("campaign artifact root must remain below project root")
        if configured.is_symlink() or destination.exists():
            raise FileExistsError(
                "refusing to reuse an existing campaign artifact root"
            )
        return destination

    def _preflight_environments(
        self,
        options: CampaignPreparationOptions,
        *,
        project_root: Path,
        source_environment: Mapping[str, str] | None,
    ) -> ChildEnvironmentRegistry:
        """Reuse strict allowlisted resolution, without saving environment values."""
        registry = ChildEnvironmentRegistry(
            options.environments,
            project_root=project_root,
            source_environment=source_environment,
        )
        registry.resolve_all(require_all_env_keys=True)
        return registry

    def _build_run_bindings(
        self,
        options: CampaignPreparationOptions,
        matrix: CampaignMatrix,
        frozen: FrozenCampaignInputs,
        frozen_external: FrozenExternalIngress | None,
        frozen_baselines: FrozenTrustedBaselines | None,
        frozen_evaluation: FrozenEvaluationProbe | None,
        frozen_behavioral: FrozenBehavioralInputs | None,
    ) -> CampaignRunBindings:
        """Attach matrix bindings to the exact frozen projection manifest."""
        bindings = matrix.bindings
        ingress_manifest = None
        if frozen_external is not None:
            ingress_manifest = frozen_external.ingress_plan_manifest
            bindings = tuple(
                binding.with_ingress_plan(
                    frozen_external.plan_set.plan_for_host_trial(binding.trial_id),
                    ingress_plan_set_id=frozen_external.plan_set.ingress_plan_set_id,
                )
                if binding.trial_id
                in {plan.host_trial_id for plan in frozen_external.plan_set.plans}
                else binding
                for binding in bindings
            )
        if frozen_evaluation is not None:
            plans_by_run = {plan.run.run_id: plan for plan in matrix.plans}
            for binding in bindings:
                plan = plans_by_run[binding.run_id]
                if plan.run.persona_id != frozen_evaluation.probe.persona_id:
                    raise ValueError(
                        "evaluation probe persona does not match every selected run"
                    )
            bindings = tuple(
                binding.with_evaluation_probe(
                    probe_id=frozen_evaluation.probe.probe_id,
                    probe_sha256=frozen_evaluation.artifact.sha256,
                )
                for binding in bindings
            )
        if frozen_baselines is not None:
            plans_by_run = {plan.run.run_id: plan for plan in matrix.plans}
            bindings = tuple(
                binding.with_trusted_baseline(
                    frozen_baselines.for_persona(plan.run.persona_id),
                    persona_id=plan.run.persona_id,
                )
                for binding in bindings
                for plan in (plans_by_run[binding.run_id],)
            )
        if frozen_behavioral is not None:
            plans_by_run = {plan.run.run_id: plan for plan in matrix.plans}
            for binding in bindings:
                plan = plans_by_run[binding.run_id]
                expected_task = frozen_behavioral.task.instantiate(
                    plan.run.scope,
                    task_id=plan.run.agent_task.task_id,
                )
                if plan.run.agent_task != expected_task:
                    raise ValueError(
                        "matrix public task differs from frozen behavioral task"
                    )
            bindings = tuple(
                binding.with_behavioral_task(
                    task_id=frozen_behavioral.task.behavioral_task_id,
                    task_sha256=frozen_behavioral.task_artifact.sha256,
                    oracle_id=frozen_behavioral.oracle.oracle_id,
                    oracle_sha256=frozen_behavioral.oracle_artifact.sha256,
                )
                for binding in bindings
            )
        bindings = CampaignRunBindings(
            campaign_id=options.campaign_id,
            preparation_sha256=options.sha256,
            projection_set_id=matrix.projection_set_id,
            projection_manifest=frozen.projection_manifest,
            ingress_plan_manifest=ingress_manifest,
            trusted_baseline_manifest=(
                None if frozen_baselines is None else frozen_baselines.artifact
            ),
            evaluation_probe=(
                None if frozen_evaluation is None else frozen_evaluation.artifact
            ),
            behavioral_task=(
                None if frozen_behavioral is None else frozen_behavioral.task_artifact
            ),
            behavioral_oracle=(
                None
                if frozen_behavioral is None
                else frozen_behavioral.oracle_artifact
            ),
            bindings=bindings,
        )
        bindings.validate_preparation(options)
        return bindings

    def _freeze_external_ingress(
        self,
        external_ingress: ExternalIngressInputs | None,
        store: CampaignArtifactStore,
    ) -> FrozenExternalIngress | None:
        if external_ingress is None:
            return None
        return freeze_external_ingress_inputs(external_ingress, store)

    def _freeze_trusted_baselines(
        self,
        trusted_baselines: TrustedBaselineInputs | None,
        store: CampaignArtifactStore,
    ) -> FrozenTrustedBaselines | None:
        if trusted_baselines is None:
            return None
        return freeze_trusted_baseline_inputs(trusted_baselines, store)

    def _freeze_evaluation_probe(
        self,
        evaluation_probe: EvaluationProbeInput | None,
        store: CampaignArtifactStore,
    ) -> FrozenEvaluationProbe | None:
        if evaluation_probe is None:
            return None
        return freeze_evaluation_probe_input(evaluation_probe, store)

    def _freeze_behavioral_inputs(
        self,
        behavioral_inputs: BehavioralInputs | None,
        store: CampaignArtifactStore,
    ) -> FrozenBehavioralInputs | None:
        if behavioral_inputs is None:
            return None
        return freeze_behavioral_inputs(behavioral_inputs, store)

    def _build_configuration(
        self,
        options: CampaignPreparationOptions,
        matrix: CampaignMatrix,
        frozen_inputs: tuple[ArtifactReference, ...],
    ) -> CampaignConfiguration:
        """Build launcher input without a self-referential configuration hash."""
        return CampaignConfiguration(
            campaign_id=options.campaign_id,
            seed=options.seed,
            runs=matrix.runs,
            environments=options.environments,
            frozen_inputs=frozen_inputs,
            artifact_root=options.artifact_root,
            max_parallel_children=options.max_parallel_children,
            fail_fast=options.fail_fast,
        )

    def _persist_preparation(
        self,
        options: CampaignPreparationOptions,
        matrix: CampaignMatrix,
        bindings: CampaignRunBindings,
        frozen: FrozenCampaignInputs,
        frozen_external: FrozenExternalIngress | None,
        frozen_baselines: FrozenTrustedBaselines | None,
        frozen_evaluation: FrozenEvaluationProbe | None,
        frozen_behavioral: FrozenBehavioralInputs | None,
        store: CampaignArtifactStore,
    ) -> tuple[CampaignConfiguration, ArtifactReference]:
        """Persist options/bindings before their referencing configuration."""
        options_ref = store.write_json(
            artifact_id="preparation:options",
            kind=ArtifactKind.OTHER,
            relative_path="preparation/options.json",
            value=options,
        )
        bindings_ref = store.write_json(
            artifact_id="preparation:run-bindings",
            kind=ArtifactKind.RUN_BINDINGS,
            relative_path="preparation/run_bindings.json",
            value=bindings,
        )
        configuration = self._build_configuration(
            options,
            matrix,
            (
                *frozen.artifacts,
                *(() if frozen_external is None else frozen_external.artifacts),
                *(() if frozen_baselines is None else (frozen_baselines.artifact,)),
                *(() if frozen_evaluation is None else (frozen_evaluation.artifact,)),
                *(() if frozen_behavioral is None else frozen_behavioral.artifacts),
                options_ref,
                bindings_ref,
            ),
        )
        configuration_ref = store.write_json(
            artifact_id="preparation:configuration",
            kind=ArtifactKind.OTHER,
            relative_path="preparation/configuration.json",
            value=configuration,
        )
        return configuration, configuration_ref

    def _validate_prepared_campaign(
        self,
        options: CampaignPreparationOptions,
        matrix: CampaignMatrix,
        bindings: CampaignRunBindings,
        configuration: CampaignConfiguration,
        configuration_ref: ArtifactReference,
        environments: ChildEnvironmentRegistry,
        store: CampaignArtifactStore,
        frozen_external: FrozenExternalIngress | None,
        frozen_baselines: FrozenTrustedBaselines | None,
        frozen_evaluation: FrozenEvaluationProbe | None,
        frozen_behavioral: FrozenBehavioralInputs | None,
    ) -> None:
        """Verify persisted bytes, contracts, and full plan-to-binding agreement."""
        if store.sealed:
            raise ValueError("preparation must leave the store unsealed")
        expected_root = (
            environments.project_root / configuration.artifact_root
        ).resolve()
        if (
            store.root != expected_root
            or configuration.environments != options.environments
            or configuration.campaign_id != options.campaign_id
            or configuration.artifact_root != options.artifact_root
            or configuration.seed != options.seed
            or configuration.max_parallel_children != options.max_parallel_children
            or configuration.fail_fast != options.fail_fast
        ):
            raise ValueError(
                "prepared environment or artifact root differs from options"
            )
        references = {
            reference.artifact_id: reference for reference in store.references()
        }
        for reference in references.values():
            store.verify(reference)
            if reference.visibility != ArtifactVisibility.EVALUATOR_ONLY:
                raise ValueError("preparation artifacts must remain evaluator-only")
        for reference in configuration.frozen_inputs:
            if references.get(reference.artifact_id) != reference:
                raise ValueError("configuration references an unfrozen artifact")
        persisted_options = CampaignPreparationOptions.model_validate_json(
            store.verify(references["preparation:options"]).read_bytes()
        )
        persisted_bindings = CampaignRunBindings.model_validate_json(
            store.verify(references["preparation:run-bindings"]).read_bytes()
        )
        persisted_configuration = CampaignConfiguration.model_validate_json(
            store.verify(configuration_ref).read_bytes()
        )
        if (
            persisted_options != options
            or persisted_bindings != bindings
            or persisted_configuration != configuration
            or configuration.runs != matrix.runs
        ):
            raise ValueError("persisted preparation differs from the validated plan")
        bindings.validate_preparation(options)
        bindings.validate_campaign(
            configuration,
            ingress_plan_set=(
                None if frozen_external is None else frozen_external.plan_set
            ),
            trusted_baselines=(
                None if frozen_baselines is None else frozen_baselines.manifest
            ),
        )
        if frozen_baselines is not None and (
            bindings.trusted_baseline_manifest != frozen_baselines.artifact
            or frozen_baselines.artifact not in configuration.frozen_inputs
        ):
            raise ValueError("prepared trusted baselines differ from bindings")
        if frozen_evaluation is not None:
            if bindings.evaluation_probe != frozen_evaluation.artifact:
                raise ValueError("prepared evaluation probe differs from binding")
        if frozen_behavioral is not None:
            if (
                bindings.behavioral_task != frozen_behavioral.task_artifact
                or bindings.behavioral_oracle != frozen_behavioral.oracle_artifact
            ):
                raise ValueError("prepared behavioral inputs differ from bindings")
            if any(
                reference not in configuration.frozen_inputs
                for reference in frozen_behavioral.artifacts
            ):
                raise ValueError("prepared behavioral inputs are absent from campaign")
        for plan in matrix.plans:
            bindings.binding_for(plan.run.run_id).validate_trial_plan(plan)


def prepare_campaign(
    options: CampaignPreparationOptions,
    *,
    project_root: str | Path,
    source_environment: Mapping[str, str] | None = None,
) -> PreparedCampaign:
    """Convenience API; equivalent to CampaignPreparation().prepare_campaign()."""
    return CampaignPreparation().prepare_campaign(
        options,
        project_root=project_root,
        source_environment=source_environment,
    )
