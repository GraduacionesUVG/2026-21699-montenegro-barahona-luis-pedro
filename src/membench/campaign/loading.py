"""Rehydrate a complete, unstarted campaign preparation from frozen artifacts.

Loading never rebuilds a matrix or reads the mutable source projection directory.
It verifies the preparation boundary, reconstructs the artifact registry, and
rechecks host-specific child environments before returning ``PreparedCampaign``.
Resume and retry semantics are intentionally outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from membench.campaign.artifacts import CampaignArtifactStore
from membench.campaign.bindings import CampaignRunBindings
from membench.campaign.environments import ChildEnvironmentRegistry
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
    CampaignConfiguration,
)
from membench.campaign.preparation import PreparedCampaign
from membench.campaign.preparation_models import (
    CampaignPreparationCompletion,
    CampaignPreparationOptions,
)
from membench.ingress.models import TrustedBaselineManifest
from membench.ingress.plans import (
    TrialIngressPlanSet,
    load_trial_ingress_plan_set,
)

_MAX_COMPLETION_BYTES = 16 * 1024 * 1024
_COMPLETION_PATH = "preparation/complete.json"
_EXECUTION_PATHS = (
    "campaign_configuration.json",
    "campaign_result.json",
    "analysis.json",
    "checksums.json",
    "children",
)


class PreparedCampaignLoadError(RuntimeError):
    """Stage-local loading failure with the original error in ``__cause__``."""

    def __init__(
        self,
        stage: str,
        *,
        campaign_root: Path | None = None,
    ) -> None:
        self.stage = stage
        self.campaign_root = campaign_root
        super().__init__(f"prepared campaign loading failed during {stage}")


@dataclass(frozen=True)
class LoadedPreparationContracts:
    """Verified preparation contracts loaded from the frozen inventory."""

    options: CampaignPreparationOptions
    configuration: CampaignConfiguration
    bindings: CampaignRunBindings


class PreparedCampaignLoader:
    """Load one immutable, previously unstarted preparation for execution."""

    def load(
        self,
        campaign_root: str | Path,
        *,
        project_root: str | Path,
        source_environment: Mapping[str, str] | None = None,
    ) -> PreparedCampaign:
        stage = "resolve_paths"
        resolved_campaign: Path | None = None
        try:
            resolved_project, resolved_campaign = self._resolve_roots(
                campaign_root,
                project_root=project_root,
            )
            stage = "read_completion"
            completion = self._load_completion(resolved_campaign)
            stage = "verify_execution_state"
            self._assert_unstarted(resolved_campaign)
            stage = "register_artifacts"
            store, completion_reference = self._open_and_register_store(
                resolved_campaign,
                completion,
            )
            stage = "load_contracts"
            contracts = self._load_contracts(store, completion)
            stage = "validate_contracts"
            ingress_plan_set, trusted_baselines = self._validate_contracts(
                contracts,
                completion,
                store,
                campaign_root=resolved_campaign,
                project_root=resolved_project,
            )
            stage = "preflight_environments"
            environments = self._load_environments(
                contracts,
                project_root=resolved_project,
                source_environment=source_environment,
            )
            return PreparedCampaign(
                configuration=contracts.configuration,
                bindings=contracts.bindings,
                completion=completion_reference,
                artifacts=store,
                environments=environments,
                ingress_plan_set=ingress_plan_set,
                trusted_baselines=trusted_baselines,
            )
        except Exception as exc:
            raise PreparedCampaignLoadError(
                stage,
                campaign_root=resolved_campaign,
            ) from exc

    def _resolve_roots(
        self,
        campaign_root: str | Path,
        *,
        project_root: str | Path,
    ) -> tuple[Path, Path]:
        project = Path(project_root).resolve()
        configured_campaign = Path(campaign_root)
        if not configured_campaign.is_absolute():
            configured_campaign = project / configured_campaign
        campaign = configured_campaign.resolve()
        if not project.is_dir():
            raise NotADirectoryError("project root must be an existing directory")
        if not campaign.is_dir():
            raise NotADirectoryError("campaign root must be an existing directory")
        if campaign == project or not campaign.is_relative_to(project):
            raise ValueError("campaign root must remain below the project root")
        return project, campaign

    def _load_completion(
        self,
        campaign_root: Path,
    ) -> CampaignPreparationCompletion:
        path = campaign_root / Path(*_COMPLETION_PATH.split("/"))
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("campaign has no regular preparation marker")
        if path.stat().st_size > _MAX_COMPLETION_BYTES:
            raise ValueError("preparation marker exceeds the size limit")
        return CampaignPreparationCompletion.model_validate_json(path.read_bytes())

    def _assert_unstarted(self, campaign_root: Path) -> None:
        present = [name for name in _EXECUTION_PATHS if (campaign_root / name).exists()]
        if present:
            raise FileExistsError(
                "campaign execution has already started: " + ", ".join(present)
            )

    def _open_and_register_store(
        self,
        campaign_root: Path,
        completion: CampaignPreparationCompletion,
    ) -> tuple[CampaignArtifactStore, ArtifactReference]:
        expected_paths = {
            *(reference.relative_path for reference in completion.artifacts),
            _COMPLETION_PATH,
        }
        observed_paths: set[str] = set()
        for path in campaign_root.rglob("*"):
            if path.is_symlink():
                raise ValueError("prepared campaign cannot contain symbolic links")
            if path.is_file():
                observed_paths.add(path.relative_to(campaign_root).as_posix())
        unexpected = sorted(observed_paths - expected_paths)
        missing = sorted(expected_paths - observed_paths)
        if unexpected or missing:
            raise ValueError(
                "prepared artifact inventory differs from disk "
                f"(missing={missing!r}, unexpected={unexpected!r})"
            )

        store = CampaignArtifactStore.open_existing(campaign_root)
        for reference in completion.artifacts:
            observed = store.register_existing(
                artifact_id=reference.artifact_id,
                kind=reference.kind,
                relative_path=reference.relative_path,
                media_type=reference.media_type,
                visibility=reference.visibility,
                expected_sha256=reference.sha256,
                expected_size_bytes=reference.size_bytes,
            )
            if observed != reference:
                raise ValueError(
                    f"prepared artifact metadata changed: {reference.artifact_id}"
                )
            store.verify(reference)

        completion_reference = store.register_existing(
            artifact_id="preparation:complete",
            kind=ArtifactKind.OTHER,
            relative_path=_COMPLETION_PATH,
            media_type="application/json",
            visibility=ArtifactVisibility.EVALUATOR_ONLY,
        )
        return store, completion_reference

    def _load_contracts(
        self,
        store: CampaignArtifactStore,
        completion: CampaignPreparationCompletion,
    ) -> LoadedPreparationContracts:
        references = {item.artifact_id: item for item in completion.artifacts}
        options = CampaignPreparationOptions.model_validate_json(
            store.verify(references["preparation:options"]).read_bytes()
        )
        configuration = CampaignConfiguration.model_validate_json(
            store.verify(references["preparation:configuration"]).read_bytes()
        )
        bindings = CampaignRunBindings.model_validate_json(
            store.verify(references["preparation:run-bindings"]).read_bytes()
        )
        return LoadedPreparationContracts(
            options=options,
            configuration=configuration,
            bindings=bindings,
        )

    def _validate_contracts(
        self,
        contracts: LoadedPreparationContracts,
        completion: CampaignPreparationCompletion,
        store: CampaignArtifactStore,
        *,
        campaign_root: Path,
        project_root: Path,
    ) -> tuple[TrialIngressPlanSet | None, TrustedBaselineManifest | None]:
        options = contracts.options
        configuration = contracts.configuration
        bindings = contracts.bindings
        if completion.configuration not in completion.artifacts:
            raise ValueError("completion configuration is not in the inventory")
        if completion.configuration.artifact_id != "preparation:configuration":
            raise ValueError("completion points to the wrong configuration artifact")
        if not (
            completion.campaign_id
            == options.campaign_id
            == configuration.campaign_id
            == bindings.campaign_id
        ):
            raise ValueError("prepared campaign IDs do not match")
        if completion.preparation_sha256 != options.sha256:
            raise ValueError("preparation options hash does not match completion")
        if completion.projection_set_id != bindings.projection_set_id:
            raise ValueError("projection set does not match completion")
        if completion.run_count != len(configuration.runs):
            raise ValueError("prepared run count does not match completion")
        if completion.purpose != options.purpose:
            raise ValueError("campaign purpose does not match completion")
        selected = {item.allocation_case_id for item in bindings.bindings}
        if selected != set(completion.selected_case_ids):
            raise ValueError("selected cases do not match completion")
        if options.case_ids is not None and selected != set(options.case_ids):
            raise ValueError("selected cases do not match preparation options")
        if configuration.environments != options.environments:
            raise ValueError("prepared child environments do not match options")
        if configuration.artifact_root != options.artifact_root:
            raise ValueError("prepared artifact roots do not match")
        expected_root = (project_root / options.artifact_root).resolve()
        if expected_root != campaign_root:
            raise ValueError("loaded path differs from the prepared artifact root")

        inventory = {item.artifact_id: item for item in completion.artifacts}
        for reference in configuration.frozen_inputs:
            if inventory.get(reference.artifact_id) != reference:
                raise ValueError("configuration references an unfrozen artifact")
            store.verify(reference)
        ingress_plan_set = self._load_ingress_plan_set(store, bindings)
        trusted_baselines = self._load_trusted_baselines(store, bindings)
        bindings.validate_preparation(options)
        bindings.validate_campaign(
            configuration,
            ingress_plan_set=ingress_plan_set,
            trusted_baselines=trusted_baselines,
        )
        cleanup = {item.cleanup_storage for item in bindings.bindings}
        if cleanup != {options.cleanup_storage}:
            raise ValueError("prepared cleanup policies are inconsistent")
        return ingress_plan_set, trusted_baselines

    def _load_ingress_plan_set(
        self,
        store: CampaignArtifactStore,
        bindings: CampaignRunBindings,
    ) -> TrialIngressPlanSet | None:
        """Load the plan set referenced by external bindings, if any."""

        manifest = bindings.ingress_plan_manifest
        if manifest is None:
            return None
        manifest_path = store.verify(manifest)
        try:
            return load_trial_ingress_plan_set(manifest_path.parent)
        except Exception as exc:
            raise ValueError("could not load frozen ingress plans") from exc

    def _load_trusted_baselines(
        self,
        store: CampaignArtifactStore,
        bindings: CampaignRunBindings,
    ) -> TrustedBaselineManifest | None:
        """Load only the exact evaluator-only baseline manifest reference."""

        reference = bindings.trusted_baseline_manifest
        if reference is None:
            return None
        try:
            return TrustedBaselineManifest.model_validate_json(
                store.verify(reference).read_bytes()
            )
        except Exception as exc:
            raise ValueError("could not load frozen trusted baselines") from exc

    def _load_environments(
        self,
        contracts: LoadedPreparationContracts,
        *,
        project_root: Path,
        source_environment: Mapping[str, str] | None,
    ) -> ChildEnvironmentRegistry:
        registry = ChildEnvironmentRegistry(
            contracts.configuration.environments,
            project_root=project_root,
            source_environment=source_environment,
        )
        registry.resolve_all(require_all_env_keys=True)
        return registry


def load_prepared_campaign(
    campaign_root: str | Path,
    *,
    project_root: str | Path,
    source_environment: Mapping[str, str] | None = None,
) -> PreparedCampaign:
    """Convenience API for :meth:`PreparedCampaignLoader.load`."""

    return PreparedCampaignLoader().load(
        campaign_root,
        project_root=project_root,
        source_environment=source_environment,
    )
