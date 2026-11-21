"""Pure preparation inputs; no files, environments, or backends are opened.

Import these contracts from this module, not the campaign package root. They
reuse trial configuration types without making campaign.models depend on trial
execution. Matrix construction and input freezing are deliberately separate work.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.agent.models import AgentRunConfiguration
from membench.campaign.artifacts import canonical_json_bytes
from membench.campaign.models import ArtifactReference, ChildEnvironmentSpec
from membench.contract import ExperimentManifest, Mechanism, ModelsConfig, WritePolicy
from membench.ingress.models import IngressMode
from membench.memory.models import CanonicalModel
from membench.qualification.models import QuiescenceConfiguration
from membench.tools.fixtures import fixture_required
from membench.trial.models import ReplayConfiguration

PreparationId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
AllocationId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CampaignPurpose = Literal["pilot", "experiment"]


def model_profile_sha256(models: ModelsConfig) -> str:
    """Identify model roles, pins, embedding dimensions, and responder tokenizer.

    This is a model-profile identity, not a fingerprint of all generation or
    mechanism settings. The complete manifest/run hashes retain those settings.
    """
    return hashlib.sha256(canonical_json_bytes(models)).hexdigest()


def execution_condition_sha256(
    manifest: ExperimentManifest,
    agent_configuration: AgentRunConfiguration,
    replay: ReplayConfiguration,
) -> str:
    """Hash static condition choices, excluding matrix-assigned identities.

    Full run/plan hashes cover attack_id, run_id, and replication. Removing only
    these template slots lets bindings verify their selected condition even
    after the matrix has instantiated a particular case and replicate.
    """
    payload = manifest.model_dump(mode="json")
    del payload["attack"]["attack_id"]
    del payload["isolation"]["run_id"]
    del payload["replication"]
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "manifest": payload,
                "agent_configuration": agent_configuration,
                "replay": replay,
            }
        )
    ).hexdigest()


class CampaignConditionSpec(CanonicalModel):
    """One explicit mechanism/policy/model-profile condition, not a cross product.

    The manifest owns model configuration: no second copy lives in a profile
    registry. model_profile_id is a readable label for those model roles.
    A future matrix builder replaces only the template's attack_id, run_id, and
    replication fields. Other choices must not be silently inferred from .env.
    """

    condition_id: PreparationId
    model_profile_id: PreparationId
    manifest_template: ExperimentManifest
    agent_configuration: AgentRunConfiguration
    replay: ReplayConfiguration = Field(default_factory=ReplayConfiguration)

    @property
    def mechanism(self) -> Mechanism:
        return self.manifest_template.condition.mechanism

    @property
    def policy(self) -> WritePolicy:
        return self.manifest_template.condition.write_policy

    @property
    def model_profile_sha256(self) -> str:
        return model_profile_sha256(self.manifest_template.models)

    @property
    def sha256(self) -> str:
        """Stable identity of this condition's result-affecting static settings."""
        return execution_condition_sha256(
            self.manifest_template, self.agent_configuration, self.replay
        )

    @model_validator(mode="after")
    def condition_must_be_executable(self) -> Self:
        if self.mechanism not in {
            Mechanism.TELEMEM,
            Mechanism.LANGMEM,
            Mechanism.DAKERA,
        }:
            raise ValueError(
                "campaign preparation supports TeleMem, LangMem, and Dakera"
            )
        if self.policy not in {WritePolicy.DIRECT, WritePolicy.NATIVE_SELECTIVE}:
            raise ValueError(
                "campaign preparation supports direct and native_selective"
            )
        models = self.manifest_template.models
        if self.mechanism == Mechanism.DAKERA and (
            models.writer is not None or models.embedding is not None
        ):
            raise ValueError("Dakera external writer and embedding models must be null")
        for role in ("writer", "responder", "embedding"):
            model = getattr(models, role)
            if model is not None and (model.provider != "ollama" or not model.digest):
                raise ValueError(
                    f"{role} requires an Ollama model with a pinned digest"
                )
        agent = self.agent_configuration
        retrieval = self.manifest_template.retrieval
        if not agent.memory_read_only:
            raise ValueError("campaign probes require a read-only agent")
        if (
            agent.retrieval_top_k != retrieval.top_k
            or agent.retrieval_threshold != retrieval.threshold
            or agent.context_token_budget != retrieval.context_token_budget
        ):
            raise ValueError("agent retrieval settings must match the manifest")
        if self.policy == WritePolicy.DIRECT and self.replay.native_options:
            raise ValueError("direct conditions cannot contain native replay options")
        return self


class ExternalIngressPreparation(CanonicalModel):
    """One frozen external-delivery source set bound to selected host trials.

    ``host_trial_ids`` maps an external delivery's trial ID to the PersonaMem
    trial ID whose replay/probe lifecycle receives that source.  Keeping the
    mapping explicit avoids assuming that separately curated external IDs share
    the PersonaMem allocation naming scheme.
    """

    external_artifact_directory: Path
    ingress_plan_set_id: PreparationId
    ingress_mode: IngressMode
    host_trial_ids: dict[PreparationId, PreparationId] = Field(min_length=1)
    selection_agent_configuration: AgentRunConfiguration | None = None

    @field_validator("external_artifact_directory", mode="before")
    @classmethod
    def source_path_must_be_nonblank(cls, value):
        if isinstance(value, str) and (not value.strip() or "\x00" in value):
            raise ValueError(
                "external ingress artifact directory must be non-blank and "
                "cannot contain NUL"
            )
        return value

    @model_validator(mode="after")
    def host_trial_mapping_must_be_deterministic(self) -> Self:
        if tuple(self.host_trial_ids) != tuple(sorted(self.host_trial_ids)):
            raise ValueError("external ingress host trial IDs must be sorted")
        if any(not value for value in self.host_trial_ids.values()):
            raise ValueError("external ingress host trial IDs cannot be blank")
        selection = self.selection_agent_configuration
        if self.ingress_mode == IngressMode.AGENT_SELECTED:
            if selection is None:
                raise ValueError(
                    "agent-selected ingress requires a selection agent configuration"
                )
            if not selection.memory_read_only:
                raise ValueError("selection agents must be memory read-only")
            if selection.allow_external_side_effects:
                raise ValueError(
                    "selection agents cannot allow external side effects"
                )
            if not selection.allowed_tools:
                raise ValueError("selection agents require frozen inbox tools")
        elif selection is not None:
            raise ValueError(
                "forced ingestion cannot carry a selection agent configuration"
            )
        return self


class CampaignPreparationOptions(CanonicalModel):
    """Serializable input for future preparation, usable from Python or a CLI.

    case_ids=None explicitly selects all included cases; an empty selection is
    an error. Conditions are enumerated explicitly so unsupported combinations
    (such as native extraction in Dakera) are never scheduled by a blind product.
    seed/response_replicates replace the templates' replication settings; native
    writer-specific seeds remain separately declared mechanism options.
    """

    schema_version: Literal[1] = 1
    campaign_id: PreparationId
    projection_directory: Path
    tool_fixture_path: Path | None = None
    external_ingress: ExternalIngressPreparation | None = None
    trusted_baseline_path: Path | None = None
    evaluation_probe_path: Path | None = None
    behavioral_task_path: Path | None = None
    behavioral_oracle_path: Path | None = None
    case_ids: tuple[AllocationId, ...] | None = None
    conditions: tuple[CampaignConditionSpec, ...] = Field(min_length=1)
    environments: tuple[ChildEnvironmentSpec, ...] = Field(min_length=1)
    seed: int = Field(ge=0)
    response_replicates: int = Field(default=1, ge=1)
    purpose: CampaignPurpose = "pilot"
    quiescence: QuiescenceConfiguration = Field(default_factory=QuiescenceConfiguration)
    cleanup_storage: bool = True
    artifact_root: str = Field(min_length=1)
    max_parallel_children: int = Field(default=1, ge=1)
    fail_fast: bool = False

    @field_validator(
        "projection_directory",
        "tool_fixture_path",
        "trusted_baseline_path",
        "evaluation_probe_path",
        "behavioral_task_path",
        "behavioral_oracle_path",
        mode="before",
    )
    @classmethod
    def source_path_must_be_nonblank(cls, value):
        if (
            value is not None
            and isinstance(value, str)
            and (not value.strip() or "\x00" in value)
        ):
            raise ValueError("source paths must be non-blank and cannot contain NUL")
        return value

    @field_validator("case_ids")
    @classmethod
    def selection_must_be_unique(cls, value):
        if value is not None:
            if not value:
                raise ValueError(
                    "case selection cannot be empty; use null for all cases"
                )
            if len(set(value)) != len(value):
                raise ValueError("case selection contains duplicate allocation IDs")
        return value

    @field_validator("artifact_root")
    @classmethod
    def output_must_be_a_portable_subdirectory(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            "\\" in value
            or ":" in value
            or "\x00" in value
            or path.is_absolute()
            or PureWindowsPath(value).drive
            or value == "."
            or ".." in path.parts
            or path.as_posix() != value
        ):
            raise ValueError("artifact_root must be a canonical relative subdirectory")
        return value

    @model_validator(mode="after")
    def conditions_and_environments_must_be_consistent(self) -> Self:
        ids = [condition.condition_id for condition in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("condition IDs must be unique")
        profiles: dict[str, str] = {}
        for condition in self.conditions:
            identity = condition.model_profile_sha256
            previous = profiles.setdefault(condition.model_profile_id, identity)
            if previous != identity:
                raise ValueError(
                    "one model_profile_id cannot describe different models"
                )
        mechanisms = [environment.mechanism for environment in self.environments]
        if len(mechanisms) != len(set(mechanisms)):
            raise ValueError("environment mechanisms must be unique")
        if set(mechanisms) != {condition.mechanism for condition in self.conditions}:
            raise ValueError("environments must match the selected mechanisms exactly")
        external = self.external_ingress
        if external is not None and external.ingress_mode == IngressMode.AGENT_SELECTED:
            assert external.selection_agent_configuration is not None
            selection = external.selection_agent_configuration
            for condition in self.conditions:
                responder = condition.agent_configuration
                if (
                    selection.retrieval_top_k != responder.retrieval_top_k
                    or selection.retrieval_threshold != responder.retrieval_threshold
                    or selection.context_token_budget != responder.context_token_budget
                ):
                    raise ValueError(
                        "selection-agent retrieval settings must match every "
                        "condition's responder settings"
                    )
        if self.tool_fixture_path is None and any(
            fixture_required(condition.agent_configuration.allowed_tools)
            for condition in self.conditions
        ):
            raise ValueError(
                "tool_fixture_path is required when contact or document tools "
                "are selected"
            )
        if self.external_ingress is not None and self.tool_fixture_path is None:
            raise ValueError(
                "tool_fixture_path is required for external ingress preparation"
            )
        behavioral_paths = (self.behavioral_task_path, self.behavioral_oracle_path)
        if any(path is None for path in behavioral_paths) and any(
            path is not None for path in behavioral_paths
        ):
            raise ValueError(
                "behavioral_task_path and behavioral_oracle_path must be "
                "present together"
            )
        return self

    @property
    def sha256(self) -> str:
        """Hash resolved option values without timestamps or filesystem access."""
        return hashlib.sha256(canonical_json_bytes(self)).hexdigest()


class CampaignPreparationCompletion(CanonicalModel):
    """Final readiness marker and integrity inventory for one preparation."""

    schema_version: Literal[1] = 1
    status: Literal["prepared"] = "prepared"
    campaign_id: PreparationId
    preparation_sha256: Sha256
    matrix_sha256: Sha256
    projection_set_id: PreparationId
    selected_case_ids: tuple[AllocationId, ...] = Field(min_length=1)
    run_count: int = Field(gt=0)
    purpose: CampaignPurpose
    configuration: ArtifactReference
    artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def inventory_must_be_unique_and_contain_configuration(self) -> Self:
        artifact_ids = [item.artifact_id for item in self.artifacts]
        paths = [item.relative_path for item in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("completion inventory contains duplicate artifact IDs")
        if len(paths) != len(set(paths)):
            raise ValueError("completion inventory contains duplicate artifact paths")
        if self.configuration not in self.artifacts:
            raise ValueError("completion configuration is absent from its inventory")
        return self
