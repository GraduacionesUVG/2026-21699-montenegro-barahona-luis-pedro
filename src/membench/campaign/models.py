"""Canonical plans and cross-process results for paired agent campaigns."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.agent.models import (
    AgentRunConfiguration,
    AgentRunResult,
    AgentRunStatus,
    AgentTask,
)
from membench.contract import ExperimentManifest, Mechanism, WritePolicy
from membench.memory.models import AdapterProvenance, CanonicalModel, MemoryScope
from membench.qualification.models import QualificationSummary
from membench.tools.fixtures import fixture_required

_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$"
_REQUEST_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TrialTwin(StrEnum):
    """Member of one independently reconstructed paired trial family."""

    CLEAN = "clean"
    MATCHED_BENIGN = "matched_benign"
    POISONED = "poisoned"


class ArtifactKind(StrEnum):
    """Stable high-level role of a sealed campaign artifact."""

    DATASET_PROJECTION = "dataset_projection"
    ATTACK_BATTERY = "attack_battery"
    TOOL_FIXTURE = "tool_fixture"
    TRIAL_INGRESS_PLAN = "trial_ingress_plan"
    EVALUATION_PROBE = "evaluation_probe"
    EVALUATION_RESULT = "evaluation_result"
    BEHAVIORAL_TASK = "behavioral_task"
    BEHAVIORAL_ORACLE = "behavioral_oracle"
    BEHAVIORAL_RESULT = "behavioral_result"
    RUN_BINDINGS = "run_bindings"
    RESOLVED_MANIFEST = "resolved_manifest"
    CHILD_REQUEST = "child_request"
    CHILD_RESULT = "child_result"
    QUALIFICATION = "qualification"
    AGENT_RESULT = "agent_result"
    ACTION_DECISION_DIAGNOSTIC = "action_decision_diagnostic"
    OUTBOX_SNAPSHOT = "outbox_snapshot"
    EVENTS = "events"
    SNAPSHOT = "snapshot"
    PROVENANCE = "provenance"
    CHECKSUMS = "checksums"
    TRUSTED_BASELINE = "trusted_baseline"
    OTHER = "other"


class ArtifactVisibility(StrEnum):
    """Whether an artifact may be visible to the simulated attacker."""

    PUBLIC = "public"
    EVALUATOR_ONLY = "evaluator_only"


class ChildRunStatus(StrEnum):
    """Operational outcome of one isolated mechanism child run."""

    SUCCEEDED = "succeeded"
    INVALIDATED = "invalidated"
    FAILED = "failed"


class CampaignStatus(StrEnum):
    """Terminal aggregate state of a campaign execution."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class ChildFailureClass(StrEnum):
    """Stable launcher-level classification for unusable child runs."""

    CONFIGURATION_INVALID = "configuration_invalid"
    ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
    QUALIFICATION_FAILED = "qualification_failed"
    PROCESS_LAUNCH_FAILED = "process_launch_failed"
    PROCESS_TIMEOUT = "process_timeout"
    CHILD_PROTOCOL_ERROR = "child_protocol_error"
    ADAPTER_FAILURE = "adapter_failure"
    AGENT_FAILURE = "agent_failure"
    ARTIFACT_FAILURE = "artifact_failure"
    UNEXPECTED_ERROR = "unexpected_error"


class ArtifactReference(CanonicalModel):
    """Portable, integrity-checked reference to one campaign artifact."""

    artifact_id: str = Field(pattern=_REQUEST_ID_PATTERN)
    kind: ArtifactKind
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    visibility: ArtifactVisibility = ArtifactVisibility.EVALUATOR_ONLY

    @field_validator("relative_path")
    @classmethod
    def artifact_path_must_be_portable_and_relative(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("artifact paths must use portable '/' separators")
        path = PurePosixPath(value)
        drive_like = bool(path.parts and path.parts[0].endswith(":"))
        if (
            path.is_absolute()
            or drive_like
            or value in {".", ".."}
            or ".." in path.parts
        ):
            raise ValueError("artifact paths must be relative and cannot traverse")
        return value


class ChildEnvironmentSpec(CanonicalModel):
    """Trusted interpreter and module used for one mechanism environment."""

    mechanism: Mechanism
    python_executable: str = Field(min_length=1)
    module: str = Field(
        default="membench.agent",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_.]*$",
    )
    working_directory: str | None = Field(default=None, min_length=1)
    timeout_seconds: float = Field(default=900.0, gt=0)
    inherited_env_keys: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("inherited_env_keys")
    @classmethod
    def environment_keys_must_be_sorted_and_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("inherited environment keys cannot be blank")
        if value != tuple(sorted(set(value))):
            raise ValueError(
                "inherited environment keys must be sorted and unique"
            )
        return value


class CampaignRunSpec(CanonicalModel):
    """One mechanism, persona, attack, twin, and response replicate."""

    schema_version: Literal[1] = 1
    campaign_id: str = Field(pattern=_ID_PATTERN)
    trial_family_id: str = Field(pattern=_ID_PATTERN)
    run_id: str = Field(pattern=_ID_PATTERN)
    twin: TrialTwin
    persona_id: str = Field(min_length=1)
    history_unit_id: str = Field(min_length=1)
    response_replicate: int = Field(default=1, ge=1)
    manifest: ExperimentManifest
    scope: MemoryScope
    agent_task: AgentTask
    agent_configuration: AgentRunConfiguration

    @property
    def mechanism(self) -> Mechanism:
        return self.manifest.condition.mechanism

    @property
    def policy(self) -> WritePolicy:
        return self.manifest.condition.write_policy

    @property
    def attack_id(self) -> str:
        return self.manifest.attack.attack_id

    @property
    def manifest_sha256(self) -> str:
        return self.manifest.sha256

    @model_validator(mode="after")
    def identities_and_scopes_must_match(self) -> Self:
        if self.manifest.isolation.run_id != self.run_id:
            raise ValueError("manifest isolation run_id must match campaign run_id")
        if self.scope.experiment_id != self.campaign_id:
            raise ValueError("memory scope experiment_id must match campaign_id")
        if self.scope.run_id != self.run_id:
            raise ValueError("memory scope run_id must match campaign run_id")
        if self.run_id not in self.scope.namespace:
            raise ValueError("memory scope namespace must contain campaign run_id")
        if self.scope.user_id != self.persona_id:
            raise ValueError("memory scope user_id must match persona_id")
        if self.agent_task.scope != self.scope:
            raise ValueError("agent task must use the campaign run memory scope")
        if self.response_replicate > self.manifest.replication.response_replicates:
            raise ValueError(
                "response_replicate exceeds the manifest replicate count"
            )
        return self


class CampaignConfiguration(CanonicalModel):
    """Frozen launcher input containing every paired child run."""

    schema_version: Literal[1] = 1
    campaign_id: str = Field(pattern=_ID_PATTERN)
    seed: int = Field(ge=0)
    runs: tuple[CampaignRunSpec, ...] = Field(min_length=1)
    environments: tuple[ChildEnvironmentSpec, ...] = Field(min_length=1)
    frozen_inputs: tuple[ArtifactReference, ...] = Field(min_length=2)
    artifact_root: str = Field(min_length=1)
    max_parallel_children: int = Field(default=1, ge=1)
    fail_fast: bool = False
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("created_at")
    @classmethod
    def created_at_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("campaign created_at must include a timezone")
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def runs_inputs_and_environments_must_form_a_fair_campaign(self) -> Self:
        if self.max_parallel_children > len(self.runs):
            raise ValueError("max_parallel_children cannot exceed scheduled runs")

        validate_campaign_runs(self.runs, campaign_id=self.campaign_id)

        environment_mechanisms = [item.mechanism for item in self.environments]
        if len(environment_mechanisms) != len(set(environment_mechanisms)):
            raise ValueError("campaign environments must be unique by mechanism")
        missing_environments = sorted(
            {run.mechanism for run in self.runs} - set(environment_mechanisms)
        )
        if missing_environments:
            missing = ", ".join(item.value for item in missing_environments)
            raise ValueError(f"campaign has no child environment for: {missing}")

        _validate_artifact_references(self.frozen_inputs, label="frozen input")
        kinds = {item.kind for item in self.frozen_inputs}
        required = {
            ArtifactKind.DATASET_PROJECTION,
            ArtifactKind.ATTACK_BATTERY,
        }
        if not required.issubset(kinds):
            missing = ", ".join(sorted(item.value for item in required - kinds))
            raise ValueError(f"campaign frozen inputs are missing: {missing}")
        if (
            any(
                fixture_required(run.agent_configuration.allowed_tools)
                for run in self.runs
            )
            and ArtifactKind.TOOL_FIXTURE not in kinds
        ):
            raise ValueError(
                "campaign frozen inputs are missing tool_fixture for selected "
                "discovery tools"
            )

        return self


class ChildRunRequest(CanonicalModel):
    """Complete JSON-serializable request sent to one isolated interpreter."""

    schema_version: Literal[1] = 1
    request_id: str = Field(pattern=_REQUEST_ID_PATTERN)
    run: CampaignRunSpec
    frozen_inputs: tuple[ArtifactReference, ...] = Field(min_length=2)
    artifact_directory: str = Field(min_length=1)
    attempt: int = Field(default=1, ge=1)
    cleanup_storage: bool = True
    issued_at: datetime = Field(default_factory=_utc_now)

    @field_validator("issued_at")
    @classmethod
    def issued_at_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("child request issued_at must include a timezone")
        return value

    @field_validator("artifact_directory")
    @classmethod
    def artifact_directory_must_be_portable(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("child artifact directory must use '/' separators")
        path = PurePosixPath(value)
        drive_like = bool(path.parts and path.parts[0].endswith(":"))
        if (
            path.is_absolute()
            or drive_like
            or value in {".", ".."}
            or ".." in path.parts
        ):
            raise ValueError(
                "child artifact directory must be relative and cannot traverse"
            )
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def frozen_inputs_must_be_complete(self) -> Self:
        _validate_artifact_references(self.frozen_inputs, label="child input")
        kinds = {item.kind for item in self.frozen_inputs}
        for required in (
            ArtifactKind.DATASET_PROJECTION,
            ArtifactKind.ATTACK_BATTERY,
        ):
            if required not in kinds:
                raise ValueError(
                    f"child request is missing {required.value} input"
                )
        if (
            fixture_required(self.run.agent_configuration.allowed_tools)
            and ArtifactKind.TOOL_FIXTURE not in kinds
        ):
            raise ValueError(
                "child request is missing tool_fixture for selected discovery tools"
            )
        return self


class ChildRunResult(CanonicalModel):
    """Terminal canonical result returned by one isolated child process."""

    schema_version: Literal[1] = 1
    request_id: str = Field(pattern=_REQUEST_ID_PATTERN)
    run: CampaignRunSpec
    status: ChildRunStatus
    qualification: QualificationSummary | None = None
    agent_result: AgentRunResult | None = None
    provenance: AdapterProvenance | None = None
    artifacts: tuple[ArtifactReference, ...] = Field(default_factory=tuple)
    failure_class: ChildFailureClass | None = None
    failure_type: str | None = Field(default=None, min_length=1)
    failure_message: str | None = Field(default=None, min_length=1)
    started_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime = Field(default_factory=_utc_now)

    @field_validator("started_at", "finished_at")
    @classmethod
    def child_timestamps_must_have_timezones(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("child result timestamps must include timezones")
        return value

    @model_validator(mode="after")
    def status_and_evidence_must_match(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("child run cannot finish before it starts")
        _validate_artifact_references(self.artifacts, label="child result")

        if self.agent_result is not None:
            if self.agent_result.task != self.run.agent_task:
                raise ValueError("agent result uses the wrong campaign task")
            if self.agent_result.configuration != self.run.agent_configuration:
                raise ValueError("agent result uses the wrong agent configuration")
        if self.qualification is not None and (
            self.qualification.mechanism != self.run.mechanism
            or self.qualification.policy != self.run.policy
        ):
            raise ValueError("qualification does not match the campaign condition")
        if self.provenance is not None:
            if self.provenance.mechanism != self.run.mechanism.value:
                raise ValueError("provenance has the wrong memory mechanism")
            if self.provenance.namespace != self.run.scope.namespace:
                raise ValueError("provenance has the wrong memory namespace")

        failures = (
            self.failure_class,
            self.failure_type,
            self.failure_message,
        )
        has_failure = any(item is not None for item in failures)
        if has_failure and any(item is None for item in failures):
            raise ValueError("child failure information must be complete")

        if self.status == ChildRunStatus.SUCCEEDED:
            if has_failure:
                raise ValueError("successful child runs cannot contain failures")
            if self.agent_result is None or self.provenance is None:
                raise ValueError(
                    "successful child runs require agent result and provenance"
                )
            if self.agent_result.status == AgentRunStatus.FAILED:
                raise ValueError("failed agent execution invalidates the child run")
            if self.qualification is not None and not self.qualification.passed:
                raise ValueError("successful child run has a failed qualification")
        elif not has_failure:
            raise ValueError("invalidated or failed child runs require a failure")
        return self


class CampaignSummary(CanonicalModel):
    """Terminal counts and status for one complete campaign."""

    campaign_id: str = Field(pattern=_ID_PATTERN)
    status: CampaignStatus
    scheduled_count: int = Field(gt=0)
    succeeded_count: int = Field(ge=0)
    invalidated_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    started_at: datetime
    finished_at: datetime

    @field_validator("started_at", "finished_at")
    @classmethod
    def summary_timestamps_must_have_timezones(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("campaign summary timestamps must include timezones")
        return value

    @model_validator(mode="after")
    def counts_and_status_must_match(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("campaign cannot finish before it starts")
        observed = (
            self.succeeded_count + self.invalidated_count + self.failed_count
        )
        if observed != self.scheduled_count:
            raise ValueError("campaign outcome counts must equal scheduled_count")
        expected = (
            CampaignStatus.COMPLETED
            if self.succeeded_count == self.scheduled_count
            else CampaignStatus.FAILED
            if self.succeeded_count == 0
            else CampaignStatus.PARTIAL
        )
        if self.status != expected:
            raise ValueError("campaign status does not match its outcome counts")
        return self


class CampaignResult(CanonicalModel):
    """Complete launcher result ready for aggregation and artifact sealing."""

    configuration: CampaignConfiguration
    children: tuple[ChildRunResult, ...]
    summary: CampaignSummary
    artifacts: tuple[ArtifactReference, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def child_results_and_summary_must_match_configuration(self) -> Self:
        if self.summary.campaign_id != self.configuration.campaign_id:
            raise ValueError("campaign summary has the wrong campaign_id")
        if self.summary.scheduled_count != len(self.configuration.runs):
            raise ValueError("campaign summary has the wrong scheduled_count")

        configured = {run.run_id: run for run in self.configuration.runs}
        observed_ids = [child.run.run_id for child in self.children]
        request_ids = [child.request_id for child in self.children]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("campaign child request IDs must be unique")
        if len(observed_ids) != len(set(observed_ids)):
            raise ValueError("campaign child run IDs must be unique")
        if set(observed_ids) != set(configured):
            raise ValueError(
                "campaign must contain exactly one result for every scheduled run"
            )
        for child in self.children:
            if child.run != configured[child.run.run_id]:
                raise ValueError("child result run specification was modified")

        counts = {
            ChildRunStatus.SUCCEEDED: 0,
            ChildRunStatus.INVALIDATED: 0,
            ChildRunStatus.FAILED: 0,
        }
        for child in self.children:
            counts[child.status] += 1
        if (
            self.summary.succeeded_count != counts[ChildRunStatus.SUCCEEDED]
            or self.summary.invalidated_count
            != counts[ChildRunStatus.INVALIDATED]
            or self.summary.failed_count != counts[ChildRunStatus.FAILED]
        ):
            raise ValueError("campaign summary counts do not match child results")
        _validate_artifact_references(self.artifacts, label="campaign result")
        return self


def _validate_artifact_references(
    artifacts: tuple[ArtifactReference, ...],
    *,
    label: str,
) -> None:
    artifact_ids = [item.artifact_id for item in artifacts]
    paths = [item.relative_path for item in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError(f"{label} artifact IDs must be unique")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} artifact paths must be unique")


def validate_campaign_runs(
    runs: tuple[CampaignRunSpec, ...], *, campaign_id: str
) -> None:
    """Validate identities, isolation, and twin fairness without artifact I/O.

    Shared by matrix construction and the final launcher configuration. Each
    run must already have passed CampaignRunSpec validation.
    """
    if not runs:
        raise ValueError("campaign must contain at least one run")
    for identities, label in (
        ([run.run_id for run in runs], "run IDs"),
        ([run.agent_task.task_id for run in runs], "agent task IDs"),
        ([run.scope.namespace for run in runs], "memory namespaces"),
    ):
        if len(identities) != len(set(identities)):
            raise ValueError(f"campaign {label} must be unique")
    if any(run.campaign_id != campaign_id for run in runs):
        raise ValueError("every run must reference the campaign_id")
    grouped: dict[tuple[str, int], list[CampaignRunSpec]] = {}
    for run in runs:
        grouped.setdefault(
            (run.trial_family_id, run.response_replicate), []
        ).append(run)
    for key, family in grouped.items():
        _validate_trial_family(key, family)


def _validate_trial_family(
    key: tuple[str, int],
    family: list[CampaignRunSpec],
) -> None:
    expected_twins = set(TrialTwin)
    observed_twins = [run.twin for run in family]
    if len(observed_twins) != len(set(observed_twins)):
        raise ValueError(f"trial family {key!r} contains duplicate twins")
    if set(observed_twins) != expected_twins:
        missing = sorted(item.value for item in expected_twins - set(observed_twins))
        raise ValueError(
            f"trial family {key!r} must contain exactly all three twins; "
            f"missing: {', '.join(missing) or 'none'}"
        )

    first = family[0]
    first_manifest = _paired_manifest_payload(first.manifest)
    for run in family[1:]:
        if run.persona_id != first.persona_id:
            raise ValueError(f"trial family {key!r} mixes personas")
        if run.history_unit_id != first.history_unit_id:
            raise ValueError(f"trial family {key!r} mixes history units")
        if _paired_manifest_payload(run.manifest) != first_manifest:
            raise ValueError(f"trial family {key!r} has unmatched manifests")
        if run.agent_configuration != first.agent_configuration:
            raise ValueError(f"trial family {key!r} has unmatched agent settings")
        comparable_task = (
            run.agent_task.instruction,
            run.agent_task.retrieval_query,
            run.agent_task.data_classification,
        )
        first_task = (
            first.agent_task.instruction,
            first.agent_task.retrieval_query,
            first.agent_task.data_classification,
        )
        if comparable_task != first_task:
            raise ValueError(f"trial family {key!r} has unmatched agent tasks")


def _paired_manifest_payload(manifest: ExperimentManifest) -> dict[str, object]:
    payload = manifest.model_dump(mode="json")
    isolation = payload.get("isolation")
    if isinstance(isolation, dict):
        isolation["run_id"] = "<paired-run-id>"
    return payload
