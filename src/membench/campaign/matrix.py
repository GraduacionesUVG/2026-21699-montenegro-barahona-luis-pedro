"""Deterministic expansion of verified projections into isolated trial plans.

No files, environment variables, clocks, backends, or subprocesses are used.
Only explicitly configured conditions are expanded; model/policy combinations
are never inferred. Import from this module rather than the package root.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Self

from pydantic import Field, model_validator

from membench.behavioral.models import BehavioralTaskTemplate
from membench.campaign.artifacts import canonical_json_bytes
from membench.campaign.bindings import RunProjectionBinding
from membench.campaign.inputs import (
    CampaignInputCase,
    CampaignInputError,
    CampaignInputs,
    validate_campaign_inputs,
)
from membench.campaign.models import CampaignRunSpec, TrialTwin, validate_campaign_runs
from membench.campaign.preparation_models import (
    CampaignConditionSpec,
    CampaignPreparationOptions,
    PreparationId,
    Sha256,
)
from membench.contract import ExperimentManifest
from membench.memory.models import CanonicalModel, MemoryScope
from membench.trial.bootstrap import build_probe_task
from membench.trial.models import TrialPlan

MATRIX_VERSION = "campaign_matrix_v1"


class CampaignMatrixError(RuntimeError):
    """Preparation inputs cannot form a complete, isolated trial matrix."""


class CampaignMatrix(CanonicalModel):
    """Plans and aligned bindings, with no private answer keys or artifact paths.

    This is preparation output, not proof of backend readiness. Replicates are
    full fresh-scope replays, not multiple responses over one shared memory.
    """

    schema_version: Literal[1] = 1
    matrix_version: Literal["campaign_matrix_v1"] = MATRIX_VERSION
    campaign_id: PreparationId
    preparation_sha256: Sha256
    projection_set_id: PreparationId
    plans: tuple[TrialPlan, ...] = Field(min_length=1)
    bindings: tuple[RunProjectionBinding, ...] = Field(min_length=1)

    @property
    def runs(self) -> tuple[CampaignRunSpec, ...]:
        """Derive launcher runs rather than storing a second copy."""
        return tuple(plan.run for plan in self.plans)

    @property
    def run_count(self) -> int:
        return len(self.plans)

    @property
    def sha256(self) -> str:
        """Stable matrix fingerprint including plan and binding content."""
        return _sha256(self)

    @model_validator(mode="after")
    def plans_and_bindings_must_agree(self) -> Self:
        validate_campaign_runs(self.runs, campaign_id=self.campaign_id)
        if len(self.plans) != len(self.bindings):
            raise ValueError("matrix requires exactly one binding per plan")
        families: dict[tuple[str, int], tuple] = {}
        for plan, binding in zip(self.plans, self.bindings, strict=True):
            binding.validate_trial_plan(plan)
            comparable = (
                binding.condition_id,
                binding.model_profile_id,
                binding.allocation_case_id,
                binding.probe_id,
                binding.replay,
                binding.quiescence,
                binding.purpose,
                binding.attack_review_status,
                binding.cleanup_storage,
            )
            key = (plan.run.trial_family_id, plan.run.response_replicate)
            if families.setdefault(key, comparable) != comparable:
                raise ValueError("matrix twin family has inconsistent bindings")
        return self


def build_campaign_matrix(
    options: CampaignPreparationOptions,
    inputs: CampaignInputs,
    *,
    behavioral_task: BehavioralTaskTemplate | None = None,
) -> CampaignMatrix:
    """Expand cases × conditions × replicates × twins without executing them.

    Ordering is projection ordinal, sorted condition ID, ascending replicate,
    then TrialTwin declaration order. IDs use canonical SHA-256, not Python's
    process-randomized hash(). Changing selection/order does not rename retained
    runs; changing their execution settings or case content does.

    Responder seed = options.seed + replicate - 1, paired across conditions and
    twins. Writer-specific seeds and model pins remain exactly as configured.
    """
    try:
        options = CampaignPreparationOptions.model_validate(
            options.model_dump(mode="json")
        )
        validate_campaign_inputs(inputs, options=options)
        if behavioral_task is not None and {
            item.case.persona_id for item in inputs.cases
        } != {behavioral_task.persona_id}:
            raise CampaignMatrixError(
                "behavioral task persona must match every selected campaign case"
            )
        expected_count = (
            len(inputs.cases)
            * len(options.conditions)
            * options.response_replicates
            * len(TrialTwin)
        )
        if options.max_parallel_children > expected_count:
            raise CampaignMatrixError(
                "max_parallel_children exceeds the matrix run count"
            )
        plans: list[TrialPlan] = []
        bindings: list[RunProjectionBinding] = []
        conditions = sorted(
            options.conditions, key=lambda condition: condition.condition_id
        )
        for item in inputs.cases:
            case_sha256 = _sha256(item)
            for condition in conditions:
                family_id = _build_family_id(
                    options, inputs, item, condition, case_sha256
                )
                for replicate in range(1, options.response_replicates + 1):
                    for twin in TrialTwin:
                        plan = _build_trial_plan(
                            options,
                            item,
                            condition,
                            family_id,
                            replicate,
                            twin,
                            behavioral_task=behavioral_task,
                        )
                        plans.append(plan)
                        bindings.append(
                            RunProjectionBinding.from_trial_plan(
                                plan,
                                condition_id=condition.condition_id,
                                model_profile_id=condition.model_profile_id,
                            )
                        )
        matrix = CampaignMatrix(
            campaign_id=options.campaign_id,
            preparation_sha256=options.sha256,
            projection_set_id=inputs.manifest.projection_set_id,
            plans=tuple(plans),
            bindings=tuple(bindings),
        )
        _validate_matrix(matrix, options, inputs)
        return matrix
    except (CampaignInputError, ValueError, KeyError) as exc:
        raise CampaignMatrixError(f"could not build campaign matrix: {exc}") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _build_family_id(
    options: CampaignPreparationOptions,
    inputs: CampaignInputs,
    item: CampaignInputCase,
    condition: CampaignConditionSpec,
    case_sha256: str,
) -> str:
    """Identify a case/condition independently of twin and replicate numbering.

    Keep IDs compact for Windows paths and backend names. Truncated 128-bit
    identities are checked for collisions when validating the completed matrix.
    Human-readable labels remain in the run/binding fields, never model prompts.
    """
    identity = {
        "version": MATRIX_VERSION,
        "campaign_id": options.campaign_id,
        "projection_set_id": inputs.manifest.projection_set_id,
        "allocation_case_id": item.case.allocation_case_id,
        "case_sha256": case_sha256,
        "condition_id": condition.condition_id,
        "condition_sha256": condition.sha256,
        "model_profile_id": condition.model_profile_id,
        "seed": options.seed,
        "purpose": options.purpose,
        "quiescence": options.quiescence,
        "cleanup_storage": options.cleanup_storage,
    }
    return f"family-m1-{_sha256(identity)[:32]}"


def _build_run_identity(family_id: str, replicate: int, twin: TrialTwin) -> str:
    return (
        "run-m1-"
        + _sha256(
            {
                "family_id": family_id,
                "replicate": replicate,
                "twin": twin,
            }
        )[:32]
    )


def _build_scope(
    options: CampaignPreparationOptions,
    condition: CampaignConditionSpec,
    item: CampaignInputCase,
    run_id: str,
) -> MemoryScope:
    return MemoryScope(
        experiment_id=options.campaign_id,
        run_id=run_id,
        user_id=item.case.persona_id,
        namespace=(
            "membench",
            "campaign",
            condition.mechanism.value,
            run_id,
            item.case.persona_id,
        ),
    )


def _instantiate_manifest(
    options: CampaignPreparationOptions,
    condition: CampaignConditionSpec,
    *,
    attack_id: str,
    run_id: str,
    replicate: int,
) -> ExperimentManifest:
    """Replace only matrix-owned template slots and validate the copied result."""
    payload = condition.manifest_template.model_dump(mode="json")
    payload["attack"]["attack_id"] = attack_id
    payload["isolation"]["run_id"] = run_id
    payload["replication"] = {
        "seed": options.seed + replicate - 1,
        "response_replicates": options.response_replicates,
    }
    return ExperimentManifest.model_validate(payload)


def _build_trial_plan(
    options: CampaignPreparationOptions,
    item: CampaignInputCase,
    condition: CampaignConditionSpec,
    family_id: str,
    replicate: int,
    twin: TrialTwin,
    *,
    behavioral_task: BehavioralTaskTemplate | None,
) -> TrialPlan:
    run_id = _build_run_identity(family_id, replicate, twin)
    scope = _build_scope(options, condition, item, run_id)
    manifest = _instantiate_manifest(
        options,
        condition,
        attack_id=item.case.attack_id,
        run_id=run_id,
        replicate=replicate,
    )
    run = CampaignRunSpec(
        campaign_id=options.campaign_id,
        trial_family_id=family_id,
        run_id=run_id,
        twin=twin,
        persona_id=item.case.persona_id,
        history_unit_id=item.case.candidate_id,
        response_replicate=replicate,
        manifest=manifest,
        scope=scope,
        agent_task=(
            build_probe_task(item.probe, scope, task_id=f"task-{run_id}")
            if behavioral_task is None
            else behavioral_task.instantiate(scope, task_id=f"task-{run_id}")
        ),
        agent_configuration=condition.agent_configuration,
    )
    plan = TrialPlan(
        run=run,
        projection=item.trial_for(twin),
        probe=item.probe,
        replay=condition.replay,
        quiescence=options.quiescence,
        purpose=options.purpose,
        attack_review_status=item.attack_spec.review_status,
        cleanup_storage=options.cleanup_storage,
    )
    # Nested dicts in otherwise frozen models must not be shared across runs.
    return TrialPlan.model_validate(plan.model_dump(mode="json"))


def _validate_matrix(
    matrix: CampaignMatrix, options: CampaignPreparationOptions, inputs: CampaignInputs
) -> None:
    """Require exact Cartesian coverage, including replicate slots and labels."""
    expected = {
        (item.case.allocation_case_id, condition.condition_id, replicate, twin)
        for item in inputs.cases
        for condition in options.conditions
        for replicate in range(1, options.response_replicates + 1)
        for twin in TrialTwin
    }
    observed = [
        (
            binding.allocation_case_id,
            binding.condition_id,
            plan.run.response_replicate,
            plan.run.twin,
        )
        for plan, binding in zip(matrix.plans, matrix.bindings, strict=True)
    ]
    if len(observed) != len(expected) or set(observed) != expected:
        raise CampaignMatrixError(
            "matrix does not cover the requested combinations exactly"
        )
