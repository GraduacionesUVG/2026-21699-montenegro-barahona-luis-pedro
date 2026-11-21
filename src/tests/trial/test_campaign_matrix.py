"""Pure matrix planning and its handoff to existing campaign/trial contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from membench.campaign.artifacts import CampaignArtifactStore, canonical_json_bytes
from membench.campaign.bindings import CampaignRunBindings
from membench.campaign.inputs import freeze_campaign_inputs, load_campaign_inputs
from membench.campaign.matrix import (
    CampaignMatrix,
    CampaignMatrixError,
    build_campaign_matrix,
)
from membench.campaign.models import CampaignConfiguration, TrialTwin
from membench.campaign.preparation_models import CampaignPreparationOptions
from membench.contract import WritePolicy, load_manifest
from membench.trial.bootstrap import build_probe_task, load_trial_plan

_ROOT = Path(__file__).resolve().parents[2]


def _options(base, **updates):
    return CampaignPreparationOptions.model_validate(
        {**base.model_dump(mode="json"), **updates}
    )


@pytest.fixture
def matrix_options(input_fixture):
    """Five supported conditions, two personas, two full-replay replicates."""
    base = input_fixture[2]
    conditions, environments = [], []
    for mechanism, filename, policies in (
        ("telemem", "example", ("direct", "native_selective")),
        ("langmem", "langmem_semantic.example", ("direct", "native_selective")),
        ("dakera", "dakera.example", ("direct",)),
    ):
        template = load_manifest(
            _ROOT / f"configs/experiments/blackbox/{filename}.yaml"
        )
        for policy in policies:
            data = base.conditions[0].model_dump(mode="json")
            data["condition_id"] = f"{mechanism}-{policy}"
            data["model_profile_id"] = (
                "qwen-agent-only" if mechanism == "dakera" else "qwen-nomic"
            )
            data["manifest_template"] = template.model_dump(mode="json")
            data["manifest_template"]["condition"]["write_policy"] = policy
            if mechanism == "telemem":
                data["manifest_template"]["condition"]["mechanism_options"]["infer"] = (
                    policy == "native_selective"
                )
            for model in data["manifest_template"]["models"].values():
                if model is not None:
                    model["digest"] = "a" * 64
            conditions.append(data)
        environments.append(
            {"mechanism": mechanism, "python_executable": f"unused-{mechanism}-python"}
        )
    return _options(
        base, conditions=conditions, environments=environments, response_replicates=2
    )


def _build(options, root):
    return build_campaign_matrix(
        options, load_campaign_inputs(options, project_root=root)
    )


def test_expands_exact_conditions_cases_replicates_and_twins(matrix_options, tmp_path):
    matrix = _build(matrix_options, tmp_path)
    assert matrix.run_count == 2 * 5 * 2 * 3 == 60
    expected = {
        (case_id, condition.condition_id, replicate, twin)
        for case_id in ("alloc-synthetic-001", "alloc-synthetic-002")
        for condition in matrix_options.conditions
        for replicate in (1, 2)
        for twin in TrialTwin
    }
    observed = {
        (
            binding.allocation_case_id,
            binding.condition_id,
            plan.run.response_replicate,
            plan.run.twin,
        )
        for plan, binding in zip(matrix.plans, matrix.bindings, strict=True)
    }
    assert observed == expected
    assert len({run.run_id for run in matrix.runs}) == matrix.run_count
    assert len({run.scope.namespace for run in matrix.runs}) == matrix.run_count
    assert len({run.agent_task.task_id for run in matrix.runs}) == matrix.run_count
    assert len({run.trial_family_id for run in matrix.runs}) == 10
    assert {
        run.policy.value for run in matrix.runs if run.mechanism.value == "dakera"
    } == {"direct"}
    assert matrix.preparation_sha256 == matrix_options.sha256


def test_replicates_have_paired_responder_seeds_and_unchanged_writer_seeds(
    matrix_options, tmp_path
):
    matrix = _build(matrix_options, tmp_path)
    for plan in matrix.plans:
        run = plan.run
        assert (
            run.manifest.replication.seed
            == matrix_options.seed + run.response_replicate - 1
        )
        assert run.manifest.replication.response_replicates == 2
        assert run.manifest.isolation.run_id == run.run_id
        assert run.run_id in run.scope.namespace
        assert run.scope.user_id == run.persona_id
        assert run.scope.experiment_id == matrix_options.campaign_id
        if run.mechanism.value == "langmem":
            assert run.manifest.condition.mechanism_options.seed == 42


def test_preserves_exact_replay_probes_and_non_slot_settings(matrix_options, tmp_path):
    inputs = load_campaign_inputs(matrix_options, project_root=tmp_path)
    options_before = matrix_options.model_dump_json()
    cases_before = tuple(item.model_dump_json() for item in inputs.cases)
    matrix = build_campaign_matrix(matrix_options, inputs)
    cases = {item.case.allocation_case_id: item for item in inputs.cases}
    conditions = {item.condition_id: item for item in matrix_options.conditions}
    for plan, binding in zip(matrix.plans, matrix.bindings, strict=True):
        case, condition = (
            cases[binding.allocation_case_id],
            conditions[binding.condition_id],
        )
        assert plan.projection == case.trial_for(plan.run.twin)
        assert plan.probe == case.probe
        assert plan.run.agent_task == build_probe_task(
            case.probe, plan.run.scope, task_id=plan.run.agent_task.task_id
        )
        assert not plan.run.agent_task.metadata
        assert plan.run.history_unit_id == case.case.candidate_id
        assert plan.run.attack_id == case.case.attack_id
        assert plan.replay == condition.replay
        assert plan.quiescence == matrix_options.quiescence
        assert plan.run.agent_configuration == condition.agent_configuration
        assert plan.run.manifest.models == condition.manifest_template.models
        expected = condition.manifest_template.model_dump(mode="json")
        actual = plan.run.manifest.model_dump(mode="json")
        for payload in (expected, actual):
            del payload["isolation"]["run_id"]
            del payload["attack"]["attack_id"]
            del payload["replication"]
        assert expected == actual
        binding.validate_trial_plan(plan)
        assert "original_option_indexes" not in plan.run.agent_task.instruction
        assert "EVALUATOR_ONLY_CANONICAL_PREFERENCE" not in plan.model_dump_json()
    assert matrix_options.model_dump_json() == options_before
    assert tuple(item.model_dump_json() for item in inputs.cases) == cases_before


def test_matrix_is_repeatable_and_roundtrips(matrix_options, tmp_path):
    first = _build(matrix_options, tmp_path)
    second = _build(matrix_options, tmp_path)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.sha256 == second.sha256
    assert CampaignMatrix.model_validate_json(first.model_dump_json()) == first


def test_condition_and_selection_order_do_not_rename_or_reorder_runs(
    matrix_options, tmp_path
):
    first = _build(matrix_options, tmp_path)
    changed = _options(
        matrix_options,
        conditions=[
            c.model_dump(mode="json") for c in reversed(matrix_options.conditions)
        ],
        case_ids=["alloc-synthetic-002", "alloc-synthetic-001"],
    )
    second = _build(changed, tmp_path)
    assert first.plans == second.plans
    assert first.bindings == second.bindings
    assert first.preparation_sha256 != second.preparation_sha256


def test_subset_retains_matching_run_ids(matrix_options, tmp_path):
    full = _build(matrix_options, tmp_path)
    subset_options = _options(matrix_options, case_ids=["alloc-synthetic-002"])
    subset = _build(subset_options, tmp_path)
    assert subset.plans == tuple(
        plan
        for plan in full.plans
        if plan.projection.allocation_case_id == "alloc-synthetic-002"
    )


@pytest.mark.parametrize(
    "change", ["campaign", "seed", "model", "batching", "quiescence", "cleanup"]
)
def test_execution_changes_get_different_run_ids(input_fixture, tmp_path, change):
    options = input_fixture[2]
    before = _build(options, tmp_path)
    data = options.model_dump(mode="json")
    if change == "campaign":
        data["campaign_id"] = "another-campaign"
    elif change == "seed":
        data["seed"] += 1
    elif change == "model":
        data["conditions"][0]["manifest_template"]["models"]["writer"]["model"] = (
            "mistral:7b"
        )
        data["conditions"][0]["manifest_template"]["models"]["writer"]["digest"] = (
            "b" * 64
        )
    elif change == "batching":
        data["conditions"][0]["replay"]["batching"] = "event_v1"
    elif change == "quiescence":
        data["quiescence"]["timeout_seconds"] += 1
    else:
        data["cleanup_storage"] = False
    after = _build(CampaignPreparationOptions.model_validate(data), tmp_path)
    assert not {run.run_id for run in before.runs} & {run.run_id for run in after.runs}


def test_writer_profile_switch_is_explicit_and_does_not_change_other_roles(
    input_fixture, tmp_path
):
    base = input_fixture[2]
    new_condition = base.conditions[0].model_dump(mode="json")
    new_condition["condition_id"] = "telemem-native-mistral"
    new_condition["model_profile_id"] = "mistral-writer-qwen-agent"
    new_condition["manifest_template"]["models"]["writer"].update(
        model="mistral:7b", digest="b" * 64
    )
    options = _options(
        base, conditions=[base.conditions[0].model_dump(mode="json"), new_condition]
    )
    matrix = _build(options, tmp_path)
    assert matrix.run_count == 12
    assert len({binding.model_profile_sha256 for binding in matrix.bindings}) == 2
    for run in matrix.runs:
        assert (
            run.manifest.models.embedding
            == base.conditions[0].manifest_template.models.embedding
        )
        assert (
            run.manifest.models.responder
            == base.conditions[0].manifest_template.models.responder
        )


@pytest.mark.parametrize(
    "change",
    ["campaign", "hash", "purpose", "missing_case", "missing_twin", "manifest"],
)
def test_matrix_rejects_stale_or_modified_inputs(input_fixture, tmp_path, change):
    options = input_fixture[2]
    inputs = load_campaign_inputs(options, project_root=tmp_path)
    if change == "campaign":
        inputs = replace(inputs, campaign_id="other")
    elif change == "hash":
        inputs = replace(inputs, preparation_sha256="f" * 64)
    elif change == "purpose":
        inputs = replace(inputs, purpose="experiment")
    elif change == "missing_case":
        inputs = replace(inputs, cases=inputs.cases[:1])
    elif change == "missing_twin":
        case = inputs.cases[0].model_copy(update={"trials": inputs.cases[0].trials[:2]})
        inputs = replace(inputs, cases=(case, inputs.cases[1]))
    else:
        inputs.manifest.artifact_sha256["probes.jsonl"] = "e" * 64
    with pytest.raises(CampaignMatrixError):
        build_campaign_matrix(options, inputs)


def test_matrix_rejects_changed_options_and_excess_parallelism(input_fixture, tmp_path):
    options = input_fixture[2]
    inputs = load_campaign_inputs(options, project_root=tmp_path)
    changed = _options(options, seed=123)
    with pytest.raises(CampaignMatrixError, match="preparation options"):
        build_campaign_matrix(changed, inputs)
    oversized = _options(options, max_parallel_children=7)
    with pytest.raises(CampaignMatrixError, match="exceeds the matrix run count"):
        _build(oversized, tmp_path)


def test_nested_native_options_are_not_shared(input_fixture, tmp_path):
    base = input_fixture[2]
    condition = base.conditions[0].model_dump(mode="json")
    condition["replay"]["native_options"] = {"nested": {"names": ["original"]}}
    options = _options(base, conditions=[condition])
    matrix = _build(options, tmp_path)
    matrix.plans[0].replay.native_options["nested"]["names"].append("changed")
    assert matrix.plans[1].replay.native_options["nested"]["names"] == ["original"]
    assert matrix.bindings[0].replay.native_options["nested"]["names"] == ["original"]
    assert options.conditions[0].replay.native_options["nested"]["names"] == [
        "original"
    ]
    with pytest.raises(ValidationError, match="frozen run binding"):
        CampaignMatrix.model_validate_json(matrix.model_dump_json())


def test_reviewed_subset_retains_experiment_and_cleanup_choices(
    input_fixture, tmp_path
):
    options = _options(
        input_fixture[2],
        case_ids=["alloc-synthetic-002"],
        purpose="experiment",
        cleanup_storage=False,
    )
    matrix = _build(options, tmp_path)
    assert matrix.run_count == 3
    for plan, binding in zip(matrix.plans, matrix.bindings, strict=True):
        assert plan.purpose == binding.purpose == "experiment"
        assert plan.attack_review_status == binding.attack_review_status == "approved"
        assert not plan.cleanup_storage
        assert not binding.cleanup_storage


def test_unvalidated_unsupported_condition_is_rejected(input_fixture, tmp_path):
    options = input_fixture[2]
    inputs = load_campaign_inputs(options, project_root=tmp_path)
    condition = options.conditions[0]
    invalid = condition.model_copy(
        update={
            "manifest_template": condition.manifest_template.model_copy(
                update={
                    "condition": condition.manifest_template.condition.model_copy(
                        update={"write_policy": WritePolicy.SHARED_GUARDED}
                    )
                }
            )
        }
    )
    with pytest.raises(CampaignMatrixError):
        build_campaign_matrix(
            options.model_copy(update={"conditions": (invalid,)}), inputs
        )


def test_matrix_calls_no_files_environment_or_subprocesses(
    input_fixture, tmp_path, monkeypatch
):
    options = input_fixture[2]
    inputs = load_campaign_inputs(options, project_root=tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("planning attempted external I/O")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "mkdir", forbidden)
        patch.setattr(os, "getenv", forbidden)
        patch.setattr(subprocess, "run", forbidden)
        assert build_campaign_matrix(options, inputs).run_count == 6


def test_id_collisions_fail_instead_of_reusing_scopes(
    input_fixture, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "membench.campaign.matrix._build_run_identity", lambda *args: "colliding-id"
    )
    with pytest.raises(CampaignMatrixError, match="run IDs must be unique"):
        _build(input_fixture[2], tmp_path)


def test_matrix_output_validates_with_campaign_bindings_and_child_plan_loader(
    matrix_options, tmp_path
):
    inputs = load_campaign_inputs(matrix_options, project_root=tmp_path)
    matrix = build_campaign_matrix(matrix_options, inputs)
    store = CampaignArtifactStore.prepare(tmp_path / matrix_options.artifact_root)
    frozen = freeze_campaign_inputs(inputs, store)
    configuration = CampaignConfiguration(
        campaign_id=matrix_options.campaign_id,
        seed=matrix_options.seed,
        runs=matrix.runs,
        environments=matrix_options.environments,
        frozen_inputs=frozen.artifacts,
        artifact_root=matrix_options.artifact_root,
    )
    inventory = CampaignRunBindings(
        campaign_id=matrix.campaign_id,
        preparation_sha256=matrix.preparation_sha256,
        projection_set_id=matrix.projection_set_id,
        projection_manifest=frozen.projection_manifest,
        bindings=matrix.bindings,
    )
    inventory.validate_campaign(configuration)
    inventory.validate_preparation(matrix_options)
    for plan in matrix.plans[:3]:
        loaded = load_trial_plan(
            plan.run,
            projection_directory=store.verify(frozen.projection_manifest).parent,
            trial_id=plan.projection.trial_id,
            replay=plan.replay,
            quiescence=plan.quiescence,
            purpose=plan.purpose,
            cleanup_storage=plan.cleanup_storage,
        )
        assert loaded == plan
        inventory.binding_for(plan.run.run_id).validate_trial_plan(loaded)


@pytest.mark.parametrize(
    "change", ["missing_twin", "duplicate_run", "missing_binding", "wrong_binding"]
)
def test_matrix_model_rejects_incomplete_or_misaligned_results(
    input_fixture, tmp_path, change
):
    matrix = _build(input_fixture[2], tmp_path)
    data = matrix.model_dump(mode="json")
    if change == "missing_twin":
        data["plans"].pop()
        data["bindings"].pop()
    elif change == "duplicate_run":
        data["plans"][1] = data["plans"][0]
    elif change == "missing_binding":
        data["bindings"].pop()
    else:
        data["bindings"].reverse()
    with pytest.raises(ValidationError):
        CampaignMatrix.model_validate(data)


def test_cold_import_and_process_hash_seed_independence(input_fixture, tmp_path):
    options = input_fixture[2]
    code = """
import sys
from pathlib import Path
from membench.campaign.matrix import build_campaign_matrix
from membench.campaign.inputs import load_campaign_inputs
from membench.campaign.preparation_models import CampaignPreparationOptions
options = CampaignPreparationOptions.model_validate_json(sys.stdin.read())
inputs = load_campaign_inputs(options, project_root=Path(sys.argv[1]))
print(build_campaign_matrix(options, inputs).sha256)
vendors = {'telemem', 'langmem', 'dakera', 'transformers', 'faiss', 'langgraph'}
assert not vendors & set(sys.modules)
"""
    results = [
        subprocess.run(
            [sys.executable, "-B", "-c", code, str(tmp_path)],
            input=options.model_dump_json(),
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=_ROOT,
            timeout=30,
        ).stdout.strip()
        for seed in ("1", "987")
    ]
    assert results == [_build(options, tmp_path).sha256] * 2
