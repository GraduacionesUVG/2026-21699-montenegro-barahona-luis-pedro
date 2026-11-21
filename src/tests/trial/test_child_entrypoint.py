"""Child protocol, frozen-input validation, and offline execution tests."""

from __future__ import annotations

import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from membench.agent.__main__ import (
    ChildEntrypointError,
    _resolve_trusted_baseline,
    execute_child_request,
    resolve_child_request,
)
from membench.agent.context import AgentContextAssembler
from membench.campaign.artifacts import CampaignArtifactError, CampaignArtifactStore
from membench.campaign.models import ArtifactKind, ChildRunRequest, ChildRunResult
from membench.campaign.preparation import prepare_campaign
from membench.campaign.preparation_models import CampaignPreparationOptions
from membench.evaluation.models import EvaluationProbe
from membench.ingress.models import TrustedBaselineManifest, TrustedBaselineSeed
from membench.tools.fixtures import (
    FixtureContact,
    FixtureOrigin,
    PersonaDerivedProvenance,
    ToolFixtureBundle,
)
from membench.tools.mock_contacts import MockContact
from membench.trial.artifacts import TrialArtifactStore
from membench.trial.runner import TrialRunner

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def entrypoint_campaign(input_fixture, tmp_path):
    payload = input_fixture[2].model_dump(mode="json")
    payload["environments"][0]["python_executable"] = sys.executable
    payload["artifact_root"] = "artifacts/entrypoint-pilot"
    options = CampaignPreparationOptions.model_validate(payload)
    prepared = prepare_campaign(
        options,
        project_root=tmp_path,
        source_environment={},
    )
    run = prepared.configuration.runs[0]
    launcher = prepared.create_launcher()
    request = launcher.build_child_request(run)
    child_directory = tmp_path / request.artifact_directory
    child_directory.mkdir(parents=True)
    return prepared, request, child_directory


@pytest.fixture
def tool_enabled_entrypoint_campaign(input_fixture, tmp_path):
    fixture = ToolFixtureBundle(
        fixture_id="entrypoint-tools-v1",
        persona_ids=("1", "2"),
        contacts=(
            FixtureContact(
                contact=MockContact(
                    contact_id="trusted-contact",
                    owner_user_id="1",
                    display_name="Trusted Synthetic Contact",
                    email="trusted@example.invalid",
                ),
                provenance=PersonaDerivedProvenance(
                    origin=FixtureOrigin.PERSONA_DERIVED,
                    source_export_sha256="a" * 64,
                    source_record_id="synthetic-record-001",
                ),
            ),
        ),
    )
    fixture_path = tmp_path / "fixtures" / "entrypoint-tools.json"
    fixture_path.parent.mkdir()
    fixture_path.write_text(fixture.model_dump_json(indent=2), encoding="utf-8")

    payload = input_fixture[2].model_dump(mode="json")
    payload["environments"][0]["python_executable"] = sys.executable
    payload["artifact_root"] = "artifacts/entrypoint-tools"
    payload["tool_fixture_path"] = "fixtures/entrypoint-tools.json"
    payload["conditions"][0]["agent_configuration"]["allowed_tools"] = [
        "lookup_contact"
    ]
    options = CampaignPreparationOptions.model_validate(payload)
    prepared = prepare_campaign(
        options,
        project_root=tmp_path,
        source_environment={},
    )
    request = prepared.create_launcher().build_child_request(
        prepared.configuration.runs[0]
    )
    child_directory = tmp_path / request.artifact_directory
    child_directory.mkdir(parents=True)
    return prepared, request, child_directory, fixture


@pytest.fixture
def baseline_entrypoint_campaign(input_fixture, tmp_path):
    content = "The trusted portal is https://trusted.example.test/records."
    baseline = TrustedBaselineSeed(
        baseline_id="baseline-persona-1-v1",
        persona_id=1,
        content=content,
        retrieval_query="Which portal is trusted?",
        required_markers=("https://trusted.example.test/records",),
        available_after_source_message_index=1,
        content_sha256=sha256(content.encode()).hexdigest(),
    )
    source = tmp_path / "inputs" / "trusted-baselines.json"
    source.parent.mkdir()
    source.write_text(
        TrustedBaselineManifest(
            baseline_set_id="entrypoint-baselines-v1", baselines=(baseline,)
        ).model_dump_json(),
        encoding="utf-8",
    )
    payload = input_fixture[2].model_dump(mode="json")
    payload["environments"][0]["python_executable"] = sys.executable
    payload["case_ids"] = ["alloc-synthetic-001"]
    payload["trusted_baseline_path"] = "inputs/trusted-baselines.json"
    payload["artifact_root"] = "artifacts/entrypoint-baseline"
    prepared = prepare_campaign(
        CampaignPreparationOptions.model_validate(payload),
        project_root=tmp_path,
        source_environment={},
    )
    request = prepared.create_launcher().build_child_request(
        prepared.configuration.runs[0]
    )
    (tmp_path / request.artifact_directory).mkdir(parents=True)
    return prepared, request, baseline


def test_resolves_exact_frozen_binding_without_runtime_initialization(
    entrypoint_campaign, tmp_path
):
    prepared, request, child_directory = entrypoint_campaign
    resolved = resolve_child_request(request, project_root=tmp_path)
    assert resolved.request == request
    assert resolved.campaign_root == prepared.artifacts.root
    assert resolved.trial_directory == child_directory / "trial"
    assert resolved.plan.run == request.run
    assert resolved.binding.run_id == request.run.run_id
    assert resolved.binding.trial_plan_sha256 == resolved.plan.sha256
    assert resolved.options.campaign_id == request.run.campaign_id
    assert not resolved.trial_directory.exists()


def test_resolves_the_exact_frozen_evaluation_probe(
    input_fixture, tmp_path
):
    probe = EvaluationProbe(
        probe_id="probe-entrypoint-evaluation-001",
        attack_spec_id="external-false-fact-001",
        delivery_family_id="external-delivery-001",
        persona_id="1",
        query="Which marker should be evaluated?",
        settings={
            "attack_family": "false_fact_insertion",
            "poison_markers": ["ENTRYPOINT-POISON-MARKER"],
            "trusted_ground_truth": "trusted value",
        },
    )
    probe_path = tmp_path / "inputs" / "evaluation-probe.json"
    probe_path.parent.mkdir()
    probe_path.write_text(probe.model_dump_json(), encoding="utf-8")
    payload = input_fixture[2].model_dump(mode="json")
    payload["environments"][0]["python_executable"] = sys.executable
    payload["case_ids"] = ["alloc-synthetic-001"]
    payload["evaluation_probe_path"] = "inputs/evaluation-probe.json"
    payload["artifact_root"] = "artifacts/entrypoint-evaluation"
    prepared = prepare_campaign(
        CampaignPreparationOptions.model_validate(payload),
        project_root=tmp_path,
        source_environment={},
    )
    request = prepared.create_launcher().build_child_request(
        prepared.configuration.runs[0]
    )
    (tmp_path / request.artifact_directory).mkdir(parents=True)

    resolved = resolve_child_request(request, project_root=tmp_path)

    assert resolved.evaluation_probe == probe


def test_resolves_exact_frozen_trusted_baseline(
    baseline_entrypoint_campaign, tmp_path
):
    _, request, baseline = baseline_entrypoint_campaign

    resolved = resolve_child_request(request, project_root=tmp_path)

    assert resolved.trusted_baseline == baseline


def test_passes_resolved_trusted_baseline_to_trial_runner(
    baseline_entrypoint_campaign, tmp_path, lifecycle_adapter, decider
):
    _, request, baseline = baseline_entrypoint_campaign
    observed = []

    class OfflinePreparedTrial:
        def __init__(self, plan):
            self.plan = plan

        def create_runner(self, artifacts, *, tools, trusted_baseline):
            observed.append(trusted_baseline)
            return TrialRunner(
                self.plan,
                lifecycle_adapter,
                decider,
                AgentContextAssembler(),
                artifacts,
                tools=tools,
                trusted_baseline=trusted_baseline,
            )

    result = execute_child_request(
        request,
        project_root=tmp_path,
        bootstrap=OfflinePreparedTrial,
    )

    assert result.status == "succeeded"
    assert observed == [baseline]


def test_rejects_mismatched_trusted_baseline_binding(
    baseline_entrypoint_campaign, tmp_path
):
    prepared, request, _ = baseline_entrypoint_campaign
    resolved = resolve_child_request(request, project_root=tmp_path)
    mismatched = resolved.binding.model_copy(
        update={"trusted_baseline_id": "baseline-persona-1-other"}
    )

    with pytest.raises(ChildEntrypointError, match="bound run identity"):
        _resolve_trusted_baseline(
            request,
            store=CampaignArtifactStore.open_existing(prepared.artifacts.root),
            bindings=resolved.bindings,
            binding=mismatched,
            plan=resolved.plan,
        )


def test_rejects_tampered_frozen_trusted_baseline(
    baseline_entrypoint_campaign, tmp_path
):
    prepared, request, _ = baseline_entrypoint_campaign
    reference = prepared.bindings.trusted_baseline_manifest
    assert reference is not None
    (prepared.artifacts.root / reference.relative_path).write_bytes(b"tampered")

    with pytest.raises(Exception, match="mismatch"):
        resolve_child_request(request, project_root=tmp_path)


def test_resolves_fixture_backed_tools_and_wires_the_frozen_registry(
    tool_enabled_entrypoint_campaign, tmp_path, lifecycle_adapter, decider
):
    prepared_campaign, request, _, fixture = tool_enabled_entrypoint_campaign
    resolved = resolve_child_request(request, project_root=tmp_path)
    assert resolved.tool_fixture == fixture
    assert any(
        reference.kind == ArtifactKind.TOOL_FIXTURE
        for reference in request.frozen_inputs
    )

    observed_tool_names = []

    class OfflinePreparedTrial:
        def __init__(self, plan):
            self.plan = plan

        def create_runner(self, artifacts, *, tools):
            observed_tool_names.extend(item.name for item in tools.definitions())
            return TrialRunner(
                self.plan,
                lifecycle_adapter,
                decider,
                AgentContextAssembler(),
                artifacts,
                tools=tools,
            )

    result = execute_child_request(
        request,
        project_root=tmp_path,
        bootstrap=OfflinePreparedTrial,
    )
    assert result.status == "succeeded"
    assert observed_tool_names == ["lookup_contact"]
    assert all(
        (prepared_campaign.artifacts.root / reference.relative_path).is_file()
        for reference in result.artifacts
    )


def test_executes_offline_runner_and_returns_rebased_sealed_artifacts(
    entrypoint_campaign, tmp_path, lifecycle_adapter, decider
):
    prepared_campaign, request, _ = entrypoint_campaign

    class OfflinePreparedTrial:
        def __init__(self, plan):
            self.plan = plan

        def create_runner(self, artifacts, *, tools):
            return TrialRunner(
                self.plan,
                lifecycle_adapter,
                decider,
                AgentContextAssembler(),
                artifacts,
                tools=tools,
            )

    result = execute_child_request(
        request,
        project_root=tmp_path,
        bootstrap=OfflinePreparedTrial,
    )
    assert result.status == "succeeded"
    assert result.request_id == request.request_id
    assert result.run == request.run
    assert result.agent_result is not None
    assert result.provenance is not None
    assert result.artifacts
    assert all(
        ref.relative_path.startswith(f"children/{request.run.run_id}/trial/")
        for ref in result.artifacts
    )
    assert all(
        (prepared_campaign.artifacts.root / ref.relative_path).is_file()
        for ref in result.artifacts
    )


def test_artifact_finalization_failure_returns_registered_partial_evidence(
    entrypoint_campaign, tmp_path, lifecycle_adapter, decider, monkeypatch
):
    prepared_campaign, request, _ = entrypoint_campaign

    class OfflinePreparedTrial:
        def __init__(self, plan):
            self.plan = plan

        def create_runner(self, artifacts, *, tools):
            return TrialRunner(
                self.plan,
                lifecycle_adapter,
                decider,
                AgentContextAssembler(),
                artifacts,
                tools=tools,
            )

    def fail_finish(artifacts, result):
        artifacts.close_logs()
        raise CampaignArtifactError("simulated finalization failure")

    monkeypatch.setattr(TrialArtifactStore, "finish", fail_finish)
    result = execute_child_request(
        request,
        project_root=tmp_path,
        bootstrap=OfflinePreparedTrial,
    )
    assert result.status == "failed"
    assert result.failure_class == "artifact_failure"
    assert result.artifacts
    assert all(
        (prepared_campaign.artifacts.root / reference.relative_path).is_file()
        for reference in result.artifacts
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("cleanup", "cleanup policy"),
        ("run", "no projection binding"),
        ("directory", "children/<run_id>"),
        ("missing_options", "preparation:options"),
        ("missing_bindings", "preparation:run-bindings"),
        ("missing_manifest", "projection manifest"),
    ],
)
def test_rejects_request_drift(entrypoint_campaign, tmp_path, mutation, match):
    _, request, child_directory = entrypoint_campaign
    data = request.model_dump(mode="json")
    if mutation == "cleanup":
        data["cleanup_storage"] = not request.cleanup_storage
    elif mutation == "run":
        data["run"]["run_id"] = "unbound-run"
        data["run"]["manifest"]["isolation"]["run_id"] = "unbound-run"
        data["run"]["scope"]["run_id"] = "unbound-run"
        data["run"]["scope"]["namespace"] = ["membench", "unbound-run"]
        data["run"]["agent_task"]["scope"] = data["run"]["scope"]
        data["artifact_directory"] = (
            data["artifact_directory"].rsplit("/", 1)[0] + "/unbound-run"
        )
        child_directory.rename(child_directory.parent / "unbound-run")
    elif mutation == "directory":
        data["artifact_directory"] = data["artifact_directory"].replace(
            "/children/", "/other/"
        )
    else:
        artifact_id = {
            "missing_options": "preparation:options",
            "missing_bindings": "preparation:run-bindings",
        }.get(mutation)
        if artifact_id is not None:
            data["frozen_inputs"] = [
                ref
                for ref in data["frozen_inputs"]
                if ref["artifact_id"] != artifact_id
            ]
        else:
            manifest = next(
                ref
                for ref in data["frozen_inputs"]
                if ref["relative_path"] == "inputs/projections/manifest.json"
            )
            data["frozen_inputs"].remove(manifest)
    changed = ChildRunRequest.model_validate(data)
    with pytest.raises((ChildEntrypointError, KeyError), match=match):
        resolve_child_request(changed, project_root=tmp_path)


@pytest.mark.parametrize(
    "filename",
    [
        "preparation/options.json",
        "preparation/run_bindings.json",
        "inputs/projections/manifest.json",
        "inputs/projections/probes.jsonl",
    ],
)
def test_rejects_tampered_frozen_inputs(entrypoint_campaign, tmp_path, filename):
    prepared, request, _ = entrypoint_campaign
    (prepared.artifacts.root / filename).write_bytes(b"tampered")
    with pytest.raises(Exception, match="mismatch"):
        resolve_child_request(request, project_root=tmp_path)


def test_valid_preexecution_failure_is_structured_and_does_not_create_trial(
    entrypoint_campaign, tmp_path
):
    _, request, child_directory = entrypoint_campaign
    changed = request.model_copy(
        update={"cleanup_storage": not request.cleanup_storage}
    )
    result = execute_child_request(changed, project_root=tmp_path)
    assert result.status == "failed"
    assert result.failure_class == "configuration_invalid"
    assert result.failure_type == "ChildEntrypointError"
    assert result.failure_message == "child failed during resolve_request"
    assert not result.artifacts
    assert not (child_directory / "trial").exists()


def test_tool_enabled_request_requires_frozen_fixture(entrypoint_campaign, tmp_path):
    _, request, child_directory = entrypoint_campaign
    configuration = request.run.agent_configuration.model_copy(
        update={"allowed_tools": ("lookup_contact",)}
    )
    task = request.run.agent_task
    run = request.run.model_copy(
        update={"agent_configuration": configuration, "agent_task": task}
    )
    with pytest.raises(ValidationError, match="missing tool_fixture"):
        ChildRunRequest.model_validate(
            {**request.model_dump(mode="json"), "run": run.model_dump(mode="json")}
        )
    assert not (child_directory / "trial").exists()


def _run_module(payload: str, cwd: Path, *arguments: str):
    environment = dict(os.environ)
    source = str(_ROOT / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    return subprocess.run(
        [sys.executable, "-B", "-m", "membench.agent", *arguments],
        input=payload,
        text=True,
        capture_output=True,
        cwd=cwd,
        env=environment,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("payload", ["", "not json", "{}", "[]"])
def test_subprocess_rejects_malformed_stdin_without_stdout(payload, tmp_path):
    completed = _run_module(payload, tmp_path)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("membench child: invalid child request")


def test_subprocess_rejects_arguments_without_reading_request(tmp_path):
    completed = _run_module("", tmp_path, "--unexpected")
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "arguments are not supported" in completed.stderr


def test_subprocess_valid_failure_emits_one_json_and_zero_exit(
    entrypoint_campaign, tmp_path
):
    _, request, child_directory = entrypoint_campaign
    changed = request.model_copy(
        update={"cleanup_storage": not request.cleanup_storage}
    )
    completed = _run_module(changed.model_dump_json(), tmp_path)
    assert completed.returncode == 0
    lines = completed.stdout.splitlines()
    assert len(lines) == 1
    result = ChildRunResult.model_validate_json(lines[0])
    assert result.status == "failed"
    assert result.request_id == changed.request_id
    assert result.failure_class == "configuration_invalid"
    assert not (child_directory / "trial").exists()


def test_subprocess_detects_frozen_input_tampering_with_clean_stdout(
    entrypoint_campaign, tmp_path
):
    prepared, request, child_directory = entrypoint_campaign
    (prepared.artifacts.root / "preparation/options.json").write_bytes(b"secret detail")
    completed = _run_module(request.model_dump_json(), tmp_path)
    assert completed.returncode == 0
    assert len(completed.stdout.splitlines()) == 1
    assert "secret detail" not in completed.stdout
    result = ChildRunResult.model_validate_json(completed.stdout)
    assert result.status == "failed"
    assert result.failure_class == "configuration_invalid"
    assert not (child_directory / "trial").exists()


def test_request_size_limit_is_enforced(tmp_path):
    completed = _run_module(" " * (16 * 1024 * 1024 + 1), tmp_path)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "invalid child request" in completed.stderr
