"""CLI coverage for offline campaign preparation."""

import json
import sys

import pytest
from typer.testing import CliRunner

from membench.agent.models import (
    AgentRunResult,
    AgentRunStatus,
    AgentTerminationReason,
)
from membench.campaign.launcher import SubprocessChildExecutor
from membench.campaign.models import ChildRunResult, ChildRunStatus
from membench.campaign.preparation_models import CampaignPreparationOptions
from membench.cli import app
from membench.memory.models import AdapterProvenance, RetrievalResult

runner = CliRunner()


def _write_options(input_fixture, tmp_path, *, inherited_env_keys=()):
    payload = input_fixture[2].model_dump(mode="json")
    payload["artifact_root"] = "artifacts/campaigns/cli-pilot"
    payload["environments"][0].update(
        python_executable=sys.executable,
        inherited_env_keys=list(inherited_env_keys),
    )
    options = CampaignPreparationOptions.model_validate(payload)
    config = tmp_path / "campaign.json"
    config.write_text(options.model_dump_json(indent=2), encoding="utf-8")
    return config, options


def _prepare_command(config, tmp_path):
    return ["campaign", "prepare", str(config), "--project-root", str(tmp_path)]


def _run_command(options, tmp_path, *extra):
    return [
        "campaign",
        "run",
        options.artifact_root,
        "--project-root",
        str(tmp_path),
        *extra,
    ]


def _successful_child(request):
    run = request.run
    return ChildRunResult(
        request_id=request.request_id,
        run=run,
        status=ChildRunStatus.SUCCEEDED,
        agent_result=AgentRunResult(
            task=run.agent_task,
            configuration=run.agent_configuration,
            retrieval_result=RetrievalResult(
                operation_id="retrieval-test",
                query_sha256=run.agent_task.retrieval_query_sha256,
                request_top_k=run.agent_configuration.retrieval_top_k,
                threshold=run.agent_configuration.retrieval_threshold,
                latency_seconds=0,
            ),
            context_sha256="0" * 64,
            status=AgentRunStatus.COMPLETED,
            termination_reason=AgentTerminationReason.FINAL_ANSWER,
            final_answer="Synthetic offline response.",
        ),
        provenance=AdapterProvenance(
            mechanism=run.mechanism.value,
            mechanism_version="test",
            adapter_version="test",
            storage_backend="fake",
            namespace=run.scope.namespace,
        ),
    )


@pytest.fixture
def successful_children(monkeypatch):
    def execute(_executor, request, _environment):
        return _successful_child(request)

    monkeypatch.setattr(SubprocessChildExecutor, "execute", execute)


def test_campaign_prepare_creates_frozen_plan_without_running(input_fixture, tmp_path):
    config, options = _write_options(input_fixture, tmp_path)

    result = runner.invoke(
        app,
        ["campaign", "prepare", str(config), "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    root = tmp_path / options.artifact_root
    assert "Campaign prepared: pilot-inputs" in result.output
    assert "Concrete runs: 6" in result.output
    assert "No trials were executed." in result.output
    assert (root / "preparation/complete.json").is_file()
    assert not (root / "children").exists()


def test_campaign_prepare_json_output_is_machine_readable(input_fixture, tmp_path):
    config, options = _write_options(input_fixture, tmp_path)

    result = runner.invoke(
        app,
        [
            "campaign",
            "prepare",
            str(config),
            "--project-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["status"] == "prepared"
    assert summary["campaign_id"] == options.campaign_id
    assert summary["run_count"] == 6
    assert summary["preparation_sha256"] == options.sha256
    assert summary["artifact_root"] == str((tmp_path / options.artifact_root).resolve())


def test_campaign_prepare_loads_only_requested_env_file(input_fixture, tmp_path):
    config, options = _write_options(
        input_fixture, tmp_path, inherited_env_keys=("CLI_TEST_PROVIDER_KEY",)
    )
    env_file = tmp_path / "campaign.env"
    env_file.write_text(
        "CLI_TEST_PROVIDER_KEY=expected\nUNRELATED_SECRET=excluded\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "campaign",
            "prepare",
            str(config),
            "--project-root",
            str(tmp_path),
            "--env-file",
            env_file.name,
        ],
    )

    assert result.exit_code == 0, result.output
    artifact_bytes = b"".join(
        path.read_bytes()
        for path in (tmp_path / options.artifact_root).rglob("*")
        if path.is_file()
    )
    assert b"expected" not in artifact_bytes
    assert b"UNRELATED_SECRET" not in artifact_bytes


def test_campaign_prepare_reports_invalid_json_without_output(input_fixture, tmp_path):
    config = tmp_path / "campaign.json"
    config.write_text("{not-json", encoding="utf-8")

    result = runner.invoke(
        app,
        ["campaign", "prepare", str(config), "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "Invalid campaign preparation input" in result.output
    assert not (tmp_path / "artifacts").exists()


def test_campaign_prepare_does_not_overwrite_existing_campaign(input_fixture, tmp_path):
    config, options = _write_options(input_fixture, tmp_path)
    command = ["campaign", "prepare", str(config), "--project-root", str(tmp_path)]
    first = runner.invoke(app, command)
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, command)

    assert second.exit_code == 1
    assert "failed during resolve_artifact_root (FileExistsError)" in second.output
    assert "Partial evidence retained" not in second.output
    assert (tmp_path / options.artifact_root / "preparation/complete.json").is_file()


def test_campaign_run_executes_aggregates_and_seals_preparation(
    input_fixture, tmp_path, successful_children
):
    config, options = _write_options(input_fixture, tmp_path)
    prepared = runner.invoke(app, _prepare_command(config, tmp_path))
    assert prepared.exit_code == 0, prepared.output

    result = runner.invoke(app, _run_command(options, tmp_path))

    assert result.exit_code == 0, result.output
    assert "Launching campaign pilot-inputs with 6 concrete runs" in result.output
    assert "Status: completed" in result.output
    assert "Succeeded: 6" in result.output
    root = tmp_path / options.artifact_root
    assert (root / "campaign_result.json").is_file()
    assert (root / "analysis.json").is_file()
    assert (root / "checksums.json").is_file()
    assert len(tuple((root / "children").glob("*/result.json"))) == 6


def test_campaign_run_json_stdout_is_one_terminal_document(
    input_fixture, tmp_path, successful_children
):
    config, options = _write_options(input_fixture, tmp_path)
    assert runner.invoke(app, _prepare_command(config, tmp_path)).exit_code == 0

    result = runner.invoke(app, _run_command(options, tmp_path, "--json"))

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["campaign_id"] == options.campaign_id
    assert summary["status"] == "completed"
    assert summary["scheduled"] == 6
    assert summary["succeeded"] == 6
    assert summary["failed"] == 0
    assert "Launching campaign" in result.stderr


def test_campaign_run_returns_one_for_canonical_child_failures(
    input_fixture, tmp_path, monkeypatch
):
    config, options = _write_options(input_fixture, tmp_path)
    assert runner.invoke(app, _prepare_command(config, tmp_path)).exit_code == 0

    def fail(_executor, request, _environment):
        return ChildRunResult(
            request_id=request.request_id,
            run=request.run,
            status=ChildRunStatus.FAILED,
            failure_class="adapter_failure",
            failure_type="SyntheticFailure",
            failure_message="offline failure",
        )

    monkeypatch.setattr(SubprocessChildExecutor, "execute", fail)
    result = runner.invoke(app, _run_command(options, tmp_path))

    assert result.exit_code == 1
    assert "Status: failed" in result.output
    assert "Failed: 6" in result.output
    root = tmp_path / options.artifact_root
    assert (root / "campaign_result.json").is_file()
    assert (root / "analysis.json").is_file()
    assert (root / "checksums.json").is_file()


def test_campaign_run_rejects_tampered_preparation_with_exit_three(
    input_fixture, tmp_path
):
    config, options = _write_options(input_fixture, tmp_path)
    assert runner.invoke(app, _prepare_command(config, tmp_path)).exit_code == 0
    root = tmp_path / options.artifact_root
    (root / "preparation/options.json").write_text("modified", encoding="utf-8")

    result = runner.invoke(app, _run_command(options, tmp_path))

    assert result.exit_code == 3
    assert "load failed during register_artifacts" in result.output
    assert not (root / "children").exists()


def test_campaign_run_is_one_shot(input_fixture, tmp_path, successful_children):
    config, options = _write_options(input_fixture, tmp_path)
    assert runner.invoke(app, _prepare_command(config, tmp_path)).exit_code == 0
    assert runner.invoke(app, _run_command(options, tmp_path)).exit_code == 0

    repeated = runner.invoke(app, _run_command(options, tmp_path))

    assert repeated.exit_code == 3
    assert "load failed during verify_execution_state" in repeated.output
