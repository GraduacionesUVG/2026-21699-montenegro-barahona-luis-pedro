import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from membench.agent.models import (
    AgentRunConfiguration,
    AgentRunResult,
    AgentRunStatus,
    AgentStep,
    AgentTask,
    AgentTerminationReason,
    FinalAnswerDecision,
)
from membench.campaign.artifacts import CampaignArtifactStore
from membench.campaign.environments import ChildEnvironmentRegistry
from membench.campaign.launcher import (
    CampaignLauncher,
    ChildProcessTimeoutError,
    SubprocessChildExecutor,
)
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    CampaignConfiguration,
    CampaignStatus,
    ChildEnvironmentSpec,
    ChildRunRequest,
    ChildRunResult,
    ChildRunStatus,
    TrialTwin,
)
from membench.contract import ExperimentManifest, Mechanism, load_manifest
from membench.memory.models import (
    AdapterProvenance,
    MemoryRecord,
    MemoryScope,
    RetrievalHit,
    RetrievalResult,
)

EXAMPLE_MANIFEST = Path("configs/experiments/blackbox/example.yaml")


def _manifest(run_id: str) -> ExperimentManifest:
    manifest = load_manifest(EXAMPLE_MANIFEST)
    return manifest.model_copy(
        update={
            "isolation": manifest.isolation.model_copy(
                update={"run_id": run_id}
            )
        }
    )


def _run(twin: TrialTwin, index: int):
    run_id = f"family-001-{twin.value}-{index}"
    scope = MemoryScope(
        experiment_id="campaign-001",
        run_id=run_id,
        user_id="persona-001",
        namespace=("membench", "telemem", run_id, "persona-001"),
    )
    task = AgentTask(
        task_id=f"task-{twin.value}-{index}",
        scope=scope,
        instruction="Answer the synthetic destination question.",
        retrieval_query="Which destination is preferred?",
    )
    from membench.campaign.models import CampaignRunSpec

    return CampaignRunSpec(
        campaign_id="campaign-001",
        trial_family_id="family-001",
        run_id=run_id,
        twin=twin,
        persona_id="persona-001",
        history_unit_id="history-001",
        manifest=_manifest(run_id),
        scope=scope,
        agent_task=task,
        agent_configuration=AgentRunConfiguration(
            max_steps=3,
            max_tool_calls=1,
            retrieval_top_k=5,
            context_token_budget=2048,
        ),
    )


def _agent_result(request: ChildRunRequest) -> AgentRunResult:
    run = request.run
    poisoned = run.twin == TrialTwin.POISONED
    answer = "Destination Beta" if poisoned else "Destination Alpha"
    memory_text = "prefers Beta" if poisoned else "prefers Alpha"
    timestamp = datetime.now(UTC)
    retrieval = RetrievalResult(
        operation_id=f"{run.run_id}:retrieve",
        query_sha256=run.agent_task.retrieval_query_sha256,
        request_top_k=run.agent_configuration.retrieval_top_k,
        hits=(
            RetrievalHit(
                record=MemoryRecord(
                    canonical_id=f"memory-{run.run_id}",
                    scope=run.scope,
                    text=memory_text,
                ),
                rank=1,
            ),
        ),
        latency_seconds=0.01,
    )
    decision = FinalAnswerDecision(content=answer)
    return AgentRunResult(
        task=run.agent_task,
        configuration=run.agent_configuration,
        retrieval_result=retrieval,
        context_sha256="c" * 64,
        status=AgentRunStatus.COMPLETED,
        termination_reason=AgentTerminationReason.FINAL_ANSWER,
        steps=(
            AgentStep(
                step_index=1,
                prompt_sha256="d" * 64,
                context_sha256="c" * 64,
                decision=decision,
                started_at=timestamp,
                finished_at=timestamp,
            ),
        ),
        final_answer=answer,
        started_at=timestamp,
        finished_at=timestamp,
    )


class _SuccessfulExecutor:
    def __init__(self) -> None:
        self.requests: list[ChildRunRequest] = []
        self._lock = Lock()

    def execute(self, request, environment):
        with self._lock:
            self.requests.append(request)
        child_directory = environment.working_directory / request.artifact_directory
        events_path = child_directory / "events.jsonl"
        content = b'{"sequence":1}\n'
        events_path.write_bytes(content)
        reference = ArtifactReference(
            artifact_id=f"events:{request.run.run_id}",
            kind=ArtifactKind.EVENTS,
            relative_path=f"children/{request.run.run_id}/events.jsonl",
            sha256=hashlib.sha256(content).hexdigest(),
            media_type="application/x-ndjson",
            size_bytes=len(content),
        )
        return _successful_child(request, artifacts=(reference,))


def _successful_child(
    request: ChildRunRequest,
    *,
    artifacts: tuple[ArtifactReference, ...] = (),
) -> ChildRunResult:
    timestamp = datetime.now(UTC)
    return ChildRunResult(
        request_id=request.request_id,
        run=request.run,
        status=ChildRunStatus.SUCCEEDED,
        agent_result=_agent_result(request),
        provenance=AdapterProvenance(
            mechanism=request.run.mechanism.value,
            mechanism_version="1.10.0",
            adapter_version="1.0.0",
            storage_backend="faiss",
            namespace=request.run.scope.namespace,
        ),
        artifacts=artifacts,
        started_at=timestamp,
        finished_at=timestamp,
    )


class _TimeoutExecutor:
    def execute(self, request, environment):
        del environment
        raise ChildProcessTimeoutError(f"synthetic timeout for {request.run.run_id}")


def _setup(
    tmp_path: Path,
    *,
    executor,
    max_parallel_children: int = 2,
):
    artifact_relative = "artifacts/runs/campaign-001"
    store = CampaignArtifactStore.prepare(tmp_path / artifact_relative)
    dataset = store.write_json(
        artifact_id="dataset-001",
        kind=ArtifactKind.DATASET_PROJECTION,
        relative_path="inputs/personas.json",
        value={"personas": ["persona-001"]},
    )
    attacks = store.write_json(
        artifact_id="attacks-001",
        kind=ArtifactKind.ATTACK_BATTERY,
        relative_path="inputs/attacks.json",
        value={"attacks": ["bb-pi-001"]},
    )
    runs = tuple(
        _run(twin, index)
        for index, twin in enumerate(TrialTwin, start=1)
    )
    configuration = CampaignConfiguration(
        campaign_id="campaign-001",
        seed=42,
        runs=runs,
        environments=(
            ChildEnvironmentSpec(
                mechanism=Mechanism.TELEMEM,
                python_executable=sys.executable,
            ),
        ),
        frozen_inputs=(dataset, attacks),
        artifact_root=artifact_relative,
        max_parallel_children=max_parallel_children,
    )
    environments = ChildEnvironmentRegistry.from_configuration(
        configuration,
        project_root=tmp_path,
        source_environment={},
    )
    launcher = CampaignLauncher(
        configuration,
        environments,
        store,
        child_executor=executor,
    )
    return launcher, store


def test_launcher_executes_children_aggregates_and_seals_evidence(
    tmp_path: Path,
) -> None:
    executor = _SuccessfulExecutor()
    launcher, store = _setup(tmp_path, executor=executor)

    result = launcher.run()

    assert result.summary.status == CampaignStatus.COMPLETED
    assert result.summary.succeeded_count == 3
    assert [request.run.run_id for request in executor.requests] != []
    assert {request.run.run_id for request in executor.requests} == {
        run.run_id for run in result.configuration.runs
    }
    assert launcher.analysis is not None
    assert launcher.analysis.excluded_run_count == 0
    condition = launcher.analysis.conditions[0]
    assert condition.poisoned_response_divergence.value == 1.0
    assert condition.benign_response_divergence.value == 0.0
    assert condition.poisoned_retrieval_divergence.value == 1.0
    assert launcher.artifact_index is not None
    assert store.sealed
    assert (store.root / "campaign_result.json").is_file()
    assert (store.root / "analysis.json").is_file()
    assert (store.root / "checksums.json").is_file()
    store.verify(launcher.artifact_index.checksums)


def test_launcher_normalizes_timeouts_and_aggregation_exclusions(
    tmp_path: Path,
) -> None:
    launcher, store = _setup(
        tmp_path,
        executor=_TimeoutExecutor(),
        max_parallel_children=1,
    )

    result = launcher.run()

    assert result.summary.status == CampaignStatus.FAILED
    assert result.summary.failed_count == 3
    assert all(
        child.failure_class.value == "process_timeout"
        for child in result.children
    )
    assert launcher.analysis is not None
    assert launcher.analysis.valid_run_count == 0
    assert launcher.analysis.excluded_run_count == 3
    condition = launcher.analysis.conditions[0]
    assert condition.valid_family_count == 0
    assert condition.poisoned_response_divergence.denominator == 0
    assert condition.poisoned_response_divergence.value is None
    assert store.sealed


def test_subprocess_executor_uses_stdin_stdout_json_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launcher, _ = _setup(
        tmp_path,
        executor=_TimeoutExecutor(),
        max_parallel_children=1,
    )
    request = launcher.build_child_request(launcher.configuration.runs[0])
    environment = launcher.environments.resolve(Mechanism.TELEMEM)
    expected = _successful_child(request)
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=expected.model_dump_json(),
            stderr="diagnostic output stays separate",
        )

    monkeypatch.setattr("membench.campaign.launcher.subprocess.run", fake_run)

    result = SubprocessChildExecutor().execute(request, environment)

    assert result == expected
    assert observed["command"] == environment.command
    assert ChildRunRequest.model_validate_json(str(observed["input"])) == request
    assert observed["cwd"] == environment.working_directory
    assert observed["env"] == environment.subprocess_environment()
    assert observed["capture_output"] is True
