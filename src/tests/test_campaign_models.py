from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from membench.agent.models import (
    AgentRunConfiguration,
    AgentRunResult,
    AgentRunStatus,
    AgentStep,
    AgentTask,
    AgentTerminationReason,
    FinalAnswerDecision,
)
from membench.campaign.aggregate import CampaignAggregator
from membench.campaign.bindings import CampaignRunBindings, RunProjectionBinding
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    CampaignConfiguration,
    CampaignResult,
    CampaignRunSpec,
    CampaignStatus,
    CampaignSummary,
    ChildEnvironmentSpec,
    ChildRunRequest,
    ChildRunResult,
    ChildRunStatus,
    TrialTwin,
)
from membench.contract import ExperimentManifest, Mechanism, load_manifest
from membench.memory.models import (
    AdapterProvenance,
    MemoryScope,
    RetrievalResult,
)

EXAMPLE_MANIFEST = Path("configs/experiments/blackbox/example.yaml")


def _artifact(
    artifact_id: str,
    kind: ArtifactKind,
    path: str,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        kind=kind,
        relative_path=path,
        sha256=("a" if kind == ArtifactKind.DATASET_PROJECTION else "b") * 64,
        media_type="application/json",
        size_bytes=128,
    )


def _inputs() -> tuple[ArtifactReference, ...]:
    return (
        _artifact(
            "dataset-001",
            ArtifactKind.DATASET_PROJECTION,
            "inputs/personamem-v2.jsonl",
        ),
        _artifact(
            "attacks-001",
            ArtifactKind.ATTACK_BATTERY,
            "inputs/attacks.json",
        ),
    )


def _manifest(run_id: str) -> ExperimentManifest:
    manifest = load_manifest(EXAMPLE_MANIFEST)
    return manifest.model_copy(
        update={
            "isolation": manifest.isolation.model_copy(
                update={"run_id": run_id}
            )
        }
    )


def _run(twin: TrialTwin, index: int) -> CampaignRunSpec:
    run_id = f"family-001-{twin.value}-{index}"
    scope = MemoryScope(
        experiment_id="campaign-001",
        run_id=run_id,
        user_id="persona-001",
        namespace=("membench", "telemem", run_id, "persona-001"),
    )
    task = AgentTask(
        task_id=f"probe-{twin.value}-{index}",
        scope=scope,
        instruction="Answer the held-out synthetic preference question.",
        retrieval_query="Which destination does this persona prefer?",
    )
    return CampaignRunSpec(
        campaign_id="campaign-001",
        trial_family_id="family-001",
        run_id=run_id,
        twin=twin,
        persona_id="persona-001",
        history_unit_id="history-001",
        response_replicate=1,
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


def _runs() -> tuple[CampaignRunSpec, ...]:
    return tuple(
        _run(twin, index)
        for index, twin in enumerate(TrialTwin, start=1)
    )


def _configuration() -> CampaignConfiguration:
    return CampaignConfiguration(
        campaign_id="campaign-001",
        seed=20260928,
        runs=_runs(),
        environments=(
            ChildEnvironmentSpec(
                mechanism=Mechanism.TELEMEM,
                python_executable=".venvs/telemem/Scripts/python.exe",
                inherited_env_keys=("OLLAMA_BASE_URL", "OLLAMA_CHAT_MODEL"),
            ),
        ),
        frozen_inputs=_inputs(),
        artifact_root="artifacts/runs/campaign-001",
    )


def _agent_result(run: CampaignRunSpec) -> AgentRunResult:
    now = datetime.now(UTC)
    decision = FinalAnswerDecision(content="The synthetic destination is Alpha.")
    retrieval = RetrievalResult(
        operation_id=f"{run.run_id}:retrieve",
        query_sha256=run.agent_task.retrieval_query_sha256,
        request_top_k=run.agent_configuration.retrieval_top_k,
        latency_seconds=0.01,
    )
    step = AgentStep(
        step_index=1,
        prompt_sha256="c" * 64,
        context_sha256="d" * 64,
        decision=decision,
        started_at=now,
        finished_at=now,
    )
    return AgentRunResult(
        task=run.agent_task,
        configuration=run.agent_configuration,
        retrieval_result=retrieval,
        context_sha256="d" * 64,
        status=AgentRunStatus.COMPLETED,
        termination_reason=AgentTerminationReason.FINAL_ANSWER,
        steps=(step,),
        final_answer=decision.content,
        started_at=now,
        finished_at=now,
    )


def _provenance(run: CampaignRunSpec) -> AdapterProvenance:
    return AdapterProvenance(
        mechanism="telemem",
        mechanism_version="1.10.0",
        adapter_version="1.0.0",
        storage_backend="faiss",
        namespace=run.scope.namespace,
    )


def test_campaign_configuration_requires_a_complete_fair_twin_family() -> None:
    configuration = _configuration()

    assert len(configuration.runs) == 3
    assert configuration.sha256 == CampaignConfiguration.model_validate_json(
        configuration.model_dump_json()
    ).sha256
    assert {run.twin for run in configuration.runs} == set(TrialTwin)

    with pytest.raises(ValidationError, match="all three twins"):
        CampaignConfiguration(
            campaign_id="campaign-001",
            seed=1,
            runs=configuration.runs[:2],
            environments=configuration.environments,
            frozen_inputs=configuration.frozen_inputs,
            artifact_root=configuration.artifact_root,
        )


def test_campaign_configuration_rejects_unfair_or_unroutable_runs() -> None:
    configuration = _configuration()
    poisoned = configuration.runs[2]
    changed = poisoned.model_copy(
        update={
            "agent_configuration": poisoned.agent_configuration.model_copy(
                update={"context_token_budget": 1024}
            )
        }
    )

    with pytest.raises(ValidationError, match="unmatched agent settings"):
        CampaignConfiguration(
            campaign_id="campaign-001",
            seed=1,
            runs=(*configuration.runs[:2], changed),
            environments=configuration.environments,
            frozen_inputs=configuration.frozen_inputs,
            artifact_root=configuration.artifact_root,
        )
    with pytest.raises(ValidationError, match="no child environment"):
        CampaignConfiguration(
            campaign_id="campaign-001",
            seed=1,
            runs=configuration.runs,
            environments=(
                ChildEnvironmentSpec(
                    mechanism=Mechanism.LANGMEM,
                    python_executable=".venvs/langgraph/Scripts/python.exe",
                ),
            ),
            frozen_inputs=configuration.frozen_inputs,
            artifact_root=configuration.artifact_root,
        )


def test_artifact_references_reject_traversal_and_child_requests_round_trip() -> None:
    with pytest.raises(ValidationError, match="cannot traverse"):
        _artifact(
            "unsafe-001",
            ArtifactKind.OTHER,
            "../outside.json",
        )

    run = _runs()[0]
    request = ChildRunRequest(
        request_id="request:001",
        run=run,
        frozen_inputs=_inputs(),
        artifact_directory=f"artifacts/runs/campaign-001/{run.run_id}",
    )
    restored = ChildRunRequest.model_validate_json(request.model_dump_json())

    assert restored == request
    assert restored.sha256 == request.sha256


def test_behavioral_artifact_kinds_are_available_for_frozen_campaign_inputs() -> None:
    references = (
        _artifact(
            "task-p277-case-summary-send-v1",
            ArtifactKind.BEHAVIORAL_TASK,
            "inputs/behavioral-task.json",
        ),
        _artifact(
            "oracle-p277-case-summary-send-v1",
            ArtifactKind.BEHAVIORAL_ORACLE,
            "inputs/behavioral-oracle.json",
        ),
        _artifact(
            "result-p277-case-summary-send-v1",
            ArtifactKind.BEHAVIORAL_RESULT,
            "results/behavioral-result.json",
        ),
    )

    assert tuple(reference.kind for reference in references) == (
        ArtifactKind.BEHAVIORAL_TASK,
        ArtifactKind.BEHAVIORAL_ORACLE,
        ArtifactKind.BEHAVIORAL_RESULT,
    )


def test_child_and_campaign_results_require_consistent_terminal_evidence() -> None:
    configuration = _configuration()
    now = datetime.now(UTC)
    children = tuple(
        ChildRunResult(
            request_id=f"request:{index}",
            run=run,
            status=ChildRunStatus.SUCCEEDED,
            agent_result=_agent_result(run),
            provenance=_provenance(run),
            started_at=now,
            finished_at=now,
        )

        for index, run in enumerate(configuration.runs, start=1)
    )
    summary = CampaignSummary(
        campaign_id=configuration.campaign_id,
        status=CampaignStatus.COMPLETED,
        scheduled_count=3,
        succeeded_count=3,
        invalidated_count=0,
        failed_count=0,
        started_at=now,
        finished_at=now,
    )

    result = CampaignResult(
        configuration=configuration,
        children=children,
        summary=summary,
    )

    assert result.summary.status == CampaignStatus.COMPLETED
    with pytest.raises(ValidationError, match="require a failure"):
        ChildRunResult(
            request_id="request:failed",
            run=configuration.runs[0],
            status=ChildRunStatus.FAILED,
            started_at=now,
            finished_at=now,
        )
    with pytest.raises(ValidationError, match="summary counts"):
        CampaignResult(
            configuration=configuration,
            children=children,
            summary=summary.model_copy(
                update={
                    "status": CampaignStatus.PARTIAL,
                    "succeeded_count": 2,
                    "failed_count": 1,
                }
            ),
        )


def test_aggregate_uses_external_attack_identity_over_neutral_host_identity() -> None:
    base = _configuration()
    family_one = tuple(
        _run(twin, index).model_copy(update={"trial_family_id": "family-ext-001"})
        for index, twin in enumerate(TrialTwin, start=10)
    )
    family_two = tuple(
        _run(twin, index).model_copy(update={"trial_family_id": "family-ext-002"})
        for index, twin in enumerate(TrialTwin, start=20)
    )
    configuration = base.model_copy(update={"runs": (*family_one, *family_two)})
    now = datetime.now(UTC)
    children = tuple(
        ChildRunResult(
            request_id=f"request:external:{index}", run=run,
            status=ChildRunStatus.SUCCEEDED, agent_result=_agent_result(run),
            provenance=_provenance(run), started_at=now, finished_at=now,
        )
        for index, run in enumerate(configuration.runs, start=1)
    )
    result = CampaignResult(
        configuration=configuration,
        children=children,
        summary=CampaignSummary(
            campaign_id=configuration.campaign_id, status=CampaignStatus.COMPLETED,
            scheduled_count=6, succeeded_count=6, invalidated_count=0,
            failed_count=0, started_at=now, finished_at=now,
        ),
    )
    bindings = CampaignRunBindings.model_construct(
        bindings=tuple(
            RunProjectionBinding.model_construct(
                run_id=run.run_id,
                external_attack_spec_id=(
                    "false-fact-277-001"
                    if run.trial_family_id == "family-ext-001"
                    else "stored-instruction-277-001"
                ),
            )
            for run in configuration.runs
        )
    )
    analysis = CampaignAggregator().aggregate(result, bindings=bindings)

    family = analysis.trial_families[0]
    assert family.projection_attack_id == configuration.runs[0].attack_id
    assert family.external_attack_spec_id == "false-fact-277-001"
    assert family.effective_attack_id == "false-fact-277-001"
    assert {item.effective_attack_id for item in analysis.conditions} == {
        "false-fact-277-001", "stored-instruction-277-001"
    }
