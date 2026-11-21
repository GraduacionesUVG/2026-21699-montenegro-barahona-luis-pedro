from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from membench.agent.models import (
    AgentRunResult,
    AgentRunStatus,
    AgentTerminationReason,
)
from membench.campaign.models import ChildFailureClass, ChildRunStatus
from membench.curation.projections import EventStage
from membench.memory.models import AdapterProvenance, MemorySnapshot, RetrievalResult
from membench.memory.policy import PolicyDispatcher
from membench.qualification.assertions import compare_snapshot_state
from membench.trial import (
    ReplayBatch,
    ReplayExecutor,
    TrialEvent,
    TrialFailure,
    TrialPlan,
    TrialResult,
    build_replay_batches,
)


def test_plan_is_roundtrippable_frozen_and_records_pilot_status(trial_plan_factory):
    plan = trial_plan_factory()
    restored = TrialPlan.model_validate_json(plan.model_dump_json())
    assert restored == plan
    assert restored.sha256 == plan.sha256
    assert restored.purpose == "pilot"
    assert restored.attack_review_status.value == "generated_unreviewed"
    with pytest.raises(ValidationError, match="frozen"):
        plan.cleanup_storage = False


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("twin", "same twin"),
        ("probe", "does not belong"),
        ("query", "held-out public probe"),
        ("writeback", "read-only"),
        ("private_labels", "Extra inputs"),
        ("unreviewed_experiment", "pilot status"),
    ],
)
def test_plan_rejects_mismatched_or_private_inputs(trial_plan_factory, mutation, match):
    payload = trial_plan_factory().model_dump(mode="json")
    if mutation == "twin":
        payload["run"]["twin"] = "clean"
    elif mutation == "probe":
        payload["probe"]["probe_id"] = "probe-another"
    elif mutation == "query":
        payload["run"]["agent_task"]["retrieval_query"] = "A different question?"
    elif mutation == "writeback":
        payload["run"]["agent_configuration"]["memory_read_only"] = False
    elif mutation == "private_labels":
        payload["private_labels"] = {"correct_answer": "evaluator only"}
    else:
        payload["purpose"] = "experiment"
    with pytest.raises(ValidationError, match=match):
        TrialPlan.model_validate(payload)


@pytest.mark.parametrize("mutation", ["content", "stage", "metadata"])
def test_batch_rejects_changed_text_cross_stage_grouping_and_label_metadata(
    trial_plan_factory, mutation
):
    batches = build_replay_batches(trial_plan_factory())
    payload = batches[0].model_dump(mode="json")
    if mutation == "content":
        payload["request"]["messages"][0]["content"] = "Altered source text"
    elif mutation == "stage":
        payload["events"][1] = batches[-1].events[0].model_dump(mode="json")
    else:
        payload["request"]["metadata"] = {"poisoned": True}
    with pytest.raises(ValidationError):
        ReplayBatch.model_validate(payload)


def test_trial_events_validate_timestamps_and_failure_status():
    payload = dict(
        sequence=1,
        run_id="run-001",
        trial_id="trial-001",
        timestamp=datetime.now(UTC),
        stage="replay",
        status="failed",
    )
    with pytest.raises(ValidationError, match="failure evidence"):
        TrialEvent(**payload)
    payload["failure"] = TrialFailure(
        stage="replay",
        failure_class=ChildFailureClass.ADAPTER_FAILURE,
        error_type="SyntheticFailure",
        error_message="write failed",
    )
    event = TrialEvent(**payload)
    assert TrialEvent.model_validate_json(event.model_dump_json()) == event
    payload["timestamp"] = datetime(2026, 1, 1)
    with pytest.raises(ValidationError, match="timezone"):
        TrialEvent(**payload)


def test_failed_trial_can_retain_partial_replay_evidence(trial_plan_factory, adapter):
    plan = trial_plan_factory()
    started = datetime.now(UTC)
    adapter.raise_on_attempt = 2
    executor = ReplayExecutor(plan, PolicyDispatcher(adapter, plan.run.policy))
    results = tuple(executor.replay_stage(EventStage.BENIGN_PREFIX))
    result = TrialResult(
        plan=plan,
        status=ChildRunStatus.FAILED,
        replay_results=results,
        failures=(results[-1].failure,),
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    assert TrialResult.model_validate_json(result.model_dump_json()) == result


def test_success_requires_full_replay_and_correct_scope(trial_plan_factory, adapter):
    plan = trial_plan_factory()
    started = datetime.now(UTC)
    executor = ReplayExecutor(plan, PolicyDispatcher(adapter, plan.run.policy))
    for stage in executor.stages:
        list(executor.replay_stage(stage))
    agent_result = AgentRunResult(
        task=plan.run.agent_task,
        configuration=plan.run.agent_configuration,
        retrieval_result=RetrievalResult(
            operation_id="retrieval-001",
            query_sha256=plan.run.agent_task.retrieval_query_sha256,
            request_top_k=plan.run.agent_configuration.retrieval_top_k,
            latency_seconds=0.0,
            hits=(),
        ),
        context_sha256="0" * 64,
        status=AgentRunStatus.COMPLETED,
        termination_reason=AgentTerminationReason.FINAL_ANSWER,
        final_answer="Trains.",
    )
    result = TrialResult(
        plan=plan,
        status=ChildRunStatus.SUCCEEDED,
        replay_results=executor.results,
        state_deltas={
            name: compare_snapshot_state(
                MemorySnapshot(scope=plan.run.scope),
                MemorySnapshot(scope=plan.run.scope),
            )
            for name in ("retrieval", "response")
        },
        agent_result=agent_result,
        provenance=AdapterProvenance(
            mechanism=plan.run.mechanism.value,
            mechanism_version="test",
            adapter_version="test",
            storage_backend="fake",
            namespace=plan.run.scope.namespace,
        ),
        started_at=started,
        finished_at=datetime.now(UTC),
    )
    payload = result.model_dump(mode="json")
    payload["replay_results"].pop()
    with pytest.raises(ValidationError, match="complete replay"):
        TrialResult.model_validate(payload)
    payload = result.model_dump(mode="json")
    payload["replay_results"][0]["batch"]["request"]["scope"]["user_id"] = "wrong-user"
    with pytest.raises(ValidationError, match="wrong memory scope"):
        TrialResult.model_validate(payload)
    payload = result.model_dump(mode="json")
    payload["state_deltas"] = {}
    with pytest.raises(ValidationError, match="both read-only checkpoints"):
        TrialResult.model_validate(payload)
