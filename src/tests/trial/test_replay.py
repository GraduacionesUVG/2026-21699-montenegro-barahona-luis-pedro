from __future__ import annotations

import pytest

from membench.campaign.models import TrialTwin
from membench.contract import WritePolicy
from membench.curation.projections import EventStage
from membench.memory.models import WriteStatus
from membench.memory.policy import PolicyDispatcher, render_conversation
from membench.trial import (
    ReplayBatching,
    ReplayExecutionError,
    ReplayExecutor,
    build_replay_batches,
)


@pytest.mark.parametrize("twin", list(TrialTwin))
@pytest.mark.parametrize("batching", list(ReplayBatching))
def test_batches_preserve_every_event_once_and_never_include_probe(
    trial_plan_factory, twin, batching
):
    plan = trial_plan_factory(twin=twin, batching=batching)
    batches = build_replay_batches(plan)

    assert tuple(e for batch in batches for e in batch.events) == (
        plan.projection.replay_events
    )
    assert batches == build_replay_batches(plan)
    assert len({b.request.operation_id for b in batches}) == len(batches)
    for batch in batches:
        assert batch.request.scope == plan.run.scope
        assert {e.stage for e in batch.events} == {batch.stage}
        assert batch.request.metadata == {}
        assert all(m.metadata == {} for m in batch.request.messages)
        assert all(m.content != plan.probe.content for m in batch.request.messages)
        assert "label-evaluator-only" not in batch.request.model_dump_json()
    assert "偶尔与朋友看电影。" in "".join(
        m.content for b in batches for m in b.request.messages
    )
    if batching == ReplayBatching.EVENT:
        assert all(len(batch.events) == 1 for batch in batches)
    else:
        lengths = [len(b.events) for b in batches]
        expected = [2, 2, 1, 2] if twin == TrialTwin.CLEAN else [2, 2, 1, 1, 1, 2]
        assert lengths == expected


def test_direct_and_native_receive_identical_role_content_batches(
    trial_plan_factory, adapter
):
    native = trial_plan_factory()
    direct = trial_plan_factory(policy=WritePolicy.DIRECT)
    native_batches = build_replay_batches(native)
    direct_batches = build_replay_batches(direct)
    assert [b.messages_sha256 for b in native_batches] == [
        b.messages_sha256 for b in direct_batches
    ]
    for plan in (direct, native):
        executor = ReplayExecutor(plan, PolicyDispatcher(adapter, plan.run.policy))
        for stage in executor.stages:
            assert all(result.succeeded for result in executor.replay_stage(stage))
        assert executor.complete
    assert [request.text for request in adapter.direct_requests] == [
        render_conversation(request.messages)
        for request in adapter.conversation_requests
    ]


def test_explicit_source_boundary_splits_a_user_turn_without_changing_content(
    trial_plan_factory,
):
    plan = trial_plan_factory(twin=TrialTwin.CLEAN, batching=ReplayBatching.USER_TURN)
    batches = build_replay_batches(plan, split_after_source_message_indexes=(1,))

    assert tuple(event for batch in batches for event in batch.events) == (
        plan.projection.replay_events
    )
    assert [event.source_message_index for event in batches[0].events] == [1]
    assert [event.source_message_index for event in batches[1].events] == [2]


def test_unknown_source_boundary_is_rejected_before_replay(trial_plan_factory):
    with pytest.raises(ReplayExecutionError, match="not present"):
        build_replay_batches(
            trial_plan_factory(), split_after_source_message_indexes=(999,)
        )


def test_stage_boundary_leaves_survival_unwritten_and_prevents_duplicates(
    trial_plan_factory, adapter
):
    plan = trial_plan_factory()
    executor = ReplayExecutor(plan, PolicyDispatcher(adapter, plan.run.policy))
    with pytest.raises(ReplayExecutionError, match="next replay stage"):
        list(executor.replay_stage(EventStage.POISONED_OVERLAY))
    assert adapter.attempts == 0

    list(executor.replay_stage(EventStage.BENIGN_PREFIX))
    assert adapter.attempts == 2
    assert not executor.complete
    with pytest.raises(ReplayExecutionError, match="next replay stage"):
        list(executor.replay_stage(EventStage.BENIGN_PREFIX))

    overlay = list(executor.replay_stage(EventStage.POISONED_OVERLAY))
    assert len(overlay) == 2
    assert all(len(result.batch.events) == 1 for result in overlay)
    assert adapter.attempts == 4
    list(executor.replay_stage(EventStage.BENIGN_SURVIVAL))
    assert executor.complete
    with pytest.raises(ReplayExecutionError, match="already complete"):
        list(executor.replay_stage(EventStage.BENIGN_SURVIVAL))
    assert adapter.attempts == 6


def test_noop_and_queued_outcomes_are_preserved_without_retries(
    trial_plan_factory, adapter
):
    plan = trial_plan_factory()
    adapter.statuses = [WriteStatus.NO_OP, WriteStatus.QUEUED]
    executor = ReplayExecutor(plan, PolicyDispatcher(adapter, plan.run.policy))
    results = tuple(executor.replay_stage(EventStage.BENIGN_PREFIX))
    assert all(result.succeeded for result in results)
    assert [result.dispatch.write_results[0].status for result in results] == [
        WriteStatus.NO_OP,
        WriteStatus.QUEUED,
    ]
    assert adapter.attempts == 2


@pytest.mark.parametrize("failure_mode", ["transport", "partial", "wrong_id"])
def test_failure_preserves_attempt_evidence_and_stops_remaining_writes(
    trial_plan_factory, adapter, failure_mode
):
    plan = trial_plan_factory(batching=ReplayBatching.EVENT)
    if failure_mode == "transport":
        adapter.raise_on_attempt = 2
    elif failure_mode == "partial":
        adapter.statuses = [WriteStatus.SUCCEEDED, WriteStatus.PARTIAL]
    else:
        adapter.bad_operation_id = True
    executor = ReplayExecutor(plan, PolicyDispatcher(adapter, plan.run.policy))
    results = list(executor.replay_stage(EventStage.BENIGN_PREFIX))
    assert not results[-1].succeeded
    assert results[-1].failure is not None
    assert adapter.attempts == (1 if failure_mode == "wrong_id" else 2)
    assert not executor.complete
    if failure_mode == "partial":
        assert results[-1].dispatch.write_results[0].status == WriteStatus.PARTIAL
        assert results[-1].failure.error_type == "PartialReplayWriteError"
    else:
        assert results[-1].dispatch is None
    attempts = adapter.attempts
    with pytest.raises(ReplayExecutionError, match="failed attempt"):
        list(executor.replay_stage(EventStage.BENIGN_PREFIX))
    assert adapter.attempts == attempts


def test_native_option_mutation_cannot_change_plan_or_recorded_requests(
    trial_plan_factory, adapter
):
    plan = trial_plan_factory(native_options={"nested": {"names": ["original"]}})
    digest = plan.sha256
    adapter.mutate_options = True
    executor = ReplayExecutor(plan, PolicyDispatcher(adapter, plan.run.policy))
    for stage in executor.stages:
        list(executor.replay_stage(stage))
    assert plan.sha256 == digest
    for result in executor.results:
        assert result.batch.request.native_options == {
            "nested": {"names": ["original"]}
        }
    exposed = executor.results[0]
    exposed.batch.request.native_options["nested"]["names"].append("consumer mutation")
    assert (
        executor.results[0].batch.request.native_options == plan.replay.native_options
    )


def test_policy_mismatch_is_rejected_before_adapter_use(trial_plan_factory, adapter):
    plan = trial_plan_factory()
    with pytest.raises(ReplayExecutionError, match="differs"):
        ReplayExecutor(plan, PolicyDispatcher(adapter, WritePolicy.DIRECT))
    assert adapter.attempts == 0


def test_policy_mutation_between_yields_does_not_route_next_write(
    trial_plan_factory, adapter
):
    plan = trial_plan_factory()
    dispatcher = PolicyDispatcher(adapter, plan.run.policy)
    executor = ReplayExecutor(plan, dispatcher)
    iterator = executor.replay_stage(EventStage.BENIGN_PREFIX)
    assert next(iterator).succeeded
    dispatcher.policy = WritePolicy.DIRECT
    assert not next(iterator).succeeded
    assert list(iterator) == []
    assert adapter.attempts == 1
    assert adapter.direct_requests == []


def test_abandoned_iterator_can_resume_without_repeating_a_write(
    trial_plan_factory, adapter
):
    plan = trial_plan_factory()
    executor = ReplayExecutor(plan, PolicyDispatcher(adapter, plan.run.policy))
    iterator = executor.replay_stage(EventStage.BENIGN_PREFIX)
    first = next(iterator)
    with pytest.raises(ReplayExecutionError, match="iterator is active"):
        list(executor.replay_stage(EventStage.BENIGN_PREFIX))
    iterator.close()
    remaining = list(executor.replay_stage(EventStage.BENIGN_PREFIX))
    assert len(remaining) == 1
    assert remaining[0].batch.batch_index == first.batch.batch_index + 1
    assert adapter.attempts == 2
