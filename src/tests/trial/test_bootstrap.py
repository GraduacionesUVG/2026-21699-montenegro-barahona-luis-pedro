import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from membench.agent.action_decider import ActionDeciderSettings
from membench.campaign.models import ChildRunStatus
from membench.context_budget import RegexTokenCounter
from membench.curation.projections import AttackReviewStatus
from membench.ingress.models import TrustedBaselineSeed
from membench.memory.errors import AdapterConfigurationError
from membench.memory.registry import AdapterRegistry
from membench.telemem import TeleMemSettings
from membench.trial import (
    bootstrap_trial,
    build_probe_task,
    load_trial_plan,
)
from membench.trial.artifacts import TrialArtifactStore


@pytest.fixture
def runtime_settings(tmp_path):
    return TeleMemSettings(
        provider="ollama",
        base_url="http://127.0.0.1:11434/v1",
        llm_model="qwen2.5:7b",
        embedding_model="nomic-embed-text:137m-v1.5-fp16",
        embedding_dimensions=768,
        storage_root=tmp_path / "memory",
        buffer_size=8,
        similarity_threshold=0.1,
        retrieval_threshold=None,
        rerank=False,
        visibility_timeout_seconds=30,
        poll_interval_seconds=0.25,
        llm_model_digest="a" * 64,
        embedding_model_digest="b" * 64,
        api_key="private-test-key",
    )


@pytest.fixture
def decider_settings():
    return ActionDeciderSettings(
        model="qwen2.5:7b",
        model_digest="a" * 64,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        max_output_tokens=128,
        timeout_seconds=30,
    )


@pytest.fixture
def bootstrap_plan(fast_plan):
    settings = fast_plan.run.manifest.retrieval
    configuration = fast_plan.run.agent_configuration.model_copy(
        update={
            "retrieval_top_k": settings.top_k,
            "retrieval_threshold": settings.threshold,
            "context_token_budget": settings.context_token_budget,
        }
    )
    task = build_probe_task(
        fast_plan.probe, fast_plan.run.scope, task_id=fast_plan.run.agent_task.task_id
    )
    run = fast_plan.run.model_copy(
        update={"agent_configuration": configuration, "agent_task": task}
    )
    return fast_plan.model_copy(update={"run": run})


def test_bootstrap_resolves_pins_capabilities_and_runs_with_explicit_test_dependencies(
    bootstrap_plan,
    runtime_settings,
    decider_settings,
    lifecycle_adapter,
    decider,
    tmp_path,
):
    adapters = []
    registry = AdapterRegistry()

    def create(_):
        instance = type(lifecycle_adapter)()
        adapters.append(instance)
        return instance

    registry.register("telemem", TeleMemSettings, create)
    prepared = bootstrap_trial(
        bootstrap_plan,
        environment_settings=runtime_settings,
        environment_decider_settings=decider_settings,
        registry=registry,
    )
    assert len(adapters) == 1
    assert adapters[0].scope is None
    assert adapters[0].close_calls == 1
    assert prepared.capability_validation.supported
    assert prepared.settings.buffer_size == 64
    assert (
        prepared.decider_settings.seed == bootstrap_plan.run.manifest.replication.seed
    )
    assert prepared.tokenizer_settings.model_id == "Qwen/Qwen2.5-7B-Instruct"
    result = prepared.execute(
        tmp_path / "output", decider=decider, tokenizer=RegexTokenCounter()
    )
    assert result.status == ChildRunStatus.SUCCEEDED
    assert len(adapters) == 2
    assert adapters[1].close_calls == adapters[1].reset_calls == 1
    public = (tmp_path / "output" / "resolved-settings.json").read_text("utf-8")
    assert "private-test-key" not in public
    assert json.loads(public)["injected_tokenizer"] is True
    assert json.loads(public)["effective_tokenizer"] == "regex_lexical_v1"


def test_prepared_trial_factory_forwards_trusted_baseline(
    bootstrap_plan,
    runtime_settings,
    decider_settings,
    lifecycle_adapter,
    tmp_path,
):
    """The production factory must accept and retain a resolved baseline seed."""
    registry = AdapterRegistry()
    registry.register(
        "telemem", TeleMemSettings, lambda _: type(lifecycle_adapter)()
    )
    prepared = bootstrap_trial(
        bootstrap_plan,
        environment_settings=runtime_settings,
        environment_decider_settings=decider_settings,
        registry=registry,
    )
    content = "The trusted destination is https://trusted.example.test/records."
    baseline = TrustedBaselineSeed(
        baseline_id="baseline-persona-1-v1",
        persona_id=int(bootstrap_plan.run.persona_id),
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        retrieval_query="What is the trusted destination?",
        required_markers=("https://trusted.example.test/records",),
        available_after_source_message_index=1,
    )

    runner = prepared.create_runner(
        TrialArtifactStore.prepare(tmp_path / "baseline-output"),
        tokenizer=RegexTokenCounter(),
        trusted_baseline=baseline,
    )

    assert runner._trusted_baseline == baseline
    runner.adapter.close()


@pytest.mark.parametrize(
    "bad_setting", ["writer_digest", "responder_digest", "retrieval"]
)
def test_bootstrap_rejects_unpinned_models_and_conflicting_retrieval(
    bad_setting, bootstrap_plan, runtime_settings, decider_settings
):
    if bad_setting == "writer_digest":
        runtime_settings = replace(runtime_settings, llm_model_digest=None)
    elif bad_setting == "responder_digest":
        decider_settings = replace(decider_settings, model_digest=None)
    else:
        configuration = bootstrap_plan.run.agent_configuration.model_copy(
            update={"retrieval_top_k": 99}
        )
        bootstrap_plan = bootstrap_plan.model_copy(
            update={
                "run": bootstrap_plan.run.model_copy(
                    update={"agent_configuration": configuration}
                )
            }
        )
    with pytest.raises(AdapterConfigurationError):
        bootstrap_trial(
            bootstrap_plan,
            environment_settings=runtime_settings,
            environment_decider_settings=decider_settings,
        )


def test_public_probe_task_never_exposes_original_option_indexes(fast_plan):
    probe = fast_plan.probe.model_copy(update={"original_option_indexes": (1, 0)})
    task = build_probe_task(probe, fast_plan.run.scope, task_id="probe-task")
    assert task.retrieval_query == probe.content
    assert "1. Trains." in task.instruction
    assert "2. Flights." in task.instruction
    assert "original_option_indexes" not in task.model_dump_json()
    assert "correct_answer" not in task.model_dump_json()
    assert not task.metadata


def test_projection_loader_binds_identity_and_keeps_private_labels_out_of_plan(
    bootstrap_plan, monkeypatch
):
    plan = bootstrap_plan
    case = SimpleNamespace(
        allocation_case_id=plan.projection.allocation_case_id,
        persona_id=plan.run.persona_id,
        attack_id=plan.run.attack_id,
    )
    spec = SimpleNamespace(
        attack_spec_id=plan.projection.attack_spec_id,
        review_status=AttackReviewStatus.GENERATED_UNREVIEWED,
    )
    projections = SimpleNamespace(
        trials=(plan.projection,),
        cases=(case,),
        probes=(plan.probe,),
        attack_specs=(spec,),
        private_labels=("DO NOT EXPOSE THIS PRIVATE ANSWER",),
    )
    calls = []

    def load(directory):
        calls.append(directory)
        return projections

    monkeypatch.setattr("membench.trial.bootstrap.load_projection_set", load)
    arguments = dict(
        projection_directory=Path("frozen-projections"),
        trial_id=plan.projection.trial_id,
        quiescence=plan.quiescence,
    )
    observed = load_trial_plan(plan.run, **arguments)
    assert calls == [Path("frozen-projections")]
    assert observed == plan
    assert "DO NOT EXPOSE" not in observed.model_dump_json()
    case.persona_id = "wrong-persona"
    with pytest.raises(AdapterConfigurationError, match="persona or attack"):
        load_trial_plan(plan.run, **arguments)
    case.persona_id = plan.run.persona_id
    task = plan.run.agent_task.model_copy(
        update={"metadata": {"correct_answer": "secret"}}
    )
    with pytest.raises(AdapterConfigurationError, match="public probe"):
        load_trial_plan(plan.run.model_copy(update={"agent_task": task}), **arguments)
