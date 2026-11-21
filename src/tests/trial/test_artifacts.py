import json

import pytest

from membench.agent.context import AgentContextAssembler
from membench.campaign.artifacts import ArtifactExistsError, CampaignArtifactError
from membench.campaign.models import ArtifactKind, ArtifactReference, ChildRunRequest
from membench.trial.artifacts import TrialArtifactStore
from membench.trial.models import TrialEvent
from membench.trial.runner import TrialRunner


def test_artifacts_preserve_unicode_scrub_settings_and_reject_overwrite(
    fast_plan, tmp_path
):
    root = tmp_path / "trial"
    store = TrialArtifactStore.prepare(root)
    store.begin(fast_plan, {"nested": {"api_key": "test-credential-value"}})
    assert "test-credential-value" not in (root / "resolved-settings.json").read_text(
        "utf-8"
    )
    store.write_evidence("unicode", {"text": "偶尔与朋友看电影。 You’re welcome."})
    assert json.loads((root / "unicode.json").read_text("utf-8"))["text"].startswith(
        "偶尔"
    )
    with pytest.raises(ArtifactExistsError):
        store.write_evidence("unicode", {"text": "different"})
    with pytest.raises(ArtifactExistsError):
        TrialArtifactStore.prepare(root)
    with pytest.raises(CampaignArtifactError):
        store.write_evidence("../escape", {})
    with pytest.raises(CampaignArtifactError, match="sequence"):
        store.emit(
            TrialEvent(
                sequence=2,
                run_id=fast_plan.run.run_id,
                trial_id=fast_plan.projection.trial_id,
                stage="initialize",
                status="started",
            )
        )
    store.close_logs()


def test_action_decision_diagnostic_is_evaluator_only_and_checksummed(
    fast_plan, lifecycle_adapter, decider, tmp_path
):
    store = TrialArtifactStore.prepare(tmp_path / "trial")
    runner = TrialRunner(
        fast_plan,
        lifecycle_adapter,
        decider,
        AgentContextAssembler(),
        store,
    )
    store.begin(fast_plan, {})
    runner._write_action_decision_diagnostic(
        "public",
        {
            "schema_version": 1,
            "task_id": fast_plan.run.agent_task.task_id,
            "step_index": 1,
            "field_presence": {"content": False, "thinking": True},
            "completion_evidence": {"thinking_sha256": "a" * 64},
        },
    )
    store.close_logs()

    reference = next(
        item
        for item in store.store.references()
        if item.kind == ArtifactKind.ACTION_DECISION_DIAGNOSTIC
    )
    assert reference.visibility.value == "evaluator_only"
    assert reference.relative_path == "public-action-decision-0001.json"
    payload = json.loads(store.store.verify(reference).read_text("utf-8"))
    assert payload["field_presence"]["thinking"] is True


def test_child_references_are_rebased_and_integrity_checked(
    fast_plan, lifecycle_adapter, decider, tmp_path
):
    root = tmp_path / "campaign"
    store = TrialArtifactStore.prepare(
        root / "children" / fast_plan.run.run_id / "trial"
    )
    result = TrialRunner(
        fast_plan, lifecycle_adapter, decider, AgentContextAssembler(), store
    ).run()
    request = ChildRunRequest(
        request_id="request-001",
        run=fast_plan.run,
        artifact_directory=f"campaign/children/{fast_plan.run.run_id}",
        frozen_inputs=tuple(
            ArtifactReference(
                artifact_id=kind.value,
                kind=kind,
                relative_path=f"inputs/{kind.value}.json",
                sha256="0" * 64,
                size_bytes=1,
                media_type="application/json",
            )
            for kind in (ArtifactKind.DATASET_PROJECTION, ArtifactKind.ATTACK_BATTERY)
        ),
    )
    child = store.child_result(request, result, campaign_root=root)
    assert child.request_id == request.request_id
    assert child.artifacts
    assert all(
        r.relative_path.startswith(f"children/{fast_plan.run.run_id}/trial/")
        for r in child.artifacts
    )
    assert all((root / r.relative_path).is_file() for r in child.artifacts)
    for ref in child.artifacts:
        assert ref.visibility.value == "evaluator_only"
    (store.store.root / "plan.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(CampaignArtifactError, match="mismatch"):
        store.child_result(request, result, campaign_root=root)
