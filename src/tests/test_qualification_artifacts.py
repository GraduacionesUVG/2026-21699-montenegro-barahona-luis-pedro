import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from membench.contract import Mechanism, WritePolicy, load_manifest
from membench.memory.models import MemoryScope
from membench.qualification.artifacts import (
    QualificationArtifactError,
    QualificationArtifactStore,
)
from membench.qualification.models import (
    QualificationEvent,
    QualificationEventStatus,
    QualificationPlan,
    QualificationReport,
    QualificationStage,
    QualificationSummary,
)


def _plan(run_id: str = "artifact-run") -> QualificationPlan:
    return QualificationPlan(
        qualification_id="baseline_semantic_v1",
        mechanism=Mechanism.TELEMEM,
        policy=WritePolicy.DIRECT,
        scope=MemoryScope(
            experiment_id="artifact-tests",
            run_id=run_id,
            user_id="user-001",
            namespace=("membench", "qualification", run_id),
        ),
        top_k=5,
    )


def _report(plan: QualificationPlan) -> QualificationReport:
    timestamp = datetime.now(UTC)
    return QualificationReport(
        plan=plan,
        summary=QualificationSummary(
            experiment_id=plan.scope.experiment_id,
            run_id=plan.scope.run_id,
            mechanism=plan.mechanism,
            policy=plan.policy,
            qualification_id=plan.qualification_id,
            passed=True,
            started_at=timestamp,
            finished_at=timestamp,
        ),
    )


def _event(plan: QualificationPlan, sequence: int) -> QualificationEvent:
    return QualificationEvent(
        sequence=sequence,
        experiment_id=plan.scope.experiment_id,
        run_id=plan.scope.run_id,
        mechanism=plan.mechanism,
        policy=plan.policy,
        qualification_id=plan.qualification_id,
        stage=QualificationStage.PREFLIGHT,
        status=QualificationEventStatus.STARTED,
    )


def test_artifact_store_streams_events_and_seals_hashed_evidence(
    tmp_path: Path,
) -> None:
    plan = _plan()
    store = QualificationArtifactStore.prepare(tmp_path, plan)
    manifest = load_manifest(
        Path("configs/experiments/blackbox/example.yaml")
    )
    store.write_manifest(manifest)
    store.write_plan(plan)
    store.write_resolved_settings(
        {"provider": "ollama", "credential_configured": True}
    )
    store.event_sink.emit(_event(plan, 1))

    index = store.seal(_report(plan))

    event_lines = (index.run_dir / "events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(event_lines) == 1
    assert json.loads(event_lines[0])["sequence"] == 1
    checksum_document = json.loads(
        (index.run_dir / "checksums.json").read_text(encoding="utf-8")
    )
    assert checksum_document["algorithm"] == "sha256"
    assert checksum_document["files"] == dict(index.sha256_by_name)
    for name, expected in index.sha256_by_name.items():
        actual = hashlib.sha256((index.run_dir / name).read_bytes()).hexdigest()
        assert actual == expected
    assert "checksums.json" not in index.sha256_by_name

    with pytest.raises(QualificationArtifactError, match="already sealed"):
        store.seal(_report(plan))
    with pytest.raises(QualificationArtifactError, match="closed"):
        store.event_sink.emit(_event(plan, 2))
    with pytest.raises(QualificationArtifactError, match="already exists"):
        QualificationArtifactStore.prepare(tmp_path, plan)


def test_event_sink_rejects_noncontiguous_sequences(tmp_path: Path) -> None:
    plan = _plan("sequence-run")
    store = QualificationArtifactStore.prepare(tmp_path, plan)

    with pytest.raises(QualificationArtifactError, match="expected 1, got 2"):
        store.event_sink.emit(_event(plan, 2))


def test_resolved_settings_reject_secret_bearing_fields(tmp_path: Path) -> None:
    store = QualificationArtifactStore.prepare(tmp_path, _plan("secret-run"))

    with pytest.raises(QualificationArtifactError, match="api_key"):
        store.write_resolved_settings({"api_key": "must-not-be-written"})

