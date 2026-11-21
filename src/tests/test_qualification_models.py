from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from membench.contract import Mechanism, WritePolicy
from membench.memory.models import MemoryScope
from membench.qualification.models import (
    QualificationAssertion,
    QualificationEvent,
    QualificationEventStatus,
    QualificationFailureClass,
    QualificationPlan,
    QualificationStage,
    QualificationSummary,
    QuiescenceConfiguration,
)


def _scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="qualification-tests",
        run_id="run-001",
        user_id="user-001",
        namespace=("membench", "qualification", "run-001"),
    )


def test_plan_is_strict_immutable_and_validates_limits() -> None:
    plan = QualificationPlan(
        qualification_id="baseline_semantic_v1",
        mechanism=Mechanism.TELEMEM,
        policy=WritePolicy.DIRECT,
        scope=_scope(),
        top_k=5,
    )

    assert plan.cleanup
    with pytest.raises(ValidationError, match="frozen"):
        plan.top_k = 2
    with pytest.raises(ValidationError, match="greater than 0"):
        QualificationPlan(
            qualification_id="baseline_semantic_v1",
            mechanism=Mechanism.TELEMEM,
            policy=WritePolicy.DIRECT,
            scope=_scope(),
            top_k=0,
        )


def test_plan_requires_secondary_scope_to_have_a_distinct_run_and_namespace() -> None:
    duplicate_run = MemoryScope(
        experiment_id="qualification-tests",
        run_id="run-001",
        user_id="user-002",
        namespace=("membench", "qualification", "alternate"),
    )

    with pytest.raises(ValidationError, match="another run_id"):
        QualificationPlan(
            qualification_id="namespace_isolation_v1",
            mechanism=Mechanism.TELEMEM,
            policy=WritePolicy.DIRECT,
            scope=_scope(),
            secondary_scope=duplicate_run,
            top_k=5,
        )


def test_quiescence_configuration_requires_achievable_timing() -> None:
    with pytest.raises(ValidationError, match="stable window cannot exceed timeout"):
        QuiescenceConfiguration(
            timeout_seconds=1,
            poll_interval_seconds=0.1,
            stable_window_seconds=2,
        )


def test_failed_event_requires_complete_failure_information() -> None:
    with pytest.raises(ValidationError, match="complete failure information"):
        QualificationEvent(
            sequence=1,
            experiment_id="qualification-tests",
            run_id="run-001",
            mechanism=Mechanism.TELEMEM,
            policy=WritePolicy.DIRECT,
            qualification_id="baseline_semantic_v1",
            stage=QualificationStage.PREFLIGHT,
            status=QualificationEventStatus.FAILED,
        )


def test_summary_passed_flag_must_match_assertions_and_failure() -> None:
    failed_assertion = QualificationAssertion(
        assertion_id="retrieval_marker",
        stage=QualificationStage.RETRIEVE,
        passed=False,
        expected=True,
        observed=False,
    )
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="passed must agree"):
        QualificationSummary(
            experiment_id="qualification-tests",
            run_id="run-001",
            mechanism=Mechanism.TELEMEM,
            policy=WritePolicy.DIRECT,
            qualification_id="baseline_semantic_v1",
            passed=True,
            assertions=(failed_assertion,),
            started_at=now,
            finished_at=now,
        )

    summary = QualificationSummary(
        experiment_id="qualification-tests",
        run_id="run-001",
        mechanism=Mechanism.TELEMEM,
        policy=WritePolicy.DIRECT,
        qualification_id="baseline_semantic_v1",
        passed=False,
        assertions=(failed_assertion,),
        failure_class=QualificationFailureClass.ASSERTION_FAILED,
        failure_type="QualificationAssertionError",
        failure_message="one assertion failed",
        started_at=now,
        finished_at=now,
    )

    assert not summary.passed


def test_failed_diagnostic_assertion_does_not_fail_summary() -> None:
    diagnostic = QualificationAssertion(
        assertion_id="retrieval_operational_unchanged:test",
        stage=QualificationStage.VERIFY_RETRIEVAL_UNCHANGED,
        passed=False,
        required=False,
        expected={"normalized_sha256": "before"},
        observed={"normalized_sha256": "after"},
    )
    now = datetime.now(UTC)

    summary = QualificationSummary(
        experiment_id="qualification-tests",
        run_id="run-001",
        mechanism=Mechanism.DAKERA,
        policy=WritePolicy.DIRECT,
        qualification_id="responder_read_only_v1",
        passed=True,
        assertions=(diagnostic,),
        started_at=now,
        finished_at=now,
    )

    assert summary.passed
    assert not summary.assertions[0].passed
    assert not summary.assertions[0].required
