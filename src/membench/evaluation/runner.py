"""Adapter-neutral execution of frozen evaluator retrieval probes."""

from __future__ import annotations

import hashlib

from membench.campaign.models import TrialTwin
from membench.evaluation.models import (
    BaselineObservation,
    EvaluationExecutionResult,
    EvaluationExposureEvidence,
    EvaluationProbe,
)
from membench.evaluation.scoring import score_retrieval
from membench.memory.models import (
    MemoryScope,
    MemorySnapshot,
    RetrievalRequest,
    RetrievalResult,
)
from membench.memory.protocol import MemoryAdapter
from membench.qualification.assertions import compare_snapshot_state


class EvaluationRunnerError(RuntimeError):
    """Raised when adapter evidence cannot be safely attributed to a probe."""


class EvaluationRunner:
    """Execute one frozen read-only probe against an initialized adapter scope.

    The runner neither initializes, flushes, nor cleans the adapter.  The trial
    lifecycle owns those write-side responsibilities.  It snapshots state on
    both sides of retrieval so a caller can invalidate or report a run if
    retrieval changed memory content.
    """

    def __init__(self, adapter: MemoryAdapter, scope: MemoryScope) -> None:
        self._adapter = adapter
        self._scope = scope

    def run(
        self,
        *,
        trial_id: str,
        twin: TrialTwin,
        probe: EvaluationProbe,
        exposure: EvaluationExposureEvidence | None = None,
        baseline_observation: BaselineObservation | None = None,
    ) -> EvaluationExecutionResult:
        """Retrieve and score one probe while preserving read-only evidence."""

        before = self._inspect("before")
        request = RetrievalRequest(
            operation_id=f"{probe.probe_id}:evaluation_retrieve",
            scope=self._scope,
            query=probe.query,
            top_k=probe.top_k,
            threshold=probe.threshold,
        )
        retrieval = self._adapter.retrieve(request)
        self._validate_retrieval(request, retrieval)
        after = self._inspect("after")
        state_delta = compare_snapshot_state(before, after)
        evaluation = score_retrieval(
            trial_id=trial_id,
            twin=twin,
            probe=probe,
            retrieval=retrieval,
            exposure=exposure,
            before_snapshot=before,
            baseline_observation=baseline_observation,
        )
        return EvaluationExecutionResult(
            scope=self._scope,
            before_snapshot=before,
            evaluation=evaluation,
            after_snapshot=after,
            state_delta=state_delta,
            content_read_only_verified=state_delta.observed_content_unchanged,
        )

    def _inspect(self, phase: str) -> MemorySnapshot:
        snapshot = self._adapter.inspect_state()
        if snapshot.scope != self._scope:
            raise EvaluationRunnerError(
                f"{phase} evaluation snapshot belongs to another memory scope"
            )
        if any(record.scope != self._scope for record in snapshot.records):
            raise EvaluationRunnerError(
                f"{phase} evaluation snapshot contains another memory scope"
            )
        return snapshot

    def _validate_retrieval(
        self,
        request: RetrievalRequest,
        retrieval: RetrievalResult,
    ) -> None:
        expected_query_sha256 = hashlib.sha256(
            request.query.encode("utf-8")
        ).hexdigest()
        if retrieval.operation_id != request.operation_id:
            raise EvaluationRunnerError("evaluation retrieval operation ID mismatched")
        if retrieval.query_sha256 != expected_query_sha256:
            raise EvaluationRunnerError("evaluation retrieval query hash mismatched")
        if retrieval.request_top_k != request.top_k:
            raise EvaluationRunnerError("evaluation retrieval top_k mismatched")
        if retrieval.threshold != request.threshold:
            raise EvaluationRunnerError("evaluation retrieval threshold mismatched")
        if any(hit.record.scope != self._scope for hit in retrieval.hits):
            raise EvaluationRunnerError(
                "evaluation retrieval contains another memory scope"
            )
