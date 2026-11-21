"""Append-only trial evidence over the existing campaign artifact store."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from membench.campaign.artifacts import (
    CampaignArtifactError,
    CampaignArtifactIndex,
    CampaignArtifactStore,
    JsonlArtifactWriter,
)
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ChildFailureClass,
    ChildRunRequest,
    ChildRunResult,
)
from membench.memory.provenance import sanitize_native_configuration
from membench.trial.models import ReplayBatchResult, TrialEvent, TrialPlan, TrialResult


class TrialArtifactStore:
    """Own a new evidence directory for exactly one trial, including failed runs.

    Call prepare with a fresh directory (e.g. children/<run>/trial). The enclosing
    launcher owns request.json/result.json. All trial evidence is evaluator-only.
    """

    def __init__(self, store: CampaignArtifactStore) -> None:
        self.store = store
        self.index: CampaignArtifactIndex | None = None
        self._plan: TrialPlan | None = None
        self._result: TrialResult | None = None
        self._events: JsonlArtifactWriter | None = None
        self._replay: JsonlArtifactWriter | None = None
        self._sequence = 0
        self._batch_index = 0

    @classmethod
    def prepare(cls, directory: str | Path) -> TrialArtifactStore:
        return cls(CampaignArtifactStore.prepare(directory))

    def begin(self, plan: TrialPlan, public_settings: Mapping[str, Any]) -> None:
        if self._plan is not None:
            raise CampaignArtifactError("trial evidence has already started")
        self._plan = plan.model_copy(deep=True)
        self.write_evidence("plan", plan)
        self.write_evidence(
            "manifest", plan.run.manifest, kind=ArtifactKind.RESOLVED_MANIFEST
        )
        self.write_evidence(
            "resolved-settings", sanitize_native_configuration(public_settings)
        )
        self._events = self._open_log("events")
        self._replay = self._open_log("replay")

    def _open_log(self, name: str) -> JsonlArtifactWriter:
        return self.store.open_jsonl(
            artifact_id=self._identity(name),
            kind=ArtifactKind.EVENTS,
            relative_path=f"{name}.jsonl",
        )

    def _identity(self, name: str) -> str:
        if self._plan is None:
            raise CampaignArtifactError("trial evidence has not started")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise CampaignArtifactError("invalid trial evidence name")
        return f"trial:{self._plan.run.run_id}:{name}"

    def write_evidence(
        self, name: str, value: Any, *, kind: ArtifactKind = ArtifactKind.OTHER
    ) -> ArtifactReference:
        return self.store.write_json(
            artifact_id=self._identity(name),
            kind=kind,
            relative_path=f"{name}.json",
            value=value,
        )

    def emit(self, event: TrialEvent) -> None:
        if self._events is None or self._plan is None:
            raise CampaignArtifactError("trial event log has not started")
        if (
            event.run_id != self._plan.run.run_id
            or event.trial_id != self._plan.projection.trial_id
            or event.sequence != self._sequence + 1
        ):
            raise CampaignArtifactError(
                "event identity or sequence does not match trial"
            )
        self._events.append(event)
        self._sequence = event.sequence

    def record_replay(self, result: ReplayBatchResult) -> None:
        if self._replay is None or self._plan is None:
            raise CampaignArtifactError("trial replay log has not started")
        if (
            result.batch.request.scope != self._plan.run.scope
            or result.batch.batch_index != self._batch_index + 1
        ):
            raise CampaignArtifactError("replay evidence has wrong scope or sequence")
        self._replay.append(result)
        self._batch_index = result.batch.batch_index

    def close_logs(self) -> None:
        """Register partial streams without deleting evidence on exceptions."""
        errors = []
        for writer in (self._events, self._replay):
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise CampaignArtifactError("could not finalize trial logs") from errors[0]

    def finish(self, result: TrialResult) -> CampaignArtifactIndex:
        if self._plan != result.plan:
            raise CampaignArtifactError("trial result does not match recorded plan")
        self.close_logs()
        self.write_evidence("trial-result", result)
        self.index = self.store.seal(artifact_id=self._identity("checksums"))
        self._result = result.model_copy(deep=True)
        return self.index

    def child_result(
        self, request: ChildRunRequest, result: TrialResult, *, campaign_root: Path
    ) -> ChildRunResult:
        """Rebase sealed references for registration by CampaignLauncher."""
        if self.index is None or result.plan.run != request.run:
            raise CampaignArtifactError("matching sealed trial evidence is required")
        if (
            result != self._result
            or request.cleanup_storage != result.plan.cleanup_storage
        ):
            raise CampaignArtifactError("child result does not match the sealed trial")
        return self._rebase_child_result(request, result, campaign_root=campaign_root)

    def partial_child_result(
        self, request: ChildRunRequest, result: TrialResult, *, campaign_root: Path
    ) -> ChildRunResult:
        """Preserve registered evidence when final sealing itself failed.

        The result must contain the artifact failure produced by TrialRunner.
        Open logs are closed by the runner before this method is called. This
        method neither seals the incomplete trial store nor invents checksums.
        """
        if self.index is not None or result.plan.run != request.run:
            raise CampaignArtifactError("matching unsealed trial evidence is required")
        if request.cleanup_storage != result.plan.cleanup_storage:
            raise CampaignArtifactError("child result does not match the partial trial")
        if not result.failures or not any(
            failure.failure_class == ChildFailureClass.ARTIFACT_FAILURE
            for failure in result.failures
        ):
            raise CampaignArtifactError(
                "partial trial result requires artifact failure evidence"
            )
        return self._rebase_child_result(request, result, campaign_root=campaign_root)

    def _rebase_child_result(
        self, request: ChildRunRequest, result: TrialResult, *, campaign_root: Path
    ) -> ChildRunResult:
        root = campaign_root.resolve()
        if self.store.root == root or not self.store.root.is_relative_to(root):
            raise CampaignArtifactError("trial evidence must be below campaign root")
        prefix = self.store.root.relative_to(root).as_posix()
        references = []
        source_references = (
            self.index.artifacts if self.index is not None else self.store.references()
        )
        for reference in source_references:
            self.store.verify(reference)
            references.append(
                reference.model_copy(
                    update={"relative_path": f"{prefix}/{reference.relative_path}"}
                )
            )
        first = result.failures[0] if result.failures else None
        return ChildRunResult(
            request_id=request.request_id,
            run=request.run,
            status=result.status,
            agent_result=result.agent_result,
            provenance=result.provenance,
            artifacts=tuple(references),
            failure_class=first.failure_class if first else None,
            failure_type=first.error_type if first else None,
            failure_message=first.error_message if first else None,
            started_at=result.started_at,
            finished_at=result.finished_at,
        )
