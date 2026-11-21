"""Cross-environment orchestration for bounded memory-agent campaigns."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import ValidationError

from membench.campaign.aggregate import CampaignAggregator, CampaignAnalysis
from membench.campaign.artifacts import (
    CampaignArtifactError,
    CampaignArtifactIndex,
    CampaignArtifactStore,
)
from membench.campaign.bindings import CampaignRunBindings
from membench.campaign.environments import (
    ChildEnvironmentError,
    ChildEnvironmentRegistry,
    EnvironmentUnavailableError,
    ResolvedChildEnvironment,
)
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    CampaignConfiguration,
    CampaignResult,
    CampaignRunSpec,
    CampaignStatus,
    CampaignSummary,
    ChildFailureClass,
    ChildRunRequest,
    ChildRunResult,
    ChildRunStatus,
)
from membench.evaluation.aggregation import EvaluationAnalysis
from membench.evaluation.loading import (
    EvaluationEvidenceLoadError,
    aggregate_sealed_evaluations,
)


class CampaignLaunchError(RuntimeError):
    """Base error for campaign-level orchestration failures."""


class CampaignFailFastError(CampaignLaunchError):
    """Raised after persisting the first non-successful fail-fast child."""

    def __init__(self, child: ChildRunResult) -> None:
        self.child = child
        super().__init__(
            f"fail-fast campaign stopped after child {child.run.run_id} "
            f"finished with status {child.status.value}"
        )


class ChildExecutionError(RuntimeError):
    """Base error raised by a child-process transport."""


class ChildProcessLaunchError(ChildExecutionError):
    """Raised when the configured child process cannot start or exits badly."""


class ChildProcessTimeoutError(ChildExecutionError):
    """Raised when a child process exceeds its environment timeout."""


class ChildProtocolError(ChildExecutionError):
    """Raised when stdout is not exactly one valid matching child result."""


class ChildProcessExecutor(Protocol):
    """Execute one request inside an already resolved child environment."""

    def execute(
        self,
        request: ChildRunRequest,
        environment: ResolvedChildEnvironment,
    ) -> ChildRunResult: ...


class SubprocessChildExecutor:
    """Exchange canonical JSON with an isolated ``python -m`` child process."""

    def execute(
        self,
        request: ChildRunRequest,
        environment: ResolvedChildEnvironment,
    ) -> ChildRunResult:
        payload = f"{request.model_dump_json()}\n"
        try:
            completed = subprocess.run(  # noqa: S603
                environment.command,
                input=payload,
                capture_output=True,
                cwd=environment.working_directory,
                env=environment.subprocess_environment(),
                encoding="utf-8",
                errors="replace",
                timeout=environment.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ChildProcessTimeoutError(
                f"child {request.run.run_id} exceeded "
                f"{environment.timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise ChildProcessLaunchError(
                f"could not start child {request.run.run_id}: {exc}"
            ) from exc

        if completed.returncode != 0:
            diagnostic = (completed.stderr or "").encode()
            diagnostic_sha256 = hashlib.sha256(diagnostic).hexdigest()
            raise ChildProcessLaunchError(
                f"child {request.run.run_id} exited with status "
                f"{completed.returncode}; stderr_sha256={diagnostic_sha256}"
            )
        output = completed.stdout.strip()
        if not output:
            raise ChildProtocolError(
                f"child {request.run.run_id} returned empty stdout"
            )
        try:
            result = ChildRunResult.model_validate_json(output)
        except ValidationError as exc:
            raise ChildProtocolError(
                f"child {request.run.run_id} returned invalid result JSON "
                f"({exc.error_count()} validation errors)"
            ) from exc
        _assert_result_matches_request(result, request)
        return result


class CampaignLauncher:
    """Launch all configured children and seal one complete evidence directory."""

    def __init__(
        self,
        configuration: CampaignConfiguration,
        environments: ChildEnvironmentRegistry,
        artifacts: CampaignArtifactStore,
        *,
        child_executor: ChildProcessExecutor | None = None,
        aggregator: CampaignAggregator | None = None,
        bindings: CampaignRunBindings | None = None,
        cleanup_storage: bool = True,
    ) -> None:
        self.configuration = configuration
        self.environments = environments
        self.artifacts = artifacts
        self.cleanup_storage = cleanup_storage
        self.child_executor = child_executor or SubprocessChildExecutor()
        self.aggregator = aggregator or CampaignAggregator()
        self.bindings = bindings
        self.analysis: CampaignAnalysis | None = None
        self.evaluation_analysis: EvaluationAnalysis | None = None
        self.artifact_index: CampaignArtifactIndex | None = None
        self._artifact_root_from_project = self._validate_artifact_root()

    def run(self) -> CampaignResult:
        """Execute, aggregate, persist, and seal one complete campaign."""

        if self.artifacts.sealed:
            raise CampaignLaunchError("campaign artifact store is already sealed")
        started_at = datetime.now(UTC)
        try:
            self._register_frozen_inputs()
            self.artifacts.write_json(
                artifact_id=(
                    f"campaign:{self.configuration.campaign_id}:configuration"
                ),
                kind=ArtifactKind.OTHER,
                relative_path="campaign_configuration.json",
                value=self.configuration,
            )
            children = self._execute_runs()
            summary = _campaign_summary(
                self.configuration.campaign_id,
                children,
                started_at,
                datetime.now(UTC),
            )
            preliminary = CampaignResult(
                configuration=self.configuration,
                children=children,
                summary=summary,
                artifacts=self.artifacts.references(),
            )
            self.analysis = self.aggregator.aggregate(
                preliminary, bindings=self.bindings
            )
            self.artifacts.write_json(
                artifact_id=f"campaign:{self.configuration.campaign_id}:analysis",
                kind=ArtifactKind.OTHER,
                relative_path="analysis.json",
                value=self.analysis,
            )
            self._persist_evaluation_analysis(children)
            result = CampaignResult(
                configuration=self.configuration,
                children=children,
                summary=summary,
                artifacts=self.artifacts.references(),
            )
            self.artifacts.write_json(
                artifact_id=f"campaign:{self.configuration.campaign_id}:result",
                kind=ArtifactKind.OTHER,
                relative_path="campaign_result.json",
                value=result,
            )
            self.artifact_index = self.artifacts.seal(
                artifact_id=(
                    f"campaign:{self.configuration.campaign_id}:checksums"
                )
            )
            return result
        except CampaignFailFastError:
            raise
        except (CampaignArtifactError, ChildEnvironmentError) as exc:
            raise CampaignLaunchError(str(exc)) from exc

    def _persist_evaluation_analysis(
        self,
        children: tuple[ChildRunResult, ...],
    ) -> None:
        """Aggregate sealed evaluator evidence only when every twin completed."""

        bindings = self.bindings
        if bindings is None or bindings.evaluation_probe is None:
            return
        if any(child.status != ChildRunStatus.SUCCEEDED for child in children):
            return
        try:
            analysis = aggregate_sealed_evaluations(
                self.artifacts,
                bindings,
                children,
            )
        except EvaluationEvidenceLoadError as exc:
            raise CampaignLaunchError(
                "could not aggregate sealed evaluation evidence"
            ) from exc
        self.evaluation_analysis = analysis
        self.artifacts.write_json(
            artifact_id=(
                f"campaign:{self.configuration.campaign_id}:evaluation-analysis"
            ),
            kind=ArtifactKind.EVALUATION_RESULT,
            relative_path="evaluation_analysis.json",
            value=analysis,
        )

    def build_child_request(
        self,
        run: CampaignRunSpec,
        *,
        attempt: int = 1,
    ) -> ChildRunRequest:
        """Build the complete request for one configured campaign run."""

        if run not in self.configuration.runs:
            raise CampaignLaunchError(
                f"run {run.run_id} does not belong to this campaign"
            )
        child_directory = (
            self._artifact_root_from_project
            / "children"
            / run.run_id
        ).as_posix()
        return ChildRunRequest(
            request_id=(
                f"{self.configuration.campaign_id}:{run.run_id}:attempt-{attempt}"
            ),
            run=run,
            frozen_inputs=self.configuration.frozen_inputs,
            artifact_directory=child_directory,
            attempt=attempt,
            cleanup_storage=self.cleanup_storage,
        )

    def run_child(self, run: CampaignRunSpec) -> ChildRunResult:
        """Execute and persist one child, primarily for controlled diagnostics."""

        child_relative_directory = f"children/{run.run_id}"
        self.artifacts.prepare_directory(child_relative_directory)
        request = self.build_child_request(run)
        self.artifacts.write_json(
            artifact_id=f"child-request:{run.run_id}",
            kind=ArtifactKind.CHILD_REQUEST,
            relative_path=f"{child_relative_directory}/request.json",
            value=request,
        )

        started_at = datetime.now(UTC)
        try:
            environment = self.environments.resolve(run.mechanism)
            result = self.child_executor.execute(request, environment)
            _assert_result_matches_request(result, request)
        except EnvironmentUnavailableError as exc:
            result = _failed_child(
                request,
                ChildFailureClass.ENVIRONMENT_UNAVAILABLE,
                exc,
                started_at,
            )
        except ChildProcessTimeoutError as exc:
            result = _failed_child(
                request,
                ChildFailureClass.PROCESS_TIMEOUT,
                exc,
                started_at,
            )
        except ChildProtocolError as exc:
            result = _failed_child(
                request,
                ChildFailureClass.CHILD_PROTOCOL_ERROR,
                exc,
                started_at,
            )
        except ChildProcessLaunchError as exc:
            result = _failed_child(
                request,
                ChildFailureClass.PROCESS_LAUNCH_FAILED,
                exc,
                started_at,
            )
        except ChildEnvironmentError as exc:
            result = _failed_child(
                request,
                ChildFailureClass.ENVIRONMENT_UNAVAILABLE,
                exc,
                started_at,
            )
        except Exception as exc:
            result = _failed_child(
                request,
                ChildFailureClass.UNEXPECTED_ERROR,
                exc,
                started_at,
            )

        registered = []
        try:
            for reference in result.artifacts:
                registered.append(
                    self.artifacts.register_existing(
                        artifact_id=reference.artifact_id,
                        kind=reference.kind,
                        relative_path=reference.relative_path,
                        media_type=reference.media_type,
                        visibility=reference.visibility,
                        expected_sha256=reference.sha256,
                        expected_size_bytes=reference.size_bytes,
                    )
                )
            result = ChildRunResult.model_validate(
                {
                    **result.model_dump(mode="json"),
                    "artifacts": [
                        item.model_dump(mode="json") for item in registered
                    ],
                }
            )
        except CampaignArtifactError as exc:
            result = _failed_child(
                request,
                ChildFailureClass.ARTIFACT_FAILURE,
                exc,
                started_at,
                artifacts=tuple(registered),
            )

        self.artifacts.write_json(
            artifact_id=f"child-result:{run.run_id}",
            kind=ArtifactKind.CHILD_RESULT,
            relative_path=f"{child_relative_directory}/result.json",
            value=result,
        )
        return result

    def _execute_runs(self) -> tuple[ChildRunResult, ...]:
        runs = self.configuration.runs
        if self.configuration.fail_fast:
            results: list[ChildRunResult] = []
            for run in runs:
                child = self.run_child(run)
                results.append(child)
                if child.status != ChildRunStatus.SUCCEEDED:
                    raise CampaignFailFastError(child)
            return tuple(results)

        if self.configuration.max_parallel_children == 1:
            return tuple(self.run_child(run) for run in runs)

        results_by_id: dict[str, ChildRunResult] = {}
        with ThreadPoolExecutor(
            max_workers=self.configuration.max_parallel_children,
            thread_name_prefix="membench-child",
        ) as executor:
            futures = {
                executor.submit(self.run_child, run): run.run_id for run in runs
            }
            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    results_by_id[run_id] = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
        return tuple(results_by_id[run.run_id] for run in runs)

    def _register_frozen_inputs(self) -> None:
        existing = {
            reference.artifact_id: reference
            for reference in self.artifacts.references()
        }
        for reference in self.configuration.frozen_inputs:
            registered = existing.get(reference.artifact_id)
            if registered is not None:
                if registered != reference:
                    raise CampaignArtifactError(
                        "frozen input conflicts with an existing artifact: "
                        f"{reference.artifact_id}"
                    )
                self.artifacts.verify(reference)
                continue
            observed = self.artifacts.register_existing(
                artifact_id=reference.artifact_id,
                kind=reference.kind,
                relative_path=reference.relative_path,
                media_type=reference.media_type,
                visibility=reference.visibility,
                expected_sha256=reference.sha256,
                expected_size_bytes=reference.size_bytes,
            )
            if observed != reference:
                raise CampaignArtifactError(
                    f"frozen input metadata changed: {reference.artifact_id}"
                )

    def _validate_artifact_root(self) -> PurePosixPath:
        configured = self.configuration.artifact_root
        if "\\" in configured:
            raise CampaignLaunchError(
                "campaign artifact_root must use portable '/' separators"
            )
        portable = PurePosixPath(configured)
        if (
            portable.is_absolute()
            or configured == "."
            or portable.as_posix() != configured
            or ".." in portable.parts
        ):
            raise CampaignLaunchError(
                "campaign artifact_root must be canonical, relative, below the "
                "project root, and cannot traverse"
            )
        expected = (
            self.environments.project_root / Path(*portable.parts)
        ).resolve()
        if expected != self.artifacts.root:
            raise CampaignLaunchError(
                "campaign artifact store root does not match artifact_root: "
                f"expected {expected}, got {self.artifacts.root}"
            )
        if (
            self.artifacts.root == self.environments.project_root
            or not self.artifacts.root.is_relative_to(
                self.environments.project_root
            )
        ):
            raise CampaignLaunchError(
                "campaign artifact store must remain below the project root"
            )
        return portable


def _assert_result_matches_request(
    result: ChildRunResult,
    request: ChildRunRequest,
) -> None:
    if result.request_id != request.request_id:
        raise ChildProtocolError(
            f"child result request_id does not match {request.request_id}"
        )
    if result.run != request.run:
        raise ChildProtocolError(
            f"child result modified run specification {request.run.run_id}"
        )


def _failed_child(
    request: ChildRunRequest,
    failure_class: ChildFailureClass,
    error: Exception,
    started_at: datetime,
    *,
    artifacts: Sequence[ArtifactReference] = (),
) -> ChildRunResult:
    return ChildRunResult(
        request_id=request.request_id,
        run=request.run,
        status=ChildRunStatus.FAILED,
        artifacts=tuple(artifacts),
        failure_class=failure_class,
        failure_type=type(error).__name__,
        failure_message=str(error) or type(error).__name__,
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )


def _campaign_summary(
    campaign_id: str,
    children: tuple[ChildRunResult, ...],
    started_at: datetime,
    finished_at: datetime,
) -> CampaignSummary:
    succeeded = sum(
        child.status == ChildRunStatus.SUCCEEDED for child in children
    )
    invalidated = sum(
        child.status == ChildRunStatus.INVALIDATED for child in children
    )
    failed = sum(child.status == ChildRunStatus.FAILED for child in children)
    status = (
        CampaignStatus.COMPLETED
        if succeeded == len(children)
        else CampaignStatus.FAILED
        if succeeded == 0
        else CampaignStatus.PARTIAL
    )
    return CampaignSummary(
        campaign_id=campaign_id,
        status=status,
        scheduled_count=len(children),
        succeeded_count=succeeded,
        invalidated_count=invalidated,
        failed_count=failed,
        started_at=started_at,
        finished_at=finished_at,
    )
