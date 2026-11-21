"""One isolated trial: replay, stabilize, probe, preserve evidence, and close."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from membench.agent.models import (
    AgentDataClassification,
    AgentDecisionContext,
    AgentRunConfiguration,
    AgentRunStatus,
    AgentStep,
    AgentTask,
    PolicyDisposition,
    ToolActionDecision,
)
from membench.agent.protocol import ActionDecider, AgentContextBuilder, ApprovalProvider
from membench.agent.runner import AgentRunner
from membench.campaign.artifacts import CampaignArtifactError, canonical_json_bytes
from membench.campaign.models import ArtifactKind, ChildFailureClass, ChildRunStatus
from membench.contract import WritePolicy
from membench.curation.projections import EventStage
from membench.evaluation.artifacts import write_evaluation_artifacts
from membench.evaluation.models import (
    BaselineObservation,
    EvaluationExposureEvidence,
    EvaluationExposureStatus,
    EvaluationProbe,
)
from membench.evaluation.runner import EvaluationRunner
from membench.ingress.agent_selected import (
    AgentSelectedIngressExecutor,
)
from membench.ingress.bridge import ExternalIngestionBridge
from membench.ingress.forced import ForcedIngressExecutor
from membench.ingress.models import (
    BaselineSeedPolicy,
    ExternalIngestionResult,
    IngressMode,
    TrustedBaselineSeed,
)
from membench.memory.errors import UnsupportedCapabilityError
from membench.memory.models import (
    DirectWriteRequest,
    MemorySnapshot,
    RetrievalRequest,
    RetrievalResult,
)
from membench.memory.policy import (
    PolicyDispatchAction,
    PolicyDispatcher,
    PolicyDispatchResult,
)
from membench.memory.protocol import MemoryAdapter
from membench.memory.requirements import validate_adapter_capabilities
from membench.qualification.assertions import compare_snapshot_state
from membench.qualification.quiescence import wait_for_quiescence
from membench.tools.mock_messaging import (
    MockMessagingOutbox,
    MockOutboxSnapshot,
    validate_simulate_send_arguments,
)
from membench.tools.models import ToolSideEffectKind, ToolStatus
from membench.tools.protocol import ToolExecutor
from membench.tools.registry import ToolRegistry
from membench.trial.artifacts import TrialArtifactStore
from membench.trial.failures import exception_causes
from membench.trial.models import (
    AgentSelectedSelectionResult,
    TrialEvent,
    TrialEventStatus,
    TrialFailure,
    TrialPlan,
    TrialResult,
    TrialStage,
    TrustedBaselineVerification,
    summarize_agent_selected_stage,
)
from membench.trial.replay import ReplayExecutor

if TYPE_CHECKING:
    from membench.ingress.resolution import ResolvedIngress


class TrialInvariantError(RuntimeError):
    """Execution occurred, but its isolation or read-only evidence is invalid."""


class TrialArtifactFailure(CampaignArtifactError):
    """Evidence could not be sealed; result retains the terminal failure details."""

    def __init__(self, result: TrialResult) -> None:
        self.result = result
        super().__init__("trial evidence could not be finalized")


class _ReplayStopped(RuntimeError):
    pass


def _canonical_json_sha256(value: object) -> str:
    """Match mock-messaging's canonical argument hash without private imports."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TrialRunner:
    """Execute once using injected dependencies; never infer attack success.

    Content changes invalidate probe execution. Operational changes (e.g. Dakera
    importance/access updates) are reported separately. Cleanup may delete only a
    scope whose initial emptiness was established by this execution.
    """

    def __init__(
        self,
        plan: TrialPlan,
        adapter: MemoryAdapter,
        decider: ActionDecider,
        context_builder: AgentContextBuilder,
        artifacts: TrialArtifactStore,
        *,
        tools: ToolExecutor | None = None,
        public_settings: Mapping[str, Any] | None = None,
        forced_ingress: ResolvedIngress | None = None,
        agent_selected_ingress: ResolvedIngress | None = None,
        selection_tools: ToolExecutor | None = None,
        replay_source_index_observer: Callable[[int], None] | None = None,
        evaluation_probe: EvaluationProbe | None = None,
        trusted_baseline: TrustedBaselineSeed | None = None,
        approval_provider: ApprovalProvider | None = None,
        outbox: MockMessagingOutbox | None = None,
    ) -> None:
        self.plan = TrialPlan.model_validate(plan.model_dump(mode="json"))
        self.adapter = adapter
        self.decider = decider
        self.context_builder = context_builder
        self.artifacts = artifacts
        if tools is None:
            tools = ToolRegistry()
            tools.freeze()
        self.tools = tools
        self.public_settings = dict(public_settings or {})
        self._forced_ingress = forced_ingress
        self._agent_selected_ingress = agent_selected_ingress
        self._selection_tools = selection_tools
        self._replay_source_index_observer = replay_source_index_observer
        self._evaluation_probe = evaluation_probe
        self._trusted_baseline = trusted_baseline
        self._approval_provider = approval_provider
        self._outbox = outbox
        if self._outbox is not None:
            definition = self.tools.resolve("simulate_send")
            if definition is None:
                raise ValueError(
                    "an outbox requires a registered simulate_send definition"
                )
            if definition.side_effect != ToolSideEffectKind.SIMULATED_WRITE:
                raise ValueError("an outbox cannot be paired with a real send tool")
        self._validate_forced_ingress()
        self._validate_agent_selected_ingress()
        self._validate_evaluation_probe()
        self._validate_trusted_baseline()
        self._used = False
        self._owns_scope = False
        self._sequence = 0
        self._stage = TrialStage.PREFLIGHT
        self._replay_stage: EventStage | None = None
        self._failures: list[TrialFailure] = []
        self._invalidated = False
        self._replay: ReplayExecutor | None = None
        self._dispatcher: PolicyDispatcher | None = None
        self._forced_ingress_executed = False
        self._forced_ingress_result: ExternalIngestionResult | None = None
        self._trusted_baseline_executed = False
        self._trusted_baseline_verification: TrustedBaselineVerification | None = None
        self._agent_selected_ingress_executed = False
        self._selection_result: AgentSelectedSelectionResult | None = None
        self._agent_result = None
        self._outbox_snapshot_captured = False
        self._provenance = None
        self._before_probe: MemorySnapshot | None = None
        self._after_retrieval: MemorySnapshot | None = None
        self._deltas = {}

    def run(self) -> TrialResult:
        if self._used:
            raise RuntimeError("TrialRunner is single-use; use a fresh isolated run")
        self._used = True
        started = datetime.now(UTC)
        try:
            self.artifacts.begin(self.plan, self.public_settings)
            if self._trusted_baseline is not None:
                self.artifacts.write_evidence(
                    "trusted-baseline-plan",
                    self._trusted_baseline,
                    kind=ArtifactKind.TRUSTED_BASELINE,
                )
            if self._forced_ingress is not None:
                self.artifacts.write_evidence(
                    "forced-ingress-plan",
                    self._forced_ingress.plan,
                    kind=ArtifactKind.TRIAL_INGRESS_PLAN,
                )
            self._perform(TrialStage.PREFLIGHT, "capabilities", self._preflight)
            self._perform(
                TrialStage.INITIALIZE,
                "initialize",
                lambda: self.adapter.initialize(self.plan.run.scope),
            )
            empty = self._perform(
                TrialStage.VERIFY_EMPTY, "initial-empty", self.adapter.verify_empty
            )
            self._require(
                empty.scope == self.plan.run.scope and empty.empty,
                "initial memory scope is not verifiably empty",
            )
            initial = self._perform(
                TrialStage.SNAPSHOT,
                "snapshot-initial",
                self.adapter.inspect_state,
                kind=ArtifactKind.SNAPSHOT,
            )
            self._validate_snapshot(initial)
            self._require(not initial.records, "initial snapshot is not empty")
            self._owns_scope = True
            self._dispatcher = PolicyDispatcher(self.adapter, self.plan.run.policy)
            self._replay = ReplayExecutor(
                self.plan,
                self._dispatcher,
                split_after_source_message_indexes=self._schedule_write_boundaries(),
            )
            for stage in self._replay.stages:
                self._replay_stage = stage
                self._stage = TrialStage.REPLAY
                self._emit(TrialEventStatus.STARTED)
                for result in self._replay.replay_stage(stage):
                    if result.failure is not None:
                        self._failures.append(result.failure)
                    self.artifacts.record_replay(result)
                    if not result.succeeded:
                        assert result.failure is not None
                        self._emit(TrialEventStatus.FAILED, failure=result.failure)
                        raise _ReplayStopped()
                    self._emit(
                        TrialEventStatus.SUCCEEDED,
                        operation_id=result.batch.request.operation_id,
                        payload={"batch_index": result.batch.batch_index},
                    )
                    self._observe_replay_source_index(result)
                    if self._trusted_baseline_is_due(result):
                        self._execute_trusted_baseline()
                    if self._forced_ingress_is_due(result):
                        self._execute_forced_ingress()
                    if self._agent_selected_ingress_is_due(result):
                        self._execute_agent_selected_ingress()
                self._before_probe = self._barrier(stage.value)
            if self._forced_ingress is not None:
                self._require(
                    self._forced_ingress_executed,
                    "forced ingress was not scheduled during replay",
                )
            if self._agent_selected_ingress is not None:
                self._require(
                    self._agent_selected_ingress_executed,
                    "agent-selected ingress was not scheduled during replay",
                )
            if self._trusted_baseline is not None:
                self._require(
                    self._trusted_baseline_executed,
                    "trusted baseline was not scheduled during replay",
                )
                self._validate_baseline_verification()
            self._replay_stage = None
            self._stage = TrialStage.RETRIEVE
            self._emit(TrialEventStatus.STARTED)
            agent = AgentRunner(
                self.adapter,
                self.decider,
                self.context_builder,
                self.tools,
                approval_provider=self._approval_provider,
                retrieval_observer=self._observe_retrieval,
                context_observer=self._observe_context,
                decision_evidence_observer=lambda evidence: self._write_action_decision_diagnostic(
                    "public", evidence
                ),
            )
            self._agent_result = agent.run(
                self.plan.run.agent_task, self.plan.run.agent_configuration
            )
            self.artifacts.write_evidence(
                "agent-result", self._agent_result, kind=ArtifactKind.AGENT_RESULT
            )
            self._seal_outbox_snapshot()
            self._stage = TrialStage.RESPOND
            if self._agent_result.status == AgentRunStatus.FAILED:
                self._record_failure(
                    RuntimeError(
                        self._agent_result.failure_message or "agent execution failed"
                    )
                )
            else:
                self._emit(TrialEventStatus.SUCCEEDED)
            after_response = self._barrier("after_response")
            assert self._after_retrieval is not None
            self._compare("response", self._after_retrieval, after_response)
            if self._evaluation_probe is not None:
                self._execute_evaluation()
        except _ReplayStopped:
            pass  # The exact adapter failure was already recorded.
        except Exception as exc:
            self._record_failure(exc)
        finally:
            try:
                self._seal_outbox_snapshot()
            except Exception as exc:
                self._record_failure(exc)
            # Persist provenance before reset even when replay or generation failed.
            if self._owns_scope:
                try:
                    provenance = self._perform(
                        TrialStage.PROVENANCE,
                        "provenance",
                        self.adapter.export_provenance,
                        kind=ArtifactKind.PROVENANCE,
                    )
                    self._require(
                        provenance.namespace == self.plan.run.scope.namespace
                        and provenance.mechanism == self.plan.run.mechanism.value,
                        "adapter provenance has the wrong mechanism or namespace",
                    )
                    self._provenance = provenance
                except Exception as exc:
                    self._record_failure(exc)
            self._cleanup()

        self._stage = TrialStage.COMPLETE
        try:
            self._emit(
                TrialEventStatus.FAILED
                if self._failures
                else TrialEventStatus.SUCCEEDED,
                # COMPLETE is a summary; individual failures retain their own stages.
                failure=(
                    self._failure("trial did not complete successfully")
                    if self._failures
                    else None
                ),
            )
        except Exception as exc:
            self._record_failure(exc)
        result = self._result(started)
        try:
            self.artifacts.finish(result)
        except Exception as exc:
            self._failures.append(self._failure(CampaignArtifactError(str(exc))))
            try:
                self.artifacts.close_logs()
            except Exception:
                pass
            raise TrialArtifactFailure(self._result(started)) from exc
        return result

    def _preflight(self) -> dict[str, Any]:
        validated = validate_adapter_capabilities(self.plan.run.manifest, self.adapter)
        baseline = self._trusted_baseline
        if (
            baseline is not None
            and baseline.seed_policy == BaselineSeedPolicy.DIRECT
            and not self.adapter.capabilities.direct_write
        ):
            raise UnsupportedCapabilityError(
                "trusted baseline direct seed requires adapter capability: "
                "direct_write",
                missing=("direct_write",),
            )
        available = {definition.name for definition in self.tools.definitions()}
        if not set(self.plan.run.agent_configuration.allowed_tools) <= available:
            raise ValueError("configured tools are not registered")
        ingress = self._agent_selected_ingress
        if ingress is not None:
            configuration = ingress.plan.selection_agent_configuration
            if configuration is None or self._selection_tools is None:
                raise ValueError("agent-selected ingress lacks runtime dependencies")
            if configuration.allowed_tools != ingress.plan.allowed_tools:
                raise ValueError("selection tools differ from the frozen ingress plan")
            selection_available = {
                definition.name for definition in self._selection_tools.definitions()
            }
            if not set(configuration.allowed_tools) <= selection_available:
                raise ValueError("configured selection tools are not registered")
        self.decider.verify()
        return validated.as_dict()

    def _validate_forced_ingress(self) -> None:
        """Reject an unresolved or non-forced delivery before adapter use."""

        ingress = self._forced_ingress
        if ingress is None:
            return
        plan = ingress.plan
        delivery = ingress.delivery
        if plan.ingress_mode != IngressMode.FORCED_INGESTION:
            raise ValueError("TrialRunner accepts only forced ingress")
        if (
            plan.host_trial_id != self.plan.projection.trial_id
            or plan.persona_id != self.plan.run.persona_id
            or plan.twin != self.plan.run.twin
            or delivery.delivery_id != plan.delivery_id
            or delivery.content_sha256 != plan.delivery_content_sha256
            or delivery.trial_id != plan.trial_id
        ):
            raise ValueError("forced ingress does not match the current trial")

    def _validate_agent_selected_ingress(self) -> None:
        """Reject a selected source that is not exactly bound to this trial."""

        ingress = self._agent_selected_ingress
        if ingress is None:
            if self._selection_tools is not None:
                raise ValueError("selection tools require agent-selected ingress")
            return
        if self._forced_ingress is not None:
            raise ValueError("a trial cannot combine forced and selected ingress")
        plan = ingress.plan
        delivery = ingress.delivery
        if (
            plan.ingress_mode != IngressMode.AGENT_SELECTED
            or ingress.selection_task is None
            or plan.selection_agent_configuration is None
            or self._selection_tools is None
        ):
            raise ValueError("agent-selected ingress is incomplete")
        if (
            plan.host_trial_id != self.plan.projection.trial_id
            or plan.persona_id != self.plan.run.persona_id
            or plan.twin != self.plan.run.twin
            or delivery.delivery_id != plan.delivery_id
            or delivery.content_sha256 != plan.delivery_content_sha256
            or delivery.trial_id != plan.trial_id
            or plan.selection_agent_configuration.allowed_tools != plan.allowed_tools
            or ingress.selection_task.allowed_tools != plan.allowed_tools
        ):
            raise ValueError("agent-selected ingress does not match the current trial")

    def _validate_evaluation_probe(self) -> None:
        """Ensure evaluator-only retrieval cannot be attributed to another user."""

        probe = self._evaluation_probe
        if probe is not None and probe.persona_id != self.plan.run.persona_id:
            raise ValueError(
                "evaluation probe persona does not match the current trial"
            )

    def _ingress_boundaries(self) -> tuple[int, ...]:
        """Split replay exactly where a configured external source becomes usable."""

        values = []
        if self._forced_ingress is not None:
            values.append(self._forced_ingress.delivery.available_after_source_message_index)
        if self._agent_selected_ingress is not None:
            values.append(
                self._agent_selected_ingress.delivery.available_after_source_message_index
            )
        return tuple(sorted(set(values)))

    def _observe_replay_source_index(self, result) -> None:
        """Advance external-source availability after one replay batch succeeds."""

        observer = self._replay_source_index_observer
        if observer is None:
            return
        source_indexes = tuple(
            event.source_message_index
            for event in result.batch.events
            if event.source_message_index is not None
        )
        if source_indexes:
            observer(max(source_indexes))

    def _forced_ingress_is_due(self, result) -> bool:
        ingress = self._forced_ingress
        if ingress is None or self._forced_ingress_executed:
            return False
        return (
            result.batch.events[-1].source_message_index
            == ingress.delivery.available_after_source_message_index
        )

    def _agent_selected_ingress_is_due(self, result) -> bool:
        ingress = self._agent_selected_ingress
        if ingress is None or self._agent_selected_ingress_executed:
            return False
        return (
            result.batch.events[-1].source_message_index
            == ingress.delivery.available_after_source_message_index
        )

    def _execute_forced_ingress(self) -> None:
        """Dispatch one frozen source, then establish its visible stable state."""

        ingress = self._forced_ingress
        dispatcher = self._dispatcher
        if ingress is None or dispatcher is None:
            raise RuntimeError("forced ingress is not configured")
        if self._forced_ingress_executed:
            raise TrialInvariantError("forced ingress was scheduled more than once")
        operation_id = (
            f"{self.plan.run.run_id}:external-ingress:{ingress.delivery.delivery_id}"
        )
        executor = ForcedIngressExecutor(ExternalIngestionBridge(dispatcher))
        result = self._perform(
            TrialStage.INGRESS,
            "forced-ingress-result",
            lambda: executor.execute(
                ingress.delivery,
                scope=self.plan.run.scope,
                operation_id=operation_id,
                native_options=dict(self.plan.replay.native_options),
            ),
        )
        self._require(
            result.request.operation_id == operation_id
            and result.request.scope == self.plan.run.scope
            and result.request.mode == IngressMode.FORCED_INGESTION
            and result.request.delivery == ingress.delivery
            and result.dispatch.policy == self.plan.run.policy
            and not result.selected_by_agent,
            "forced ingress dispatch does not match the bound delivery",
        )
        self._forced_ingress_executed = True
        self._forced_ingress_result = result
        self._barrier("external_ingress")
        self._stage = TrialStage.REPLAY

    def _execute_agent_selected_ingress(self) -> None:
        """Run one frozen inbox task at its exact replay availability boundary."""

        ingress = self._agent_selected_ingress
        dispatcher = self._dispatcher
        tools = self._selection_tools
        if (
            ingress is None
            or dispatcher is None
            or tools is None
            or ingress.selection_task is None
            or ingress.plan.selection_agent_configuration is None
        ):
            raise RuntimeError("agent-selected ingress is not configured")
        if self._agent_selected_ingress_executed:
            raise TrialInvariantError(
                "agent-selected ingress was scheduled more than once"
            )

        before_selection = self._barrier("before_external_selection")
        task = AgentTask(
            task_id=f"selection:{ingress.selection_task.selection_task_id}",
            scope=self.plan.run.scope,
            instruction=ingress.selection_task.instruction,
            retrieval_query=ingress.selection_task.retrieval_query,
            data_classification=AgentDataClassification.INTERNAL,
            metadata={
                "stage": "external_source_selection_v1",
                "selection_task_id": ingress.selection_task.selection_task_id,
                "delivery_family_id": ingress.plan.delivery_family_id,
            },
        )
        configuration = ingress.plan.selection_agent_configuration
        self._stage = TrialStage.SELECTION
        self._emit(TrialEventStatus.STARTED)
        agent = AgentRunner(
            self.adapter,
            self.decider,
            self.context_builder,
            tools,
            retrieval_observer=lambda retrieval: self._observe_selection_retrieval(
                task, configuration, retrieval
            ),
            context_observer=lambda context: self.artifacts.write_evidence(
                f"selection-prompt-{context.step_index:04d}", context
            ),
            decision_evidence_observer=lambda evidence: self._write_action_decision_diagnostic(
                "selection", evidence
            ),
        )
        agent_result = agent.run(task, configuration)
        self.artifacts.write_evidence(
            "selection-agent-result", agent_result, kind=ArtifactKind.AGENT_RESULT
        )
        for step in agent_result.steps:
            if step.tool_result is not None:
                self.artifacts.write_evidence(
                    f"selection-tool-step-{step.step_index:04d}", step.tool_result
                )
        after_selection = self._barrier("after_external_selection")
        self._compare("selection", before_selection, after_selection)

        executor = AgentSelectedIngressExecutor(
            ExternalIngestionBridge(dispatcher), ingress.delivery
        )
        attempts = []
        for step in agent_result.steps:
            if step.tool_result is None or step.tool_result.tool_name != "read_message":
                continue
            attempt = self._perform(
                TrialStage.INGRESS,
                f"selection-ingestion-attempt-{step.step_index:04d}",
                lambda step=step: executor.observe_read(
                    step.tool_result,
                    scope=self.plan.run.scope,
                    operation_id=(
                        f"{self.plan.run.run_id}:agent-selected-ingress:"
                        f"{ingress.delivery.delivery_id}:step-{step.step_index}"
                    ),
                    native_options=dict(self.plan.replay.native_options),
                ),
            )
            attempts.append(attempt)
        self._selection_result = summarize_agent_selected_stage(
            ingress.selection_task,
            ingress.delivery,
            agent_result,
            tuple(attempts),
        )
        self.artifacts.write_evidence("selection-result", self._selection_result)
        if executor.ingested:
            self._barrier("agent_selected_ingress")
        if agent_result.status == AgentRunStatus.FAILED:
            raise RuntimeError(
                agent_result.failure_message or "selection agent execution failed"
            )
        self._agent_selected_ingress_executed = True
        self._stage = TrialStage.SELECTION
        self._emit(TrialEventStatus.SUCCEEDED)
        self._stage = TrialStage.REPLAY

    def _observe_selection_retrieval(
        self,
        task: AgentTask,
        configuration: AgentRunConfiguration,
        retrieval: RetrievalResult,
    ) -> None:
        """Record the stage-private retrieval without conflating it with the probe."""

        self.artifacts.write_evidence("selection-retrieval", retrieval)
        self._require(
            retrieval.operation_id == f"{task.task_id}:agent_retrieve"
            and retrieval.query_sha256 == task.retrieval_query_sha256
            and retrieval.request_top_k == configuration.retrieval_top_k
            and retrieval.threshold == configuration.retrieval_threshold
            and all(hit.record.scope == task.scope for hit in retrieval.hits),
            "selection retrieval identity, settings, or scope do not match",
        )

    def _write_action_decision_diagnostic(
        self,
        phase: Literal["public", "selection"],
        evidence: Mapping[str, Any],
    ) -> None:
        """Seal one evaluator-only provider-shape record for a decision step."""

        step_index = evidence.get("step_index")
        if not isinstance(step_index, int) or step_index < 1:
            raise TrialInvariantError("action-decision diagnostic has invalid step index")
        expected_task_id = (
            self.plan.run.agent_task.task_id if phase == "public" else None
        )
        if expected_task_id is not None and evidence.get("task_id") != expected_task_id:
            raise TrialInvariantError(
                "public action-decision diagnostic has the wrong task ID"
            )
        self.artifacts.write_evidence(
            f"{phase}-action-decision-{step_index:04d}",
            dict(evidence),
            kind=ArtifactKind.ACTION_DECISION_DIAGNOSTIC,
        )

    def _seal_outbox_snapshot(self) -> None:
        """Persist one verified evaluator-only view of public simulated sends."""

        if self._outbox is None or self._outbox_snapshot_captured:
            return
        self._outbox_snapshot_captured = True
        snapshot = self._outbox.snapshot()
        self._validate_outbox_snapshot(snapshot)
        self.artifacts.write_evidence(
            "outbox-snapshot",
            snapshot,
            kind=ArtifactKind.OUTBOX_SNAPSHOT,
        )

    def _validate_outbox_snapshot(self, snapshot: MockOutboxSnapshot) -> None:
        """Link every sealed simulated send to approved public-agent evidence."""

        self._require(
            snapshot.scope == self.plan.run.scope,
            "outbox snapshot has the wrong memory scope",
        )
        result = self._agent_result
        if result is None:
            self._require(
                not snapshot.messages,
                "outbox has messages but the public agent produced no result",
            )
            return

        self._require(
            result.task == self.plan.run.agent_task,
            "outbox validation received a result for a different public task",
        )
        matching_steps: dict[str, tuple[ToolActionDecision, AgentStep]] = {}
        for step in result.steps:
            if not isinstance(step.decision, ToolActionDecision):
                continue
            call = step.decision.call
            if call.tool_name != "simulate_send":
                continue
            self._require(
                call.call_id not in matching_steps,
                "public agent reused a simulate_send call ID",
            )
            matching_steps[call.call_id] = (step.decision, step)

        for message in snapshot.messages:
            self._require(
                message.scope == self.plan.run.scope,
                "outbox message has the wrong memory scope",
            )
            self._require(
                message.task_id == result.task.task_id,
                "outbox message was not created by the public agent task",
            )
            pair = matching_steps.get(message.call_id)
            self._require(
                pair is not None,
                "outbox message has no matching public-agent tool step",
            )
            decision, step = pair
            call = decision.call
            self._require(
                step.policy_decision is not None
                and step.policy_decision.disposition == PolicyDisposition.ALLOW,
                "outbox message lacks an allowed policy decision",
            )
            self._require(
                step.approval_decision is not None
                and step.approval_decision.approved
                and step.policy_decision.approval_id
                == step.approval_decision.approval_id,
                "outbox message lacks a matching approved decision",
            )
            self._require(
                step.tool_result is not None
                and step.tool_result.status == ToolStatus.SUCCEEDED
                and step.tool_result.tool_name == "simulate_send",
                "outbox message lacks a successful simulate_send result",
            )
            self._require(
                call.idempotency_key == message.idempotency_key,
                "outbox idempotency key differs from the public tool call",
            )

            arguments = validate_simulate_send_arguments(dict(call.arguments))
            self._require(
                _canonical_json_sha256(arguments) == message.arguments_sha256,
                "outbox arguments hash differs from the public tool call",
            )
            self._require(
                arguments["recipient"] == message.recipient,
                "outbox recipient differs from the public tool call",
            )
            self._require(
                arguments.get("subject") == message.subject
                and arguments.get("body") == message.body
                and tuple(arguments.get("attachment_ids", []))
                == message.attachment_ids,
                "outbox content differs from the public tool call",
            )
            output = step.tool_result.output
            self._require(
                output.get("status") == "simulated"
                and output.get("message_id") == message.message_id,
                "successful simulate_send result does not match the outbox message",
            )

    def _execute_evaluation(self) -> None:
        """Run and persist the frozen read-only probe before lifecycle cleanup."""

        probe = self._evaluation_probe
        if probe is None:
            return
        self._stage = TrialStage.EVALUATE
        self._emit(TrialEventStatus.STARTED)
        execution = EvaluationRunner(self.adapter, self.plan.run.scope).run(
            trial_id=self.plan.projection.trial_id,
            twin=self.plan.run.twin,
            probe=probe,
            exposure=self._evaluation_exposure(),
            baseline_observation=self._evaluation_baseline_observation(),
        )
        write_evaluation_artifacts(self.artifacts.store, execution)
        self._require(
            execution.content_read_only_verified,
            "evaluation retrieval changed observable memory content",
        )
        self._emit(TrialEventStatus.SUCCEEDED)

    def _evaluation_exposure(self) -> EvaluationExposureEvidence:
        """Describe whether the trial offered its external payload to memory."""

        if self._forced_ingress is not None:
            return EvaluationExposureEvidence(
                status=(EvaluationExposureStatus.EXPOSED if self._forced_ingress_result
                        else EvaluationExposureStatus.EXPOSURE_FAILED),
                ingress_mode=IngressMode.FORCED_INGESTION.value,
            )
        if self._agent_selected_ingress is not None:
            result = self._selection_result
            exposed = bool(result and result.outcome.value == "ingested")
            return EvaluationExposureEvidence(
                status=(EvaluationExposureStatus.EXPOSED if exposed
                        else EvaluationExposureStatus.NOT_EXPOSED),
                ingress_mode=IngressMode.AGENT_SELECTED.value,
                detail=None if result is None else result.outcome.value,
            )
        return EvaluationExposureEvidence(
            status=EvaluationExposureStatus.NOT_APPLICABLE,
        )

    def _evaluation_baseline_observation(self) -> BaselineObservation | None:
        """Translate pre-ingress verification into evaluator-only evidence."""

        baseline = self._trusted_baseline
        verification = self._trusted_baseline_verification
        if baseline is None or verification is None:
            return None
        return BaselineObservation(
            baseline_id=baseline.baseline_id,
            expected_markers=baseline.required_markers,
            observed_markers=verification.observed_markers,
            retrieved_before_ingress=verification.was_retrieved_before_ingress,
            retrieval_evidence_sha256=verification.retrieval_result_sha256,
        )

    def _perform(
        self,
        stage: TrialStage,
        name: str,
        action: Callable[[], Any],
        *,
        kind: ArtifactKind = ArtifactKind.OTHER,
    ) -> Any:
        self._stage = stage
        self._emit(TrialEventStatus.STARTED)
        value = action()
        if value is not None:
            self.artifacts.write_evidence(name, value, kind=kind)
        self._emit(TrialEventStatus.SUCCEEDED)
        return value

    def _barrier(self, phase: str) -> MemorySnapshot:
        flush = self._perform(
            TrialStage.FLUSH, f"flush-{phase}", self.adapter.flush_and_wait
        )
        if not flush.completed:
            raise TimeoutError("pending memory writes did not become visible")
        self._require(
            all(
                record.scope == self.plan.run.scope for record in flush.visible_records
            ),
            "flush returned records from another memory scope",
        )
        evidence = self._perform(
            TrialStage.QUIESCENCE,
            f"quiescence-{phase}",
            lambda: wait_for_quiescence(
                self.adapter.inspect_state,
                self.plan.quiescence,
                phase=phase,
                expected_scope=self.plan.run.scope,
            ),
        )
        if not evidence.stabilized:
            raise TimeoutError("memory state did not reach quiescence")
        self._validate_snapshot(evidence.final_snapshot)
        self.artifacts.write_evidence(
            f"snapshot-{phase}", evidence.final_snapshot, kind=ArtifactKind.SNAPSHOT
        )
        return evidence.final_snapshot

    def _observe_retrieval(self, retrieval: RetrievalResult) -> None:
        self._stage = TrialStage.RETRIEVE
        self.artifacts.write_evidence("retrieval", retrieval)
        task = self.plan.run.agent_task
        config = self.plan.run.agent_configuration
        self._require(
            retrieval.operation_id == f"{task.task_id}:agent_retrieve"
            and retrieval.query_sha256 == task.retrieval_query_sha256
            and retrieval.request_top_k == config.retrieval_top_k
            and retrieval.threshold == config.retrieval_threshold
            and all(hit.record.scope == task.scope for hit in retrieval.hits),
            "retrieval identity, settings, or scope do not match the trial",
        )
        self._emit(TrialEventStatus.SUCCEEDED)
        self._after_retrieval = self._barrier("after_retrieval")
        assert self._before_probe is not None
        self._compare("retrieval", self._before_probe, self._after_retrieval)
        self._stage = TrialStage.RESPOND
        self._emit(TrialEventStatus.STARTED)

    def _observe_context(self, context: AgentDecisionContext) -> None:
        self.artifacts.write_evidence(f"prompt-{context.step_index:04d}", context)

    def _compare(
        self,
        phase: Literal["selection", "retrieval", "response"],
        before: MemorySnapshot,
        after: MemorySnapshot,
    ) -> None:
        self._stage = TrialStage.VERIFY_READ_ONLY
        delta = compare_snapshot_state(before, after)
        self._deltas[phase] = delta
        self.artifacts.write_evidence(f"state-delta-{phase}", delta)
        self._require(
            delta.observed_content_unchanged,
            f"{phase} changed observable memory content",
        )
        self._emit(
            TrialEventStatus.SUCCEEDED,
            payload={
                "phase": phase,
                "comparison_level": delta.comparison_level,
                "observed_content_unchanged": (
                    delta.observed_content_unchanged
                ),
                "observed_full_state_unchanged": (
                    delta.observed_full_state_unchanged
                ),
                "content_unchanged": delta.content_unchanged,
                "full_state_unchanged": delta.full_state_unchanged,
                "before_reported_total_count": (
                    delta.before_reported_total_count
                ),
                "after_reported_total_count": (
                    delta.after_reported_total_count
                ),
            },
        )

    def _validate_snapshot(self, snapshot: MemorySnapshot) -> None:
        correct_scope = (
            snapshot.scope == self.plan.run.scope
            and all(
                record.scope == self.plan.run.scope
                for record in snapshot.records
            )
        )
        self._require(correct_scope, "snapshot contains a different memory scope")

        if self.plan.quiescence.enumeration_policy == "exact":
            self._require(snapshot.complete, "snapshot enumeration is incomplete")

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise TrialInvariantError(message)

    def _cleanup(self) -> None:
        self._replay_stage = None
        if self._owns_scope and self.plan.cleanup_storage:
            self._stage = TrialStage.RESET
            try:
                reset = self.adapter.reset_and_verify_empty()
                self.artifacts.write_evidence("reset", reset)
                self._require(
                    reset.scope == self.plan.run.scope and reset.empty,
                    "cleanup did not verify an empty trial scope",
                )
                self._emit(TrialEventStatus.SUCCEEDED)
            except Exception as exc:
                self._record_failure(exc)
        self._stage = TrialStage.CLOSE
        try:
            self.adapter.close()
            self._emit(TrialEventStatus.SUCCEEDED)
        except Exception as exc:
            self._record_failure(exc)

    def _failure(self, error: Exception | str) -> TrialFailure:
        if isinstance(error, str):
            # COMPLETE emits a compact summary event after one or more concrete
            # failures have already been recorded. A summary message is not an
            # exception and therefore has no ``__cause__`` chain to inspect.
            category = (
                self._failures[0].failure_class
                if self._failures
                else ChildFailureClass.ADAPTER_FAILURE
            )
            return TrialFailure(
                stage=self._stage,
                failure_class=category,
                error_type="TrialFailureSummary",
                error_message=error,
            )

        category = ChildFailureClass.ADAPTER_FAILURE
        if isinstance(error, CampaignArtifactError):
            category = ChildFailureClass.ARTIFACT_FAILURE
        elif self._stage == TrialStage.PREFLIGHT:
            category = ChildFailureClass.CONFIGURATION_INVALID
        elif self._stage in {TrialStage.RESPOND, TrialStage.RETRIEVE}:
            category = ChildFailureClass.AGENT_FAILURE
        return TrialFailure(
            stage=self._stage,
            failure_class=category,
            error_type=type(error).__name__,
            error_message=str(error) or type(error).__name__,
            causes=exception_causes(error),
        )

    def _record_failure(self, error: Exception) -> None:
        self._invalidated |= isinstance(error, TrialInvariantError)
        failure = self._failure(error)
        self._failures.append(failure)
        try:
            self._emit(TrialEventStatus.FAILED, failure=failure)
        except Exception as logging_error:
            self._failures.append(
                TrialFailure(
                    stage=self._stage,
                    failure_class=ChildFailureClass.ARTIFACT_FAILURE,
                    error_type=type(logging_error).__name__,
                    error_message=str(logging_error) or "trial event logging failed",
                )
            )

    def _emit(
        self,
        status: TrialEventStatus,
        *,
        failure: TrialFailure | None = None,
        operation_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        event = TrialEvent(
            sequence=self._sequence + 1,
            run_id=self.plan.run.run_id,
            trial_id=self.plan.projection.trial_id,
            stage=self._stage,
            status=status,
            replay_stage=self._replay_stage,
            operation_id=operation_id,
            payload=payload or {},
            failure=failure,
        )
        self.artifacts.emit(event)
        self._sequence += 1

    def _result(self, started: datetime) -> TrialResult:
        return TrialResult(
            plan=self.plan,
            status=(
                ChildRunStatus.INVALIDATED
                if self._invalidated
                else ChildRunStatus.FAILED
                if self._failures
                else ChildRunStatus.SUCCEEDED
            ),
            replay_results=self._replay.results if self._replay is not None else (),
            selection_result=self._selection_result,
            trusted_baseline_verification=self._trusted_baseline_verification,
            agent_result=self._agent_result,
            provenance=self._provenance,
            failures=tuple(self._failures),
            state_deltas=self._deltas,
            started_at=started,
            finished_at=datetime.now(UTC),
        )

    def _validate_trusted_baseline(self) -> None:
        """Reject a seed that cannot be safely established before ingress."""

        baseline = self._trusted_baseline
        if baseline is None:
            return
        if baseline.persona_id != int(self.plan.run.persona_id):
            raise ValueError(
                "trusted baseline persona does not match the current trial"
            )
        source_indexes = {
            event.source_message_index
            for event in self.plan.projection.replay_events
            if event.source_message_index is not None
        }
        if baseline.available_after_source_message_index not in source_indexes:
            raise ValueError("trusted baseline schedule is absent from replay")
        ingress_indexes = tuple(
            ingress.delivery.available_after_source_message_index
            for ingress in (self._forced_ingress, self._agent_selected_ingress)
            if ingress is not None
        )
        if any(
            baseline.available_after_source_message_index >= ingress_index
            for ingress_index in ingress_indexes
        ):
            raise ValueError(
                "trusted baseline must be scheduled before external ingress"
            )

    def _schedule_write_boundaries(self) -> tuple[int, ...]:
        """Split replay after every scheduled non-replay memory write."""

        values = list(self._ingress_boundaries())
        if self._trusted_baseline is not None:
            values.append(self._trusted_baseline.available_after_source_message_index)
        return tuple(sorted(set(values)))

    def _trusted_baseline_is_due(self, replay_result) -> bool:
        baseline = self._trusted_baseline
        if baseline is None or self._trusted_baseline_executed:
            return False
        return (
            replay_result.batch.events[-1].source_message_index
            == baseline.available_after_source_message_index
        )

    def _execute_trusted_baseline(self) -> None:
        """Write and verify a frozen trusted seed before any external attack."""

        baseline = self._trusted_baseline
        if baseline is None:
            raise RuntimeError("trusted baseline is not configured")
        if self._trusted_baseline_executed:
            raise TrialInvariantError("trusted baseline was scheduled more than once")
        if self._forced_ingress_executed or self._agent_selected_ingress_executed:
            raise TrialInvariantError("trusted baseline cannot execute after ingress")

        operation_id = f"{self.plan.run.run_id}:trusted-baseline:{baseline.baseline_id}"
        dispatch = self._perform(
            TrialStage.INGRESS,
            "trusted-baseline-direct-write",
            lambda: PolicyDispatchResult(
                operation_id=operation_id,
                policy=WritePolicy.DIRECT,
                action=PolicyDispatchAction.DIRECT_WRITE,
                write_results=(
                    self.adapter.write_direct(
                        DirectWriteRequest(
                            operation_id=operation_id,
                            scope=self.plan.run.scope,
                            text=baseline.content,
                            source_event_id=(
                                f"trusted-baseline:{baseline.baseline_id}"
                            ),
                            metadata={
                                "content_trust": "trusted_baseline",
                                "baseline_id": baseline.baseline_id,
                                "content_sha256": baseline.content_sha256,
                                "seed_policy": baseline.seed_policy.value,
                            },
                        )
                    ),
                ),
            ),
        )
        self._require(
            dispatch.operation_id == operation_id
            and dispatch.policy == WritePolicy.DIRECT
            and dispatch.action == PolicyDispatchAction.DIRECT_WRITE
            and all(
                result.operation_id == operation_id
                for result in dispatch.write_results
            ),
            "trusted baseline dispatch does not match the bound seed",
        )
        self._trusted_baseline_executed = True
        self._barrier("trusted_baseline")
        self._verify_trusted_baseline()
        self._stage = TrialStage.REPLAY

    def _verify_trusted_baseline(self) -> None:
        """Prove seed visibility and read-only verification before ingress."""

        baseline = self._trusted_baseline
        if baseline is None or not self._trusted_baseline_executed:
            raise RuntimeError("trusted baseline has not been written")
        if self._forced_ingress_executed or self._agent_selected_ingress_executed:
            raise TrialInvariantError(
                "trusted baseline verification occurred after ingress"
            )
        before = self._perform(
            TrialStage.SNAPSHOT,
            "snapshot-before-trusted-baseline-verification",
            self.adapter.inspect_state,
            kind=ArtifactKind.SNAPSHOT,
        )
        self._validate_snapshot(before)
        request = RetrievalRequest(
            operation_id=(
                f"{self.plan.run.run_id}:trusted-baseline:"
                f"{baseline.baseline_id}:verify"
            ),
            scope=self.plan.run.scope,
            query=baseline.retrieval_query,
            top_k=max(10, len(baseline.required_markers)),
        )
        retrieval = self._perform(
            TrialStage.RETRIEVE,
            "trusted-baseline-retrieval",
            lambda: self.adapter.retrieve(request),
        )
        self._require(
            retrieval.operation_id == request.operation_id
            and retrieval.query_sha256 == request.query_sha256
            and retrieval.request_top_k == request.top_k
            and all(hit.record.scope == self.plan.run.scope for hit in retrieval.hits),
            "trusted baseline retrieval identity or scope does not match",
        )
        after = self._barrier("trusted_baseline_verification")
        delta = compare_snapshot_state(before, after)
        self.artifacts.write_evidence(
            "state-delta-trusted-baseline-verification", delta
        )
        self._require(
            delta.observed_content_unchanged,
            "trusted baseline verification changed observable memory content",
        )
        retrieved_text = "\n".join(hit.record.text for hit in retrieval.hits)
        observed = tuple(
            marker for marker in baseline.required_markers if marker in retrieved_text
        )
        forbidden = tuple(
            marker
            for marker in baseline.forbidden_pre_ingress_markers
            if marker in retrieved_text
        )
        self._trusted_baseline_verification = TrustedBaselineVerification(
            baseline_id=baseline.baseline_id,
            was_written=True,
            was_retrieved_before_ingress=(
                set(observed) == set(baseline.required_markers) and not forbidden
            ),
            retrieval_result_sha256=hashlib.sha256(
                canonical_json_bytes(retrieval)
            ).hexdigest(),
            observed_markers=observed,
        )
        self.artifacts.write_evidence(
            "trusted-baseline-verification",
            self._trusted_baseline_verification,
            kind=ArtifactKind.TRUSTED_BASELINE,
        )
        self._validate_baseline_verification()

    def _validate_baseline_verification(self) -> None:
        """Do not let a missing seed be misreported as attack resistance."""

        baseline = self._trusted_baseline
        verification = self._trusted_baseline_verification
        if baseline is None:
            return
        self._require(
            verification is not None
            and verification.baseline_id == baseline.baseline_id
            and verification.was_written
            and verification.was_retrieved_before_ingress
            and set(verification.observed_markers) == set(baseline.required_markers),
            "trusted baseline was not established before external ingress",
        )
