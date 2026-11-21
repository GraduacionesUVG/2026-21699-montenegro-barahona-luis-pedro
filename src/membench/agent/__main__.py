"""One-request/one-result child entry point for an isolated campaign run.

The launcher writes one ChildRunRequest JSON document to stdin. For a valid
request, this process always emits exactly one ChildRunResult JSON document to
stdout and returns zero, even when validation or trial execution fails. Invalid
stdin is a protocol failure: it is diagnosed on stderr and returns non-zero.

Diagnostics and incidental dependency output are redirected to stderr. Frozen
input hashes, preparation bindings, public projection content, and the complete
trial plan are verified before a memory adapter is initialized.
"""

from __future__ import annotations

import contextlib
import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from membench.agent.approval import SyntheticSendApprovalProvider
from membench.agent.protocol import ApprovalProvider
from membench.behavioral.loading import (
    BehavioralInputLoadError,
    validate_behavioral_pair,
)
from membench.behavioral.models import (
    BehavioralTaskTemplate,
    CaseSummaryBehavioralOracle,
)
from membench.campaign.artifacts import CampaignArtifactStore, canonical_json_bytes
from membench.campaign.bindings import CampaignRunBindings, RunProjectionBinding
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
    ChildFailureClass,
    ChildRunRequest,
    ChildRunResult,
    ChildRunStatus,
)
from membench.campaign.preparation_models import CampaignPreparationOptions
from membench.evaluation.models import EvaluationProbe
from membench.ingress.models import (
    ExternalSourceKind,
    IngressMode,
    TrustedBaselineManifest,
    TrustedBaselineSeed,
)
from membench.ingress.resolution import ResolvedIngress, resolve_frozen_ingress
from membench.tools.fixtures import (
    ToolFixtureBundle,
    ToolFixtureError,
    ToolRuntime,
    build_tool_runtime,
    fixture_required,
    parse_tool_fixture,
)
from membench.tools.mock_inbox import MockInbox, MockInboxConfigurationError
from membench.tools.registry import ToolRegistry
from membench.trial.artifacts import TrialArtifactStore
from membench.trial.bootstrap import PreparedTrial, bootstrap_trial, load_trial_plan
from membench.trial.models import TrialPlan
from membench.trial.runner import TrialArtifactFailure

_MAX_REQUEST_BYTES = 16 * 1024 * 1024


class ChildEntrypointError(RuntimeError):
    """A valid request could not be resolved into a trusted isolated trial."""


class TrialBootstrap(Protocol):
    def __call__(self, plan: TrialPlan) -> PreparedTrial: ...


@dataclass(frozen=True)
class ResolvedChildRequest:
    """Trusted paths and contracts resolved from one verified request."""

    request: ChildRunRequest
    project_root: Path
    campaign_root: Path
    trial_directory: Path
    options: CampaignPreparationOptions
    bindings: CampaignRunBindings
    binding: RunProjectionBinding
    plan: TrialPlan
    tool_fixture: ToolFixtureBundle | None = field(repr=False)
    ingress: ResolvedIngress | None = field(default=None, repr=False)
    trusted_baseline: TrustedBaselineSeed | None = field(default=None, repr=False)
    evaluation_probe: EvaluationProbe | None = field(default=None, repr=False)
    behavioral_task: BehavioralTaskTemplate | None = field(default=None, repr=False)
    behavioral_oracle: CaseSummaryBehavioralOracle | None = field(
        default=None, repr=False
    )


def resolve_child_request(
    request: ChildRunRequest, *, project_root: str | Path
) -> ResolvedChildRequest:
    """Verify frozen evidence and reconstruct the exact bound trial plan.

    No adapter, model, tokenizer, tool, or backend is initialized here.
    """
    request = ChildRunRequest.model_validate(request.model_dump(mode="json"))
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ChildEntrypointError("project root is not an existing directory")

    child_directory = (root / request.artifact_directory).resolve()
    if child_directory == root or not child_directory.is_relative_to(root):
        raise ChildEntrypointError("child artifact directory escapes the project root")
    if (
        child_directory.name != request.run.run_id
        or child_directory.parent.name != "children"
    ):
        raise ChildEntrypointError("child artifact directory must be children/<run_id>")
    campaign_root = child_directory.parent.parent
    if campaign_root == root or not campaign_root.is_relative_to(root):
        raise ChildEntrypointError("campaign root must remain below the project root")
    if not campaign_root.is_dir() or not child_directory.is_dir():
        raise ChildEntrypointError("prepared campaign or child directory is missing")
    trial_directory = child_directory / "trial"
    if trial_directory.exists() or trial_directory.is_symlink():
        raise ChildEntrypointError("trial evidence directory already exists")

    store = CampaignArtifactStore.open_existing(campaign_root)
    for reference in request.frozen_inputs:
        store.verify(reference)
    options_reference = _single_reference(request, artifact_id="preparation:options")
    bindings_reference = _single_reference(
        request,
        artifact_id="preparation:run-bindings",
        kind=ArtifactKind.RUN_BINDINGS,
    )
    options = CampaignPreparationOptions.model_validate_json(
        store.verify(options_reference).read_bytes()
    )
    bindings = CampaignRunBindings.model_validate_json(
        store.verify(bindings_reference).read_bytes()
    )
    expected_campaign_root = (root / options.artifact_root).resolve()
    if (
        expected_campaign_root != campaign_root
        or request.run.campaign_id != options.campaign_id
    ):
        raise ChildEntrypointError(
            "request campaign or artifact root differs from preparation"
        )
    bindings.validate_preparation(options)
    binding = bindings.binding_for(request.run.run_id)
    if request.cleanup_storage != binding.cleanup_storage:
        raise ChildEntrypointError("request cleanup policy differs from binding")
    if bindings.projection_manifest not in request.frozen_inputs:
        raise ChildEntrypointError("bound projection manifest is not a child input")
    projection_manifest = store.verify(bindings.projection_manifest)
    plan = load_trial_plan(
        request.run,
        projection_directory=projection_manifest.parent,
        trial_id=binding.trial_id,
        replay=binding.replay,
        quiescence=binding.quiescence,
        purpose=binding.purpose,
        cleanup_storage=binding.cleanup_storage,
    )
    binding.validate_trial_plan(plan)
    tool_fixture = _resolve_tool_fixture(request, store=store, plan=plan)
    ingress = resolve_frozen_ingress(
        store=store,
        frozen_inputs=request.frozen_inputs,
        bindings=bindings,
        binding=binding,
        run=request.run,
        fixture=tool_fixture,
    )
    trusted_baseline = _resolve_trusted_baseline(
        request,
        store=store,
        bindings=bindings,
        binding=binding,
        plan=plan,
    )
    behavioral_task, behavioral_oracle = _resolve_behavioral_inputs(
        request,
        store=store,
        bindings=bindings,
        binding=binding,
        plan=plan,
        ingress=ingress,
    )
    evaluation_probe = _resolve_evaluation_probe(
        request,
        store=store,
        bindings=bindings,
        binding=binding,
        plan=plan,
        ingress=ingress,
    )
    return ResolvedChildRequest(
        request=request,
        project_root=root,
        campaign_root=campaign_root,
        trial_directory=trial_directory,
        options=options,
        bindings=bindings,
        binding=binding,
        plan=plan,
        tool_fixture=tool_fixture,
        ingress=ingress,
        trusted_baseline=trusted_baseline,
        evaluation_probe=evaluation_probe,
        behavioral_task=behavioral_task,
        behavioral_oracle=behavioral_oracle,
    )


def execute_child_request(
    request: ChildRunRequest,
    *,
    project_root: str | Path,
    bootstrap: TrialBootstrap = bootstrap_trial,
    tools_factory: Callable[[TrialPlan], ToolRegistry] | None = None,
) -> ChildRunResult:
    """Resolve, bootstrap, and execute one trial with sealed evidence.

    Dependency injection is intentionally limited to offline tests. Production
    uses the normal adapter registry, environment settings, Ollama responder,
    and configured tokenizer from bootstrap_trial().
    """
    started = datetime.now(UTC)
    stage = "resolve_request"
    try:
        resolved = resolve_child_request(request, project_root=project_root)
        stage = "configure_tools"
        selected_inbox = _agent_selected_inbox(resolved)
        tool_runtime: ToolRuntime | None = None
        if tools_factory is not None:
            tools = tools_factory(resolved.plan)
        else:
            tool_runtime = _default_tools(
                resolved.plan,
                fixture=resolved.tool_fixture,
            )
            tools = tool_runtime.registry
        approval_provider = _public_approval_provider(resolved.plan, tools)
        selection_tools = (
            None
            if selected_inbox is None
            else _selection_tools(
                resolved.plan,
                ingress=resolved.ingress,
                fixture=resolved.tool_fixture,
                inbox=selected_inbox,
            )
        )
        stage = "bootstrap_trial"
        prepared = bootstrap(resolved.plan)
        stage = "execute_trial"
        artifacts = TrialArtifactStore.prepare(resolved.trial_directory)
        try:
            runner_options = {"tools": tools}
            if approval_provider is not None:
                runner_options["approval_provider"] = approval_provider
            if tool_runtime is not None and tool_runtime.outbox is not None:
                runner_options["outbox"] = tool_runtime.outbox
            if resolved.ingress is not None:
                if resolved.ingress.plan.ingress_mode == IngressMode.FORCED_INGESTION:
                    runner_options["forced_ingress"] = resolved.ingress
                elif selected_inbox is not None:
                    runner_options["agent_selected_ingress"] = resolved.ingress
                    runner_options["selection_tools"] = selection_tools
                    runner_options["replay_source_index_observer"] = (
                        selected_inbox.advance_to_source_message_index
                    )
            if resolved.evaluation_probe is not None:
                runner_options["evaluation_probe"] = resolved.evaluation_probe
            if resolved.trusted_baseline is not None:
                runner_options["trusted_baseline"] = resolved.trusted_baseline
            result = prepared.create_runner(artifacts, **runner_options).run()
        except TrialArtifactFailure as exc:
            return artifacts.partial_child_result(
                request, exc.result, campaign_root=resolved.campaign_root
            )
        stage = "build_child_result"
        return artifacts.child_result(
            request, result, campaign_root=resolved.campaign_root
        )
    except Exception as exc:
        return _failed_result(request, stage=stage, error=exc, started_at=started)


def main(argv: list[str] | None = None) -> int:
    """Read stdin and emit only canonical result JSON on stdout."""
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        _diagnostic("command-line arguments are not supported")
        return 2
    try:
        payload = sys.stdin.read(_MAX_REQUEST_BYTES + 1)
        if len(payload.encode("utf-8")) > _MAX_REQUEST_BYTES:
            raise ValueError("child request exceeds the size limit")
        if not payload.strip():
            raise ValueError("child request stdin is empty")
        request = ChildRunRequest.model_validate_json(payload)
    except (UnicodeError, ValueError, ValidationError) as exc:
        _diagnostic(f"invalid child request ({type(exc).__name__})")
        return 2

    # Vendor SDKs may write progress text. Keep stdout as a single JSON value.
    with contextlib.redirect_stdout(sys.stderr):
        result = execute_child_request(request, project_root=Path.cwd())
    sys.stdout.write(result.model_dump_json() + "\n")
    sys.stdout.flush()
    return 0


def _single_reference(
    request: ChildRunRequest,
    *,
    artifact_id: str,
    kind: ArtifactKind | None = None,
) -> ArtifactReference:
    matches = tuple(
        reference
        for reference in request.frozen_inputs
        if reference.artifact_id == artifact_id
        and (kind is None or reference.kind == kind)
    )
    if len(matches) != 1:
        raise ChildEntrypointError(
            f"child inputs require exactly one {artifact_id} artifact"
        )
    return matches[0]


def _resolve_tool_fixture(
    request: ChildRunRequest,
    *,
    store: CampaignArtifactStore,
    plan: TrialPlan,
) -> ToolFixtureBundle | None:
    """Load the one exact fixture artifact allowed to define discovery data."""

    references = tuple(
        reference
        for reference in request.frozen_inputs
        if reference.kind == ArtifactKind.TOOL_FIXTURE
    )
    if len(references) > 1:
        raise ChildEntrypointError("child inputs contain multiple tool fixtures")
    if not references:
        if fixture_required(plan.run.agent_configuration.allowed_tools):
            raise ChildEntrypointError(
                "selected discovery tools require a frozen tool fixture"
            )
        return None

    reference = references[0]
    if (
        reference.artifact_id != "inputs:tool-fixture"
        or reference.relative_path != "inputs/tools/fixture.json"
        or reference.media_type != "application/json"
    ):
        raise ChildEntrypointError("tool fixture reference has an unexpected identity")
    try:
        fixture = parse_tool_fixture(store.verify(reference).read_bytes())
        if fixture_required(plan.run.agent_configuration.allowed_tools):
            fixture.validate_for_scope(
                plan.run.scope, plan.run.agent_configuration.allowed_tools
            )
    except ToolFixtureError as exc:
        raise ChildEntrypointError(f"invalid frozen tool fixture: {exc}") from exc
    return fixture


def _resolve_behavioral_inputs(
    request: ChildRunRequest,
    *,
    store: CampaignArtifactStore,
    bindings: CampaignRunBindings,
    binding: RunProjectionBinding,
    plan: TrialPlan,
    ingress: ResolvedIngress | None,
) -> tuple[BehavioralTaskTemplate | None, CaseSummaryBehavioralOracle | None]:
    """Resolve a task for the agent while retaining its oracle as private evidence."""

    if not binding.has_behavioral_task:
        if (
            bindings.behavioral_task is not None
            or bindings.behavioral_oracle is not None
        ):
            raise ChildEntrypointError("campaign has unbound behavioral artifacts")
        return None, None
    task_reference = bindings.behavioral_task
    oracle_reference = bindings.behavioral_oracle
    if (
        task_reference is None
        or oracle_reference is None
        or task_reference not in request.frozen_inputs
        or oracle_reference not in request.frozen_inputs
    ):
        raise ChildEntrypointError("bound behavioral inputs are not child inputs")
    try:
        task = BehavioralTaskTemplate.model_validate_json(
            store.verify(task_reference).read_bytes()
        )
        oracle = CaseSummaryBehavioralOracle.model_validate_json(
            store.verify(oracle_reference).read_bytes()
        )
        validate_behavioral_pair(task, oracle)
    except (BehavioralInputLoadError, OSError, ValueError) as exc:
        raise ChildEntrypointError("could not load frozen behavioral inputs") from exc
    if (
        task.behavioral_task_id != binding.behavioral_task_id
        or task_reference.sha256 != binding.behavioral_task_sha256
        or oracle.oracle_id != binding.behavioral_oracle_id
        or oracle_reference.sha256 != binding.behavioral_oracle_sha256
        or task.persona_id != plan.run.persona_id
    ):
        raise ChildEntrypointError(
            "behavioral inputs do not match the bound run identity"
        )
    if ingress is None or (
        task.attack_spec_id != ingress.plan.attack_spec_id
        or task.delivery_family_id != ingress.plan.delivery_family_id
    ):
        raise ChildEntrypointError(
            "behavioral task does not match the bound external delivery"
        )
    expected_task = task.instantiate(
        plan.run.scope,
        task_id=plan.run.agent_task.task_id,
    )
    if plan.run.agent_task != expected_task:
        raise ChildEntrypointError(
            "run-scoped agent task does not match the frozen behavioral task"
        )
    return task, oracle


def _resolve_evaluation_probe(
    request: ChildRunRequest,
    *,
    store: CampaignArtifactStore,
    bindings: CampaignRunBindings,
    binding: RunProjectionBinding,
    plan: TrialPlan,
    ingress: ResolvedIngress | None,
) -> EvaluationProbe | None:
    """Load only the evaluator probe exactly bound to this isolated run."""

    if not binding.has_evaluation_probe:
        if bindings.evaluation_probe is not None:
            raise ChildEntrypointError(
                "campaign has an unbound evaluation-probe artifact"
            )
        return None
    reference = bindings.evaluation_probe
    if reference is None or reference not in request.frozen_inputs:
        raise ChildEntrypointError("bound evaluation probe is not a child input")
    try:
        content = store.verify(reference).read_bytes()
        probe = EvaluationProbe.model_validate_json(content)
    except (OSError, ValueError) as exc:
        raise ChildEntrypointError("could not load frozen evaluation probe") from exc
    if (
        probe.probe_id != binding.evaluation_probe_id
        or reference.sha256 != binding.evaluation_probe_sha256
        or probe.persona_id != plan.run.persona_id
    ):
        raise ChildEntrypointError(
            "evaluation probe does not match the bound run identity"
        )
    if ingress is not None and (
        probe.attack_spec_id != ingress.plan.attack_spec_id
        or probe.delivery_family_id != ingress.plan.delivery_family_id
    ):
        raise ChildEntrypointError(
            "evaluation probe does not match the bound external delivery"
        )
    return probe


def _resolve_trusted_baseline(
    request: ChildRunRequest,
    *,
    store: CampaignArtifactStore,
    bindings: CampaignRunBindings,
    binding: RunProjectionBinding,
    plan: TrialPlan,
) -> TrustedBaselineSeed | None:
    """Load the one sealed pre-ingress seed bound to this run, if any.

    The manifest is campaign-level evidence while the identifier and canonical
    seed digest are run-level bindings.  Checking both keeps a valid manifest
    from being substituted with another persona's otherwise-valid seed.
    """

    if not binding.has_trusted_baseline:
        if bindings.trusted_baseline_manifest is not None:
            raise ChildEntrypointError(
                "campaign has an unbound trusted-baseline manifest"
            )
        return None

    reference = bindings.trusted_baseline_manifest
    if (
        reference is None
        or reference not in request.frozen_inputs
        or reference.kind != ArtifactKind.TRUSTED_BASELINE
        or reference.visibility != ArtifactVisibility.EVALUATOR_ONLY
        or reference.media_type != "application/json"
    ):
        raise ChildEntrypointError(
            "bound trusted baseline is not evaluator-only frozen JSON evidence"
        )
    try:
        # verify() checks the sealed artifact digest before its JSON is parsed.
        manifest = TrustedBaselineManifest.model_validate_json(
            store.verify(reference).read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise ChildEntrypointError(
            "could not load frozen trusted baseline manifest"
        ) from exc

    matches = tuple(
        seed
        for seed in manifest.baselines
        if seed.persona_id == int(plan.run.persona_id)
    )
    if len(matches) != 1:
        raise ChildEntrypointError(
            "frozen trusted baseline does not contain one seed for the run persona"
        )
    seed = matches[0]
    seed_sha256 = hashlib.sha256(canonical_json_bytes(seed)).hexdigest()
    if (
        seed.baseline_id != binding.trusted_baseline_id
        or seed_sha256 != binding.trusted_baseline_sha256
    ):
        raise ChildEntrypointError(
            "frozen trusted baseline does not match the bound run identity"
        )
    return seed


def _agent_selected_inbox(resolved: ResolvedChildRequest) -> MockInbox | None:
    """Build the one frozen email inbox permitted for agent-selected ingress."""

    ingress = resolved.ingress
    if ingress is None or ingress.plan.ingress_mode != IngressMode.AGENT_SELECTED:
        return None
    if ingress.selection_task is None:
        raise ChildEntrypointError("agent-selected ingress has no selection task")
    configuration = ingress.plan.selection_agent_configuration
    if configuration is None:
        raise ChildEntrypointError(
            "agent-selected ingress has no frozen selection configuration"
        )
    configured = set(configuration.allowed_tools)
    required = set(ingress.plan.allowed_tools)
    if not required <= configured:
        raise ChildEntrypointError(
            "agent-selected ingress tools are absent from the agent configuration"
        )
    if ingress.delivery.source_kind != ExternalSourceKind.EMAIL:
        raise ChildEntrypointError(
            "agent-selected runtime currently supports email only"
        )
    try:
        return MockInbox((ingress.delivery,))
    except MockInboxConfigurationError as exc:
        raise ChildEntrypointError("could not configure frozen mock inbox") from exc


def _default_tools(
    plan: TrialPlan,
    *,
    fixture: ToolFixtureBundle | None = None,
) -> ToolRuntime:
    """Construct one complete per-run trusted tool runtime."""

    try:
        return build_tool_runtime(
            scope=plan.run.scope,
            allowed_tools=plan.run.agent_configuration.allowed_tools,
            fixture=fixture,
        )
    except ToolFixtureError as exc:
        raise ChildEntrypointError(f"could not configure trial tools: {exc}") from exc


def _public_approval_provider(
    plan: TrialPlan,
    tools: ToolRegistry,
) -> ApprovalProvider | None:
    """Bind deterministic approval only to the public simulated-send tool."""

    if "simulate_send" not in plan.run.agent_configuration.allowed_tools:
        return None
    definition = tools.resolve("simulate_send")
    if definition is None:
        raise ChildEntrypointError(
            "simulate_send is allowed but has no registered tool definition"
        )
    return SyntheticSendApprovalProvider(plan.run.scope, definition)


def _selection_tools(
    plan: TrialPlan,
    *,
    ingress: ResolvedIngress | None,
    fixture: ToolFixtureBundle | None,
    inbox: MockInbox,
) -> ToolRegistry:
    """Construct the isolated inbox-only registry for a frozen selection task."""

    if ingress is None or ingress.plan.selection_agent_configuration is None:
        raise ChildEntrypointError("agent-selected ingress is not fully resolved")
    try:
        return build_tool_runtime(
            scope=plan.run.scope,
            allowed_tools=ingress.plan.selection_agent_configuration.allowed_tools,
            fixture=fixture,
            inbox=inbox,
        ).registry
    except ToolFixtureError as exc:
        raise ChildEntrypointError(
            f"could not configure selection tools: {exc}"
        ) from exc


def _failed_result(
    request: ChildRunRequest,
    *,
    stage: str,
    error: Exception,
    started_at: datetime,
) -> ChildRunResult:
    failure_class = (
        ChildFailureClass.CONFIGURATION_INVALID
        if stage in {"resolve_request", "configure_tools", "bootstrap_trial"}
        else ChildFailureClass.ARTIFACT_FAILURE
        if stage == "build_child_result"
        else ChildFailureClass.UNEXPECTED_ERROR
    )
    return ChildRunResult(
        request_id=request.request_id,
        run=request.run,
        status=ChildRunStatus.FAILED,
        failure_class=failure_class,
        failure_type=type(error).__name__,
        failure_message=f"child failed during {stage}",
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )


def _diagnostic(message: str) -> None:
    sys.stderr.write(f"membench child: {message}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())
