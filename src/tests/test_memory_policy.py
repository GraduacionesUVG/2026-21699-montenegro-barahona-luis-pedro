from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from membench.contract import WritePolicy
from membench.memory.capabilities import AdapterCapabilities
from membench.memory.errors import (
    PolicyDependencyError,
    UnsupportedCapabilityError,
)
from membench.memory.models import (
    MemoryMessage,
    MemoryScope,
    MemoryType,
    MessageRole,
    WriteResult,
    WriteStatus,
)
from membench.memory.policy import (
    GuardAction,
    GuardDecision,
    MemoryCandidate,
    PolicyDispatchAction,
    PolicyDispatcher,
    PolicyWriteRequest,
    render_conversation,
)


def _capabilities(
    *,
    direct: bool = True,
    native: bool = True,
) -> AdapterCapabilities:
    return AdapterCapabilities(
        direct_write=direct,
        native_conversation_write=native,
        semantic_memory=True,
        episodic_memory=False,
        procedural_memory=False,
        working_memory=False,
        updates=False,
        deletion=False,
        ttl=False,
        metadata_filters=False,
        threshold=False,
        reranking=False,
        hybrid_search=False,
        associations=False,
    )


class RecordingAdapter:
    def __init__(
        self,
        *,
        direct: bool = True,
        native: bool = True,
    ) -> None:
        self.capabilities = _capabilities(direct=direct, native=native)
        self.direct_requests = []
        self.conversation_requests = []

    def write_direct(self, request):
        self.direct_requests.append(request)
        return WriteResult(
            operation_id=request.operation_id,
            status=WriteStatus.SUCCEEDED,
        )

    def write_conversation(self, request):
        self.conversation_requests.append(request)
        return WriteResult(
            operation_id=request.operation_id,
            status=WriteStatus.QUEUED,
        )


class FixedGuard:
    def __init__(self, decision: GuardDecision) -> None:
        self.decision = decision
        self.requests: list[PolicyWriteRequest] = []

    def evaluate(self, request: PolicyWriteRequest) -> GuardDecision:
        self.requests.append(request)
        return self.decision


@pytest.fixture
def scope() -> MemoryScope:
    return MemoryScope(
        experiment_id="experiment-1",
        run_id="run-1",
        user_id="persona-1",
        namespace=("membench", "run-1"),
    )


@pytest.fixture
def policy_request(scope: MemoryScope) -> PolicyWriteRequest:
    return PolicyWriteRequest(
        operation_id="event-1",
        scope=scope,
        messages=(
            MemoryMessage(role=MessageRole.USER, content="I prefer tea."),
            MemoryMessage(role=MessageRole.ASSISTANT, content="Understood."),
        ),
        source_event_id="source-1",
        memory_type=MemoryType.SEMANTIC,
        metadata={"event_kind": "benign"},
        native_options={"prompt": "fixed native prompt"},
    )


def test_direct_policy_serializes_and_writes_once(
    policy_request: PolicyWriteRequest,
) -> None:
    adapter = RecordingAdapter()
    dispatcher = PolicyDispatcher(adapter, WritePolicy.DIRECT)  # type: ignore[arg-type]

    result = dispatcher.dispatch(policy_request)

    assert result.action == PolicyDispatchAction.DIRECT_WRITE
    assert result.submitted_write_count == 1
    assert len(adapter.direct_requests) == 1
    written = adapter.direct_requests[0]
    assert written.operation_id == "event-1"
    assert written.text == "user: I prefer tea.\nassistant: Understood."
    assert written.memory_type == MemoryType.SEMANTIC
    assert written.metadata == {"event_kind": "benign"}
    assert not adapter.conversation_requests


def test_native_policy_preserves_conversation_and_options(
    policy_request: PolicyWriteRequest,
) -> None:
    adapter = RecordingAdapter()
    dispatcher = PolicyDispatcher(  # type: ignore[arg-type]
        adapter,
        WritePolicy.NATIVE_SELECTIVE,
    )

    result = dispatcher.dispatch(policy_request)

    assert result.action == PolicyDispatchAction.NATIVE_CONVERSATION_WRITE
    assert len(adapter.conversation_requests) == 1
    written = adapter.conversation_requests[0]
    assert written.messages == policy_request.messages
    assert written.native_options == {"prompt": "fixed native prompt"}
    assert not adapter.direct_requests


def test_shared_guard_writes_each_approved_candidate(
    policy_request: PolicyWriteRequest,
) -> None:
    adapter = RecordingAdapter()
    guard = FixedGuard(
        GuardDecision(
            action=GuardAction.WRITE,
            candidates=(
                MemoryCandidate(text="Likes tea."),
                MemoryCandidate(
                    text="Avoids coffee.",
                    memory_type=MemoryType.EPISODIC,
                    metadata={"candidate": 2},
                ),
            ),
            metadata={"prompt_version": "guard-v1"},
        )
    )
    dispatcher = PolicyDispatcher(  # type: ignore[arg-type]
        adapter,
        WritePolicy.SHARED_GUARDED,
        guard=guard,
    )

    result = dispatcher.dispatch(policy_request)

    assert result.action == PolicyDispatchAction.GUARDED_WRITE
    assert result.submitted_write_count == 2
    assert guard.requests == [policy_request]
    assert [item.operation_id for item in adapter.direct_requests] == [
        "event-1:guard:001",
        "event-1:guard:002",
    ]
    assert adapter.direct_requests[0].memory_type == MemoryType.SEMANTIC
    assert adapter.direct_requests[1].memory_type == MemoryType.EPISODIC
    assert adapter.direct_requests[1].metadata == {
        "event_kind": "benign",
        "candidate": 2,
    }


@pytest.mark.parametrize(
    ("action", "dispatch_action"),
    [
        (GuardAction.NO_WRITE, PolicyDispatchAction.NO_WRITE),
        (GuardAction.QUARANTINE, PolicyDispatchAction.QUARANTINE),
    ],
)
def test_guard_rejection_paths_do_not_call_adapter(
    policy_request: PolicyWriteRequest,
    action: GuardAction,
    dispatch_action: PolicyDispatchAction,
) -> None:
    adapter = RecordingAdapter()
    guard = FixedGuard(
        GuardDecision(action=action, reason_code="unsafe_memory_candidate")
    )
    dispatcher = PolicyDispatcher(  # type: ignore[arg-type]
        adapter,
        WritePolicy.SHARED_GUARDED,
        guard=guard,
    )

    result = dispatcher.dispatch(policy_request)

    assert result.action == dispatch_action
    assert result.submitted_write_count == 0
    assert result.guard_decision is guard.decision
    assert not adapter.direct_requests
    assert not adapter.conversation_requests


def test_policy_dependencies_are_validated_before_dispatch() -> None:
    with pytest.raises(UnsupportedCapabilityError) as direct_error:
        PolicyDispatcher(  # type: ignore[arg-type]
            RecordingAdapter(direct=False),
            WritePolicy.DIRECT,
        )
    assert direct_error.value.missing == ("direct_write",)

    with pytest.raises(UnsupportedCapabilityError) as native_error:
        PolicyDispatcher(  # type: ignore[arg-type]
            RecordingAdapter(native=False),
            WritePolicy.NATIVE_SELECTIVE,
        )
    assert native_error.value.missing == ("native_conversation_write",)

    with pytest.raises(PolicyDependencyError, match="requires an admission guard"):
        PolicyDispatcher(  # type: ignore[arg-type]
            RecordingAdapter(),
            WritePolicy.SHARED_GUARDED,
        )


def test_guard_is_rejected_for_policies_that_do_not_use_it() -> None:
    guard = FixedGuard(
        GuardDecision(action=GuardAction.NO_WRITE, reason_code="not_relevant")
    )

    with pytest.raises(PolicyDependencyError, match="only be configured"):
        PolicyDispatcher(  # type: ignore[arg-type]
            RecordingAdapter(),
            WritePolicy.DIRECT,
            guard=guard,
        )


def test_guard_decision_schema_enforces_action_payload() -> None:
    with pytest.raises(ValidationError, match="require candidates"):
        GuardDecision(action=GuardAction.WRITE)
    with pytest.raises(ValidationError, match="require a reason code"):
        GuardDecision(action=GuardAction.NO_WRITE)
    with pytest.raises(ValidationError, match="cannot have candidates"):
        GuardDecision(
            action=GuardAction.QUARANTINE,
            reason_code="unsafe",
            candidates=(MemoryCandidate(text="must not be written"),),
        )


def test_invalid_guard_implementation_is_rejected() -> None:
    with pytest.raises(PolicyDependencyError, match="must implement"):
        PolicyDispatcher(  # type: ignore[arg-type]
            RecordingAdapter(),
            WritePolicy.SHARED_GUARDED,
            guard=SimpleNamespace(),  # type: ignore[arg-type]
        )


def test_render_conversation_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        render_conversation(())
