"""Immutable write-policy dispatch over the common memory adapter protocol."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field, model_validator

from membench.contract import WritePolicy
from membench.memory.errors import (
    PolicyDependencyError,
    UnsupportedCapabilityError,
)
from membench.memory.models import (
    CanonicalModel,
    ConversationWriteRequest,
    DirectWriteRequest,
    JsonObject,
    MemoryMessage,
    MemoryScope,
    MemoryType,
    WriteResult,
)
from membench.memory.protocol import MemoryAdapter


class GuardAction(StrEnum):
    """Admission outcome returned by the shared guard."""

    WRITE = "write"
    NO_WRITE = "no_write"
    QUARANTINE = "quarantine"


class PolicyDispatchAction(StrEnum):
    """Concrete route taken for one policy request."""

    DIRECT_WRITE = "direct_write"
    GUARDED_WRITE = "guarded_write"
    NATIVE_CONVERSATION_WRITE = "native_conversation_write"
    NO_WRITE = "no_write"
    QUARANTINE = "quarantine"


class MemoryCandidate(CanonicalModel):
    """One normalized memory proposed by the shared admission guard."""

    text: str = Field(min_length=1)
    memory_type: MemoryType | None = None
    metadata: JsonObject = Field(default_factory=dict)


class PolicyWriteRequest(CanonicalModel):
    """Mechanism-neutral conversation event supplied to a write policy."""

    operation_id: str = Field(min_length=1)
    scope: MemoryScope
    messages: tuple[MemoryMessage, ...] = Field(min_length=1)
    source_event_id: str = Field(min_length=1)
    memory_type: MemoryType = MemoryType.UNSPECIFIED
    metadata: JsonObject = Field(default_factory=dict)
    native_options: JsonObject = Field(default_factory=dict)


class GuardDecision(CanonicalModel):
    """Version-independent output contract for a shared admission guard."""

    action: GuardAction
    candidates: tuple[MemoryCandidate, ...] = Field(default_factory=tuple)
    reason_code: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9_.-]*$",
    )
    latency_seconds: float = Field(default=0.0, ge=0)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def action_must_match_payload(self) -> GuardDecision:
        if self.action == GuardAction.WRITE:
            if not self.candidates:
                raise ValueError("WRITE guard decisions require candidates")
            if self.reason_code is not None:
                raise ValueError("WRITE guard decisions cannot have a reason code")
        else:
            if self.candidates:
                raise ValueError(
                    "NO_WRITE and QUARANTINE guard decisions cannot have candidates"
                )
            if self.reason_code is None:
                raise ValueError(
                    "NO_WRITE and QUARANTINE guard decisions require a reason code"
                )
        return self


@runtime_checkable
class AdmissionGuard(Protocol):
    """Dependency boundary for the versioned shared admission policy."""

    def evaluate(self, request: PolicyWriteRequest) -> GuardDecision:
        """Return a schema-validated decision without writing memory."""

        ...


class PolicyDispatchResult(CanonicalModel):
    """Auditable result of routing one event through its immutable policy."""

    operation_id: str = Field(min_length=1)
    policy: WritePolicy
    action: PolicyDispatchAction
    write_results: tuple[WriteResult, ...] = Field(default_factory=tuple)
    guard_decision: GuardDecision | None = None

    @property
    def submitted_write_count(self) -> int:
        return len(self.write_results)

    @model_validator(mode="after")
    def route_must_match_payload(self) -> PolicyDispatchResult:
        expected_policy = {
            PolicyDispatchAction.DIRECT_WRITE: WritePolicy.DIRECT,
            PolicyDispatchAction.NATIVE_CONVERSATION_WRITE: (
                WritePolicy.NATIVE_SELECTIVE
            ),
            PolicyDispatchAction.GUARDED_WRITE: WritePolicy.SHARED_GUARDED,
            PolicyDispatchAction.NO_WRITE: WritePolicy.SHARED_GUARDED,
            PolicyDispatchAction.QUARANTINE: WritePolicy.SHARED_GUARDED,
        }[self.action]
        if self.policy != expected_policy:
            raise ValueError(
                f"{self.action.value} action requires {expected_policy.value} policy"
            )

        guarded_actions = {
            PolicyDispatchAction.GUARDED_WRITE,
            PolicyDispatchAction.NO_WRITE,
            PolicyDispatchAction.QUARANTINE,
        }
        if self.action in guarded_actions:
            if self.policy != WritePolicy.SHARED_GUARDED:
                raise ValueError("guarded actions require the shared_guarded policy")
            if self.guard_decision is None:
                raise ValueError("guarded actions require a guard decision")
        elif self.guard_decision is not None:
            raise ValueError(
                "direct and native actions cannot include a guard decision"
            )

        expected_guard_action = {
            PolicyDispatchAction.GUARDED_WRITE: GuardAction.WRITE,
            PolicyDispatchAction.NO_WRITE: GuardAction.NO_WRITE,
            PolicyDispatchAction.QUARANTINE: GuardAction.QUARANTINE,
        }.get(self.action)
        if (
            self.guard_decision is not None
            and self.guard_decision.action != expected_guard_action
        ):
            raise ValueError("dispatch action must agree with the guard decision")

        if (
            self.action
            in {
                PolicyDispatchAction.NO_WRITE,
                PolicyDispatchAction.QUARANTINE,
            }
            and self.write_results
        ):
            raise ValueError("non-writing policy actions cannot contain write results")
        if (
            self.action
            in {
                PolicyDispatchAction.DIRECT_WRITE,
                PolicyDispatchAction.NATIVE_CONVERSATION_WRITE,
                PolicyDispatchAction.GUARDED_WRITE,
            }
            and not self.write_results
        ):
            raise ValueError("writing policy actions require write results")
        return self


class PolicyDispatcher:
    """Bind one write policy and route events through a memory adapter."""

    def __init__(
        self,
        adapter: MemoryAdapter,
        policy: WritePolicy,
        *,
        guard: AdmissionGuard | None = None,
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.guard = guard
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        if self.policy == WritePolicy.SHARED_GUARDED:
            self._require_capability("direct_write")
            if self.guard is None:
                raise PolicyDependencyError(
                    "shared_guarded policy requires an admission guard"
                )
            if not isinstance(self.guard, AdmissionGuard):
                raise PolicyDependencyError(
                    "shared_guarded guard must implement evaluate(request)"
                )
            return

        if self.guard is not None:
            raise PolicyDependencyError(
                "an admission guard may only be configured for shared_guarded"
            )
        if self.policy == WritePolicy.DIRECT:
            self._require_capability("direct_write")
        elif self.policy == WritePolicy.NATIVE_SELECTIVE:
            self._require_capability("native_conversation_write")

    def _require_capability(self, capability: str) -> None:
        if self.adapter.capabilities[capability]:
            return
        raise UnsupportedCapabilityError(
            f"{self.policy.value} policy requires adapter capability: {capability}",
            missing=(capability,),
        )

    def dispatch(self, request: PolicyWriteRequest) -> PolicyDispatchResult:
        """Route one event without flushing or retrieving from the adapter."""
        if self.policy == WritePolicy.DIRECT:
            return self._dispatch_direct(request)
        if self.policy == WritePolicy.SHARED_GUARDED:
            return self._dispatch_guarded(request)
        if self.policy == WritePolicy.NATIVE_SELECTIVE:
            return self._dispatch_native(request)
        raise PolicyDependencyError(f"unsupported write policy: {self.policy!r}")

    def _dispatch_direct(
        self,
        request: PolicyWriteRequest,
    ) -> PolicyDispatchResult:
        result = self.adapter.write_direct(
            DirectWriteRequest(
                operation_id=request.operation_id,
                scope=request.scope,
                text=render_conversation(request.messages),
                memory_type=request.memory_type,
                source_event_id=request.source_event_id,
                metadata=request.metadata,
            )
        )
        return PolicyDispatchResult(
            operation_id=request.operation_id,
            policy=self.policy,
            action=PolicyDispatchAction.DIRECT_WRITE,
            write_results=(result,),
        )

    def _dispatch_guarded(
        self,
        request: PolicyWriteRequest,
    ) -> PolicyDispatchResult:
        if self.guard is None:  # Protected by construction; keeps narrowing explicit.
            raise PolicyDependencyError(
                "shared_guarded policy requires an admission guard"
            )
        decision = self.guard.evaluate(request)
        if not isinstance(decision, GuardDecision):
            raise PolicyDependencyError("admission guard returned an invalid decision")

        if decision.action == GuardAction.NO_WRITE:
            return PolicyDispatchResult(
                operation_id=request.operation_id,
                policy=self.policy,
                action=PolicyDispatchAction.NO_WRITE,
                guard_decision=decision,
            )
        if decision.action == GuardAction.QUARANTINE:
            return PolicyDispatchResult(
                operation_id=request.operation_id,
                policy=self.policy,
                action=PolicyDispatchAction.QUARANTINE,
                guard_decision=decision,
            )

        writes = tuple(
            self.adapter.write_direct(
                DirectWriteRequest(
                    operation_id=_candidate_operation_id(
                        request.operation_id,
                        index,
                    ),
                    scope=request.scope,
                    text=candidate.text,
                    memory_type=candidate.memory_type or request.memory_type,
                    source_event_id=request.source_event_id,
                    metadata={**request.metadata, **candidate.metadata},
                )
            )
            for index, candidate in enumerate(decision.candidates, start=1)
        )
        return PolicyDispatchResult(
            operation_id=request.operation_id,
            policy=self.policy,
            action=PolicyDispatchAction.GUARDED_WRITE,
            write_results=writes,
            guard_decision=decision,
        )

    def _dispatch_native(
        self,
        request: PolicyWriteRequest,
    ) -> PolicyDispatchResult:
        result = self.adapter.write_conversation(
            ConversationWriteRequest(
                operation_id=request.operation_id,
                scope=request.scope,
                messages=request.messages,
                source_event_id=request.source_event_id,
                native_options=request.native_options,
            )
        )
        return PolicyDispatchResult(
            operation_id=request.operation_id,
            policy=self.policy,
            action=PolicyDispatchAction.NATIVE_CONVERSATION_WRITE,
            write_results=(result,),
        )


def render_conversation(messages: tuple[MemoryMessage, ...]) -> str:
    """Serialize direct-policy input identically for every mechanism."""
    if not messages:
        raise ValueError("at least one message is required")
    return "\n".join(f"{message.role.value}: {message.content}" for message in messages)


def _candidate_operation_id(operation_id: str, index: int) -> str:
    return f"{operation_id}:guard:{index:03d}"
