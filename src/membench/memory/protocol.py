"""Structural protocol implemented by all memory-mechanism adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from membench.memory.capabilities import AdapterCapabilities
from membench.memory.models import (
    AdapterProvenance,
    ConversationWriteRequest,
    DirectWriteRequest,
    EmptyStateResult,
    FlushResult,
    MemoryScope,
    MemorySnapshot,
    RetrievalRequest,
    RetrievalResult,
    WriteResult,
)


@runtime_checkable
class MemoryAdapter(Protocol):
    """Backend-neutral lifecycle required by an agent or experiment runner.

    The protocol is structural: a TeleMem, LangMem, or Dakera adapter satisfies
    it by exposing these methods and does not need to inherit from this class.
    """

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Report supported optional features without probing them during a run."""

        ...

    def initialize(self, scope: MemoryScope) -> None:
        """Bind resources to exactly one scope; do not silently clear its state."""

        ...

    def verify_empty(self) -> EmptyStateResult:
        """Prove that the bound scope has no visible or pending memories."""

        ...

    def write_direct(self, request: DirectWriteRequest) -> WriteResult:
        """Store one caller-approved memory without native LLM extraction."""

        ...

    def write_conversation(self, request: ConversationWriteRequest) -> WriteResult:
        """Let the mechanism extract, consolidate, or skip conversation memories."""

        ...

    def flush_and_wait(self) -> FlushResult:
        """Block until acknowledged writes are query-visible or timeout."""

        ...

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Search only the bound scope and return normalized ranked hits."""

        ...

    def inspect_state(self) -> MemorySnapshot:
        """Enumerate evaluator-visible state without a semantic query."""

        ...
        
    def reset_and_verify_empty(self) -> EmptyStateResult:
        """Delete only the bound scope and return evidence that it is empty."""

        ...
  
    def export_provenance(self) -> AdapterProvenance:
        """Return non-secret versions and resolved settings for reproducibility."""

        ...

    def close(self) -> None:
        """Release owned resources; repeated calls must be safe."""

        ...
