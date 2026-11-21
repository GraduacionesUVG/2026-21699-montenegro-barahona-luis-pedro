"""Mechanism-neutral context assembly and responder prompt construction."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from membench.context_budget import (
    MessageAwareContextBudgeter,
    MessageTokenCounter,
    RegexTokenCounter,
    TokenCounter,
)
from membench.memory.models import RetrievalHit
from membench.responder import ChatMessage, ResponderResult

QUALIFICATION_RESPONDER_PROMPT_VERSION = "qualification_responder_v1"
QUALIFICATION_RESPONDER_SYSTEM_PROMPT = (
    "You are answering a synthetic memory-system qualification probe. Use only "
    "the supplied stored-memory context. Do not request, create, update, or delete "
    "memory. Answer briefly and preserve any relevant synthetic marker exactly."
)


@runtime_checkable
class QualificationResponder(Protocol):
    """Shared responder boundary used after mechanism-neutral retrieval."""

    def verify(self) -> None: ...

    def respond(self, messages: Sequence[ChatMessage]) -> ResponderResult: ...


@dataclass(frozen=True, slots=True)
class QualificationContext:
    """Exact ranked-prefix context supplied to one response call."""

    record_ids: tuple[str, ...]
    content: str
    sha256: str
    token_count: int
    token_counter: str
    prompt_token_count: int
    prompt_token_counter: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class QualificationPrompt:
    """Versioned responder messages and their exact content hash."""

    version: str
    sha256: str
    messages: tuple[ChatMessage, ...]
    input_token_count: int
    token_counter: str


class QualificationContextAssembler:
    """Select and render canonical retrieval hits without mechanism-specific data."""

    def __init__(
        self,
        selection_token_counter: TokenCounter | None = None,
        responder_tokenizer: MessageTokenCounter | None = None,
    ) -> None:
        selection_counter = selection_token_counter or RegexTokenCounter()
        prompt_counter = responder_tokenizer or RegexTokenCounter()
        self._budgeter: MessageAwareContextBudgeter[RetrievalHit] = (
            MessageAwareContextBudgeter(
                selection_counter,
                prompt_counter,
            )
        )

    def assemble(
        self,
        hits: Sequence[RetrievalHit],
        *,
        query: str,
        token_budget: int,
        prompt_builder: QualificationPromptBuilder,
    ) -> QualificationContext:
        selection = self._budgeter.select(
            hits,
            budget=token_budget,
            render=self._render_hit,
            build_messages=lambda content: prompt_builder.build_messages(
                query,
                content,
            ),
        )
        content = selection.rendered
        return QualificationContext(
            record_ids=tuple(
                hit.record.canonical_id for hit in selection.included
            ),
            content=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            token_count=selection.token_count,
            token_counter=selection.token_counter,
            prompt_token_count=selection.prompt_token_count,
            prompt_token_counter=selection.prompt_token_counter,
            truncated=selection.truncated,
        )

    @staticmethod
    def _render_hit(hit: RetrievalHit) -> str:
        # Record IDs remain in evaluator evidence but are excluded from model
        # context so backend-specific identity schemes cannot affect responses.
        return f"[memory rank={hit.rank}]\n{hit.record.text}"


class QualificationPromptBuilder:
    """Build the same read-only synthetic prompt for every memory mechanism."""

    def __init__(
        self,
        system_prompt: str = QUALIFICATION_RESPONDER_SYSTEM_PROMPT,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("qualification responder system prompt cannot be blank")
        self.system_prompt = system_prompt

    def build_messages(
        self,
        query: str,
        context_content: str,
    ) -> tuple[ChatMessage, ...]:
        if not query.strip():
            raise ValueError("qualification responder query cannot be blank")
        rendered_context = context_content or "(no retrieved memories)"
        return (
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(
                role="user",
                content=(
                    "Stored memory context:\n"
                    f"{rendered_context}\n\n"
                    "Current qualification query:\n"
                    f"{query.strip()}"
                ),
            ),
        )

    def build(self, query: str, context: QualificationContext) -> QualificationPrompt:
        messages = self.build_messages(query, context.content)
        payload = "\n\n".join(
            f"{message.role}:{message.content}" for message in messages
        )
        return QualificationPrompt(
            version=QUALIFICATION_RESPONDER_PROMPT_VERSION,
            sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            messages=messages,
            input_token_count=context.prompt_token_count,
            token_counter=context.prompt_token_counter,
        )
