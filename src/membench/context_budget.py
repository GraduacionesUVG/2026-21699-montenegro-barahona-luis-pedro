"""Deterministic context selection with an injectable token counter."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol


class TokenCounter(Protocol):
    """Count model input units and name the counting implementation."""

    @property
    def name(self) -> str: ...

    def count(self, text: str) -> int: ...


class TokenizableMessage(Protocol):
    """Minimal chat-message shape required by a responder tokenizer."""

    @property
    def role(self) -> str: ...

    @property
    def content(self) -> str: ...


class MessageTokenCounter(TokenCounter, Protocol):
    """Count both unwrapped text and a complete provider-facing chat prompt."""

    def count_messages(
        self,
        messages: Sequence[TokenizableMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> int: ...


class RegexTokenCounter:
    """Dependency-free deterministic approximation for local development."""

    _pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    @property
    def name(self) -> str:
        return "regex_lexical_v1"

    def count(self, text: str) -> int:
        return len(self._pattern.findall(text))

    def count_messages(
        self,
        messages: Sequence[TokenizableMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        """Approximate a chat prompt while retaining deterministic test behavior."""
        rendered = "\n".join(
            f"<{message.role}>\n{message.content}\n</{message.role}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered = f"{rendered}\n<assistant>"
        return self.count(rendered)


@dataclass(frozen=True)
class ContextSelection[T]:
    included: tuple[T, ...]
    rendered: str
    token_count: int
    token_counter: str
    truncated: bool


@dataclass(frozen=True)
class MessageAwareContextSelection[T]:
    """Ranked context selected against the complete responder prompt budget."""

    included: tuple[T, ...]
    rendered: str
    token_count: int
    token_counter: str
    prompt_token_count: int
    prompt_token_counter: str
    truncated: bool


class ContextBudgeter[T]:
    """Select a ranked prefix of complete records within a fixed budget."""

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self.token_counter = token_counter or RegexTokenCounter()

    def select(
        self,
        items: Sequence[T],
        *,
        budget: int,
        render: Callable[[T], str],
    ) -> ContextSelection[T]:
        if budget <= 0:
            raise ValueError("context token budget must be positive")
        included: list[T] = []
        rendered_items: list[str] = []
        for item in items:
            candidate_items = [*rendered_items, render(item)]
            candidate = "\n".join(candidate_items)
            if self.token_counter.count(candidate) > budget:
                break
            included.append(item)
            rendered_items = candidate_items
        rendered = "\n".join(rendered_items)
        return ContextSelection(
            included=tuple(included),
            rendered=rendered,
            token_count=self.token_counter.count(rendered),
            token_counter=self.token_counter.name,
            truncated=len(included) < len(items),
        )


class MessageAwareContextBudgeter[T]:
    """Select a ranked prefix by counting each complete candidate chat prompt."""

    def __init__(
        self,
        selection_token_counter: TokenCounter,
        responder_tokenizer: MessageTokenCounter,
    ) -> None:
        self.selection_token_counter = selection_token_counter
        self.responder_tokenizer = responder_tokenizer

    def select(
        self,
        items: Sequence[T],
        *,
        budget: int,
        render: Callable[[T], str],
        build_messages: Callable[[str], Sequence[TokenizableMessage]],
    ) -> MessageAwareContextSelection[T]:
        if budget <= 0:
            raise ValueError("context token budget must be positive")

        empty_prompt_count = self.responder_tokenizer.count_messages(
            build_messages("")
        )
        if empty_prompt_count > budget:
            raise ValueError(
                "fixed responder prompt exceeds the complete input token budget: "
                f"{empty_prompt_count} > {budget}"
            )

        included: list[T] = []
        rendered_items: list[str] = []
        prompt_token_count = empty_prompt_count
        for item in items:
            candidate_items = [*rendered_items, render(item)]
            candidate = "\n".join(candidate_items)
            candidate_prompt_count = self.responder_tokenizer.count_messages(
                build_messages(candidate)
            )
            if candidate_prompt_count > budget:
                break
            included.append(item)
            rendered_items = candidate_items
            prompt_token_count = candidate_prompt_count

        rendered = "\n".join(rendered_items)
        return MessageAwareContextSelection(
            included=tuple(included),
            rendered=rendered,
            token_count=self.selection_token_counter.count(rendered),
            token_counter=self.selection_token_counter.name,
            prompt_token_count=prompt_token_count,
            prompt_token_counter=self.responder_tokenizer.name,
            truncated=len(included) < len(items),
        )
