from collections.abc import Sequence

import pytest

from membench.context_budget import (
    ContextBudgeter,
    MessageAwareContextBudgeter,
    RegexTokenCounter,
    TokenizableMessage,
)
from membench.responder import ChatMessage


class SyntheticMessageCounter:
    @property
    def name(self) -> str:
        return "synthetic_chat_v1"

    def count(self, text: str) -> int:
        return len(text.split())

    def count_messages(
        self,
        messages: Sequence[TokenizableMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        content_tokens = sum(len(message.content.split()) for message in messages)
        return 10 + content_tokens + int(add_generation_prompt)


def test_budgeter_keeps_a_ranked_prefix_of_complete_items() -> None:
    budgeter: ContextBudgeter[str] = ContextBudgeter(RegexTokenCounter())

    selection = budgeter.select(
        ["first memory", "second memory", "third memory"],
        budget=4,
        render=lambda item: item,
    )

    assert selection.included == ("first memory", "second memory")
    assert selection.rendered == "first memory\nsecond memory"
    assert selection.token_count == 4
    assert selection.token_counter == "regex_lexical_v1"
    assert selection.truncated


def test_message_aware_budgeter_counts_the_complete_candidate_prompt() -> None:
    counter = SyntheticMessageCounter()
    budgeter: MessageAwareContextBudgeter[str] = MessageAwareContextBudgeter(
        counter,
        counter,
    )

    selection = budgeter.select(
        ["first memory", "second memory"],
        budget=15,
        render=lambda item: item,
        build_messages=lambda context: (
            ChatMessage(role="system", content="fixed instruction"),
            ChatMessage(role="user", content=context or "empty"),
        ),
    )

    assert selection.included == ("first memory",)
    assert selection.rendered == "first memory"
    assert selection.token_count == 2
    assert selection.prompt_token_count == 15
    assert selection.token_counter == "synthetic_chat_v1"
    assert selection.prompt_token_counter == "synthetic_chat_v1"
    assert selection.truncated


def test_message_aware_budgeter_rejects_fixed_prompt_over_budget() -> None:
    counter = SyntheticMessageCounter()
    budgeter: MessageAwareContextBudgeter[str] = MessageAwareContextBudgeter(
        counter,
        counter,
    )

    with pytest.raises(ValueError, match="fixed responder prompt exceeds"):
        budgeter.select(
            [],
            budget=5,
            render=str,
            build_messages=lambda context: (
                ChatMessage(role="user", content=context or "empty"),
            ),
        )
