from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from membench.personamem_v2 import (
    PersonaMemCase,
    PersonaMemMessage,
    PersonaMemProbe,
    PersonaMemReplayEvent,
)
from membench.responder import ChatMessage, ResponderResult
from membench.telemem import TeleMemRecord
from membench.telemem_agent import (
    TeleMemAgent,
    TeleMemAgentError,
    TeleMemAgentSettings,
)

REVISION = "a" * 40


def _record(identifier: str, text: str) -> TeleMemRecord:
    return TeleMemRecord(
        vendor_id=identifier,
        text=text,
        score=0.9,
        event="ADD",
        user_id="persona-001",
        run_id="run-001",
        role="user",
        created_at="2026-09-08T00:00:00Z",
        updated_at=None,
        metadata={},
        raw_value={"id": identifier, "memory": text},
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.user_id = "persona-001"
        self.records: list[TeleMemRecord] = []
        self.infer_values: list[bool] = []
        self.closed = False

    def initialize(self, *, require_empty: bool = True) -> FakeRuntime:
        if require_empty and self.records:
            raise RuntimeError("not empty")
        return self

    def ingest_benign(
        self,
        messages: list[dict[str, str]],
        *,
        infer: bool,
        metadata: dict[str, Any],
        source_event_id: str,
    ) -> list[TeleMemRecord]:
        self.infer_values.append(infer)
        record = _record(source_event_id, messages[0]["content"])
        self.records.append(record)
        return [record]

    def flush_and_wait(self) -> list[TeleMemRecord]:
        return list(self.records)

    def retrieve(self, _query: str, *, top_k: int) -> list[TeleMemRecord]:
        return list(self.records[:top_k])

    def inspect_state(self) -> list[TeleMemRecord]:
        return list(self.records)

    def close(self) -> None:
        self.closed = True


class FakeResponder:
    def __init__(self) -> None:
        self.verified = False
        self.messages: tuple[ChatMessage, ...] = ()

    def verify(self) -> None:
        self.verified = True

    def respond(self, messages: Sequence[ChatMessage]) -> ResponderResult:
        self.messages = tuple(messages)
        return ResponderResult(
            content="The memory supports blue. Final Answer: A",
            provider="ollama",
            model="test-model",
            model_digest="sha256-test",
            prompt_tokens=40,
            completion_tokens=3,
            finish_reason="stop",
            latency_seconds=0.01,
            raw_response={"done": True},
        )


def _case(persona_id: str = "persona-001") -> PersonaMemCase:
    event = PersonaMemReplayEvent(
        dataset_revision=REVISION,
        subset="blackbox_battery_v1",
        source_record_id="row-001",
        history_unit_id="history-001",
        persona_id=persona_id,
        event_id="event-001",
        sequence=0,
        messages=(
            PersonaMemMessage(
                role="user", content="Blue works well with my desk."
            ),
            PersonaMemMessage(role="assistant", content="Good choice."),
        ),
    )
    probe = PersonaMemProbe(
        dataset_revision=REVISION,
        subset="blackbox_battery_v1",
        source_record_id="row-001",
        history_unit_id="history-001",
        persona_id=persona_id,
        event_id="probe-001",
        query="Which notebook best suits me?",
        options=("A blue notebook", "A red notebook"),
    )
    return PersonaMemCase(
        history_unit_id="history-001",
        persona_id=persona_id,
        replay_events=(event,),
        probe=probe,
    )


def test_agent_replays_history_and_answers_probe_without_writeback() -> None:
    runtime = FakeRuntime()
    responder = FakeResponder()
    agent = TeleMemAgent(
        runtime,
        responder,
        TeleMemAgentSettings(top_k=5, context_token_budget=100),
    ).initialize()
    case = _case()

    visible = agent.replay_case(case, infer=True)
    count_before_probe = len(runtime.records)
    turn = agent.answer_probe(case.probe)

    assert visible[0].vendor_id == "event-001"
    assert runtime.infer_values == [True]
    assert len(runtime.records) == count_before_probe
    assert turn.response.content.endswith("Final Answer: A")
    assert turn.memory_unchanged
    assert turn.context_record_ids == ("event-001",)
    assert "Which notebook best suits me?" in responder.messages[1].content
    assert "A. A blue notebook" in responder.messages[1].content
    assert responder.verified

    agent.close()
    assert runtime.closed


def test_agent_rejects_a_case_for_another_persona() -> None:
    agent = TeleMemAgent(
        FakeRuntime(),
        FakeResponder(),
        TeleMemAgentSettings(top_k=5, context_token_budget=100),
    ).initialize(verify_responder=False)

    with pytest.raises(TeleMemAgentError, match="does not match"):
        agent.replay_case(_case("persona-002"), infer=False)
