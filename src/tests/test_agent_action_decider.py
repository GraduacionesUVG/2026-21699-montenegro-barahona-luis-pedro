import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from membench.agent.action_decider import (
    ActionDeciderSettings,
    ActionDecisionCompletion,
    ActionDecisionGenerationError,
    ActionDecisionValidationError,
    OllamaActionDecider,
    OllamaActionDecisionBackend,
    action_decision_json_schema,
)
from membench.agent.models import (
    AgentDecisionContext,
    AgentMemoryContext,
    AgentPromptMessage,
    AgentPromptRole,
    FinalAnswerDecision,
    ToolActionDecision,
)
from membench.agent.protocol import ActionDecider
from membench.tools.models import ToolDefinition


def _settings() -> ActionDeciderSettings:
    return ActionDeciderSettings(
        model="qwen-test:7b",
        model_digest="sha256-test",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        max_output_tokens=256,
        timeout_seconds=30,
    )


def _context() -> AgentDecisionContext:
    memory_content = "The synthetic destination is alpha@example.invalid."
    memory_hash = hashlib.sha256(memory_content.encode("utf-8")).hexdigest()
    memory = AgentMemoryContext(
        record_ids=("record-001",),
        content=memory_content,
        sha256=memory_hash,
        token_count=7,
        token_counter="test_counter",
        initial_prompt_token_count=40,
        prompt_token_counter="test_counter",
    )
    messages = (
        AgentPromptMessage(
            role=AgentPromptRole.SYSTEM,
            content="Return one structured JSON decision.",
        ),
        AgentPromptMessage(
            role=AgentPromptRole.USER,
            content="Find the synthetic destination.",
        ),
    )
    payload = "\n\n".join(
        f"{message.role.value}:{message.content}" for message in messages
    )
    return AgentDecisionContext(
        task_id="task-001",
        step_index=1,
        prompt_version="test_prompt_v1",
        messages=messages,
        available_tools=(
            ToolDefinition(
                name="lookup_contact",
                description="Look up a synthetic contact.",
            ),
        ),
        memory_context=memory,
        prompt_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        prompt_token_count=40,
        prompt_token_counter="test_counter",
    )


class FakeBackend:
    def __init__(self, completion: ActionDecisionCompletion) -> None:
        self.completion = completion
        self.messages: tuple[AgentPromptMessage, ...] = ()
        self.schema: Mapping[str, Any] = {}
        self.settings: ActionDeciderSettings | None = None

    def complete(
        self,
        messages: Sequence[AgentPromptMessage],
        schema: Mapping[str, Any],
        settings: ActionDeciderSettings,
        tools: Sequence[ToolDefinition],
    ) -> ActionDecisionCompletion:
        self.messages = tuple(messages)
        self.schema = schema
        self.settings = settings
        return self.completion


def _completion(
    decision: dict[str, Any] | None,
    *,
    finish_reason: str = "stop",
) -> ActionDecisionCompletion:
    content = None if decision is None else json.dumps({"decision": decision})
    return ActionDecisionCompletion(
        content=content,
        prompt_tokens=50,
        completion_tokens=12,
        finish_reason=finish_reason,
        raw_response={"done": True},
    )


def test_settings_resolve_only_ollama_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "membench.agent.action_decider.load_dotenv",
        lambda **_kwargs: None,
    )
    monkeypatch.setenv("RESPONDER_PROVIDER", "openai")
    monkeypatch.setenv("RESPONDER_MODEL", "cloud-model")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "local-model:fixed")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL_DIGEST", "sha256-local")
    monkeypatch.setenv("AGENT_DECIDER_OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("AGENT_DECIDER_THINK", "false")
    monkeypatch.setenv("AGENT_DECIDER_CAPTURE_RAW_RESPONSE", "true")

    settings = ActionDeciderSettings.from_env()

    assert settings.model == "local-model:fixed"
    assert settings.model_digest == "sha256-local"
    assert settings.base_url == "http://localhost:11434"
    assert settings.think is False
    assert settings.capture_raw_response is True
    assert settings.public_config()["provider"] == "ollama"


def test_schema_requires_a_discriminated_decision_kind() -> None:
    schema = action_decision_json_schema()

    assert schema["type"] == "object"
    assert schema["required"] == ["decision"]
    for name in (
        "ToolActionDecision",
        "FinalAnswerDecision",
        "RefusalDecision",
    ):
        assert "kind" in schema["$defs"][name]["required"]


def test_decider_parses_final_answer_and_records_trace() -> None:
    context = _context()
    backend = FakeBackend(
        _completion(
            {
                "kind": "final_answer",
                "content": "Use alpha@example.invalid.",
            }
        )
    )
    decider = OllamaActionDecider(_settings(), backend=backend)

    decision = decider.decide(context)

    assert isinstance(decider, ActionDecider)
    assert isinstance(decision, FinalAnswerDecision)
    assert decision.content == "Use alpha@example.invalid."
    assert backend.messages == context.messages
    assert decider.last_trace is not None
    assert decider.last_trace.prompt_sha256 == context.prompt_sha256
    assert decider.last_trace.context_sha256 == context.memory_context.sha256
    assert decider.last_trace.prompt_tokens == 50
    assert decider.last_trace.parse_outcome == "valid"
    assert decider.last_trace.field_presence == {}
    assert decider.public_config()["decision_schema_sha256"] == (
        decider.schema_sha256
    )


def test_decider_preserves_tool_proposal_for_runner_policy() -> None:
    backend = FakeBackend(
        _completion(
            {
                "kind": "tool_action",
                "call": {
                    "call_id": "call-001",
                    "tool_name": "lookup_contact",
                    "arguments": {"persona_id": "persona-001"},
                    "idempotency_key": "task-001:call-001",
                },
                "decision_summary": "Look up the synthetic contact.",
            }
        )
    )

    decision = OllamaActionDecider(_settings(), backend=backend).decide(_context())

    assert isinstance(decision, ToolActionDecision)
    assert decision.call.tool_name == "lookup_contact"
    assert decision.call.arguments == {"persona_id": "persona-001"}


def test_decider_rejects_invalid_or_truncated_output() -> None:
    invalid = FakeBackend(
        ActionDecisionCompletion(
            content='{"decision":{"kind":"not_supported"}}',
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
            raw_response={},
        )
    )
    truncated = FakeBackend(
        ActionDecisionCompletion(
            content='{"decision":',
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="length",
            raw_response={},
        )
    )

    with pytest.raises(ActionDecisionValidationError, match="canonical"):
        OllamaActionDecider(_settings(), backend=invalid).decide(_context())
    with pytest.raises(ActionDecisionGenerationError, match="incomplete"):
        OllamaActionDecider(_settings(), backend=truncated).decide(_context())


def test_ollama_backend_sends_the_schema_in_native_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post_json(
        url: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> dict[str, Any]:
        captured.update(
            url=url,
            payload=payload,
            headers=headers,
            timeout=timeout,
        )
        return {
            "message": {
                "content": json.dumps(
                    {
                        "decision": {
                            "kind": "refusal",
                            "reason": "The task is not permitted.",
                        }
                    }
                )
            },
            "done_reason": "stop",
            "prompt_eval_count": 30,
            "eval_count": 8,
        }

    monkeypatch.setattr(
        "membench.agent.action_decider._post_json",
        fake_post_json,
    )
    context = _context()
    schema = action_decision_json_schema()

    completion = OllamaActionDecisionBackend().complete(
        context.messages,
        schema,
        _settings(),
        context.available_tools,
    )

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["headers"] is None
    assert captured["payload"]["format"] == schema
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["temperature"] == 0
    assert completion.prompt_tokens == 30
    assert completion.completion_tokens == 8
    assert captured["payload"]["think"] is False
    assert captured["payload"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_contact",
                "description": "Look up a synthetic contact.",
                "parameters": {},
            },
        }
    ]


def test_backend_records_field_presence_without_retaining_thinking_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post_json(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "message": {
                "content": "",
                "thinking": "private intermediate reasoning",
                "tool_calls": [{"function": {"name": "lookup_contact"}}],
            },
            "done_reason": "stop",
        }

    monkeypatch.setattr("membench.agent.action_decider._post_json", fake_post_json)
    completion = OllamaActionDecisionBackend().complete(
        _context().messages,
        action_decision_json_schema(),
        _settings(),
        _context().available_tools,
    )

    evidence = completion.diagnostic_evidence(include_raw_response=False)
    assert completion.field_presence == {
        "message": True,
        "content": False,
        "thinking": True,
        "tool_calls": True,
    }
    assert evidence["thinking_character_count"] == len("private intermediate reasoning")
    assert "private intermediate reasoning" not in json.dumps(evidence)
    assert "raw_response" not in evidence
    assert evidence["tool_call_count"] == 1


def test_decider_retains_evidence_when_provider_content_is_empty() -> None:
    decider = OllamaActionDecider(
        _settings(),
        backend=FakeBackend(
            ActionDecisionCompletion(
                content=None,
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                raw_response={"message": {"thinking": "hidden"}},
                field_presence={"message": True, "content": False, "thinking": True},
                thinking_character_count=6,
                thinking_sha256="digest",
            )
        ),
    )

    with pytest.raises(ActionDecisionGenerationError, match="no decision content"):
        decider.decide(_context())

    assert decider.last_trace is not None
    assert decider.last_trace.parse_outcome == "empty_content"
    assert decider.last_trace.field_presence["thinking"] is True
    assert "raw_response" not in decider.last_trace.completion_evidence


def test_decider_normalizes_one_ollama_native_tool_call() -> None:
    context = _context()
    decider = OllamaActionDecider(
        _settings(),
        backend=FakeBackend(
            ActionDecisionCompletion(
                content="",
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="stop",
                raw_response={
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-001",
                                "function": {
                                    "name": "lookup_contact",
                                    "arguments": {"persona_id": "persona-001"},
                                },
                            }
                        ],
                    }
                },
                field_presence={"content": False, "tool_calls": True},
            )
        ),
    )

    decision = decider.decide(context)

    assert isinstance(decision, ToolActionDecision)
    assert decision.call.call_id == "call-001"
    assert decision.call.tool_name == "lookup_contact"
    assert decision.call.idempotency_key == "task-001:call-001"
