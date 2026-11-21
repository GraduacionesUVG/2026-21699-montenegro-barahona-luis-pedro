import pytest

from membench.responder import (
    BackendCompletion,
    ChatMessage,
    DedicatedResponder,
    ResponderError,
    ResponderSettings,
)


class FakeBackend:
    def __init__(self) -> None:
        self.messages: tuple[ChatMessage, ...] = ()

    def complete(
        self, messages: tuple[ChatMessage, ...], _settings: ResponderSettings
    ) -> BackendCompletion:
        self.messages = tuple(messages)
        return BackendCompletion(
            content="Use the blue notebook.",
            prompt_tokens=21,
            completion_tokens=5,
            finish_reason="stop",
            raw_response={"id": "response-001"},
        )


def test_ollama_responder_uses_separate_frozen_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESPONDER_PROVIDER", "ollama")
    monkeypatch.setenv("RESPONDER_MODEL", "responder-model:fixed")
    monkeypatch.setenv("RESPONDER_MODEL_DIGEST", "sha256-responder")
    monkeypatch.setenv("RESPONDER_OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("RESPONDER_TEMPERATURE", "0")
    monkeypatch.setenv("RESPONDER_SEED", "7")
    monkeypatch.setenv("RESPONDER_MAX_OUTPUT_TOKENS", "128")
    monkeypatch.setenv("RESPONDER_TIMEOUT_SECONDS", "30")

    settings = ResponderSettings.from_env()

    assert settings.model == "responder-model:fixed"
    assert settings.model_digest == "sha256-responder"
    assert settings.base_url == "http://localhost:11434"
    assert settings.seed == 7
    assert settings.max_output_tokens == 128


def test_dedicated_responder_normalizes_backend_result() -> None:
    settings = ResponderSettings(
        provider="ollama",
        model="test-model",
        model_digest="sha256-test",
        base_url="http://localhost:11434",
        temperature=0,
        seed=42,
        max_output_tokens=64,
        timeout_seconds=10,
    )
    backend = FakeBackend()
    responder = DedicatedResponder(settings, backend=backend)
    messages = (
        ChatMessage(role="system", content="Be helpful."),
        ChatMessage(role="user", content="Which notebook?"),
    )

    result = responder.respond(messages)

    assert backend.messages == messages
    assert result.content == "Use the blue notebook."
    assert result.prompt_tokens == 21
    assert result.completion_tokens == 5
    assert result.model_digest == "sha256-test"
    assert result.latency_seconds >= 0


def test_openai_responder_requires_explicit_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESPONDER_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr("membench.responder.load_dotenv", lambda **_kwargs: None)

    with pytest.raises(ResponderError, match="OPENAI_API_KEY"):
        ResponderSettings.from_env()
