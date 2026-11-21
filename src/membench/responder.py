"""Deterministic chat responder independent from memory implementations."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from membench.ollama import OllamaService


class ResponderError(RuntimeError):
    """Raised when responder configuration or generation fails."""


@dataclass(frozen=True)
class ChatMessage:
    """One validated provider-neutral chat message."""

    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("chat role must be system, user, or assistant")
        if not self.content.strip():
            raise ValueError("chat message content must not be empty")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ResponderSettings:
    """Resolved generation settings for the dedicated response model."""

    provider: str
    model: str
    model_digest: str | None
    base_url: str
    temperature: float
    seed: int
    max_output_tokens: int
    timeout_seconds: float
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> ResponderSettings:
        load_dotenv(override=False)
        provider = os.getenv("RESPONDER_PROVIDER", "ollama").strip().lower()
        explicit_model = (os.getenv("RESPONDER_MODEL") or "").strip()
        explicit_digest = (os.getenv("RESPONDER_MODEL_DIGEST") or "").strip()

        if provider == "ollama":
            fallback_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
            model = explicit_model or fallback_model
            digest = explicit_digest or (
                os.getenv("OLLAMA_CHAT_MODEL_DIGEST", "").strip()
                if model == fallback_model
                else ""
            )
            base_url = os.getenv(
                "RESPONDER_OLLAMA_BASE_URL",
                os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            )
            api_key = None
        elif provider == "openai":
            model = explicit_model or os.getenv("RESPONDER_OPENAI_MODEL", "gpt-5-mini")
            digest = explicit_digest
            base_url = (
                os.getenv("RESPONDER_OPENAI_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )
            api_key = os.getenv("OPENAI_API_KEY") or None
        else:
            raise ValueError("RESPONDER_PROVIDER must be 'ollama' or 'openai'")

        settings = cls(
            provider=provider,
            model=model.strip(),
            model_digest=digest or None,
            base_url=base_url.rstrip("/"),
            temperature=float(os.getenv("RESPONDER_TEMPERATURE", "0")),
            seed=int(os.getenv("RESPONDER_SEED", "42")),
            max_output_tokens=int(os.getenv("RESPONDER_MAX_OUTPUT_TOKENS", "512")),
            timeout_seconds=float(os.getenv("RESPONDER_TIMEOUT_SECONDS", "180")),
            api_key=api_key,
        )
        settings.validate()
        return settings

    @property
    def credential_configured(self) -> bool:
        return bool(self.api_key)

    def validate(self) -> None:
        if not self.model:
            raise ResponderError("responder model must not be empty")
        if not self.base_url:
            raise ResponderError("responder base URL must not be empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("RESPONDER_TEMPERATURE must be between zero and two")
        if self.max_output_tokens <= 0:
            raise ValueError("RESPONDER_MAX_OUTPUT_TOKENS must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("RESPONDER_TIMEOUT_SECONDS must be positive")
        if self.provider == "openai" and not self.api_key:
            raise ResponderError(
                "OPENAI_API_KEY is required when RESPONDER_PROVIDER=openai"
            )

    def public_config(self) -> dict[str, Any]:
        """Return responder provenance without exposing credentials."""
        return {
            "provider": self.provider,
            "model": self.model,
            "model_digest": self.model_digest,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "credential_configured": self.credential_configured,
        }


@dataclass(frozen=True)
class BackendCompletion:
    """Provider response before experiment timing and provenance are attached."""

    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class ResponderResult:
    """One completed response with stable provider and timing metadata."""

    content: str
    provider: str
    model: str
    model_digest: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    latency_seconds: float
    raw_response: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "model_digest": self.model_digest,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "latency_seconds": self.latency_seconds,
            "raw_response": self.raw_response,
        }


class ChatBackend(Protocol):
    """Small dependency-injection boundary for one chat provider."""

    def complete(
        self, messages: Sequence[ChatMessage], settings: ResponderSettings
    ) -> BackendCompletion: ...


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None,
    timeout: float,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=request_headers,
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ResponderError(f"responder returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ResponderError(f"responder is unavailable at {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise ResponderError("responder returned a non-object JSON response")
    return result


class OllamaChatBackend:
    """Generate responses through Ollama's native chat endpoint."""

    def complete(
        self, messages: Sequence[ChatMessage], settings: ResponderSettings
    ) -> BackendCompletion:
        response = _post_json(
            f"{settings.base_url}/api/chat",
            {
                "model": settings.model,
                "messages": [message.as_dict() for message in messages],
                "stream": False,
                "options": {
                    "temperature": settings.temperature,
                    "seed": settings.seed,
                    "num_predict": settings.max_output_tokens,
                },
            },
            headers=None,
            timeout=settings.timeout_seconds,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ResponderError("Ollama returned no response content")
        return BackendCompletion(
            content=content,
            prompt_tokens=_optional_int(response.get("prompt_eval_count")),
            completion_tokens=_optional_int(response.get("eval_count")),
            finish_reason=response.get("done_reason"),
            raw_response=response,
        )


class OpenAIChatBackend:
    """Generate responses through an OpenAI-compatible chat endpoint."""

    def complete(
        self, messages: Sequence[ChatMessage], settings: ResponderSettings
    ) -> BackendCompletion:
        if not settings.api_key:
            raise ResponderError("OpenAI responder credential is not configured")
        response = _post_json(
            f"{settings.base_url}/chat/completions",
            {
                "model": settings.model,
                "messages": [message.as_dict() for message in messages],
                "temperature": settings.temperature,
                "seed": settings.seed,
                "max_completion_tokens": settings.max_output_tokens,
            },
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout_seconds,
        )
        choices = response.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ResponderError("OpenAI-compatible endpoint returned no content")
        usage = response.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        return BackendCompletion(
            content=content,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            finish_reason=(
                first.get("finish_reason") if isinstance(first, Mapping) else None
            ),
            raw_response=response,
        )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class DedicatedResponder:
    """Validate and execute one frozen response-model configuration."""

    def __init__(
        self,
        settings: ResponderSettings,
        *,
        backend: ChatBackend | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self._backend = backend or self._default_backend(settings.provider)

    @staticmethod
    def _default_backend(provider: str) -> ChatBackend:
        if provider == "ollama":
            return OllamaChatBackend()
        if provider == "openai":
            return OpenAIChatBackend()
        raise ValueError(f"unsupported responder provider: {provider}")

    def verify(self) -> None:
        """Verify local model identity without issuing a completion."""
        self.settings.validate()
        if self.settings.provider != "ollama":
            return
        service = OllamaService(
            self.settings.base_url, timeout=self.settings.timeout_seconds
        )
        if not service.has_model(self.settings.model):
            raise ResponderError(
                f"Ollama responder model is not installed: {self.settings.model}"
            )
        service.verify_digest(self.settings.model, self.settings.model_digest)

    def respond(self, messages: Sequence[ChatMessage]) -> ResponderResult:
        if not messages:
            raise ValueError("at least one responder message is required")
        started = time.perf_counter()
        try:
            completion = self._backend.complete(messages, self.settings)
        except ResponderError:
            raise
        except Exception as exc:
            raise ResponderError(f"responder generation failed: {exc}") from exc
        latency = time.perf_counter() - started
        return ResponderResult(
            content=completion.content,
            provider=self.settings.provider,
            model=self.settings.model,
            model_digest=self.settings.model_digest,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            finish_reason=completion.finish_reason,
            latency_seconds=latency,
            raw_response=completion.raw_response,
        )
