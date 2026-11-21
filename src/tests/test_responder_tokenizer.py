import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from membench.context_budget import RegexTokenCounter
from membench.responder import ChatMessage
from membench.responder_tokenizer import (
    PINNED_TRANSFORMERS_VERSION,
    HuggingFaceAutoTokenizer,
    RegexResponderTokenizerSettings,
    ResponderTokenizerError,
    ResponderTokenizerSettings,
    create_responder_tokenizer,
)

_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


class FakeAutoTokenizer:
    is_fast = True
    chat_template = "{{ messages }}{% if add_generation_prompt %}assistant{% endif %}"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(range(len(text.split())))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize
        content_tokens = sum(
            len(message["content"].split()) for message in messages
        )
        return list(range(content_tokens + 3 + int(add_generation_prompt)))


def _settings() -> ResponderTokenizerSettings:
    return ResponderTokenizerSettings(
        implementation="huggingface_auto",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        revision=_REVISION,
        transformers_version=PINNED_TRANSFORMERS_VERSION,
        use_fast=True,
    )


def test_huggingface_tokenizer_counts_text_and_complete_messages() -> None:
    vendor = FakeAutoTokenizer()
    counter = HuggingFaceAutoTokenizer(
        _settings(),
        vendor,
        installed_transformers_version=PINNED_TRANSFORMERS_VERSION,
    )
    messages = (
        ChatMessage(role="system", content="Use supplied memories."),
        ChatMessage(role="user", content="Which notebook is mine?"),
    )

    assert counter.count("one two three") == 3
    assert counter.count_messages(messages) == 11
    expected_hash = hashlib.sha256(
        vendor.chat_template.encode("utf-8")
    ).hexdigest()
    assert counter.chat_template_sha256 == expected_hash
    assert _REVISION in counter.name
    assert f"transformers={PINNED_TRANSFORMERS_VERSION}" in counter.name
    assert expected_hash in counter.name

    with pytest.raises(ValueError, match="at least one message"):
        counter.count_messages(())


@pytest.mark.parametrize(
    ("vendor_updates", "version", "message"),
    [
        ({"is_fast": False}, PINNED_TRANSFORMERS_VERSION, "fast tokenizer"),
        ({"chat_template": None}, PINNED_TRANSFORMERS_VERSION, "chat template"),
        ({}, "0.0.0", "version mismatch"),
    ],
)
def test_huggingface_tokenizer_rejects_runtime_drift(
    vendor_updates: dict[str, Any],
    version: str,
    message: str,
) -> None:
    vendor = FakeAutoTokenizer()
    for name, value in vendor_updates.items():
        setattr(vendor, name, value)

    with pytest.raises(ResponderTokenizerError, match=message):
        HuggingFaceAutoTokenizer(
            _settings(),
            vendor,
            installed_transformers_version=version,
        )


def test_regex_tokenizer_requires_explicit_smoke_configuration() -> None:
    counter = create_responder_tokenizer(RegexResponderTokenizerSettings())

    assert isinstance(counter, RegexTokenCounter)


def test_loader_resolves_pinned_snapshot_before_autotokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "snapshot"
    snapshot_calls: list[dict[str, Any]] = []
    tokenizer_calls: list[tuple[str, dict[str, Any]]] = []

    fake_hub = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: Any) -> str:
        snapshot_calls.append(kwargs)
        return str(snapshot_path)

    fake_hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]

    class FakeAutoFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> FakeAutoTokenizer:
            tokenizer_calls.append((path, kwargs))
            return FakeAutoTokenizer()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoFactory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "membench.responder_tokenizer.importlib.metadata.version",
        lambda package: PINNED_TRANSFORMERS_VERSION,
    )
    settings = replace(
        _settings(),
        cache_dir=tmp_path / "cache",
        local_files_only=True,
    )

    counter = HuggingFaceAutoTokenizer.load(settings)

    assert isinstance(counter, HuggingFaceAutoTokenizer)
    assert snapshot_calls[0]["repo_id"] == settings.model_id
    assert snapshot_calls[0]["revision"] == settings.revision
    assert snapshot_calls[0]["local_files_only"] is True
    assert "tokenizer.json" in snapshot_calls[0]["allow_patterns"]
    assert tokenizer_calls == [
        (
            str(snapshot_path),
            {
                "use_fast": True,
                "trust_remote_code": False,
                "local_files_only": True,
            },
        )
    ]
