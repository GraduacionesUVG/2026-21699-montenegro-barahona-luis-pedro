"""Pinned responder tokenizers used for selection and complete-prompt budgeting."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from membench.context_budget import (
    MessageTokenCounter,
    RegexTokenCounter,
    TokenizableMessage,
)

PINNED_TRANSFORMERS_VERSION = "4.57.6"
_TOKENIZER_FILE_PATTERNS = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class ResponderTokenizerError(RuntimeError):
    """Raised when a frozen responder tokenizer cannot be loaded faithfully."""


@dataclass(frozen=True, slots=True)
class ResponderTokenizerSettings:
    """Result-affecting identity plus operational Hugging Face cache settings."""

    implementation: Literal["huggingface_auto"]
    model_id: str
    revision: str
    transformers_version: str
    use_fast: bool
    cache_dir: Path | None = None
    local_files_only: bool = False

    @classmethod
    def from_manifest(
        cls,
        *,
        implementation: str,
        model_id: str,
        revision: str,
        transformers_version: str,
        use_fast: bool,
    ) -> ResponderTokenizerSettings:
        """Combine immutable manifest identity with non-semantic cache controls."""
        load_dotenv(override=False)
        cache_text = os.getenv("RESPONDER_TOKENIZER_CACHE_DIR", "").strip()
        local_text = os.getenv(
            "RESPONDER_TOKENIZER_LOCAL_FILES_ONLY", "false"
        ).strip().casefold()
        if local_text not in {"true", "false"}:
            raise ValueError(
                "RESPONDER_TOKENIZER_LOCAL_FILES_ONLY must be true or false"
            )
        if implementation != "huggingface_auto":
            raise ValueError("unsupported responder tokenizer implementation")
        settings = cls(
            implementation=implementation,
            model_id=model_id,
            revision=revision,
            transformers_version=transformers_version,
            use_fast=use_fast,
            cache_dir=Path(cache_text) if cache_text else None,
            local_files_only=local_text == "true",
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.implementation != "huggingface_auto":
            raise ValueError("unsupported responder tokenizer implementation")
        if not self.model_id.strip():
            raise ValueError("responder tokenizer model_id must not be blank")
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError(
                "responder tokenizer revision must be an immutable 40-character "
                "lowercase hexadecimal commit"
            )
        if self.transformers_version != PINNED_TRANSFORMERS_VERSION:
            raise ValueError(
                "responder tokenizer requires transformers=="
                f"{PINNED_TRANSFORMERS_VERSION}"
            )
        if not self.use_fast:
            raise ValueError("responder tokenizer must use the frozen fast tokenizer")

    def public_config(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "model_id": self.model_id,
            "revision": self.revision,
            "transformers_version": self.transformers_version,
            "use_fast": self.use_fast,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
            "local_files_only": self.local_files_only,
        }


@dataclass(frozen=True, slots=True)
class RegexResponderTokenizerSettings:
    """Explicit approximation configuration for dependency-free smoke runs."""

    implementation: Literal["regex_lexical_v1"] = "regex_lexical_v1"

    def public_config(self) -> dict[str, object]:
        return {"implementation": self.implementation}


type ResponderTokenizerConfiguration = (
    ResponderTokenizerSettings | RegexResponderTokenizerSettings
)


class HuggingFaceAutoTokenizer:
    """Count text and complete chats with one pinned fast AutoTokenizer."""

    def __init__(
        self,
        settings: ResponderTokenizerSettings,
        tokenizer: Any,
        *,
        installed_transformers_version: str,
    ) -> None:
        settings.validate()
        if installed_transformers_version != settings.transformers_version:
            raise ResponderTokenizerError(
                "transformers version mismatch: expected "
                f"{settings.transformers_version}, got "
                f"{installed_transformers_version}"
            )
        if settings.use_fast and not bool(getattr(tokenizer, "is_fast", False)):
            raise ResponderTokenizerError(
                "Hugging Face did not return the required fast tokenizer"
            )
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template.strip():
            raise ResponderTokenizerError(
                "responder tokenizer does not define a chat template"
            )
        self.settings = settings
        self._tokenizer = tokenizer
        self.chat_template_sha256 = hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()

    @classmethod
    def load(
        cls,
        settings: ResponderTokenizerSettings,
    ) -> HuggingFaceAutoTokenizer:
        """Load only tokenizer assets and reject package or identity drift."""
        settings.validate()
        try:
            installed_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ResponderTokenizerError(
                "transformers is not installed; sync the responder-tokenizer "
                "dependency group"
            ) from exc
        if installed_version != settings.transformers_version:
            raise ResponderTokenizerError(
                "transformers version mismatch: expected "
                f"{settings.transformers_version}, got {installed_version}"
        )
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ResponderTokenizerError(
                "transformers or huggingface-hub could not import tokenizer support"
            ) from exc
        try:
            snapshot_path = snapshot_download(
                repo_id=settings.model_id,
                revision=settings.revision,
                cache_dir=(
                    str(settings.cache_dir)
                    if settings.cache_dir is not None
                    else None
                ),
                local_files_only=settings.local_files_only,
                allow_patterns=list(_TOKENIZER_FILE_PATTERNS),
            )
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot_path,
                use_fast=settings.use_fast,
                trust_remote_code=False,
                local_files_only=True,
            )
        except Exception as exc:
            raise ResponderTokenizerError(
                "failed to load pinned Hugging Face responder tokenizer "
                f"{settings.model_id}@{settings.revision}: {exc}"
            ) from exc
        return cls(
            settings,
            tokenizer,
            installed_transformers_version=installed_version,
        )

    @property
    def name(self) -> str:
        return (
            "huggingface_auto_v1:"
            f"{self.settings.model_id}@{self.settings.revision}:"
            f"transformers={self.settings.transformers_version}:"
            f"chat_template_sha256={self.chat_template_sha256}"
        )

    def count(self, text: str) -> int:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return len(token_ids)

    def count_messages(
        self,
        messages: Sequence[TokenizableMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        if not messages:
            raise ValueError("at least one message is required for tokenization")
        payload = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]
        token_ids = self._tokenizer.apply_chat_template(
            payload,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return len(token_ids)

    def public_config(self) -> dict[str, object]:
        return {
            **self.settings.public_config(),
            "chat_template_sha256": self.chat_template_sha256,
            "counter_name": self.name,
        }


def create_responder_tokenizer(
    settings: ResponderTokenizerConfiguration,
) -> MessageTokenCounter:
    """Create the declared counter without silently changing implementations."""
    if isinstance(settings, ResponderTokenizerSettings):
        return HuggingFaceAutoTokenizer.load(settings)
    if isinstance(settings, RegexResponderTokenizerSettings):
        return RegexTokenCounter()
    raise TypeError("unsupported responder tokenizer settings")
