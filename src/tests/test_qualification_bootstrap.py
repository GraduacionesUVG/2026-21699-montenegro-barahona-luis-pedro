from pathlib import Path

import pytest

from membench.contract import ExperimentManifest, Mechanism, load_manifest
from membench.memory.errors import AdapterConfigurationError
from membench.qualification.bootstrap import (
    bootstrap_qualification,
    resolve_responder_settings,
    resolve_responder_tokenizer_settings,
    resolve_settings,
)
from membench.responder import ResponderSettings
from membench.responder_tokenizer import ResponderTokenizerSettings
from membench.telemem import TeleMemSettings

_CHAT_DIGEST = "a" * 64
_EMBEDDING_DIGEST = "b" * 64


def _telemem_settings() -> TeleMemSettings:
    return TeleMemSettings(
        provider="ollama",
        base_url="http://127.0.0.1:11434/v1",
        llm_model="qwen2.5:7b",
        embedding_model="nomic-embed-text:137m-v1.5-fp16",
        embedding_dimensions=768,
        storage_root=Path("artifacts/memory/telemem"),
        buffer_size=8,
        similarity_threshold=0.1,
        retrieval_threshold=0.2,
        rerank=True,
        visibility_timeout_seconds=30,
        poll_interval_seconds=0.25,
        llm_model_digest=_CHAT_DIGEST,
        embedding_model_digest=_EMBEDDING_DIGEST,
        api_key="ollama",
    )


def _responder_settings() -> ResponderSettings:
    return ResponderSettings(
        provider="ollama",
        model="qwen2.5:7b",
        model_digest=_CHAT_DIGEST,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        max_output_tokens=128,
        timeout_seconds=30,
    )


def test_resolve_settings_overlays_manifest_values_and_keeps_secrets() -> None:
    manifest = load_manifest(
        Path("configs/experiments/blackbox/example.yaml")
    )

    resolved = resolve_settings(
        manifest,
        environment_settings=_telemem_settings(),
    )

    assert isinstance(resolved, TeleMemSettings)
    assert resolved.api_key == "ollama"
    assert resolved.buffer_size == 64
    assert resolved.similarity_threshold == 0.95
    assert resolved.retrieval_threshold is None
    assert not resolved.rerank
    assert resolved.llm_model_digest == _CHAT_DIGEST
    assert resolved.embedding_model_digest == _EMBEDDING_DIGEST


def test_resolve_settings_rejects_provider_mismatch() -> None:
    manifest = load_manifest(
        Path("configs/experiments/blackbox/example.yaml")
    )
    mismatched = TeleMemSettings(
        **{
            **_telemem_settings().__dict__,
            "provider": "openai",
        }
    )

    with pytest.raises(AdapterConfigurationError, match="does not match"):
        resolve_settings(manifest, environment_settings=mismatched)


def test_bootstrap_builds_plan_and_fresh_uninitialized_adapters() -> None:
    manifest = load_manifest(
        Path("configs/experiments/blackbox/example.yaml")
    )

    prepared = bootstrap_qualification(
        manifest,
        environment_settings=_telemem_settings(),
    )

    assert prepared.plan.mechanism == Mechanism.TELEMEM
    assert prepared.plan.scope.run_id == manifest.isolation.run_id
    assert prepared.plan.manifest_sha256 == manifest.sha256
    assert prepared.capability_validation.supported
    first = prepared.adapter_factory.create()
    second = prepared.adapter_factory.create()
    assert first is not second


def test_bootstrap_requires_pinned_ollama_model_digests() -> None:
    manifest = load_manifest(
        Path("configs/experiments/blackbox/example.yaml")
    )
    unpinned = TeleMemSettings(
        **{
            **_telemem_settings().__dict__,
            "llm_model_digest": None,
        }
    )

    with pytest.raises(AdapterConfigurationError, match="immutable Ollama"):
        bootstrap_qualification(manifest, environment_settings=unpinned)


def test_bootstrap_derives_a_distinct_scope_for_isolation() -> None:
    source = load_manifest(
        Path("configs/experiments/blackbox/example.yaml")
    ).model_dump(mode="json")
    source["condition"]["write_policy"] = "direct"
    source["condition"]["mechanism_options"]["infer"] = False
    manifest = ExperimentManifest.model_validate(source)

    prepared = bootstrap_qualification(
        manifest,
        qualification_id="namespace_isolation_v1",
        environment_settings=_telemem_settings(),
    )

    secondary = prepared.plan.secondary_scope
    assert secondary is not None
    assert secondary.experiment_id == prepared.plan.scope.experiment_id
    assert secondary.run_id != prepared.plan.scope.run_id
    assert secondary.user_id != prepared.plan.scope.user_id
    assert secondary.namespace != prepared.plan.scope.namespace


def test_bootstrap_resolves_responder_qualification_settings() -> None:
    source = load_manifest(
        Path("configs/experiments/blackbox/example.yaml")
    ).model_dump(mode="json")
    source["condition"]["write_policy"] = "direct"
    source["condition"]["mechanism_options"]["infer"] = False
    manifest = ExperimentManifest.model_validate(source)

    resolved = resolve_responder_settings(
        manifest,
        environment_settings=_responder_settings(),
    )
    tokenizer = resolve_responder_tokenizer_settings(manifest)
    prepared = bootstrap_qualification(
        manifest,
        qualification_id="responder_read_only_v1",
        environment_settings=_telemem_settings(),
        environment_responder_settings=_responder_settings(),
    )

    assert resolved.model == manifest.models.responder.model
    assert resolved.model_digest == _CHAT_DIGEST
    assert resolved.seed == manifest.replication.seed
    assert prepared.responder_settings == resolved
    assert isinstance(tokenizer, ResponderTokenizerSettings)
    assert prepared.responder_tokenizer_settings == tokenizer
    assert tokenizer.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert tokenizer.transformers_version == "4.57.6"
    assert prepared.plan.responder_required
    assert (
        prepared.plan.context_token_budget
        == manifest.retrieval.context_token_budget
    )
