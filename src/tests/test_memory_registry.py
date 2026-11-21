from pathlib import Path

import pytest

from membench.contract import Mechanism
from membench.memory.adapters.telemem_adapter import TeleMemAdapter
from membench.memory.errors import AdapterConfigurationError
from membench.memory.registry import (
    AdapterRegistry,
    available_mechanisms,
    create,
)
from membench.telemem import TeleMemSettings


class ExampleSettings:
    pass


class ExampleAdapter:
    def __init__(self, settings: ExampleSettings) -> None:
        self.settings = settings


def _telemem_settings() -> TeleMemSettings:
    return TeleMemSettings(
        provider="ollama",
        base_url="http://127.0.0.1:11434/v1",
        llm_model="qwen2.5:7b",
        embedding_model="nomic-embed-text:137m-v1.5-fp16",
        embedding_dimensions=768,
        storage_root=Path("artifacts/memory/telemem"),
        buffer_size=64,
        similarity_threshold=0.95,
        retrieval_threshold=None,
        rerank=False,
        visibility_timeout_seconds=30,
        poll_interval_seconds=0.25,
        llm_model_digest=None,
        embedding_model_digest=None,
        api_key="ollama",
    )


def test_custom_registry_constructs_aliases_without_initializing_services() -> None:
    registry = AdapterRegistry()
    registry.register(
        "example",
        ExampleSettings,
        ExampleAdapter,
        aliases=("sample",),
    )
    settings = ExampleSettings()

    adapter = registry.create("sample", settings)

    assert isinstance(adapter, ExampleAdapter)
    assert adapter.settings is settings
    assert registry.available_mechanisms() == ("example",)
    assert registry.available_mechanisms(include_aliases=True) == (
        "example",
        "sample",
    )


def test_registry_rejects_duplicates_unknown_mechanisms_and_wrong_settings() -> None:
    registry = AdapterRegistry()
    registry.register("example", ExampleSettings, ExampleAdapter)

    with pytest.raises(AdapterConfigurationError, match="already registered"):
        registry.register("example", ExampleSettings, ExampleAdapter)
    with pytest.raises(AdapterConfigurationError, match="no memory adapter"):
        registry.create("unknown", ExampleSettings())
    with pytest.raises(AdapterConfigurationError, match="ExampleSettings"):
        registry.create("example", object())


def test_default_registry_uses_manifest_names_and_reports_memanto_unavailable() -> None:
    settings = _telemem_settings()

    adapter = create(Mechanism.TELEMEM, settings)

    assert isinstance(adapter, TeleMemAdapter)
    assert adapter.settings is settings
    assert available_mechanisms() == ("dakera", "langmem", "telemem")
    assert "langgraph" in available_mechanisms(include_aliases=True)
    with pytest.raises(AdapterConfigurationError, match="memanto"):
        create(Mechanism.MEMANTO, settings)
