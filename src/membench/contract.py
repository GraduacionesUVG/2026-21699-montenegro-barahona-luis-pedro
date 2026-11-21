"""Typed, versioned contract for resolved black-box experiment manifests."""

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class Mechanism(StrEnum):
    TELEMEM = "telemem"
    MEMANTO = "memanto"
    LANGMEM = "langmem"
    DAKERA = "dakera"


class WritePolicy(StrEnum):
    DIRECT = "direct"
    SHARED_GUARDED = "shared_guarded"
    NATIVE_SELECTIVE = "native_selective"


SUPPORTED_POLICIES: dict[Mechanism, frozenset[WritePolicy]] = {
    Mechanism.TELEMEM: frozenset(WritePolicy),
    Mechanism.MEMANTO: frozenset(WritePolicy),
    Mechanism.LANGMEM: frozenset(WritePolicy),
    # Dakera's public memory API stores the candidate supplied by the caller.
    # Until a separately versioned conversation-selection writer is qualified,
    # describing that path as mechanism-native selective writing would be false.
    Mechanism.DAKERA: frozenset(
        {WritePolicy.DIRECT, WritePolicy.SHARED_GUARDED}
    ),
}

LANGMEM_WRITER_IMPLEMENTATION = "langmem_store_manager_v1"
LANGMEM_WRITER_VERSION = "0.0.30"
DAKERA_SDK_VERSION = "0.12.12"
_IMMUTABLE_REVISION_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"
_MODEL_DIGEST_PATTERN = r"^(?:sha256:)?[0-9a-f]{64}$"
_DAKERA_IMAGE_PATTERN = r"^ghcr\.io/dakera-ai/dakera@sha256:[0-9a-f]{64}$"


class DatasetConfig(StrictModel):
    name: str = Field(pattern=r"^personamem_v2$")
    revision: str = Field(pattern=_IMMUTABLE_REVISION_PATTERN)
    subset: str = Field(pattern=r"^blackbox_battery_v1$")


class TeleMemOptions(StrictModel):
    """Result-affecting TeleMem settings declared by a resolved run."""

    infer: bool
    buffer_size: int = Field(gt=0)
    similarity_threshold: float = Field(ge=0, le=1)
    rerank: bool


class LangMemOptions(StrictModel):
    """Pinned LangMem writer and LangGraph storage behavior."""

    writer_implementation: str = Field(
        pattern=rf"^{re.escape(LANGMEM_WRITER_IMPLEMENTATION)}$"
    )
    writer_version: str = Field(
        pattern=rf"^{re.escape(LANGMEM_WRITER_VERSION)}$"
    )
    memory_types: tuple[
        Literal["semantic", "episodic", "procedural"], ...
    ] = Field(min_length=1)
    procedural_prompt_optimization: Literal[False]
    storage_backend: Literal["memory", "sqlite", "postgres"]
    enable_deletes: bool
    query_limit: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    temperature: float = Field(ge=0, le=2)
    seed: int = Field(ge=0)

    @field_validator("memory_types")
    @classmethod
    def memory_types_must_be_unique(
        cls,
        value: tuple[Literal["semantic", "episodic", "procedural"], ...],
    ) -> tuple[Literal["semantic", "episodic", "procedural"], ...]:
        if len(value) != len(set(value)):
            raise ValueError("LangMem memory_types must not contain duplicates")
        return value


class DakeraOptions(StrictModel):
    """Pinned Dakera SDK, server image, storage, and retrieval behavior."""

    sdk_version: str = Field(pattern=rf"^{re.escape(DAKERA_SDK_VERSION)}$")
    server_image: str = Field(pattern=_DAKERA_IMAGE_PATTERN)
    memory_type: Literal["semantic", "episodic", "procedural", "working"]
    importance: float = Field(ge=0, le=1)
    routing: Literal["auto", "vector", "bm25", "hybrid"]
    rerank: bool
    neighborhood: bool
    include_associated: bool
    ttl_seconds: int | None = Field(default=None, gt=0)


type MechanismOptions = (
    TeleMemOptions | LangMemOptions | DakeraOptions | dict[str, Any]
)


class ConditionConfig(StrictModel):
    mechanism: Mechanism
    mechanism_version: str = Field(min_length=1)
    write_policy: WritePolicy
    mechanism_options: MechanismOptions

    @model_validator(mode="before")
    @classmethod
    def parse_mechanism_options(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        try:
            mechanism = Mechanism(result.get("mechanism"))
        except (TypeError, ValueError):
            return result
        options = result.get("mechanism_options", {})
        option_model: type[StrictModel] | None = {
            Mechanism.TELEMEM: TeleMemOptions,
            Mechanism.LANGMEM: LangMemOptions,
            Mechanism.DAKERA: DakeraOptions,
        }.get(mechanism)
        if option_model is not None:
            result["mechanism_options"] = option_model.model_validate(options)
        return result

    @model_validator(mode="after")
    def policy_must_be_supported(self) -> "ConditionConfig":
        if self.write_policy not in SUPPORTED_POLICIES[self.mechanism]:
            raise ValueError(
                f"{self.mechanism.value} does not support "
                f"{self.write_policy.value}"
            )
        if self.mechanism == Mechanism.TELEMEM:
            if not isinstance(self.mechanism_options, TeleMemOptions):
                raise ValueError("TeleMem requires typed mechanism_options")
            expected_infer = self.write_policy == WritePolicy.NATIVE_SELECTIVE
            if self.mechanism_options.infer != expected_infer:
                raise ValueError(
                    "TeleMem infer must be true only for native_selective"
                )
        if self.mechanism == Mechanism.LANGMEM:
            if not isinstance(self.mechanism_options, LangMemOptions):
                raise ValueError("LangMem requires typed mechanism_options")
            if self.mechanism_version != self.mechanism_options.writer_version:
                raise ValueError(
                    "LangMem mechanism_version must equal its writer_version"
                )
        if self.mechanism == Mechanism.DAKERA and not isinstance(
            self.mechanism_options, DakeraOptions
        ):
            raise ValueError("Dakera requires typed mechanism_options")
        return self


class ModelRef(StrictModel):
    provider: Literal["ollama", "openai"]
    model: str = Field(min_length=1)
    digest: str | None = Field(default=None, pattern=_MODEL_DIGEST_PATTERN)


class EmbeddingModelRef(ModelRef):
    dimensions: int = Field(gt=0)


class HuggingFaceResponderTokenizerRef(StrictModel):
    """Immutable AutoTokenizer identity used by the responder condition."""

    implementation: Literal["huggingface_auto"]
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    transformers_version: Literal["4.57.6"]
    use_fast: Literal[True]


class RegexResponderTokenizerRef(StrictModel):
    """Dependency-free tokenizer allowed only for approximate smoke runs."""

    implementation: Literal["regex_lexical_v1"]


type ResponderTokenizerRef = Annotated[
    HuggingFaceResponderTokenizerRef | RegexResponderTokenizerRef,
    Field(discriminator="implementation"),
]


class ResponderModelRef(ModelRef):
    tokenizer: ResponderTokenizerRef


class ModelsConfig(StrictModel):
    writer: ModelRef | None
    responder: ResponderModelRef
    embedding: EmbeddingModelRef | None


class RetrievalConfig(StrictModel):
    top_k: int = Field(gt=0)
    threshold: float | None = None
    context_token_budget: int = Field(gt=0)


class AttackConfig(StrictModel):
    knowledge: str = Field(pattern=r"^black_box$")
    delivery: str = Field(pattern=r"^conversation$")
    battery: str = Field(pattern=r"^pmv2_blackbox_battery_v1$")
    attack_id: str = Field(pattern=r"^bb-(pi|opc)-\d{3}$")


class IsolationConfig(StrictModel):
    run_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    config_ref: str = "planning/config.md"


class ReplicationConfig(StrictModel):
    seed: int = Field(ge=0)
    response_replicates: int = Field(gt=0)


class ExperimentManifest(StrictModel):
    schema_version: int = Field(ge=1, le=1)
    experiment_track: str = Field(pattern=r"^blackbox_memory_poisoning$")
    dataset: DatasetConfig
    condition: ConditionConfig
    models: ModelsConfig
    retrieval: RetrievalConfig
    attack: AttackConfig
    isolation: IsolationConfig
    replication: ReplicationConfig

    @model_validator(mode="after")
    def required_models_must_match_the_mechanism(self) -> "ExperimentManifest":
        if self.condition.mechanism in {
            Mechanism.TELEMEM,
            Mechanism.LANGMEM,
            Mechanism.MEMANTO,
        }:
            if self.models.writer is None:
                raise ValueError(
                    f"{self.condition.mechanism.value} requires a writer model"
                )
            if self.models.embedding is None:
                raise ValueError(
                    f"{self.condition.mechanism.value} requires an embedding model"
                )
        return self

    @property
    def sha256(self) -> str:
        """Hash the complete validated manifest using canonical JSON."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> ExperimentManifest:
    """Load and validate one experiment manifest."""
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return ExperimentManifest.model_validate(data)
