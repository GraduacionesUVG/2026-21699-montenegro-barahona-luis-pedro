"""Resolve manifests into reproducible settings and qualification runners."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from membench.canary_data import CanarySuite
from membench.context_budget import MessageTokenCounter
from membench.contract import (
    DakeraOptions,
    ExperimentManifest,
    HuggingFaceResponderTokenizerRef,
    LangMemOptions,
    Mechanism,
    RegexResponderTokenizerRef,
    TeleMemOptions,
    WritePolicy,
    load_manifest,
)
from membench.dakera import (
    DakeraMemoryType,
    DakeraRouting,
    DakeraSettings,
)
from membench.langmem import LangMemSettings
from membench.langmem import MemoryType as LangMemMemoryType
from membench.memory.errors import (
    AdapterConfigurationError,
    UnsupportedCapabilityError,
)
from membench.memory.models import MemoryScope, MemoryType
from membench.memory.policy import AdmissionGuard
from membench.memory.protocol import MemoryAdapter
from membench.memory.registry import DEFAULT_REGISTRY, AdapterRegistry
from membench.memory.requirements import (
    CapabilityValidation,
    validate_adapter_capabilities,
)
from membench.qualification.artifacts import (
    QualificationArtifactIndex,
    QualificationArtifactStore,
)
from membench.qualification.events import QualificationEventSink
from membench.qualification.models import (
    QualificationPlan,
    QualificationReport,
)
from membench.qualification.responder_stage import (
    QualificationContextAssembler,
    QualificationResponder,
)
from membench.qualification.runner import QualificationRunner
from membench.responder import DedicatedResponder, ResponderSettings
from membench.responder_tokenizer import (
    RegexResponderTokenizerSettings,
    ResponderTokenizerConfiguration,
    ResponderTokenizerSettings,
    create_responder_tokenizer,
)
from membench.telemem import TeleMemSettings

type AdapterSettings = TeleMemSettings | LangMemSettings | DakeraSettings

_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class QualificationAdapterFactory:
    """Create fresh, equivalently configured adapters for isolated scopes."""

    mechanism: Mechanism
    settings: AdapterSettings
    registry: AdapterRegistry = field(
        default=DEFAULT_REGISTRY,
        repr=False,
        compare=False,
    )

    def create(self) -> MemoryAdapter:
        """Construct an uninitialized adapter and verify its protocol surface."""
        adapter = self.registry.create(self.mechanism, self.settings)
        if not isinstance(adapter, MemoryAdapter):
            raise AdapterConfigurationError(
                f"registered {self.mechanism.value} factory did not return a "
                "MemoryAdapter"
            )
        return adapter


@dataclass(frozen=True, slots=True)
class PreparedQualification:
    """Fully resolved inputs that are safe to execute without more defaults."""

    manifest: ExperimentManifest
    settings: AdapterSettings
    plan: QualificationPlan
    suite: CanarySuite
    capability_validation: CapabilityValidation
    adapter_factory: QualificationAdapterFactory
    responder_settings: ResponderSettings | None = field(
        default=None,
        repr=False,
    )
    responder_tokenizer_settings: ResponderTokenizerConfiguration | None = field(
        default=None,
        repr=False,
    )

    def create_runner(
        self,
        *,
        event_sink: QualificationEventSink | None = None,
        guard: AdmissionGuard | None = None,
        responder: QualificationResponder | None = None,
        responder_tokenizer: MessageTokenCounter | None = None,
    ) -> QualificationRunner:
        """Create a runner with a fresh adapter for one lifecycle execution."""
        resolved_responder = responder
        if resolved_responder is None and self.responder_settings is not None:
            resolved_responder = DedicatedResponder(self.responder_settings)
        resolved_tokenizer = responder_tokenizer
        if (
            resolved_tokenizer is None
            and self.responder_tokenizer_settings is not None
        ):
            resolved_tokenizer = create_responder_tokenizer(
                self.responder_tokenizer_settings
            )
        context_assembler = None
        if resolved_tokenizer is not None:
            context_assembler = QualificationContextAssembler(
                selection_token_counter=resolved_tokenizer,
                responder_tokenizer=resolved_tokenizer,
            )
        return QualificationRunner(
            self.adapter_factory.create(),
            self.suite,
            event_sink=event_sink,
            guard=guard,
            secondary_adapter_factory=self.adapter_factory.create,
            responder=resolved_responder,
            context_assembler=context_assembler,
        )

    def execute(
        self,
        artifact_root: Path,
        *,
        guard: AdmissionGuard | None = None,
        responder: QualificationResponder | None = None,
        responder_tokenizer: MessageTokenCounter | None = None,
    ) -> tuple[QualificationReport, QualificationArtifactIndex]:
        """Run once and seal all qualification evidence under a new directory."""
        artifacts = QualificationArtifactStore.prepare(artifact_root, self.plan)
        artifacts.write_manifest(self.manifest)
        artifacts.write_plan(self.plan)
        artifacts.write_resolved_settings(self.settings.public_config())
        if self.responder_settings is not None:
            artifacts.write_responder_settings(
                self.responder_settings.public_config()
            )
        if self.responder_tokenizer_settings is not None:
            artifacts.write_responder_tokenizer_settings(
                self.responder_tokenizer_settings.public_config()
            )
        artifacts.write_capability_validation(self.capability_validation)
        runner = self.create_runner(
            event_sink=artifacts.event_sink,
            guard=guard,
            responder=responder,
            responder_tokenizer=responder_tokenizer,
        )
        report = runner.run(self.plan)
        return report, artifacts.seal(report)


def bootstrap_qualification(
    manifest: ExperimentManifest,
    *,
    qualification_id: str = "baseline_semantic_v1",
    user_id: str = "qualification-user",
    environment_settings: AdapterSettings | None = None,
    environment_responder_settings: ResponderSettings | None = None,
    registry: AdapterRegistry = DEFAULT_REGISTRY,
    suite: CanarySuite | None = None,
) -> PreparedQualification:
    """Resolve, validate, and prepare one side-effect-free qualification."""
    if not _IDENTIFIER_PATTERN.fullmatch(user_id):
        raise AdapterConfigurationError(
            "qualification user_id must start with an alphanumeric character and "
            "contain only letters, numbers, '.', '_' or '-'"
        )

    loaded_suite = suite or CanarySuite.load()
    try:
        qualification = loaded_suite.qualification(qualification_id)
    except KeyError as exc:
        raise AdapterConfigurationError(str(exc)) from exc
    if manifest.condition.write_policy not in qualification.supported_policies:
        raise AdapterConfigurationError(
            f"qualification {qualification_id!r} does not support policy "
            f"{manifest.condition.write_policy.value!r}"
        )

    settings = resolve_settings(manifest, environment_settings=environment_settings)
    responder_settings = None
    responder_tokenizer_settings = None
    if qualification.requires_responder:
        responder_settings = resolve_responder_settings(
            manifest,
            environment_settings=environment_responder_settings,
        )
        responder_tokenizer_settings = resolve_responder_tokenizer_settings(
            manifest
        )
    elif environment_responder_settings is not None:
        raise AdapterConfigurationError(
            "responder settings were supplied to a qualification that does not "
            "use a responder"
        )
    factory = QualificationAdapterFactory(
        mechanism=manifest.condition.mechanism,
        settings=settings,
        registry=registry,
    )
    probe = factory.create()
    capability_validation = validate_adapter_capabilities(manifest, probe)
    missing = tuple(
        name
        for name in qualification.required_capabilities
        if not probe.capabilities[name]
    )
    if missing:
        raise UnsupportedCapabilityError(
            f"{manifest.condition.mechanism.value} adapter cannot run "
            f"qualification {qualification_id!r}; missing capabilities: "
            + ", ".join(missing),
            missing=missing,
        )

    run_id = manifest.isolation.run_id
    scope = MemoryScope(
        experiment_id=f"{manifest.experiment_track}-qualification",
        run_id=run_id,
        user_id=user_id,
        namespace=(
            "membench",
            "qualification",
            manifest.condition.mechanism.value,
            run_id,
            user_id,
        ),
    )
    secondary_scope = None
    if qualification.requires_secondary_scope:
        secondary_run_id = f"{run_id}-peer"
        secondary_user_id = f"{user_id}-peer"
        secondary_scope = MemoryScope(
            experiment_id=scope.experiment_id,
            run_id=secondary_run_id,
            user_id=secondary_user_id,
            namespace=(
                "membench",
                "qualification",
                manifest.condition.mechanism.value,
                secondary_run_id,
                secondary_user_id,
            ),
        )
    plan = QualificationPlan(
        qualification_id=qualification_id,
        mechanism=manifest.condition.mechanism,
        policy=manifest.condition.write_policy,
        scope=scope,
        secondary_scope=secondary_scope,
        top_k=manifest.retrieval.top_k,
        threshold=manifest.retrieval.threshold,
        context_token_budget=manifest.retrieval.context_token_budget,
        responder_required=qualification.requires_responder,
        responder_provider=(
            responder_settings.provider if responder_settings is not None else None
        ),
        responder_model=(
            responder_settings.model if responder_settings is not None else None
        ),
        responder_model_digest=(
            responder_settings.model_digest
            if responder_settings is not None
            else None
        ),
        cleanup=True,
        manifest_sha256=manifest.sha256,
        native_options_by_canary=_native_options(
            manifest,
            loaded_suite,
            qualification_id,
        ),
    )
    return PreparedQualification(
        manifest=manifest,
        settings=settings,
        plan=plan,
        suite=loaded_suite,
        capability_validation=capability_validation,
        adapter_factory=factory,
        responder_settings=responder_settings,
        responder_tokenizer_settings=responder_tokenizer_settings,
    )


def bootstrap_from_path(
    manifest_path: Path,
    **kwargs: object,
) -> PreparedQualification:
    """Load a manifest from disk and pass it through the normal bootstrap."""
    return bootstrap_qualification(load_manifest(manifest_path), **kwargs)


def resolve_settings(
    manifest: ExperimentManifest,
    *,
    environment_settings: AdapterSettings | None = None,
) -> AdapterSettings:
    """Overlay immutable manifest choices onto secret-bearing environment settings."""
    mechanism = manifest.condition.mechanism
    if mechanism == Mechanism.MEMANTO:
        raise AdapterConfigurationError(
            "Memanto has no registered adapter or resolved settings implementation"
        )
    if mechanism == Mechanism.TELEMEM:
        base = _settings_or_env(environment_settings, TeleMemSettings)
        return _resolve_telemem(manifest, base)
    if mechanism == Mechanism.LANGMEM:
        base = _settings_or_env(environment_settings, LangMemSettings)
        return _resolve_langmem(manifest, base)
    if mechanism == Mechanism.DAKERA:
        base = _settings_or_env(environment_settings, DakeraSettings)
        return _resolve_dakera(manifest, base)
    raise AdapterConfigurationError(f"unsupported mechanism: {mechanism.value}")


def resolve_responder_settings(
    manifest: ExperimentManifest,
    *,
    environment_settings: ResponderSettings | None = None,
) -> ResponderSettings:
    """Overlay the manifest responder identity onto endpoint configuration."""
    if environment_settings is None:
        try:
            base = ResponderSettings.from_env()
        except (RuntimeError, ValueError) as exc:
            raise AdapterConfigurationError(
                f"could not load ResponderSettings from the environment: {exc}"
            ) from exc
    else:
        base = environment_settings
    model = manifest.models.responder
    _require_provider("responder", model.provider, base.provider)
    resolved = replace(
        base,
        model=model.model,
        model_digest=_resolve_digest(
            provider=model.provider,
            model=model.model,
            manifest_digest=model.digest,
            environment_model=base.model,
            environment_digest=base.model_digest,
            role="responder",
        ),
        seed=manifest.replication.seed,
    )
    try:
        resolved.validate()
    except (RuntimeError, ValueError) as exc:
        raise AdapterConfigurationError(
            f"resolved ResponderSettings is invalid: {exc}"
        ) from exc
    return resolved


def resolve_responder_tokenizer_settings(
    manifest: ExperimentManifest,
) -> ResponderTokenizerConfiguration:
    """Resolve the manifest's frozen tokenizer plus operational cache policy."""
    tokenizer = manifest.models.responder.tokenizer
    if isinstance(tokenizer, HuggingFaceResponderTokenizerRef):
        try:
            return ResponderTokenizerSettings.from_manifest(
                implementation=tokenizer.implementation,
                model_id=tokenizer.model_id,
                revision=tokenizer.revision,
                transformers_version=tokenizer.transformers_version,
                use_fast=tokenizer.use_fast,
            )
        except (RuntimeError, ValueError) as exc:
            raise AdapterConfigurationError(
                f"resolved responder tokenizer is invalid: {exc}"
            ) from exc
    if isinstance(tokenizer, RegexResponderTokenizerRef):
        return RegexResponderTokenizerSettings()
    raise AdapterConfigurationError(
        "manifest responder tokenizer implementation is unsupported"
    )


def _settings_or_env[SettingsT: AdapterSettings](
    supplied: AdapterSettings | None,
    expected: type[SettingsT],
) -> SettingsT:
    if supplied is None:
        try:
            return expected.from_env()
        except (RuntimeError, ValueError) as exc:
            raise AdapterConfigurationError(
                f"could not load {expected.__name__} from the environment: {exc}"
            ) from exc
    if not isinstance(supplied, expected):
        raise AdapterConfigurationError(
            f"{expected.__name__} is required for this manifest; got "
            f"{type(supplied).__name__}"
        )
    return supplied


def _resolve_telemem(
    manifest: ExperimentManifest,
    base: TeleMemSettings,
) -> TeleMemSettings:
    options = manifest.condition.mechanism_options
    if not isinstance(options, TeleMemOptions):
        raise AdapterConfigurationError("TeleMem mechanism options are unresolved")
    writer = manifest.models.writer
    embedding = manifest.models.embedding
    if writer is None or embedding is None:
        raise AdapterConfigurationError("TeleMem requires writer and embedding models")
    _require_provider("TeleMem writer", writer.provider, base.provider)
    _require_provider("TeleMem embedding", embedding.provider, base.provider)
    resolved = replace(
        base,
        llm_model=writer.model,
        llm_model_digest=_resolve_digest(
            provider=writer.provider,
            model=writer.model,
            manifest_digest=writer.digest,
            environment_model=base.llm_model,
            environment_digest=base.llm_model_digest,
            role="TeleMem writer",
        ),
        embedding_model=embedding.model,
        embedding_model_digest=_resolve_digest(
            provider=embedding.provider,
            model=embedding.model,
            manifest_digest=embedding.digest,
            environment_model=base.embedding_model,
            environment_digest=base.embedding_model_digest,
            role="TeleMem embedder",
        ),
        embedding_dimensions=embedding.dimensions,
        buffer_size=options.buffer_size,
        similarity_threshold=options.similarity_threshold,
        retrieval_threshold=manifest.retrieval.threshold,
        rerank=options.rerank,
    )
    _validate_runtime(resolved)
    return resolved


def _resolve_langmem(
    manifest: ExperimentManifest,
    base: LangMemSettings,
) -> LangMemSettings:
    options = manifest.condition.mechanism_options
    if not isinstance(options, LangMemOptions):
        raise AdapterConfigurationError("LangMem mechanism options are unresolved")
    writer = manifest.models.writer
    embedding = manifest.models.embedding
    if writer is None or embedding is None:
        raise AdapterConfigurationError("LangMem requires writer and embedding models")
    _require_provider("LangMem writer", writer.provider, base.writer_provider)
    _require_provider(
        "LangMem embedding",
        embedding.provider,
        base.embedding_provider,
    )
    resolved = replace(
        base,
        writer_model=writer.model,
        writer_model_digest=_resolve_digest(
            provider=writer.provider,
            model=writer.model,
            manifest_digest=writer.digest,
            environment_model=base.writer_model,
            environment_digest=base.writer_model_digest,
            role="LangMem writer",
        ),
        embedding_model=embedding.model,
        embedding_model_digest=_resolve_digest(
            provider=embedding.provider,
            model=embedding.model,
            manifest_digest=embedding.digest,
            environment_model=base.embedding_model,
            environment_digest=base.embedding_model_digest,
            role="LangMem embedder",
        ),
        embedding_dimensions=embedding.dimensions,
        memory_types=tuple(LangMemMemoryType(item) for item in options.memory_types),
        enable_deletes=options.enable_deletes,
        query_limit=options.query_limit,
        max_steps=options.max_steps,
        temperature=options.temperature,
        seed=options.seed,
        storage_backend=options.storage_backend,
    )
    _validate_runtime(resolved)
    return resolved


def _resolve_dakera(
    manifest: ExperimentManifest,
    base: DakeraSettings,
) -> DakeraSettings:
    options = manifest.condition.mechanism_options
    if not isinstance(options, DakeraOptions):
        raise AdapterConfigurationError("Dakera mechanism options are unresolved")
    resolved = replace(
        base,
        memory_type=DakeraMemoryType(options.memory_type),
        importance=options.importance,
        top_k=manifest.retrieval.top_k,
        routing=DakeraRouting(options.routing),
        rerank=options.rerank,
        neighborhood=options.neighborhood,
        include_associated=options.include_associated,
        ttl_seconds=options.ttl_seconds,
        expected_server_version=manifest.condition.mechanism_version,
        server_image=options.server_image,
    )
    _validate_runtime(resolved)
    return resolved


def _require_provider(role: str, requested: str, configured: str) -> None:
    if requested != configured:
        raise AdapterConfigurationError(
            f"{role} provider {requested!r} does not match configured provider "
            f"{configured!r}"
        )


def _resolve_digest(
    *,
    provider: str,
    model: str,
    manifest_digest: str | None,
    environment_model: str,
    environment_digest: str | None,
    role: str,
) -> str | None:
    digest = manifest_digest
    if digest is None and model == environment_model:
        digest = environment_digest
    if provider != "ollama":
        return digest
    if digest is None:
        raise AdapterConfigurationError(
            f"{role} model {model!r} requires an immutable Ollama SHA-256 digest"
        )
    normalized = digest.removeprefix("sha256:").casefold()
    if not _DIGEST_PATTERN.fullmatch(normalized):
        raise AdapterConfigurationError(
            f"{role} digest must contain exactly 64 hexadecimal characters"
        )
    return normalized


def _validate_runtime(settings: AdapterSettings) -> None:
    try:
        settings.validate_runtime()
    except (RuntimeError, ValueError) as exc:
        raise AdapterConfigurationError(
            f"resolved {type(settings).__name__} is invalid: {exc}"
        ) from exc


def _native_options(
    manifest: ExperimentManifest,
    suite: CanarySuite,
    qualification_id: str,
) -> dict[str, dict[str, object]]:
    if manifest.condition.write_policy != WritePolicy.NATIVE_SELECTIVE:
        return {}
    options: dict[str, dict[str, object]] = {}
    for canary in suite.qualification_canaries(qualification_id):
        if manifest.condition.mechanism == Mechanism.LANGMEM:
            if canary.memory_type in {
                MemoryType.SEMANTIC,
                MemoryType.EPISODIC,
                MemoryType.PROCEDURAL,
            }:
                options[canary.canary_id] = {
                    "memory_type": canary.memory_type.value
                }
        elif (
            manifest.condition.mechanism == Mechanism.TELEMEM
            and canary.memory_type == MemoryType.PROCEDURAL
        ):
            options[canary.canary_id] = {"memory_type": "procedural_memory"}
    return options
