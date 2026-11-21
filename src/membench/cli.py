"""Command-line entry point for experiment setup and validation."""

import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from dotenv import dotenv_values
from pydantic import ValidationError

from membench.campaign.launcher import CampaignFailFastError, CampaignLaunchError
from membench.campaign.loading import (
    PreparedCampaignLoadError,
    load_prepared_campaign,
)
from membench.campaign.models import CampaignStatus
from membench.campaign.preparation import CampaignPreparationError, prepare_campaign
from membench.campaign.preparation_models import CampaignPreparationOptions
from membench.contract import SUPPORTED_POLICIES, Mechanism, load_manifest
from membench.curation.delivery_projections import (
    DELIVERY_MANIFEST_FILE,
    DELIVERY_PROJECTIONS_FILE,
    DeliveryProjectionError,
    export_delivery_projection_set,
)
from membench.curation.external_attack_spec_drafts import (
    ExternalAttackSpecDraftError,
    load_external_attack_spec_draft_set,
)
from membench.curation.external_attack_specs import (
    EXTERNAL_ATTACK_MANIFEST_FILE,
    EXTERNAL_ATTACK_SPECS_FILE,
    ExternalAttackSpecError,
    export_external_attack_spec_set,
)
from membench.curation.external_delivery_recipes import (
    ExternalDeliveryRecipeError,
    load_external_delivery_recipe_set,
    render_external_delivery_projection_set,
)
from membench.langgraph_store import (
    LangGraphStoreError,
    LangGraphStoreSettings,
    database_status,
    qualify_in_memory,
    qualify_store,
    run_postgres_service,
)
from membench.langgraph_store import package_versions as langgraph_versions
from membench.langmem import LangMemError, LangMemSettings
from membench.langmem import package_versions as langmem_versions
from membench.langmem import qualify as qualify_langmem
from membench.memory.requirements import requirements_for
from membench.ollama import OllamaError, OllamaService, OllamaSettings
from membench.telemem import TeleMemError, TeleMemSettings
from membench.telemem import package_version as telemem_version
from membench.telemem import qualify as qualify_telemem
from membench.tools.fixtures import ToolFixtureError, load_tool_fixture

app = typer.Typer(help="Black-box memory-poisoning experiment harness.")
manifest_app = typer.Typer(help="Inspect and validate experiment manifests.")
ollama_app = typer.Typer(help="Inspect and qualify the shared Ollama endpoint.")
telemem_app = typer.Typer(help="Inspect and qualify the isolated TeleMem runtime.")
langgraph_app = typer.Typer(help="Manage and qualify the persistent LangGraph store.")
langmem_app = typer.Typer(
    help="Inspect and qualify LangMem selective writing over LangGraph stores."
)
campaign_app = typer.Typer(
    help="Prepare and execute reproducible cross-mechanism campaigns."
)
external_app = typer.Typer(
    help="Build frozen external-source attack specifications and deliveries."
)
app.add_typer(manifest_app, name="manifest")
app.add_typer(ollama_app, name="ollama")
app.add_typer(telemem_app, name="telemem")
app.add_typer(langgraph_app, name="langgraph")
app.add_typer(langmem_app, name="langmem")
app.add_typer(campaign_app, name="campaign")
app.add_typer(external_app, name="external")


@external_app.command("build")
def external_build(
    attack_draft: Annotated[
        Path,
        typer.Option("--attack-draft", help="Editable external attack-spec JSON."),
    ],
    recipe: Annotated[
        Path,
        typer.Option("--recipe", help="Editable external delivery-recipe JSON."),
    ],
    fixture: Annotated[
        Path,
        typer.Option("--fixture", help="Frozen tool-fixture JSON."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="New directory for the frozen external artifact set.",
        ),
    ],
    allow_unreviewed: Annotated[
        bool,
        typer.Option(
            "--allow-unreviewed",
            help=(
                "Permit a draft marked generated_unreviewed for local pipeline checks."
            ),
        ),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable build summary."),
    ] = False,
) -> None:
    """Validate inputs, render twin deliveries, then freeze a fresh artifact set."""
    if output_dir.exists():
        typer.echo(f"Output directory already exists: {output_dir}", err=True)
        raise typer.Exit(2)
    try:
        draft = load_external_attack_spec_draft_set(attack_draft)
        fixture_bundle = load_tool_fixture(fixture)
        recipes = load_external_delivery_recipe_set(recipe)
        attack_set = draft.freeze()
        deliveries = render_external_delivery_projection_set(
            recipes,
            fixture=fixture_bundle,
            attack_set=attack_set,
            require_approved_specs=not allow_unreviewed,
        )
    except (
        ExternalAttackSpecDraftError,
        ExternalDeliveryRecipeError,
        ToolFixtureError,
        ValueError,
    ) as exc:
        typer.echo(f"External artifact build failed: {exc}", err=True)
        raise typer.Exit(2) from exc

    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        attack_manifest = export_external_attack_spec_set(attack_set, output_dir)
        delivery_manifest = export_delivery_projection_set(deliveries, output_dir)
    except (OSError, ExternalAttackSpecError, DeliveryProjectionError) as exc:
        typer.echo(f"External artifact export failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    summary = {
        "schema_version": 1,
        "status": "frozen",
        "output_dir": str(output_dir),
        "attack_set_id": attack_set.attack_set_id,
        "attack_spec_count": attack_manifest.spec_count,
        "attack_specs_sha256": attack_manifest.specs_sha256,
        "delivery_set_id": deliveries.delivery_set_id,
        "delivery_family_count": delivery_manifest.family_count,
        "deliveries_sha256": delivery_manifest.families_sha256,
        "allow_unreviewed": allow_unreviewed,
    }
    if output_json:
        typer.echo(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return
    typer.echo(f"External artifact set frozen: {output_dir}")
    typer.echo(f"Attack specs: {attack_manifest.spec_count}")
    typer.echo(f"Delivery families: {delivery_manifest.family_count}")
    typer.echo(
        "Files: "
        f"{EXTERNAL_ATTACK_SPECS_FILE}, {EXTERNAL_ATTACK_MANIFEST_FILE}, "
        f"{DELIVERY_PROJECTIONS_FILE}, {DELIVERY_MANIFEST_FILE}"
    )


def _campaign_source_environment(
    env_file: Path | None, *, project_root: Path
) -> dict[str, str]:
    """Build preparation's environment snapshot without mutating ``os.environ``."""
    source = dict(os.environ)
    if env_file is None:
        return source
    resolved = env_file if env_file.is_absolute() else project_root / env_file
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"environment file does not exist: {resolved}")
    # Match load_dotenv(override=False): exported values take precedence over .env.
    for key, value in dotenv_values(resolved).items():
        if value is not None:
            source.setdefault(key, value)
    return source


@campaign_app.command("prepare")
def campaign_prepare(
    config: Annotated[
        Path,
        typer.Argument(help="JSON file containing CampaignPreparationOptions."),
    ],
    project_root: Annotated[
        Path,
        typer.Option(
            "--project-root",
            help=(
                "Project root used to resolve projection, environment, and "
                "artifact paths."
            ),
        ),
    ] = Path("."),
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            help=(
                "Optional dotenv file, resolved from the project root. Only "
                "variables allowlisted by each child environment can enter child "
                "processes."
            ),
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable preparation summary."),
    ] = False,
) -> None:
    """Validate and freeze a campaign plan without running any trials."""
    root = project_root.resolve()
    config_path = config if config.is_absolute() else Path.cwd() / config
    config_path = config_path.resolve()
    try:
        if not root.is_dir():
            raise NotADirectoryError(f"project root is not a directory: {root}")
        options = CampaignPreparationOptions.model_validate_json(
            config_path.read_bytes()
        )
        source_environment = _campaign_source_environment(env_file, project_root=root)
    except (OSError, ValidationError, ValueError) as exc:
        typer.echo(f"Invalid campaign preparation input: {exc}", err=True)
        raise typer.Exit(2) from exc

    try:
        prepared = prepare_campaign(
            options,
            project_root=root,
            source_environment=source_environment,
        )
    except CampaignPreparationError as exc:
        cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else "unknown"
        typer.echo(
            f"Campaign preparation failed during {exc.stage} ({cause}).",
            err=True,
        )
        if exc.artifact_root is not None:
            typer.echo(f"Partial evidence retained at: {exc.artifact_root}", err=True)
        raise typer.Exit(1) from exc

    summary = {
        "schema_version": 1,
        "status": "prepared",
        "campaign_id": prepared.configuration.campaign_id,
        "artifact_root": str(prepared.artifacts.root),
        "run_count": len(prepared.configuration.runs),
        "preparation_sha256": options.sha256,
        "completion_sha256": prepared.completion.sha256,
    }
    if output_json:
        typer.echo(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return
    typer.echo(f"Campaign prepared: {summary['campaign_id']}")
    typer.echo(f"Artifact root: {summary['artifact_root']}")
    typer.echo(f"Concrete runs: {summary['run_count']}")
    typer.echo(f"Preparation SHA-256: {summary['preparation_sha256']}")
    typer.echo("No trials were executed.")


def _emit_campaign_run_summary(
    summary: Mapping[str, object], *, output_json: bool
) -> None:
    """Emit one compact terminal summary; full evidence remains on disk."""
    if output_json:
        typer.echo(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return
    typer.echo(f"Campaign finished: {summary['campaign_id']}")
    typer.echo(f"Status: {summary['status']}")
    typer.echo(f"Scheduled: {summary['scheduled']}")
    typer.echo(f"Succeeded: {summary['succeeded']}")
    typer.echo(f"Invalidated: {summary['invalidated']}")
    typer.echo(f"Failed: {summary['failed']}")
    typer.echo(f"Artifacts: {summary['artifact_root']}")


@campaign_app.command("run")
def campaign_run(
    campaign_root: Annotated[
        Path,
        typer.Argument(help="Directory containing a completed campaign preparation."),
    ],
    project_root: Annotated[
        Path,
        typer.Option(
            "--project-root",
            help="Project root used to validate the frozen campaign and child paths.",
        ),
    ] = Path("."),
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            help=(
                "Optional dotenv file, resolved from the project root. Only "
                "variables frozen in each child environment allowlist are inherited."
            ),
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit one machine-readable terminal summary."),
    ] = False,
) -> None:
    """Execute, aggregate, and seal one complete, unstarted preparation."""
    root = project_root.resolve()
    try:
        if not root.is_dir():
            raise NotADirectoryError(f"project root is not a directory: {root}")
        source_environment = _campaign_source_environment(env_file, project_root=root)
    except (OSError, ValueError) as exc:
        typer.echo(f"Invalid campaign execution input: {exc}", err=True)
        raise typer.Exit(2) from exc

    try:
        prepared = load_prepared_campaign(
            campaign_root,
            project_root=root,
            source_environment=source_environment,
        )
    except PreparedCampaignLoadError as exc:
        cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else "unknown"
        typer.echo(
            f"Prepared campaign load failed during {exc.stage} ({cause}).",
            err=True,
        )
        raise typer.Exit(3) from exc

    run_count = len(prepared.configuration.runs)
    if output_json:
        typer.echo(
            f"Launching campaign {prepared.configuration.campaign_id} "
            f"with {run_count} concrete runs.",
            err=True,
        )
    else:
        typer.echo(
            f"Launching campaign {prepared.configuration.campaign_id} "
            f"with {run_count} concrete runs..."
        )

    try:
        launcher = prepared.create_launcher()
        result = launcher.run()
    except CampaignFailFastError as exc:
        typer.echo(
            f"Campaign stopped by fail-fast after {exc.child.run.run_id} "
            f"finished with status {exc.child.status.value}.",
            err=True,
        )
        typer.echo(f"Partial evidence retained at: {prepared.artifacts.root}", err=True)
        raise typer.Exit(1) from exc
    except CampaignLaunchError as exc:
        cause = type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__
        typer.echo(
            f"Campaign launch failed ({cause}).",
            err=True,
        )
        typer.echo(f"Partial evidence retained at: {prepared.artifacts.root}", err=True)
        raise typer.Exit(1) from exc
    except KeyboardInterrupt as exc:
        typer.echo("Campaign interrupted; partial evidence was retained.", err=True)
        raise typer.Exit(130) from exc

    campaign_root_path = prepared.artifacts.root
    summary = {
        "schema_version": 1,
        "campaign_id": result.summary.campaign_id,
        "status": result.summary.status.value,
        "scheduled": result.summary.scheduled_count,
        "succeeded": result.summary.succeeded_count,
        "invalidated": result.summary.invalidated_count,
        "failed": result.summary.failed_count,
        "artifact_root": str(campaign_root_path),
        "analysis": str(campaign_root_path / "analysis.json"),
        "result": str(campaign_root_path / "campaign_result.json"),
        "checksums": str(campaign_root_path / "checksums.json"),
    }
    _emit_campaign_run_summary(summary, output_json=output_json)
    if result.summary.status != CampaignStatus.COMPLETED:
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Report whether the core development environment is ready."""
    typer.echo(f"Python: {platform.python_version()}")
    typer.echo(f"Executable: {sys.executable}")
    typer.echo("Core packages:")
    for package in ("pydantic", "PyYAML", "typer"):
        typer.echo(f"  {package}: {importlib.metadata.version(package)}")
    if sys.version_info[:2] != (3, 12):
        typer.echo("ERROR: this project requires Python 3.12.x", err=True)
        raise typer.Exit(1)
    typer.echo("Environment is ready.")


@app.command("matrix")
def show_matrix() -> None:
    """Print supported mechanism/write-policy combinations."""
    for mechanism in Mechanism:
        policies = ", ".join(sorted(p.value for p in SUPPORTED_POLICIES[mechanism]))
        if mechanism == Mechanism.LANGMEM:
            policies += " (native_selective requires langmem_store_manager_v1)"
        typer.echo(f"{mechanism.value}: {policies}")


@manifest_app.command("validate")
def validate_manifest(path: Path) -> None:
    """Validate a resolved experiment manifest against schema version 1."""
    try:
        manifest = load_manifest(path)
    except (OSError, ValidationError, ValueError) as exc:
        typer.echo(f"Invalid manifest: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"Valid manifest: {manifest.isolation.run_id} "
        f"({manifest.condition.mechanism.value}/"
        f"{manifest.condition.write_policy.value})"
    )
    required = ", ".join(requirements_for(manifest))
    typer.echo(f"Required adapter capabilities: {required}")


@ollama_app.command("status")
def ollama_status() -> None:
    """Show endpoint, server version, and locally installed model digests."""
    settings = OllamaSettings.from_env()
    service = OllamaService(settings.base_url)
    try:
        version = service.version()
        models = service.models()
    except (OllamaError, ValueError) as exc:
        typer.echo(f"Ollama check failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Endpoint: {settings.base_url}")
    typer.echo(f"OpenAI-compatible endpoint: {settings.openai_base_url}")
    typer.echo(f"Server version: {version}")
    if not models:
        typer.echo("Models: none installed")
        return
    typer.echo("Models:")
    for model in models:
        name = model.get("name", "unknown")
        digest = str(model.get("digest", "unknown"))
        typer.echo(f"  {name}: {digest}")


@ollama_app.command("verify")
def ollama_verify() -> None:
    """Exercise the configured chat and embedding models through native APIs."""
    settings = OllamaSettings.from_env()
    service = OllamaService(settings.base_url, timeout=180.0)
    try:
        required = {settings.chat_model, settings.embedding_model}
        missing = sorted(model for model in required if not service.has_model(model))
        if missing:
            raise OllamaError(f"missing configured models: {', '.join(missing)}")
        service.verify_digest(settings.chat_model, settings.chat_model_digest)
        service.verify_digest(settings.embedding_model, settings.embedding_model_digest)
        vector = service.embed(
            settings.embedding_model, "Ollama memory qualification probe"
        )
        if len(vector) != settings.embedding_dimensions:
            raise OllamaError(
                f"{settings.embedding_model} returned {len(vector)} dimensions; "
                f"expected {settings.embedding_dimensions}"
            )
        reply = service.chat(settings.chat_model, "Reply with exactly: OLLAMA_READY")
        if reply.strip() != "OLLAMA_READY":
            raise OllamaError(
                f"chat qualification returned an unexpected response: {reply!r}"
            )
    except (OllamaError, ValueError) as exc:
        typer.echo(f"Ollama qualification failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Chat model ready: {settings.chat_model}")
    typer.echo(
        f"Embedding model ready: {settings.embedding_model} ({len(vector)} dimensions)"
    )


@telemem_app.command("status")
def telemem_status() -> None:
    """Show the resolved TeleMem provider without importing vendor runtime state."""
    try:
        settings = TeleMemSettings.from_env()
        version = telemem_version()
    except (TeleMemError, ValueError) as exc:
        typer.echo(f"TeleMem check failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"TeleMem version: {version}")
    typer.echo(f"Provider: {settings.provider}")
    typer.echo(f"Endpoint: {settings.base_url}")
    typer.echo(f"Chat model: {settings.llm_model}")
    typer.echo(
        f"Embedding model: {settings.embedding_model} "
        f"({settings.embedding_dimensions} dimensions)"
    )
    typer.echo(f"Storage root: {settings.storage_root}")
    if settings.provider == "openai":
        configured = "yes" if settings.credential_configured else "no"
        typer.echo(f"OpenAI credential configured: {configured}")


@telemem_app.command("verify")
def telemem_verify(
    run_id: str = typer.Option(
        "",
        help=(
            "Unique run identifier; an isolated qualification ID is generated "
            "by default."
        ),
    ),
) -> None:
    """Run isolated direct and native-selective TeleMem write/search probes."""
    if not run_id:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"qualification-{timestamp}"
    try:
        settings = TeleMemSettings.from_env()
        result = qualify_telemem(settings, run_id)
    except (TeleMemError, ValueError, OSError) as exc:
        typer.echo(f"TeleMem qualification failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"TeleMem qualification passed: {result.run_id}")
    typer.echo(f"Provider: {settings.provider}")
    typer.echo(f"Storage: {result.storage_path}")
    typer.echo(
        "Direct path: "
        f"{result.direct_write_count} write(s), "
        f"{result.direct_search_count} search result(s)"
    )
    typer.echo(
        "Native-selective path: "
        f"{result.native_write_count} write(s), "
        f"{result.native_search_count} search result(s)"
    )


@langgraph_app.command("up")
def langgraph_up() -> None:
    """Start the pinned PostgreSQL/pgvector service and wait for health."""
    try:
        settings = LangGraphStoreSettings.from_env()
        output = run_postgres_service(settings, "up")
    except (LangGraphStoreError, ValueError) as exc:
        typer.echo(f"LangGraph PostgreSQL start failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    if output:
        typer.echo(output)
    typer.echo(f"LangGraph PostgreSQL is ready: {settings.redacted_dsn}")


@langgraph_app.command("stop")
def langgraph_stop() -> None:
    """Stop PostgreSQL without deleting its persistent volume."""
    try:
        settings = LangGraphStoreSettings.from_env()
        output = run_postgres_service(settings, "stop")
    except (LangGraphStoreError, ValueError) as exc:
        typer.echo(f"LangGraph PostgreSQL stop failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    if output:
        typer.echo(output)
    typer.echo("LangGraph PostgreSQL stopped; its volume was retained.")


@langgraph_app.command("status")
def langgraph_status() -> None:
    """Show package, database, and embedding status without exposing credentials."""
    try:
        settings = LangGraphStoreSettings.from_env()
        versions = langgraph_versions()
        server_version, vector_version = database_status(settings)
    except (LangGraphStoreError, ValueError) as exc:
        typer.echo(f"LangGraph store check failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    for package, version in versions.items():
        typer.echo(f"{package}: {version}")
    typer.echo(f"PostgreSQL: {server_version}")
    typer.echo(f"pgvector: {vector_version or 'not installed'}")
    typer.echo(f"Database: {settings.redacted_dsn}")
    typer.echo(f"Embedding endpoint: {settings.embedding_base_url}")
    typer.echo(
        f"Embedding model: {settings.embedding_model} "
        f"({settings.embedding_dimensions} dimensions)"
    )


@langgraph_app.command("verify")
def langgraph_verify(
    run_id: str = typer.Option(
        "",
        help=(
            "Unique run identifier; an isolated qualification ID is generated "
            "by default."
        ),
    ),
) -> None:
    """Run persistent write, read, semantic search, and isolation probes."""
    if not run_id:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"qualification-{timestamp}"
    try:
        settings = LangGraphStoreSettings.from_env()
        result = qualify_store(settings, run_id)
    except (LangGraphStoreError, ValueError) as exc:
        typer.echo(f"LangGraph store qualification failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"LangGraph store qualification passed: {result.run_id}")
    typer.echo(f"Namespace: {'/'.join(result.namespace)}")
    typer.echo(f"Exact read: {result.exact_read}")
    typer.echo(f"Semantic results: {result.semantic_results}")
    typer.echo(f"Isolated namespace empty: {result.isolated_namespace_empty}")


@langgraph_app.command("smoke")
def langgraph_smoke(
    run_id: str = typer.Option("smoke", help="Namespace identifier for the smoke run."),
) -> None:
    """Verify LangGraph and Ollama using a non-persistent in-memory store."""
    try:
        settings = LangGraphStoreSettings.from_env()
        result = qualify_in_memory(settings, run_id)
    except (LangGraphStoreError, ValueError) as exc:
        typer.echo(f"LangGraph in-memory smoke test failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo("LangGraph in-memory smoke test passed (non-persistent).")
    typer.echo(f"Namespace: {'/'.join(result.namespace)}")
    typer.echo(f"Exact read: {result.exact_read}")
    typer.echo(f"Semantic results: {result.semantic_results}")
    typer.echo(f"Isolated namespace empty: {result.isolated_namespace_empty}")


@langmem_app.command("status")
def langmem_status() -> None:
    """Show the resolved LangMem writer, schemas, and storage backend."""
    try:
        settings = LangMemSettings.from_env()
        versions = langmem_versions()
    except (LangMemError, ValueError) as exc:
        typer.echo(f"LangMem check failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    for package, version in versions.items():
        typer.echo(f"{package}: {version}")
    typer.echo(f"Writer: {settings.writer_provider}/{settings.writer_model}")
    typer.echo(
        "Embedding: "
        f"{settings.embedding_provider}/{settings.embedding_model} "
        f"({settings.embedding_dimensions} dimensions)"
    )
    typer.echo(
        "Memory types: "
        + ", ".join(memory_type.value for memory_type in settings.memory_types)
    )
    typer.echo(f"Storage backend: {settings.storage_backend}")
    if settings.storage_backend == "sqlite":
        typer.echo(f"SQLite root: {settings.sqlite_root}")
    if "openai" in {settings.writer_provider, settings.embedding_provider}:
        configured = "yes" if settings.credential_configured else "no"
        typer.echo(f"OpenAI credential configured: {configured}")
    typer.echo("Procedural prompt optimization: disabled")


def _run_langmem_qualification(settings: LangMemSettings, run_id: str) -> None:
    try:
        result = qualify_langmem(settings, run_id)
    except (LangMemError, ValueError, OSError) as exc:
        typer.echo(f"LangMem qualification failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"LangMem qualification passed: {result.run_id}")
    typer.echo(f"Storage backend: {result.storage_backend}")
    if result.storage_path is not None:
        typer.echo(f"Storage: {result.storage_path}")
    for memory_type, count in result.write_counts.items():
        typer.echo(
            f"{memory_type}: {count} write(s), "
            f"{result.retrieval_counts[memory_type]} retrieval result(s)"
        )
    typer.echo(f"Isolated namespaces empty: {result.isolated_namespaces_empty}")


@langmem_app.command("verify")
def langmem_verify(
    run_id: str = typer.Option(
        "",
        help=(
            "Unique run identifier; an isolated qualification ID is generated "
            "by default."
        ),
    ),
) -> None:
    """Qualify LangMem with the configured persistent LangGraph store."""
    if not run_id:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"qualification-{timestamp}"
    try:
        settings = LangMemSettings.from_env()
    except (LangMemError, ValueError) as exc:
        typer.echo(f"LangMem configuration failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    _run_langmem_qualification(settings, run_id)


@langmem_app.command("smoke")
def langmem_smoke(
    run_id: str = typer.Option(
        "smoke", help="Namespace identifier for the non-persistent smoke run."
    ),
) -> None:
    """Qualify the writer with an in-memory LangGraph Store."""
    try:
        settings = LangMemSettings.from_env().with_storage_backend("memory")
    except (LangMemError, ValueError) as exc:
        typer.echo(f"LangMem configuration failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    _run_langmem_qualification(settings, run_id)
