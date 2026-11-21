"""Preparation is offline; launcher integration uses a recording fake child."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from membench.campaign.artifacts import ArtifactIntegrityError, CampaignArtifactStore
from membench.campaign.bindings import CampaignRunBindings
from membench.campaign.models import (
    ArtifactKind,
    ArtifactVisibility,
    CampaignConfiguration,
    ChildRunResult,
)
from membench.campaign.preparation import (
    CampaignPreparation,
    CampaignPreparationError,
    prepare_campaign,
)
from membench.campaign.preparation_models import CampaignPreparationOptions
from membench.contract import load_manifest
from membench.evaluation.models import EvaluationProbe
from membench.tools.fixtures import (
    FixtureContact,
    FixtureOrigin,
    PersonaDerivedProvenance,
    ToolFixtureBundle,
)
from membench.tools.mock_contacts import MockContact


@pytest.fixture
def preparation_options(input_fixture):
    payload = input_fixture[2].model_dump(mode="json")
    payload["environments"][0].update(
        python_executable=sys.executable,
        inherited_env_keys=["TEST_PROVIDER_KEY"],
    )
    payload["cleanup_storage"] = False
    return CampaignPreparationOptions.model_validate(payload)


def _prepare(options, root, **kwargs):
    return prepare_campaign(
        options,
        project_root=root,
        source_environment={"TEST_PROVIDER_KEY": "not-a-real-secret"},
        **kwargs,
    )


def _write_fixture(path: Path) -> ToolFixtureBundle:
    fixture = ToolFixtureBundle(
        fixture_id="campaign-tools-v1",
        persona_ids=("1", "2"),
        contacts=(
            FixtureContact(
                contact=MockContact(
                    contact_id="trusted-contact",
                    owner_user_id="1",
                    display_name="Trusted Synthetic Contact",
                    email="trusted@example.invalid",
                ),
                provenance=PersonaDerivedProvenance(
                    origin=FixtureOrigin.PERSONA_DERIVED,
                    source_export_sha256="a" * 64,
                    source_record_id="synthetic-record-001",
                ),
            ),
        ),
    )
    path.parent.mkdir()
    path.write_text(fixture.model_dump_json(indent=2), encoding="utf-8")
    return fixture


def test_freezes_and_binds_an_optional_evaluation_probe(
    preparation_options, tmp_path
):
    probe = EvaluationProbe(
        probe_id="probe-evaluation-preparation-001",
        attack_spec_id="external-false-fact-001",
        delivery_family_id="external-delivery-001",
        persona_id="1",
        query="What fact should be retrieved?",
        settings={
            "attack_family": "false_fact_insertion",
            "poison_markers": ["UNIQUE-POISON-MARKER"],
            "trusted_ground_truth": "trusted value",
        },
    )
    source = tmp_path / "inputs" / "evaluation-probe.json"
    source.parent.mkdir()
    source.write_text(probe.model_dump_json(), encoding="utf-8")
    options = CampaignPreparationOptions.model_validate(
        {
            **preparation_options.model_dump(mode="json"),
            "case_ids": ["alloc-synthetic-001"],
            "evaluation_probe_path": "inputs/evaluation-probe.json",
            "artifact_root": "artifacts/evaluation-probe-pilot",
        }
    )

    prepared = _prepare(options, tmp_path)

    reference = prepared.bindings.evaluation_probe
    assert reference is not None
    assert reference.kind == ArtifactKind.EVALUATION_PROBE
    assert reference in prepared.configuration.frozen_inputs
    assert all(binding.has_evaluation_probe for binding in prepared.bindings.bindings)
    assert {
        binding.evaluation_probe_id for binding in prepared.bindings.bindings
    } == {probe.probe_id}
    assert {
        binding.evaluation_probe_sha256 for binding in prepared.bindings.bindings
    } == {reference.sha256}


def test_prepares_complete_verified_inventory_without_exposing_secrets(
    preparation_options, tmp_path
):
    options = preparation_options
    prepared = _prepare(options, tmp_path)
    store = prepared.artifacts
    assert store.root == (tmp_path / options.artifact_root).resolve()
    assert not store.sealed
    assert len(prepared.configuration.runs) == 6
    assert prepared.configuration.max_parallel_children == options.max_parallel_children
    assert prepared.configuration.fail_fast == options.fail_fast
    assert (
        len(store.references()) == 13
    )  # Nine input files + options/bindings/config/marker.
    for reference in store.references():
        assert reference.visibility == ArtifactVisibility.EVALUATOR_ONLY
        content = store.verify(reference).read_bytes()
        assert b"not-a-real-secret" not in content
    assert "not-a-real-secret" not in repr(prepared)
    assert not (store.root / "campaign_configuration.json").exists()
    assert not (store.root / "children").exists()
    assert not (store.root / "checksums.json").exists()
    assert (
        CampaignConfiguration.model_validate_json(
            (store.root / "preparation/configuration.json").read_bytes()
        )
        == prepared.configuration
    )
    assert (
        CampaignRunBindings.model_validate_json(
            (store.root / "preparation/run_bindings.json").read_bytes()
        )
        == prepared.bindings
    )
    prepared.bindings.validate_preparation(options)
    prepared.bindings.validate_campaign(prepared.configuration)
    completion = json.loads(store.verify(prepared.completion).read_bytes())
    assert completion["status"] == "prepared"
    assert completion["run_count"] == 6
    assert completion["purpose"] == "pilot"
    assert completion["preparation_sha256"] == options.sha256
    assert len(completion["artifacts"]) == 12
    assert [ref.kind for ref in prepared.configuration.frozen_inputs].count(
        ArtifactKind.RUN_BINDINGS
    ) == 1


def test_preparation_carries_one_frozen_optional_tool_fixture_to_children(
    preparation_options, tmp_path
):
    fixture_path = tmp_path / "fixtures" / "campaign-tools.json"
    fixture = _write_fixture(fixture_path)
    payload = preparation_options.model_dump(mode="json")
    payload["tool_fixture_path"] = "fixtures/campaign-tools.json"
    payload["conditions"][0]["agent_configuration"]["allowed_tools"] = [
        "lookup_contact"
    ]
    payload["artifact_root"] = "artifacts/with-tools"
    options = CampaignPreparationOptions.model_validate(payload)

    prepared = _prepare(options, tmp_path)
    fixture_reference = next(
        reference
        for reference in prepared.configuration.frozen_inputs
        if reference.kind == ArtifactKind.TOOL_FIXTURE
    )
    assert prepared.artifacts.verify(fixture_reference).read_bytes() == (
        fixture_path.read_bytes()
    )
    assert fixture_reference in prepared.create_launcher().build_child_request(
        prepared.configuration.runs[0]
    ).frozen_inputs
    selection = json.loads(
        (prepared.artifacts.root / "inputs/projections/selection.json").read_bytes()
    )
    assert selection["tool_fixture"]["fixture_sha256"] == fixture.sha256


def test_preparation_does_not_start_processes_or_modify_sources(
    preparation_options, input_fixture, tmp_path, monkeypatch
):
    source = input_fixture[1]
    before = {p.name: p.read_bytes() for p in source.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation attempted execution")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    prepared = _prepare(preparation_options, tmp_path)
    assert prepared.create_launcher().configuration == prepared.configuration
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


@pytest.mark.parametrize(
    "problem,stage",
    [
        ("options", "validate_options"),
        ("source", "load_inputs"),
        ("matrix", "build_matrix"),
        ("interpreter", "preflight_environments"),
        ("working_directory", "preflight_environments"),
        ("missing_env", "preflight_environments"),
    ],
)
def test_preflight_failures_leave_no_output(
    preparation_options, tmp_path, problem, stage
):
    options = preparation_options
    environment = {"TEST_PROVIDER_KEY": "not-a-real-secret"}
    if problem == "options":
        options = options.model_copy(update={"seed": -1})
    elif problem == "missing_env":
        environment = {}
    else:
        payload = options.model_dump(mode="json")
        if problem == "source":
            payload["projection_directory"] = "missing-projections"
        elif problem == "matrix":
            payload["max_parallel_children"] = 100
        else:
            key = (
                "python_executable" if problem == "interpreter" else "working_directory"
            )
            payload["environments"][0][key] = "missing-path"
        options = CampaignPreparationOptions.model_validate(payload)
    with pytest.raises(CampaignPreparationError) as caught:
        prepare_campaign(options, project_root=tmp_path, source_environment=environment)
    assert caught.value.stage == stage
    assert caught.value.artifact_root is None
    assert caught.value.__cause__ is not None
    assert not (tmp_path / options.artifact_root).exists()


def test_existing_output_is_not_overwritten(preparation_options, tmp_path):
    prepared = _prepare(preparation_options, tmp_path)
    before = {
        p.relative_to(prepared.artifacts.root): p.read_bytes()
        for p in prepared.artifacts.root.rglob("*")
        if p.is_file()
    }
    with pytest.raises(CampaignPreparationError) as caught:
        _prepare(preparation_options, tmp_path)
    assert caught.value.stage == "resolve_artifact_root"
    assert caught.value.artifact_root is None
    after = {
        p.relative_to(prepared.artifacts.root): p.read_bytes()
        for p in prepared.artifacts.root.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_resolved_output_cannot_escape_project_root(
    preparation_options, tmp_path, monkeypatch
):
    configured = tmp_path / preparation_options.artifact_root
    resolve = Path.resolve

    def redirected(path, *args, **kwargs):
        if path == configured:
            return tmp_path.parent / "outside-project"
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(CampaignPreparationError) as caught:
        _prepare(preparation_options, tmp_path)
    assert caught.value.stage == "resolve_artifact_root"
    assert not configured.exists()


@pytest.mark.parametrize(
    "method,stage",
    [
        ("_build_run_bindings", "build_run_bindings"),
        ("_persist_preparation", "persist_preparation"),
        ("_validate_prepared_campaign", "validate_prepared_campaign"),
    ],
)
def test_late_failures_retain_partial_evidence_without_ready_marker(
    preparation_options, tmp_path, monkeypatch, method, stage
):
    coordinator = CampaignPreparation()

    def fail(*args, **kwargs):
        raise RuntimeError("sensitive-exception-content")

    monkeypatch.setattr(coordinator, method, fail)
    with pytest.raises(CampaignPreparationError) as caught:
        coordinator.prepare_campaign(
            preparation_options,
            project_root=tmp_path,
            source_environment={"TEST_PROVIDER_KEY": "x"},
        )
    assert caught.value.stage == stage
    assert "sensitive-exception-content" not in str(caught.value)
    root = caught.value.artifact_root
    assert root == (tmp_path / preparation_options.artifact_root).resolve()
    assert (root / "inputs/projections/manifest.json").is_file()
    assert not (root / "preparation/complete.json").exists()


def test_disk_tampering_before_final_validation_prevents_completion(
    preparation_options, tmp_path, monkeypatch
):
    coordinator = CampaignPreparation()
    original = coordinator._persist_preparation

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        (
            tmp_path / preparation_options.artifact_root / "preparation/options.json"
        ).write_bytes(b"tampered")
        return result

    monkeypatch.setattr(coordinator, "_persist_preparation", corrupt)
    with pytest.raises(CampaignPreparationError) as caught:
        coordinator.prepare_campaign(
            preparation_options,
            project_root=tmp_path,
            source_environment={"TEST_PROVIDER_KEY": "x"},
        )
    assert caught.value.stage == "validate_prepared_campaign"
    assert isinstance(caught.value.__cause__, ArtifactIntegrityError)
    assert not (caught.value.artifact_root / "preparation/complete.json").exists()


def test_launcher_handoff_preserves_cleanup_and_does_not_collide_with_artifacts(
    preparation_options, tmp_path
):
    class FakeChild:
        def __init__(self):
            self.requests = []

        def execute(self, request, environment):
            self.requests.append(request)
            return ChildRunResult(
                request_id=request.request_id,
                run=request.run,
                status="failed",
                failure_class="unexpected_error",
                failure_type="SyntheticFailure",
                failure_message="Offline wiring test; no backend was executed.",
            )

    prepared = _prepare(preparation_options, tmp_path)
    child = FakeChild()
    launcher = prepared.create_launcher(child_executor=child)
    assert launcher.artifacts is prepared.artifacts
    result = launcher.run()
    assert len(child.requests) == 6
    assert all(not request.cleanup_storage for request in child.requests)
    assert result.summary.failed_count == 6
    assert prepared.artifacts.sealed
    assert (prepared.artifacts.root / "campaign_configuration.json").exists()
    assert (prepared.artifacts.root / "preparation/configuration.json").exists()
    with pytest.raises(ValueError, match="already sealed"):
        prepared.create_launcher()


def test_handoff_rejects_tampered_artifacts(preparation_options, tmp_path):
    prepared = _prepare(preparation_options, tmp_path)
    (prepared.artifacts.root / "preparation/run_bindings.json").write_bytes(b"modified")
    with pytest.raises(ArtifactIntegrityError):
        prepared.create_launcher()


def test_handoff_rejects_changed_in_memory_configuration(preparation_options, tmp_path):
    prepared = _prepare(preparation_options, tmp_path)
    changed = replace(
        prepared,
        configuration=prepared.configuration.model_copy(update={"fail_fast": True}),
    )
    with pytest.raises(ValueError, match="frozen preparation"):
        changed.create_launcher()


@pytest.mark.parametrize(
    "mechanism,filename",
    [
        ("langmem", "langmem_semantic.example.yaml"),
        ("dakera", "dakera.example.yaml"),
    ],
)
def test_other_mechanisms_prepare_without_their_sdk_or_server(
    preparation_options, tmp_path, mechanism, filename
):
    template = load_manifest(Path("configs/experiments/blackbox") / filename)
    payload = preparation_options.model_dump(mode="json")
    payload["conditions"][0]["condition_id"] = f"{mechanism}-condition"
    payload["conditions"][0]["manifest_template"] = template.model_dump(mode="json")
    for model in payload["conditions"][0]["manifest_template"]["models"].values():
        if model is not None:
            model["digest"] = "a" * 64
    payload["environments"][0]["mechanism"] = mechanism
    prepared = _prepare(CampaignPreparationOptions.model_validate(payload), tmp_path)
    assert {run.mechanism.value for run in prepared.configuration.runs} == {mechanism}
    assert len(prepared.bindings.bindings) == 6


def test_partial_freeze_is_retained_and_not_marked_complete(
    preparation_options, tmp_path, monkeypatch
):
    original = CampaignArtifactStore.write_bytes

    def fail_midway(store, **kwargs):
        if kwargs["relative_path"].endswith("cases.jsonl"):
            raise OSError("simulated disk failure")
        return original(store, **kwargs)

    monkeypatch.setattr(CampaignArtifactStore, "write_bytes", fail_midway)
    with pytest.raises(CampaignPreparationError) as caught:
        _prepare(preparation_options, tmp_path)
    assert caught.value.stage == "freeze_inputs"
    assert (caught.value.artifact_root / "inputs/projections/manifest.json").exists()
    assert not (caught.value.artifact_root / "preparation/complete.json").exists()


def test_completion_write_failure_is_not_success(
    preparation_options, tmp_path, monkeypatch
):
    original = CampaignArtifactStore.write_json

    def fail_completion(store, **kwargs):
        if kwargs["relative_path"] == "preparation/complete.json":
            raise OSError("simulated completion-write failure")
        return original(store, **kwargs)

    monkeypatch.setattr(CampaignArtifactStore, "write_json", fail_completion)
    with pytest.raises(CampaignPreparationError) as caught:
        _prepare(preparation_options, tmp_path)
    assert caught.value.stage == "record_completion"
    assert (caught.value.artifact_root / "preparation/configuration.json").exists()
    assert not (caught.value.artifact_root / "preparation/complete.json").exists()


def test_environment_snapshot_is_allowlisted_and_not_updated_after_preflight(
    preparation_options, tmp_path
):
    supplied = {"TEST_PROVIDER_KEY": "first-secret", "UNRELATED_SECRET": "excluded"}
    prepared = prepare_campaign(
        preparation_options, project_root=tmp_path, source_environment=supplied
    )
    supplied["TEST_PROVIDER_KEY"] = "changed"
    resolved = prepared.environments.resolve("telemem").subprocess_environment()
    assert resolved["TEST_PROVIDER_KEY"] == "first-secret"
    assert "UNRELATED_SECRET" not in resolved
    assert resolved["PYTHON_DOTENV_DISABLED"] == "1"


def test_completion_is_written_last(preparation_options, tmp_path, monkeypatch):
    writes = []
    original = CampaignArtifactStore.write_bytes

    def record(store, **kwargs):
        writes.append(kwargs["relative_path"])
        return original(store, **kwargs)

    monkeypatch.setattr(CampaignArtifactStore, "write_bytes", record)
    _prepare(preparation_options, tmp_path)
    assert writes[-1] == "preparation/complete.json"


def test_experiment_subset_and_launcher_settings_are_preserved(
    preparation_options, tmp_path
):
    payload = preparation_options.model_dump(mode="json")
    payload.update(
        purpose="experiment",
        case_ids=["alloc-synthetic-002"],
        fail_fast=True,
        max_parallel_children=2,
        cleanup_storage=True,
    )
    prepared = _prepare(CampaignPreparationOptions.model_validate(payload), tmp_path)
    assert len(prepared.configuration.runs) == 3
    assert prepared.configuration.fail_fast
    assert prepared.configuration.max_parallel_children == 2
    launcher = prepared.create_launcher()
    for run in prepared.configuration.runs:
        assert launcher.build_child_request(run).cleanup_storage
    assert all(
        binding.purpose == "experiment" for binding in prepared.bindings.bindings
    )
