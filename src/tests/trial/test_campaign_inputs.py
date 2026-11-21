"""Offline campaign input checks using synthetic projections, not private data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from membench.campaign.artifacts import (
    ArtifactExistsError,
    ArtifactIntegrityError,
    ArtifactSealedError,
    CampaignArtifactStore,
)
from membench.campaign.inputs import (
    CampaignInputError,
    _validate_links,
    freeze_campaign_inputs,
    load_campaign_inputs,
)
from membench.campaign.models import (
    ArtifactKind,
    ArtifactVisibility,
)
from membench.context_budget import RegexTokenCounter
from membench.curation.projections import (
    PROJECTION_FILES,
    ProjectionKind,
    ProjectionSetError,
    TrialProjection,
    export_projection_set,
    load_projection_set,
    parse_projection_set,
)
from membench.tools.fixtures import (
    FixtureContact,
    FixtureOrigin,
    PersonaDerivedProvenance,
    ToolFixtureBundle,
)
from membench.tools.mock_contacts import MockContact


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _projection_hash(value):
    return _hash(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _write_tool_fixture(tmp_path, *, persona_ids=("1", "2")):
    """Create a deliberately formatted fixture to test byte-preserving freezing."""

    fixture = ToolFixtureBundle(
        fixture_id="synthetic-tools-v1",
        persona_ids=persona_ids,
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
    path = tmp_path / "fixtures" / "tool-fixture.json"
    path.parent.mkdir()
    path.write_text(fixture.model_dump_json(indent=4), encoding="utf-8")
    return path.relative_to(tmp_path), path, fixture


def _options_with_contact_lookup(options, fixture_path=None):
    payload = options.model_dump(mode="json")
    payload["conditions"][0]["agent_configuration"]["allowed_tools"] = [
        "lookup_contact"
    ]
    payload["tool_fixture_path"] = fixture_path
    return payload


def test_load_all_cases_is_read_only_and_contains_all_twins(input_fixture, tmp_path):
    source, directory, options = input_fixture
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    inputs = load_campaign_inputs(options, project_root=tmp_path)
    assert inputs.case_ids == ("alloc-synthetic-001", "alloc-synthetic-002")
    assert inputs.source_directory == directory.resolve()
    assert inputs.preparation_sha256 == options.sha256
    assert inputs.manifest == source.manifest
    assert sum(len(item.trials) for item in inputs.cases) == 6
    assert inputs.cases[0].trial_for("clean").twin == "clean"
    with pytest.raises(KeyError):
        inputs.cases[0].trial_for("absent")
    assert "EVALUATOR_ONLY" not in repr(inputs)
    for item in inputs.cases:
        assert "canonical_preference" not in item.model_dump_json()
        assert "correct_answer" not in item.model_dump_json()
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    assert not (tmp_path / options.artifact_root).exists()


def test_neutral_external_host_requires_equal_twin_replay(
    input_fixture, tmp_path
):
    source, directory, options = input_fixture
    original_case = source.cases[0]
    original_spec = source.attack_specs[0]
    clean = next(
        trial
        for trial in source.trials
        if (
            trial.allocation_case_id == original_case.allocation_case_id
            and trial.twin == "clean"
        )
    )
    neutral_case = original_case.model_copy(
        update={
            "projection_kind": ProjectionKind.NEUTRAL_EXTERNAL_HOST,
            "source_allocation_case_id": original_case.allocation_case_id,
        }
    )
    neutral_spec = original_spec.__class__.model_validate(
        {
            **original_spec.model_dump(mode="json"),
            "projection_kind": ProjectionKind.NEUTRAL_EXTERNAL_HOST,
            "matched_benign_overlay": [],
            "poisoned_overlay": [],
            "token_delta_by_event": [],
            "payload_sha256": _projection_hash(
                {"matched_benign": [], "poisoned": []}
            ),
        }
    )

    neutral_trials = []
    for twin in ("clean", "matched_benign", "poisoned"):
        trial_id = f"{original_case.allocation_case_id}--{twin.replace('_', '-')}"
        events = []
        for ordinal, event in enumerate(clean.replay_events, start=1):
            event_payload = event.model_dump(mode="json")
            event_payload["event_id"] = f"{trial_id}.event-{ordinal:04d}"
            events.append(event_payload)
        neutral_trials.append(
            TrialProjection.model_validate(
                {
                    **clean.model_dump(mode="json"),
                    "projection_kind": ProjectionKind.NEUTRAL_EXTERNAL_HOST,
                    "trial_id": trial_id,
                    "twin": twin,
                    "replay_events": events,
                    "replay_sha256": _projection_hash(events),
                }
            )
        )

    neutral_source = replace(
        source,
        cases=(neutral_case, *source.cases[1:]),
        attack_specs=(neutral_spec, *source.attack_specs[1:]),
        trials=(
            *neutral_trials,
            *(
                trial
                for trial in source.trials
                if trial.allocation_case_id != original_case.allocation_case_id
            ),
        ),
    )
    _validate_links(neutral_source)
    export_projection_set(neutral_source, directory, overwrite=True)
    inputs = load_campaign_inputs(options, project_root=tmp_path)
    assert inputs.cases[0].case.projection_kind == ProjectionKind.NEUTRAL_EXTERNAL_HOST

    altered = neutral_trials[-1]
    event_payloads = [event.model_dump(mode="json") for event in altered.replay_events]
    event_payloads[0].update(
        content="different benign content",
        content_sha256=_hash("different benign content"),
        token_count=RegexTokenCounter().count("different benign content"),
    )
    altered = TrialProjection.model_validate(
        {
            **altered.model_dump(mode="json"),
            "replay_events": event_payloads,
            "replay_sha256": _projection_hash(event_payloads),
        }
    )
    unequal_source = replace(
        neutral_source,
        trials=(*neutral_trials[:2], altered, *neutral_source.trials[3:]),
    )
    with pytest.raises(CampaignInputError, match="identical memory-visible replay"):
        _validate_links(unequal_source)


def test_fixture_backed_tools_require_an_optional_fixture_path(input_fixture):
    from pydantic import ValidationError

    from membench.campaign.preparation_models import CampaignPreparationOptions

    with pytest.raises(ValidationError, match="tool_fixture_path is required"):
        CampaignPreparationOptions.model_validate(
            _options_with_contact_lookup(input_fixture[2])
        )


def test_tool_fixture_is_validated_and_frozen_as_exact_evaluator_only_bytes(
    input_fixture, tmp_path
):
    from membench.campaign.preparation_models import CampaignPreparationOptions

    fixture_relative_path, fixture_path, fixture = _write_tool_fixture(tmp_path)
    options = CampaignPreparationOptions.model_validate(
        _options_with_contact_lookup(input_fixture[2], str(fixture_relative_path))
    )

    inputs = load_campaign_inputs(options, project_root=tmp_path)
    assert inputs.tool_fixture == fixture
    store = CampaignArtifactStore.prepare(tmp_path / "frozen-with-tool-fixture")
    frozen = freeze_campaign_inputs(inputs, store)

    assert frozen.tool_fixture is not None
    assert frozen.tool_fixture.kind == ArtifactKind.TOOL_FIXTURE
    assert frozen.tool_fixture.visibility == ArtifactVisibility.EVALUATOR_ONLY
    assert store.verify(frozen.tool_fixture).read_bytes() == fixture_path.read_bytes()
    assert frozen.tool_fixture in frozen.artifacts
    selection = json.loads(store.verify(frozen.selection).read_bytes())
    assert selection["tool_fixture"]["fixture_id"] == fixture.fixture_id
    assert selection["tool_fixture"]["fixture_sha256"] == fixture.sha256
    assert selection["tool_fixture"]["artifact"] == frozen.tool_fixture.model_dump(
        mode="json"
    )


def test_tool_fixture_must_cover_every_selected_persona(input_fixture, tmp_path):
    from membench.campaign.preparation_models import CampaignPreparationOptions

    fixture_relative_path, _, _ = _write_tool_fixture(tmp_path, persona_ids=("1",))
    options = CampaignPreparationOptions.model_validate(
        _options_with_contact_lookup(input_fixture[2], str(fixture_relative_path))
    )

    with pytest.raises(
        CampaignInputError, match="does not declare selected personas: 2"
    ):
        load_campaign_inputs(options, project_root=tmp_path)


def test_selection_keeps_source_order_not_argument_order(input_fixture, tmp_path):
    _, _, options = input_fixture
    reversed_options = options.model_copy(
        update={"case_ids": ("alloc-synthetic-002", "alloc-synthetic-001")}
    )
    assert load_campaign_inputs(reversed_options, project_root=tmp_path).case_ids == (
        "alloc-synthetic-001",
        "alloc-synthetic-002",
    )
    subset = options.model_copy(update={"case_ids": ("alloc-synthetic-002",)})
    assert (
        load_campaign_inputs(subset, project_root=tmp_path).case_ids == subset.case_ids
    )


@pytest.mark.parametrize(
    "case_id,match", [("alloc-missing", "unknown"), ("alloc-excluded", "excluded")]
)
def test_unknown_and_excluded_cases_fail(input_fixture, tmp_path, case_id, match):
    options = input_fixture[2].model_copy(update={"case_ids": (case_id,)})
    with pytest.raises(CampaignInputError, match=match):
        load_campaign_inputs(options, project_root=tmp_path)


def test_experiment_review_gate_applies_to_selected_cases(input_fixture, tmp_path):
    options = input_fixture[2].model_copy(update={"purpose": "experiment"})
    with pytest.raises(CampaignInputError, match="unreviewed.*alloc-synthetic-001"):
        load_campaign_inputs(options, project_root=tmp_path)
    reviewed = options.model_copy(update={"case_ids": ("alloc-synthetic-002",)})
    inputs = load_campaign_inputs(reviewed, project_root=tmp_path)
    assert inputs.purpose == "experiment"
    assert not inputs.manifest.execution_ready  # Unselected case still awaits review.


def test_relative_paths_use_project_root_and_absolute_paths_work(
    input_fixture, tmp_path, monkeypatch
):
    _, directory, options = input_fixture
    monkeypatch.chdir(directory)
    relative = load_campaign_inputs(options, project_root=tmp_path)
    absolute = load_campaign_inputs(
        options.model_copy(update={"projection_directory": directory}),
        project_root=directory,
    )
    assert absolute.cases == relative.cases


@pytest.mark.parametrize("filename", PROJECTION_FILES)
def test_missing_projection_file_fails(input_fixture, tmp_path, filename):
    _, directory, options = input_fixture
    (directory / filename).unlink()
    with pytest.raises(CampaignInputError, match="missing or unsafe"):
        load_campaign_inputs(options, project_root=tmp_path)


def test_hash_mismatch_fails_before_creating_artifacts(input_fixture, tmp_path):
    _, directory, options = input_fixture
    with (directory / "probes.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(CampaignInputError, match="integrity failed: probes.jsonl"):
        load_campaign_inputs(options, project_root=tmp_path)
    assert not (tmp_path / options.artifact_root).exists()


@pytest.mark.parametrize(
    "kind", ["case", "trial", "label", "duplicate", "benign", "overlay", "indexes"]
)
def test_valid_hashes_do_not_hide_invalid_links(input_fixture, tmp_path, kind):
    source, directory, options = input_fixture
    if kind == "case":
        case = source.cases[0].model_copy(update={"probe_id": source.cases[1].probe_id})
        source = replace(source, cases=(case, source.cases[1]))
    elif kind == "trial":
        trial = source.trials[0].model_copy(update={"attack_spec_id": "wrong-spec"})
        source = replace(source, trials=(trial, *source.trials[1:]))
    elif kind == "label":
        label = source.private_labels[0].model_copy(update={"correct_display_index": 1})
        source = replace(source, private_labels=(label, source.private_labels[1]))
    elif kind == "duplicate":
        trial = source.trials[0].model_copy(
            update={"trial_id": source.trials[1].trial_id}
        )
        source = replace(source, trials=(trial, *source.trials[1:]))
    else:
        trial = source.trials[2]
        events = [event.model_dump(mode="json") for event in trial.replay_events]
        index = 4 if kind == "overlay" else 0
        if kind == "indexes":
            events[index]["source_message_index"] = 0
        else:
            events[index].update(
                content="Different text.",
                content_sha256=_hash("Different text."),
                token_count=RegexTokenCounter().count("Different text."),
            )
        trial = TrialProjection.model_validate(
            {
                **trial.model_dump(mode="json"),
                "replay_events": events,
                "replay_sha256": _projection_hash(events),
            }
        )
        source = replace(source, trials=(*source.trials[:2], trial, *source.trials[3:]))
    export_projection_set(source, directory, overwrite=True)
    # The existing loader validates hashes/counts; campaign adds cross-file checks.
    assert load_projection_set(directory).manifest == source.manifest
    with pytest.raises(CampaignInputError):
        load_campaign_inputs(options, project_root=tmp_path)


def test_freeze_exact_bytes_with_subset_and_evaluator_only_labels(
    input_fixture, tmp_path
):
    source, directory, options = input_fixture
    # Formatting of metadata files is preserved, not regenerated.
    for name in ("manifest.json", "config.json"):
        path = directory / name
        path.write_bytes(
            json.dumps(json.loads(path.read_bytes()), ensure_ascii=False, indent=4)
            .replace("\n", "\r\n")
            .encode("utf-8")
        )
    options = options.model_copy(update={"case_ids": ("alloc-synthetic-002",)})
    inputs = load_campaign_inputs(options, project_root=tmp_path)
    store = CampaignArtifactStore.prepare(tmp_path / "frozen")
    frozen = freeze_campaign_inputs(inputs, store)
    assert len(frozen.artifacts) == 9
    assert not store.sealed
    assert frozen.attack_battery.kind == ArtifactKind.ATTACK_BATTERY
    assert frozen.projection_manifest.kind == ArtifactKind.DATASET_PROJECTION
    for reference in frozen.artifacts:
        assert reference.visibility == ArtifactVisibility.EVALUATOR_ONLY
        path = store.verify(reference)
        if path.name in PROJECTION_FILES:
            assert path.read_bytes() == (directory / path.name).read_bytes()
    selection = json.loads(store.verify(frozen.selection).read_bytes())
    assert selection["selected_case_ids"] == ["alloc-synthetic-002"]
    assert len(selection["selected_trial_ids"]) == 3
    assert selection["preparation_sha256"] == options.sha256
    assert selection["projection_manifest"] == frozen.projection_manifest.model_dump(
        mode="json"
    )
    assert (
        load_projection_set(store.verify(frozen.projection_manifest).parent).manifest
        == source.manifest
    )
    store.write_json(
        artifact_id="later",
        kind=ArtifactKind.OTHER,
        relative_path="later.json",
        value={},
    )
    with pytest.raises(ArtifactExistsError):
        freeze_campaign_inputs(inputs, store)


def test_freeze_uses_verified_snapshot_not_later_source_edits(input_fixture, tmp_path):
    _, directory, options = input_fixture
    inputs = load_campaign_inputs(options, project_root=tmp_path)
    original = (directory / "private_labels.jsonl").read_bytes()
    (directory / "private_labels.jsonl").write_bytes(b"changed after load")
    store = CampaignArtifactStore.prepare(tmp_path / "frozen")
    frozen = freeze_campaign_inputs(inputs, store)
    reference = next(
        ref
        for ref in frozen.artifacts
        if ref.relative_path.endswith("private_labels.jsonl")
    )
    assert store.verify(reference).read_bytes() == original


def test_changed_in_memory_inputs_are_rejected_before_writing(input_fixture, tmp_path):
    inputs = load_campaign_inputs(input_fixture[2], project_root=tmp_path)
    inputs.manifest.artifact_sha256["probes.jsonl"] = "f" * 64
    store = CampaignArtifactStore.prepare(tmp_path / "frozen")
    with pytest.raises(CampaignInputError, match="byte snapshot"):
        freeze_campaign_inputs(inputs, store)
    assert not tuple(store.root.iterdir())


def test_sealed_store_and_frozen_file_tampering(input_fixture, tmp_path):
    inputs = load_campaign_inputs(input_fixture[2], project_root=tmp_path)
    store = CampaignArtifactStore.prepare(tmp_path / "frozen")
    frozen = freeze_campaign_inputs(inputs, store)
    store.seal()
    with pytest.raises(ArtifactSealedError):
        freeze_campaign_inputs(inputs, store)
    store.verify(frozen.attack_battery).write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        store.verify(frozen.attack_battery)


def test_snapshot_parser_reuses_loader_validation(input_fixture):
    source, directory, _ = input_fixture
    payloads = {name: (directory / name).read_bytes() for name in PROJECTION_FILES}
    assert parse_projection_set(payloads).manifest == source.manifest
    with pytest.raises(ProjectionSetError, match="all expected files"):
        parse_projection_set(
            {name: value for name, value in payloads.items() if name != "config.json"}
        )
    with pytest.raises(ProjectionSetError, match="config or manifest"):
        parse_projection_set({**payloads, "config.json": b"not json"})
