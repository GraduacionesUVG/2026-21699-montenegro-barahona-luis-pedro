"""A prepared campaign can be rehydrated without rebuilding its matrix."""

import json
import shutil
import sys

import pytest

from membench.campaign.loading import (
    PreparedCampaignLoader,
    PreparedCampaignLoadError,
    load_prepared_campaign,
)
from membench.campaign.preparation import prepare_campaign
from membench.campaign.preparation_models import CampaignPreparationOptions


@pytest.fixture
def loadable_campaign(input_fixture, tmp_path):
    payload = input_fixture[2].model_dump(mode="json")
    payload["artifact_root"] = "artifacts/campaigns/loadable-pilot"
    payload["environments"][0].update(
        python_executable=sys.executable,
        inherited_env_keys=["LOADER_PROVIDER_KEY"],
    )
    options = CampaignPreparationOptions.model_validate(payload)
    environment = {"LOADER_PROVIDER_KEY": "not-a-real-secret"}
    prepared = prepare_campaign(
        options,
        project_root=tmp_path,
        source_environment=environment,
    )
    return prepared, options, environment


def test_loads_verified_preparation_after_original_process_boundary(
    loadable_campaign, input_fixture, tmp_path
):
    prepared, options, environment = loadable_campaign
    # Frozen inputs, not the mutable projection source, are authoritative.
    shutil.rmtree(input_fixture[1])

    loaded = load_prepared_campaign(
        options.artifact_root,
        project_root=tmp_path,
        source_environment=environment,
    )

    assert loaded.configuration == prepared.configuration
    assert loaded.bindings == prepared.bindings
    assert loaded.artifacts.root == prepared.artifacts.root
    assert len(loaded.artifacts.references()) == 13
    assert loaded.create_launcher().configuration == prepared.configuration


def test_missing_completion_marker_is_not_loadable(loadable_campaign, tmp_path):
    prepared, _, environment = loadable_campaign
    (prepared.artifacts.root / "preparation/complete.json").unlink()

    with pytest.raises(PreparedCampaignLoadError) as caught:
        load_prepared_campaign(
            prepared.artifacts.root,
            project_root=tmp_path,
            source_environment=environment,
        )

    assert caught.value.stage == "read_completion"
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_modified_frozen_artifact_fails_registration(loadable_campaign, tmp_path):
    prepared, _, environment = loadable_campaign
    (prepared.artifacts.root / "preparation/options.json").write_bytes(b"modified")

    with pytest.raises(PreparedCampaignLoadError) as caught:
        load_prepared_campaign(
            prepared.artifacts.root,
            project_root=tmp_path,
            source_environment=environment,
        )

    assert caught.value.stage == "register_artifacts"


@pytest.mark.parametrize(
    "execution_path",
    [
        "campaign_configuration.json",
        "campaign_result.json",
        "analysis.json",
        "checksums.json",
        "children",
    ],
)
def test_started_or_sealed_campaign_is_rejected(
    loadable_campaign, tmp_path, execution_path
):
    prepared, _, environment = loadable_campaign
    path = prepared.artifacts.root / execution_path
    if execution_path == "children":
        path.mkdir()
    else:
        path.write_text("{}", encoding="utf-8")

    with pytest.raises(PreparedCampaignLoadError) as caught:
        load_prepared_campaign(
            prepared.artifacts.root,
            project_root=tmp_path,
            source_environment=environment,
        )

    assert caught.value.stage == "verify_execution_state"
    assert isinstance(caught.value.__cause__, FileExistsError)


def test_unregistered_file_is_rejected(loadable_campaign, tmp_path):
    prepared, _, environment = loadable_campaign
    (prepared.artifacts.root / "unexpected.txt").write_text("x", encoding="utf-8")

    with pytest.raises(PreparedCampaignLoadError) as caught:
        PreparedCampaignLoader().load(
            prepared.artifacts.root,
            project_root=tmp_path,
            source_environment=environment,
        )

    assert caught.value.stage == "register_artifacts"


def test_changed_completion_contract_is_rejected(loadable_campaign, tmp_path):
    prepared, _, environment = loadable_campaign
    path = prepared.artifacts.root / "preparation/complete.json"
    completion = json.loads(path.read_bytes())
    completion["campaign_id"] = "different-campaign"
    path.write_text(json.dumps(completion), encoding="utf-8")

    with pytest.raises(PreparedCampaignLoadError) as caught:
        load_prepared_campaign(
            prepared.artifacts.root,
            project_root=tmp_path,
            source_environment=environment,
        )

    assert caught.value.stage == "validate_contracts"


def test_environment_is_rechecked_when_loading(loadable_campaign, tmp_path):
    prepared, _, _ = loadable_campaign

    with pytest.raises(PreparedCampaignLoadError) as caught:
        load_prepared_campaign(
            prepared.artifacts.root,
            project_root=tmp_path,
            source_environment={},
        )

    assert caught.value.stage == "preflight_environments"


def test_campaign_root_must_match_frozen_artifact_root(loadable_campaign, tmp_path):
    prepared, _, environment = loadable_campaign
    moved = tmp_path / "artifacts/campaigns/moved-pilot"
    prepared.artifacts.root.rename(moved)

    with pytest.raises(PreparedCampaignLoadError) as caught:
        load_prepared_campaign(
            moved,
            project_root=tmp_path,
            source_environment=environment,
        )

    assert caught.value.stage == "validate_contracts"
