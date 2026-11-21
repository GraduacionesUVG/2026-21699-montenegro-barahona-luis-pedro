"""Preparation/binding contracts reuse the public synthetic trial fixtures."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from membench.campaign.artifacts import canonical_json_bytes
from membench.campaign.bindings import CampaignRunBindings, RunProjectionBinding
from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    CampaignConfiguration,
    ChildEnvironmentSpec,
    TrialTwin,
)
from membench.campaign.preparation_models import (
    CampaignConditionSpec,
    CampaignPreparationOptions,
    model_profile_sha256,
)
from membench.contract import load_manifest
from membench.trial.models import TrialPlan


def _configured(plan):
    data = plan.model_dump(mode="json")
    for model in data["run"]["manifest"]["models"].values():
        if model is not None:
            model["digest"] = "a" * 64
    data["run"]["agent_configuration"]["retrieval_top_k"] = 5
    return TrialPlan.model_validate(data)


def _condition(plan):
    return CampaignConditionSpec(
        condition_id="telemem-native-qwen",
        model_profile_id="qwen-writer-qwen-agent-nomic",
        manifest_template=plan.run.manifest,
        agent_configuration=plan.run.agent_configuration,
        replay=plan.replay,
    )


def _options(plan):
    return CampaignPreparationOptions(
        campaign_id=plan.run.campaign_id,
        projection_directory="data/blackbox/projections-v1",
        case_ids=(plan.projection.allocation_case_id,),
        conditions=(_condition(plan),),
        environments=(
            ChildEnvironmentSpec(
                mechanism="telemem",
                python_executable=".venvs/telemem/Scripts/python.exe",
            ),
        ),
        seed=20261006,
        artifact_root="artifacts/campaigns/pilot-001",
    )


def _binding(plan):
    condition = _condition(plan)
    return RunProjectionBinding.from_trial_plan(
        plan,
        condition_id=condition.condition_id,
        model_profile_id=condition.model_profile_id,
    )


@pytest.fixture
def prepared_contracts(trial_plan_factory):
    plans = tuple(_configured(trial_plan_factory(twin=twin)) for twin in TrialTwin)
    options = _options(plans[0])
    projection_manifest = ArtifactReference(
        artifact_id="inputs:projection-manifest",
        kind="dataset_projection",
        relative_path="inputs/projections/manifest.json",
        sha256="b" * 64,
        size_bytes=1024,
        media_type="application/json",
    )
    configuration = CampaignConfiguration(
        campaign_id=options.campaign_id,
        seed=options.seed,
        runs=tuple(plan.run for plan in plans),
        environments=options.environments,
        frozen_inputs=(
            projection_manifest,
            ArtifactReference(
                artifact_id="inputs:attack-specs",
                kind="attack_battery",
                relative_path="inputs/projections/attack_specs.jsonl",
                sha256="c" * 64,
                size_bytes=2048,
                media_type="application/jsonl",
            ),
        ),
        artifact_root=options.artifact_root,
    )
    inventory = CampaignRunBindings(
        campaign_id=options.campaign_id,
        preparation_sha256=options.sha256,
        projection_set_id="synthetic-projections-v1",
        projection_manifest=projection_manifest,
        bindings=tuple(_binding(plan) for plan in plans),
    )
    return options, plans, configuration, inventory


def test_options_accept_json_arguments_and_have_stable_hashes(prepared_contracts):
    options, _, _, _ = prepared_contracts
    parsed = CampaignPreparationOptions.model_validate_json(options.model_dump_json())
    assert parsed == options
    assert parsed.sha256 == options.sha256
    assert isinstance(parsed.projection_directory, Path)
    assert parsed.conditions[0].mechanism.value == "telemem"
    assert parsed.conditions[0].policy.value == "native_selective"
    assert parsed.purpose == "pilot"
    assert parsed.cleanup_storage
    assert parsed.conditions[0].replay.batching.value == "user_turn_v1"
    with pytest.raises(ValidationError, match="frozen"):
        parsed.seed = 99


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("case_ids", [], "selection cannot be empty"),
        ("case_ids", ["alloc-a", "alloc-a"], "duplicate allocation"),
        ("projection_directory", "  ", "non-blank"),
        ("conditions", [], "at least 1"),
        ("seed", -1, "greater than or equal"),
        ("response_replicates", 0, "greater than or equal"),
        ("api_key", "must-not-be-accepted", "Extra inputs"),
    ],
)
def test_options_reject_invalid_or_secret_fields(
    prepared_contracts, field, value, match
):
    data = prepared_contracts[0].model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError, match=match):
        CampaignPreparationOptions.model_validate(data)


@pytest.mark.parametrize(
    "path",
    [
        ".",
        "../campaign",
        "/campaign",
        "C:/campaign",
        "C:campaign",
        "artifacts\\campaign",
        "artifacts//campaign",
        "artifacts/../campaign",
    ],
)
def test_options_reject_unsafe_output_paths(prepared_contracts, path):
    data = prepared_contracts[0].model_dump(mode="json")
    data["artifact_root"] = path
    with pytest.raises(ValidationError, match="relative subdirectory"):
        CampaignPreparationOptions.model_validate(data)


def test_all_cases_is_explicit_and_no_files_or_environments_are_resolved(
    prepared_contracts,
):
    data = prepared_contracts[0].model_dump(mode="json")
    data["case_ids"] = None
    data["projection_directory"] = "nonexistent/projections"
    data["environments"][0]["python_executable"] = "nonexistent/python.exe"
    options = CampaignPreparationOptions.model_validate(data)
    assert options.case_ids is None
    # These are pure contracts; actual availability belongs to preflight.
    assert options.projection_directory == Path("nonexistent/projections")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_condition", "condition IDs"),
        ("missing_environment", "selected mechanisms"),
        ("duplicate_environment", "environment mechanisms"),
        ("ambiguous_profile", "different models"),
    ],
)
def test_condition_and_environment_identities_are_unambiguous(
    prepared_contracts, mutation, match
):
    data = prepared_contracts[0].model_dump(mode="json")
    if mutation == "duplicate_condition":
        data["conditions"] *= 2
    elif mutation == "duplicate_environment":
        data["environments"] *= 2
    elif mutation == "missing_environment":
        data["environments"][0]["mechanism"] = "langmem"
    else:
        other = _condition(prepared_contracts[1][0]).model_dump(mode="json")
        other["condition_id"] = "mistral-writer"
        other["manifest_template"]["models"]["writer"]["model"] = "mistral:7b"
        data["conditions"].append(other)
    with pytest.raises(ValidationError, match=match):
        CampaignPreparationOptions.model_validate(data)


@pytest.mark.parametrize("role", ["writer", "responder", "embedding"])
def test_each_external_model_requires_an_explicit_pin(prepared_contracts, role):
    data = prepared_contracts[0].conditions[0].model_dump(mode="json")
    data["manifest_template"]["models"][role]["digest"] = None
    with pytest.raises(ValidationError, match=f"{role} requires.*pinned digest"):
        CampaignConditionSpec.model_validate(data)


@pytest.mark.parametrize(
    "change,match",
    [
        ("openai", "Ollama"),
        ("writeback", "read-only"),
        ("top_k", "retrieval settings"),
        ("direct_options", "native replay"),
        ("guard", "direct and native_selective"),
    ],
)
def test_conditions_reject_unsupported_execution(prepared_contracts, change, match):
    data = prepared_contracts[0].conditions[0].model_dump(mode="json")
    if change == "openai":
        data["manifest_template"]["models"]["responder"]["provider"] = "openai"
    elif change == "writeback":
        data["agent_configuration"]["memory_read_only"] = False
    elif change == "top_k":
        data["agent_configuration"]["retrieval_top_k"] = 100
    else:
        data["manifest_template"]["condition"]["write_policy"] = (
            "shared_guarded" if change == "guard" else "direct"
        )
        data["manifest_template"]["condition"]["mechanism_options"]["infer"] = False
        if change == "direct_options":
            data["replay"]["native_options"] = {"extra": True}
    with pytest.raises(ValidationError, match=match):
        CampaignConditionSpec.model_validate(data)


@pytest.mark.parametrize("mechanism", ["langmem", "dakera"])
def test_other_mechanisms_use_their_existing_model_roles(prepared_contracts, mechanism):
    filename = "langmem_semantic" if mechanism == "langmem" else "dakera"
    template = load_manifest(
        Path(f"configs/experiments/blackbox/{filename}.example.yaml")
    )
    data = prepared_contracts[0].conditions[0].model_dump(mode="json")
    data["manifest_template"] = template.model_dump(mode="json")
    for model in data["manifest_template"]["models"].values():
        if model is not None:
            model["digest"] = "d" * 64
    condition = CampaignConditionSpec.model_validate(data)
    assert condition.mechanism.value == mechanism
    if mechanism == "dakera":
        assert condition.manifest_template.models.writer is None
        assert condition.manifest_template.models.embedding is None
        data["manifest_template"]["models"]["embedding"] = (
            prepared_contracts[0]
            .conditions[0]
            .manifest_template.models.embedding.model_dump()
        )
        with pytest.raises(ValidationError, match="must be null"):
            CampaignConditionSpec.model_validate(data)
        data["manifest_template"]["models"]["embedding"] = None
        data["manifest_template"]["condition"]["write_policy"] = "native_selective"
        with pytest.raises(ValidationError, match="does not support native_selective"):
            CampaignConditionSpec.model_validate(data)


def test_writer_profiles_can_change_without_changing_agent_or_embedder(
    prepared_contracts,
):
    qwen = prepared_contracts[0].conditions[0]
    data = qwen.model_dump(mode="json")
    data["condition_id"] = "telemem-native-mistral"
    data["model_profile_id"] = "mistral-writer-qwen-agent-nomic"
    data["manifest_template"]["models"]["writer"]["model"] = "mistral:7b"
    data["manifest_template"]["models"]["writer"]["digest"] = "e" * 64
    mistral = CampaignConditionSpec.model_validate(data)
    assert mistral.model_profile_sha256 != qwen.model_profile_sha256
    assert (
        mistral.manifest_template.models.responder
        == qwen.manifest_template.models.responder
    )
    assert (
        mistral.manifest_template.models.embedding
        == qwen.manifest_template.models.embedding
    )
    options_data = prepared_contracts[0].model_dump(mode="json")
    options_data["conditions"].append(mistral.model_dump(mode="json"))
    assert len(CampaignPreparationOptions.model_validate(options_data).conditions) == 2


def test_bindings_roundtrip_and_validate_exact_campaign_and_plan(prepared_contracts):
    options, plans, configuration, inventory = prepared_contracts
    parsed = CampaignRunBindings.model_validate_json(inventory.model_dump_json())
    assert parsed == inventory
    assert parsed.sha256 == inventory.sha256
    parsed.validate_campaign(configuration)
    parsed.validate_preparation(options)
    for plan in plans:
        binding = parsed.binding_for(plan.run.run_id)
        binding.validate_trial_plan(plan)
        assert binding.trial_id == plan.projection.trial_id
        assert binding.model_profile_sha256 == model_profile_sha256(
            plan.run.manifest.models
        )
        assert binding.trial_plan_sha256 == plan.sha256
    assert "private_labels" not in parsed.model_dump_json()
    assert "correct_answer" not in parsed.model_dump_json()
    with pytest.raises(KeyError, match="no projection binding"):
        parsed.binding_for("unknown")
    assert ArtifactKind.RUN_BINDINGS.value == "run_bindings"


@pytest.mark.parametrize("change", ["probe", "replay", "cleanup", "purpose", "run"])
def test_binding_rejects_reconstructed_plan_drift(prepared_contracts, change):
    plan = prepared_contracts[1][0]
    binding = _binding(plan)
    data = plan.model_dump(mode="json")
    if change == "probe":
        data["probe"]["options"][0] = "A changed option."
        data["probe"]["option_sha256"][0] = hashlib.sha256(
            b"A changed option."
        ).hexdigest()
    elif change == "replay":
        data["replay"]["batching"] = "event_v1"
    elif change == "cleanup":
        data["cleanup_storage"] = False
    elif change == "purpose":
        data["purpose"] = "experiment"
        data["attack_review_status"] = "approved"
    else:
        data["run"]["agent_configuration"]["max_steps"] = 6
    changed = TrialPlan.model_validate(data)
    with pytest.raises(ValueError, match="frozen run binding"):
        binding.validate_trial_plan(changed)


def test_bindings_do_not_share_mutable_options_with_plans_or_consumers(
    trial_plan_factory,
):
    plan = _configured(trial_plan_factory(native_options={"nested": {"values": [1]}}))
    binding = _binding(plan)
    plan.replay.native_options["nested"]["values"].append(2)
    assert binding.replay.native_options == {"nested": {"values": [1]}}


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing", "cover every"),
        ("wrong_run_hash", "does not match campaign run"),
        ("wrong_twin", "does not match campaign run"),
        ("family_options", "inconsistent projection"),
        ("campaign", "different campaign"),
        ("source", "not a frozen"),
    ],
)
def test_inventory_rejects_wrong_campaign_bindings(prepared_contracts, change, match):
    _, _, configuration, inventory = prepared_contracts
    data = inventory.model_dump(mode="json")
    if change == "missing":
        data["bindings"].pop()
    elif change == "wrong_run_hash":
        data["bindings"][0]["run_spec_sha256"] = "f" * 64
    elif change == "wrong_twin":
        data["bindings"][0]["twin"] = "poisoned"
    elif change == "family_options":
        data["bindings"][0]["cleanup_storage"] = False
    elif change == "campaign":
        data["campaign_id"] = "another-campaign"
    else:
        data["projection_manifest"]["sha256"] = "f" * 64
    changed = CampaignRunBindings.model_validate(data)
    with pytest.raises(ValueError, match=match):
        changed.validate_campaign(configuration)


@pytest.mark.parametrize(
    "change,match",
    [
        ("duplicates", "binding IDs"),
        ("public", "evaluator-only"),
        ("profile", "different models"),
        ("unreviewed", "pilot status"),
    ],
)
def test_inventory_rejects_ambiguous_or_unreviewed_bindings(
    prepared_contracts, change, match
):
    data = prepared_contracts[3].model_dump(mode="json")
    if change == "duplicates":
        data["bindings"].append(data["bindings"][0])
    elif change == "public":
        data["projection_manifest"]["visibility"] = "public"
    elif change == "profile":
        data["bindings"][0]["model_profile_sha256"] = "f" * 64
    else:
        data["bindings"][0]["purpose"] = "experiment"
    with pytest.raises(ValidationError, match=match):
        CampaignRunBindings.model_validate(data)


def test_inventory_rejects_changed_options_and_condition_references(prepared_contracts):
    options, _, _, inventory = prepared_contracts
    changed = options.model_copy(update={"seed": 100})
    with pytest.raises(ValueError, match="frozen preparation"):
        inventory.validate_preparation(changed)
    data = inventory.model_dump(mode="json")
    data["bindings"][0]["condition_id"] = "unknown-condition"
    with pytest.raises(ValueError, match="unselected condition"):
        CampaignRunBindings.model_validate(data).validate_preparation(options)


def test_hashes_use_shared_utf8_canonicalization(prepared_contracts):
    options, plans, _, inventory = prepared_contracts
    assert options.sha256 == hashlib.sha256(canonical_json_bytes(options)).hexdigest()
    assert (
        inventory.bindings[0].run_spec_sha256
        == hashlib.sha256(canonical_json_bytes(plans[0].run)).hexdigest()
    )


def test_condition_hash_ignores_only_matrix_slots(prepared_contracts):
    condition = prepared_contracts[0].conditions[0]
    data = condition.model_dump(mode="json")
    data["manifest_template"]["attack"]["attack_id"] = "bb-pi-099"
    data["manifest_template"]["isolation"]["run_id"] = "assigned-run"
    data["manifest_template"]["replication"]["seed"] = 42
    data["manifest_template"]["replication"]["response_replicates"] = 3
    assigned = CampaignConditionSpec.model_validate(data)
    assert assigned.sha256 == condition.sha256
    data["manifest_template"]["condition"]["mechanism_options"]["buffer_size"] = 32
    assert CampaignConditionSpec.model_validate(data).sha256 != condition.sha256


def test_relabeling_a_direct_run_as_native_is_rejected(prepared_contracts):
    options, _, _, inventory = prepared_contracts
    data = options.model_dump(mode="json")
    condition = data["conditions"][0]["manifest_template"]["condition"]
    condition["write_policy"] = "direct"
    condition["mechanism_options"]["infer"] = False
    direct = CampaignPreparationOptions.model_validate(data)
    changed = inventory.model_copy(update={"preparation_sha256": direct.sha256})
    with pytest.raises(ValueError, match="execution settings differ"):
        changed.validate_preparation(direct)


def test_same_projection_can_be_bound_to_independent_model_conditions(
    prepared_contracts,
):
    options, plans, configuration, inventory = prepared_contracts
    alternate_plans = []
    alternate_bindings = []
    for plan in plans:
        data = plan.model_dump(mode="json")
        run = data["run"]
        run_id = run["run_id"] + "-mistral"
        run["run_id"] = run_id
        run["trial_family_id"] += "-mistral"
        run["manifest"]["isolation"]["run_id"] = run_id
        run["manifest"]["models"]["writer"]["model"] = "mistral:7b"
        run["manifest"]["models"]["writer"]["digest"] = "f" * 64
        run["scope"]["run_id"] = run_id
        run["scope"]["namespace"] = ["membench", run_id, run["persona_id"]]
        run["agent_task"]["scope"] = run["scope"]
        run["agent_task"]["task_id"] += "-mistral"
        alternate = TrialPlan.model_validate(data)
        alternate_plans.append(alternate)
        alternate_bindings.append(
            RunProjectionBinding.from_trial_plan(
                alternate,
                condition_id="telemem-native-mistral",
                model_profile_id="mistral-writer-qwen-agent-nomic",
            )
        )
    alternate_condition = _condition(alternate_plans[0]).model_copy(
        update={
            "condition_id": "telemem-native-mistral",
            "model_profile_id": "mistral-writer-qwen-agent-nomic",
        }
    )
    expanded_options = CampaignPreparationOptions.model_validate(
        {
            **options.model_dump(mode="json"),
            "conditions": [*options.conditions, alternate_condition],
        }
    )
    expanded_configuration = CampaignConfiguration.model_validate(
        {
            **configuration.model_dump(mode="json"),
            "runs": [*configuration.runs, *(plan.run for plan in alternate_plans)],
        }
    )
    expanded = CampaignRunBindings.model_validate(
        {
            **inventory.model_dump(mode="json"),
            "preparation_sha256": expanded_options.sha256,
            "bindings": [*inventory.bindings, *alternate_bindings],
        }
    )
    expanded.validate_campaign(expanded_configuration)
    expanded.validate_preparation(expanded_options)
    assert len(expanded.bindings) == 6
    assert len({binding.trial_id for binding in expanded.bindings}) == 3
    assert len({binding.model_profile_sha256 for binding in expanded.bindings}) == 2
    detached = expanded.binding_for(plans[0].run.run_id)
    detached.replay.native_options["changed"] = True
    assert not expanded.bindings[0].replay.native_options
