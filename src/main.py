from pathlib import Path

from membench.scenario.compiler import load_scenario_spec
from membench.scenario.profiles import load_condition_profile_registry
from membench.scenario.runner import (
    ScenarioGeneratedCampaignSources,
    prepare_generated_scenario_campaign,
)

root = Path(__file__).resolve().parent
 
# TODO: CHANGE THE SCENARIO ALWAYS WHEN CHANGING THE ATTACK
spec = load_scenario_spec(
    root / "configs/scenarios/p371-ir-smoke-v1.json" 
)
profiles = load_condition_profile_registry(
    root / "configs/scenario_condition_profiles.json"
)

# TODO CHange fixtures when using Information Replacement and Stored Instruction
# For p277 information replacement use: p277-reentry-report-tools-v1.json
# For p277 stored instruction use: p277-external-tools-v1.json
# Use oracles just for stored instruction
sources = ScenarioGeneratedCampaignSources(
    fixture_path=Path("configs/fixtures/p371-external-tools-v1.json"),
    generated_input_directory=Path(
        "artifacts/s/poc4-tlm-in"
    ),
    external_artifact_directory=Path(
        "artifacts/s/poc4-tlm-ext"
    ),
    artifact_root="artifacts/c/poc4-tlm",  
    # behavioral_oracle_path=Path(
    #     "configs/behavioral_oracles/p371-si-smoke-v1.json"
    # ),
    # behavioral_task_path=Path(
    #     "configs/behavioral_tasks/p371-si-smoke-v1.json"
    # )
)

prepared = prepare_generated_scenario_campaign(
    spec,
    projection_directory=root / "data/blackbox/neutral-external-hosts-v1",
    condition_profile_registry=profiles,
    sources=sources,
    project_root=root,
    env_file=root / ".env"
)

print(prepared.configuration.campaign_id)
run = prepared.configuration.runs[0]
binding = prepared.bindings.bindings[0]

print("campaign:", prepared.configuration.campaign_id)
print("condition profile:", binding.condition_id)
print("mechanism:", run.manifest.condition.mechanism)
print("write policy:", run.manifest.condition.write_policy)
print("model profile:", binding.model_profile_id)
print("runs:", len(prepared.configuration.runs))