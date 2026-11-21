"""Single-file authored scenario contracts.

Scenario specifications are the human-editable source for an experiment.  A
future compiler renders the existing frozen delivery, baseline, evaluation,
and campaign contracts from these models; users do not maintain those
identities or hashes by hand.
"""

from membench.scenario.compiler import (
    NeutralHostEvidence,
    ScenarioCompilerError,
    ScenarioDryRunPlan,
    ScenarioMatrixRun,
    compile_dry_run,
    load_and_compile_dry_run,
    load_neutral_host,
    load_scenario_spec,
)
from membench.scenario.external import (
    ScenarioExternalArtifacts,
    ScenarioExternalRenderError,
    export_forced_external_artifacts,
    render_forced_external_artifacts,
)
from membench.scenario.ids import ScenarioIds, derive_scenario_ids
from membench.scenario.models import (
    ScenarioAttackFamily,
    ScenarioBaseline,
    ScenarioBehavioral,
    ScenarioEvaluation,
    ScenarioExternalAttack,
    ScenarioIdentity,
    ScenarioIngress,
    ScenarioMatrix,
    ScenarioSpec,
    ScenarioSubject,
    ScenarioVariants,
)
from membench.scenario.profiles import (
    ScenarioConditionProfile,
    ScenarioConditionProfileRegistry,
    ScenarioProfileError,
    load_condition_profile_registry,
)
from membench.scenario.runner import (
    ScenarioBuildError,
    ScenarioCampaignBuild,
    ScenarioCampaignSources,
    build_campaign_options,
    prepare_scenario_campaign,
)
from membench.scenario.validation import (
    ScenarioValidationError,
    ScenarioValidationReport,
    validate_scenario_spec,
)

__all__ = [
    "NeutralHostEvidence",
    "ScenarioAttackFamily",
    "ScenarioBaseline",
    "ScenarioBehavioral",
    "ScenarioBuildError",
    "ScenarioCampaignBuild",
    "ScenarioCampaignSources",
    "ScenarioCompilerError",
    "ScenarioConditionProfile",
    "ScenarioConditionProfileRegistry",
    "ScenarioDryRunPlan",
    "ScenarioExternalArtifacts",
    "ScenarioExternalAttack",
    "ScenarioExternalRenderError",
    "ScenarioEvaluation",
    "ScenarioIdentity",
    "ScenarioIngress",
    "ScenarioIds",
    "ScenarioMatrix",
    "ScenarioMatrixRun",
    "ScenarioProfileError",
    "ScenarioSpec",
    "ScenarioSubject",
    "ScenarioVariants",
    "ScenarioValidationError",
    "ScenarioValidationReport",
    "compile_dry_run",
    "build_campaign_options",
    "derive_scenario_ids",
    "export_forced_external_artifacts",
    "load_and_compile_dry_run",
    "load_condition_profile_registry",
    "load_neutral_host",
    "load_scenario_spec",
    "prepare_scenario_campaign",
    "render_forced_external_artifacts",
    "validate_scenario_spec",
]
