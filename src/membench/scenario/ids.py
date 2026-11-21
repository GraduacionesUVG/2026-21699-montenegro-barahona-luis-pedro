"""Deterministic internal identities derived from one :class:`ScenarioSpec`."""

from __future__ import annotations

from dataclasses import dataclass

from membench.campaign.models import TrialTwin
from membench.scenario.models import ScenarioSpec


@dataclass(frozen=True)
class ScenarioIds:
    """Generated names used by the future scenario compiler.

    None of these values belong in authored scenario JSON.  They are stable for
    a given scenario slug/version and are deliberately free of content hashes.
    """

    scenario_key: str
    attack_set_id: str
    delivery_set_id: str
    recipe_set_id: str
    attack_spec_id: str
    delivery_family_id: str
    baseline_id: str
    evaluation_probe_id: str
    selection_task_id: str
    behavioral_task_id: str
    behavioral_oracle_id: str
    ingress_plan_set_id: str

    def delivery_id(self, twin: TrialTwin) -> str:
        return f"{self.delivery_family_id}.{twin.value}"

    def campaign_id(self, condition_profile: str) -> str:
        return f"{self.scenario_key}-{condition_profile}"


def derive_scenario_ids(spec: ScenarioSpec) -> ScenarioIds:
    """Derive every compiler-owned identity from a validated scenario."""

    key = f"{spec.scenario.slug}-v{spec.scenario.version}"
    return ScenarioIds(
        scenario_key=key,
        attack_set_id=f"{key}-attacks",
        delivery_set_id=f"{key}-deliveries",
        recipe_set_id=f"{key}-recipes",
        attack_spec_id=key,
        delivery_family_id=f"delivery-{key}",
        baseline_id=f"baseline-{key}",
        evaluation_probe_id=f"probe-{key}",
        selection_task_id=f"select-{key}",
        behavioral_task_id=f"task-{key}",
        behavioral_oracle_id=f"oracle-{key}",
        ingress_plan_set_id=f"{key}-{spec.ingress.mode.value}",
    )
