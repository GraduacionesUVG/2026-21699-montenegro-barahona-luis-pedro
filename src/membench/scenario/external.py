"""Render forced-ingress external artifacts directly from a ScenarioSpec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from membench.curation.delivery_projections import (
    DeliveryProjectionSet,
    export_delivery_projection_set,
)
from membench.curation.external_attack_spec_drafts import (
    ExternalAttackSpecDraft,
    ExternalAttackSpecDraftSet,
)
from membench.curation.external_attack_specs import export_external_attack_spec_set
from membench.curation.external_delivery_recipes import (
    DeliveryPayloads,
    ExternalDeliveryRecipe,
    ExternalDeliveryRecipeSet,
    render_external_delivery_projection_set,
)
from membench.curation.projections import AttackReviewStatus
from membench.ingress.models import IngressMode
from membench.ingress.plans import (
    TrialIngressPlanSet,
    build_trial_ingress_plan_set,
    export_trial_ingress_plan_set,
)
from membench.scenario.compiler import ScenarioDryRunPlan
from membench.scenario.models import ScenarioSpec
from membench.tools.fixtures import ToolFixtureBundle


class ScenarioExternalRenderError(RuntimeError):
    """Scenario content cannot form a valid frozen external ingress family."""


@dataclass(frozen=True)
class ScenarioExternalArtifacts:
    """In-memory immutable external artifacts, ready to export or prepare."""

    attack_set: ExternalAttackSpecDraftSet
    deliveries: DeliveryProjectionSet
    ingress_plans: TrialIngressPlanSet
    host_trial_ids: dict[str, str]


def render_forced_external_artifacts(
    spec: ScenarioSpec,
    *,
    dry_run: ScenarioDryRunPlan,
    fixture: ToolFixtureBundle,
) -> ScenarioExternalArtifacts:
    """Render the three twin emails and their host bindings without I/O."""

    external = spec.external_attack
    if external is None:
        raise ScenarioExternalRenderError("scenario requires external_attack metadata")
    if spec.ingress.mode != IngressMode.FORCED_INGESTION:
        raise ScenarioExternalRenderError("only forced ingestion is supported")
    if fixture.fixture_id == "":
        raise ScenarioExternalRenderError("fixture ID cannot be blank")
    ids = dry_run.ids
    allocation_case_id = f"alloc-external-{ids.scenario_key}"
    draft_set = ExternalAttackSpecDraftSet(
        attack_set_id=ids.attack_set_id,
        projection_set_id=spec.subject.projection_set_id,
        specs=(
            ExternalAttackSpecDraft(
                spec_ordinal=1,
                attack_spec_id=ids.attack_spec_id,
                allocation_case_id=allocation_case_id,
                attack_id=external.attack_id,
                attack_methodology=external.methodology,
                objective=external.objective,
                variant=external.variant,
                source_kind=spec.ingress.source_kind,
                insert_after_source_message_index=(
                    spec.ingress.available_after_source_message_index
                ),
                target_reference=external.target_reference,
                public_attacker_view=external.public_attacker_view,
                review_status=AttackReviewStatus.APPROVED,
            ),
        ),
    )
    attack_set = draft_set.freeze()
    recipes = ExternalDeliveryRecipeSet(
        recipe_set_id=ids.recipe_set_id,
        delivery_set_id=ids.delivery_set_id,
        attack_set_id=ids.attack_set_id,
        projection_set_id=spec.subject.projection_set_id,
        fixture_id=fixture.fixture_id,
        fixture_sha256=fixture.sha256,
        recipes=(
            ExternalDeliveryRecipe(
                recipe_ordinal=1,
                recipe_id=ids.attack_spec_id,
                attack_spec_id=ids.attack_spec_id,
                persona_id=str(spec.subject.persona_id),
                delivery_family_id=ids.delivery_family_id,
                source_kind=spec.ingress.source_kind,
                source_id=f"source-{ids.scenario_key}",
                title=spec.ingress.title,
                trusted_contact_id=external.trusted_contact_id,
                attacker_contact_id=external.attacker_contact_id,
                trusted_document_id=external.trusted_document_id,
                available_after_source_message_index=(
                    spec.ingress.available_after_source_message_index
                ),
                payloads=DeliveryPayloads(
                    clean=spec.variants.clean,
                    matched_benign=spec.variants.matched_benign,
                    poisoned=spec.variants.poisoned,
                ),
            ),
        ),
    )
    deliveries = render_external_delivery_projection_set(
        recipes, fixture=fixture, attack_set=attack_set
    )
    delivery_trial_ids = tuple(
        delivery.trial_id for delivery in deliveries.families[0].deliveries
    )
    hosts = dict(
        zip(delivery_trial_ids, dry_run.neutral_host.trial_ids, strict=True)
    )
    plans = build_trial_ingress_plan_set(
        ingress_plan_set_id=ids.ingress_plan_set_id,
        ingress_mode=IngressMode.FORCED_INGESTION,
        delivery_set=deliveries,
        attack_set=attack_set,
        fixture=fixture,
        host_trial_ids=hosts,
    )
    return ScenarioExternalArtifacts(draft_set, deliveries, plans, hosts)


def export_forced_external_artifacts(
    artifacts: ScenarioExternalArtifacts, destination: str | Path
) -> None:
    """Freeze rendered attack specs, deliveries, and plans in one new directory."""

    root = Path(destination)
    attack_set = artifacts.attack_set.freeze()
    export_external_attack_spec_set(attack_set, root)
    export_delivery_projection_set(artifacts.deliveries, root)
    export_trial_ingress_plan_set(artifacts.ingress_plans, root)
