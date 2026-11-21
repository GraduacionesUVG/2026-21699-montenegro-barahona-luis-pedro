"""Editable recipes and deterministic rendering for external deliveries.

Recipes deliberately hold the human-reviewed source bodies and fixture resource
references.  Rendering performs no text generation: it resolves those stable
references, validates the attack contract, and creates the immutable delivery
models consumed by the ingress layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from membench.campaign.models import TrialTwin
from membench.curation.delivery_projections import (
    DeliveryProjectionFamily,
    DeliveryProjectionSet,
    build_delivery_projection_set,
    validate_delivery_attack_links,
)
from membench.curation.external_attack_specs import (
    ExternalAttackSpec,
    ExternalAttackSpecSet,
)
from membench.curation.projections import AttackReviewStatus
from membench.ingress.models import (
    ExternalDelivery,
    ExternalSourceKind,
    SourceSelectionTask,
)
from membench.memory.models import CanonicalModel
from membench.tools.fixtures import ToolFixtureBundle

_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_SET_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ExternalDeliveryRecipeError(ValueError):
    """Raised when editable delivery recipes cannot be safely rendered."""


class DeliveryPayloads(CanonicalModel):
    """Exact human-reviewed source content for one aligned twin family."""

    clean: str = Field(min_length=1, max_length=100_000)
    matched_benign: str = Field(min_length=1, max_length=100_000)
    poisoned: str = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def payloads_must_be_distinct_and_not_leak_whole_payloads(self) -> Self:
        values = (self.clean, self.matched_benign, self.poisoned)
        if len(set(values)) != len(values):
            raise ValueError("delivery payloads must be distinct across twins")
        if self.poisoned in self.clean or self.poisoned in self.matched_benign:
            raise ValueError("poisoned payload must not occur in a control body")
        return self


class SourceSelectionRecipe(CanonicalModel):
    """Public task used when the external source is agent-selected."""

    selection_task_id: str = Field(pattern=_ID_PATTERN)
    instruction: str = Field(min_length=1, max_length=10_000)
    retrieval_query: str = Field(min_length=1, max_length=256)


class ExternalDeliveryRecipe(CanonicalModel):
    """Editable binding between one attack spec and one fixture workflow."""

    schema_version: Literal[1] = 1
    recipe_ordinal: int = Field(ge=1)
    recipe_id: str = Field(pattern=_ID_PATTERN)
    attack_spec_id: str = Field(pattern=_ID_PATTERN)
    persona_id: str = Field(pattern=r"^[0-9]+$")
    delivery_family_id: str = Field(pattern=_ID_PATTERN)
    source_kind: ExternalSourceKind
    source_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=1, max_length=300)
    trusted_contact_id: str = Field(pattern=_ID_PATTERN)
    attacker_contact_id: str = Field(pattern=_ID_PATTERN)
    trusted_document_id: str = Field(pattern=_ID_PATTERN)
    available_after_source_message_index: int = Field(ge=0)
    payloads: DeliveryPayloads
    selection: SourceSelectionRecipe | None = None

    @model_validator(mode="after")
    def distinct_fixture_resources_are_required(self) -> Self:
        if self.trusted_contact_id == self.attacker_contact_id:
            raise ValueError("trusted and attacker contact IDs must differ")
        return self


class ExternalDeliveryRecipeSet(CanonicalModel):
    """One editable, fixture-bound collection of external delivery recipes."""

    schema_version: Literal[1] = 1
    recipe_set_id: str = Field(pattern=_SET_ID_PATTERN)
    delivery_set_id: str = Field(pattern=_SET_ID_PATTERN)
    attack_set_id: str = Field(pattern=_SET_ID_PATTERN)
    projection_set_id: str = Field(pattern=_SET_ID_PATTERN)
    fixture_id: str = Field(pattern=_ID_PATTERN)
    fixture_sha256: str = Field(pattern=_SHA256_PATTERN)
    recipes: tuple[ExternalDeliveryRecipe, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def recipes_must_be_unique_and_deterministic(self) -> Self:
        ordinals = tuple(recipe.recipe_ordinal for recipe in self.recipes)
        if ordinals != tuple(range(1, len(self.recipes) + 1)):
            raise ValueError("recipe ordinals must be contiguous and start at one")
        for attribute in (
            "recipe_id",
            "attack_spec_id",
            "delivery_family_id",
            "source_id",
        ):
            values = [getattr(recipe, attribute) for recipe in self.recipes]
            if len(values) != len(set(values)):
                raise ValueError(f"delivery recipes require unique {attribute}")
        tasks = [recipe.selection for recipe in self.recipes if recipe.selection]
        task_ids = [task.selection_task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("delivery recipes require unique selection task IDs")
        return self

    @property
    def sha256(self) -> str:
        """Return the stable identity of the exact editable recipe content."""

        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_external_delivery_recipe_set(
    path: str | Path,
) -> ExternalDeliveryRecipeSet:
    """Load one editable JSON recipe set without rendering any deliveries."""

    source = Path(path)
    try:
        return ExternalDeliveryRecipeSet.model_validate_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise ExternalDeliveryRecipeError(
            f"could not load external delivery recipes: {source}"
        ) from exc


def render_external_delivery_projection_set(
    recipes: ExternalDeliveryRecipeSet,
    *,
    fixture: ToolFixtureBundle,
    attack_set: ExternalAttackSpecSet,
    require_approved_specs: bool = True,
) -> DeliveryProjectionSet:
    """Resolve recipes into immutable, hash-checked external delivery twins.

    The caller supplies frozen fixture and attack-spec artifacts.  This function
    intentionally does not generate, paraphrase, or otherwise alter a recipe's
    body text.
    """

    _validate_set_bindings(recipes, fixture=fixture, attack_set=attack_set)
    families = tuple(
        _render_family(
            recipe,
            fixture=fixture,
            attack_spec=attack_set.spec_for(recipe.attack_spec_id),
            require_approved_specs=require_approved_specs,
        )
        for recipe in recipes.recipes
    )
    rendered = build_delivery_projection_set(
        delivery_set_id=recipes.delivery_set_id,
        attack_set_id=recipes.attack_set_id,
        projection_set_id=recipes.projection_set_id,
        families=families,
    )
    validate_delivery_attack_links(rendered, attack_set)
    return rendered


def _validate_set_bindings(
    recipes: ExternalDeliveryRecipeSet,
    *,
    fixture: ToolFixtureBundle,
    attack_set: ExternalAttackSpecSet,
) -> None:
    if recipes.fixture_id != fixture.fixture_id:
        raise ExternalDeliveryRecipeError("recipe and fixture IDs differ")
    if recipes.fixture_sha256 != fixture.sha256:
        raise ExternalDeliveryRecipeError("recipe fixture hash does not match")
    if recipes.attack_set_id != attack_set.attack_set_id:
        raise ExternalDeliveryRecipeError("recipe and attack set IDs differ")
    if recipes.projection_set_id != attack_set.projection_set_id:
        raise ExternalDeliveryRecipeError("recipe and attack projection-set IDs differ")


def _render_family(
    recipe: ExternalDeliveryRecipe,
    *,
    fixture: ToolFixtureBundle,
    attack_spec: ExternalAttackSpec,
    require_approved_specs: bool,
) -> DeliveryProjectionFamily:
    _validate_recipe_against_spec(recipe, attack_spec, require_approved_specs)
    sender = _validate_fixture_resources(recipe, fixture)
    delivery_values = (
        (TrialTwin.CLEAN, recipe.payloads.clean),
        (TrialTwin.MATCHED_BENIGN, recipe.payloads.matched_benign),
        (TrialTwin.POISONED, recipe.payloads.poisoned),
    )
    deliveries = tuple(
        _delivery_from_recipe(
            recipe,
            attack_spec=attack_spec,
            twin=twin,
            content=content,
            sender=sender,
        )
        for twin, content in delivery_values
    )
    selection_task = _selection_task_from_recipe(recipe, attack_spec)
    return DeliveryProjectionFamily(
        family_ordinal=recipe.recipe_ordinal,
        delivery_family_id=recipe.delivery_family_id,
        allocation_case_id=attack_spec.allocation_case_id,
        attack_spec_id=attack_spec.attack_spec_id,
        attack_spec_sha256=attack_spec.sha256,
        attack_methodology=attack_spec.attack_methodology,
        attack_objective=attack_spec.objective,
        attack_variant=attack_spec.variant,
        target_reference_sha256=attack_spec.target_reference_sha256,
        source_kind=recipe.source_kind,
        persona_id=recipe.persona_id,
        fixture_id=fixture.fixture_id,
        fixture_sha256=fixture.sha256,
        deliveries=deliveries,
        selection_task=selection_task,
    )


def _validate_recipe_against_spec(
    recipe: ExternalDeliveryRecipe,
    spec: ExternalAttackSpec,
    require_approved_specs: bool,
) -> None:
    if recipe.source_kind != spec.source_kind:
        raise ExternalDeliveryRecipeError(
            f"recipe source kind differs from attack spec: {recipe.recipe_id}"
        )
    if (
        recipe.available_after_source_message_index
        != spec.insert_after_source_message_index
    ):
        raise ExternalDeliveryRecipeError(
            "recipe availability index differs from attack-spec insertion index: "
            f"{recipe.recipe_id}"
        )
    if require_approved_specs and spec.review_status != AttackReviewStatus.APPROVED:
        raise ExternalDeliveryRecipeError(
            f"attack spec is not approved: {spec.attack_spec_id}"
        )


def _validate_fixture_resources(
    recipe: ExternalDeliveryRecipe,
    fixture: ToolFixtureBundle,
) -> str | None:
    if recipe.persona_id not in fixture.persona_ids:
        raise ExternalDeliveryRecipeError(
            f"recipe persona is not declared by fixture: {recipe.persona_id}"
        )
    contacts = {
        (item.contact.owner_user_id, item.contact.contact_id): item.contact
        for item in fixture.contacts
    }
    trusted = contacts.get((recipe.persona_id, recipe.trusted_contact_id))
    attacker = contacts.get((recipe.persona_id, recipe.attacker_contact_id))
    if recipe.trusted_contact_id == recipe.attacker_contact_id:
        raise ExternalDeliveryRecipeError(
            "recipe trusted and attacker contacts must differ"
        )
    if trusted is None:
        raise ExternalDeliveryRecipeError("recipe trusted contact is unavailable")
    if attacker is None:
        raise ExternalDeliveryRecipeError("recipe attacker contact is unavailable")
    document = next(
        (
            item.document
            for item in fixture.documents
            if item.document.document_id == recipe.trusted_document_id
            and recipe.persona_id in item.document.authorized_user_ids
        ),
        None,
    )
    if document is None:
        raise ExternalDeliveryRecipeError("recipe trusted document is unavailable")
    if not document.approved:
        raise ExternalDeliveryRecipeError("recipe trusted document is not approved")
    if recipe.trusted_contact_id not in document.approved_recipient_contact_ids:
        raise ExternalDeliveryRecipeError(
            "recipe trusted contact is not approved by its trusted document"
        )
    if recipe.attacker_contact_id in document.approved_recipient_contact_ids:
        raise ExternalDeliveryRecipeError(
            "recipe attacker contact is approved by its trusted document"
        )
    if recipe.source_kind == ExternalSourceKind.EMAIL:
        if attacker.email is None:
            raise ExternalDeliveryRecipeError(
                "email recipe attacker contact has no email"
            )
        return attacker.email
    return None


def _delivery_from_recipe(
    recipe: ExternalDeliveryRecipe,
    *,
    attack_spec: ExternalAttackSpec,
    twin: TrialTwin,
    content: str,
    sender: str | None,
) -> ExternalDelivery:
    return ExternalDelivery(
        delivery_id=f"{recipe.delivery_family_id}.{twin.value}",
        delivery_family_id=recipe.delivery_family_id,
        trial_id=(f"{attack_spec.allocation_case_id}--{twin.value.replace('_', '-')}"),
        allocation_case_id=attack_spec.allocation_case_id,
        twin=twin,
        source_kind=recipe.source_kind,
        source_id=recipe.source_id,
        title=recipe.title,
        sender=sender,
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        available_after_source_message_index=recipe.available_after_source_message_index,
    )


def _selection_task_from_recipe(
    recipe: ExternalDeliveryRecipe,
    spec: ExternalAttackSpec,
) -> SourceSelectionTask | None:
    if recipe.selection is None:
        return None
    tools = (
        ("read_message", "search_messages")
        if recipe.source_kind == ExternalSourceKind.EMAIL
        else ("read_document", "search_documents")
    )
    return SourceSelectionTask(
        selection_task_id=recipe.selection.selection_task_id,
        delivery_family_id=recipe.delivery_family_id,
        allocation_case_id=spec.allocation_case_id,
        instruction=recipe.selection.instruction,
        retrieval_query=recipe.selection.retrieval_query,
        allowed_tools=tools,
    )
