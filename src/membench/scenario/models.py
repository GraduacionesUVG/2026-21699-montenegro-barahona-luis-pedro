"""Human-authored, hash-free contracts for one memory-poisoning scenario.

These models intentionally avoid legacy implementation identities such as
attack-spec IDs, delivery IDs, ingress-plan IDs, campaign IDs, SHA-256 values,
and artifact paths.  The scenario compiler will derive those values from the
stable ``scenario.slug`` and ``scenario.version``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.curation.external_attack_specs import (
    ExternalAttackMethodology,
    ExternalAttackObjective,
)
from membench.ingress.models import ExternalSourceKind, IngressMode
from membench.memory.models import CanonicalModel

_SCENARIO_SLUG_PATTERN = r"^[a-z][a-z0-9-]*$"
_HOST_CASE_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
_PROFILE_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_CONTACT_PATTERN = r"^[a-z][a-z0-9_.-]*$"


class ScenarioAttackFamily(StrEnum):
    """Attack families that one scenario compiler can render."""

    FALSE_FACT_INSERTION = "false_fact_insertion"
    STORED_INSTRUCTION = "stored_instruction_persistence"
    INFORMATION_REPLACEMENT = "information_replacement"


class ScenarioIdentity(CanonicalModel):
    """The only versioned identity a scenario author must maintain."""

    slug: str = Field(pattern=_SCENARIO_SLUG_PATTERN)
    version: int = Field(ge=1)
    purpose: str = Field(default="pilot", min_length=1, max_length=120)


class ScenarioSubject(CanonicalModel):
    """Persona and neutral replay host shared by all generated twins."""

    persona_id: int = Field(ge=0)
    neutral_host_case_id: str = Field(pattern=_HOST_CASE_PATTERN)
    projection_set_id: str = Field(
        default="pmv2-neutral-external-hosts-v1", pattern=_HOST_CASE_PATTERN
    )


class ScenarioBaseline(CanonicalModel):
    """Trusted value seeded before external ingress.

    Required/forbidden markers and all baseline IDs are derived from the
    evaluation section.  This prevents the same value being manually copied
    into several mutable files.
    """

    content: str = Field(min_length=1, max_length=100_000)
    retrieval_query: str = Field(min_length=1, max_length=10_000)
    seed_after_source_message_index: int = Field(ge=0)
    seed_policy: Literal["direct"] = "direct"


class ScenarioIngress(CanonicalModel):
    """How the external source is introduced to the memory mechanism."""

    source_kind: ExternalSourceKind = ExternalSourceKind.EMAIL
    mode: IngressMode = IngressMode.FORCED_INGESTION
    available_after_source_message_index: int = Field(ge=0)
    title: str = Field(min_length=1, max_length=300)
    sender_contact_id: str | None = Field(default=None, pattern=_CONTACT_PATTERN)
    fixture_path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def email_requires_a_fixture_backed_sender(self) -> Self:
        if (
            self.source_kind == ExternalSourceKind.EMAIL
            and self.sender_contact_id is None
        ):
            raise ValueError("email ingress requires sender_contact_id")
        if self.sender_contact_id is not None and self.fixture_path is None:
            raise ValueError("sender_contact_id requires fixture_path")
        return self


class ScenarioExternalAttack(CanonicalModel):
    """Evaluator-reviewed metadata required to freeze an external email family."""

    attack_id: str = Field(pattern=r"^bb-[a-z0-9]+-[0-9]{3}$")
    methodology: ExternalAttackMethodology
    objective: ExternalAttackObjective
    variant: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target_reference: str = Field(min_length=1, max_length=100_000)
    public_attacker_view: str = Field(min_length=1, max_length=100_000)
    trusted_contact_id: str = Field(pattern=_CONTACT_PATTERN)
    attacker_contact_id: str = Field(pattern=_CONTACT_PATTERN)
    trusted_document_id: str = Field(pattern=_CONTACT_PATTERN)

    @model_validator(mode="after")
    def trusted_and_attacker_contacts_must_differ(self) -> Self:
        if self.trusted_contact_id == self.attacker_contact_id:
            raise ValueError("trusted and attacker contact IDs must differ")
        return self


class ScenarioVariants(CanonicalModel):
    """The exact human-reviewed content for aligned experimental twins."""

    clean: str = Field(min_length=1, max_length=100_000)
    matched_benign: str = Field(min_length=1, max_length=100_000)
    poisoned: str = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def bodies_must_be_distinct(self) -> Self:
        values = (self.clean, self.matched_benign, self.poisoned)
        if len(values) != len(set(values)):
            raise ValueError("clean, matched_benign, and poisoned bodies must differ")
        return self


class ScenarioEvaluation(CanonicalModel):
    """Deterministic retrieval question and the values that classify it."""

    query: str = Field(min_length=1, max_length=10_000)
    original_markers: tuple[str, ...] = Field(min_length=1)
    replacement_markers: tuple[str, ...] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("original_markers", "replacement_markers")
    @classmethod
    def markers_must_be_nonblank_and_unique(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(
            not marker.strip() or "\n" in marker or "\r" in marker
            for marker in value
        ):
            raise ValueError("evaluation markers must be nonblank single-line text")
        if len(value) != len(set(value)):
            raise ValueError("evaluation markers must be unique")
        return value

    @model_validator(mode="after")
    def original_and_replacement_markers_must_differ(self) -> Self:
        if set(self.original_markers) & set(self.replacement_markers):
            raise ValueError("original and replacement markers must differ")
        return self


class ScenarioMatrix(CanonicalModel):
    """Named immutable runtime profiles selected for this scenario."""

    condition_profiles: tuple[str, ...] = Field(min_length=1)
    response_replicates: int = Field(default=1, ge=1, le=100)

    @field_validator("condition_profiles")
    @classmethod
    def profiles_must_be_unique_and_well_formed(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not _matches_profile(identifier) for identifier in value):
            raise ValueError("condition profile names are malformed")
        if len(value) != len(set(value)):
            raise ValueError("condition profile names must be unique")
        return value


class ScenarioBehavioral(CanonicalModel):
    """Optional downstream action test; absent for retrieval-only experiments."""

    enabled: bool = False
    instruction: str | None = Field(default=None, min_length=1, max_length=10_000)
    retrieval_query: str | None = Field(default=None, min_length=1, max_length=10_000)
    maximum_messages: int = Field(default=1, ge=1, le=10)

    @model_validator(mode="after")
    def enabled_behavior_requires_public_task_content(self) -> Self:
        fields = (self.instruction, self.retrieval_query)
        if self.enabled and any(value is None for value in fields):
            raise ValueError(
                "enabled behavioral evaluation requires instruction and retrieval_query"
            )
        if not self.enabled and any(value is not None for value in fields):
            raise ValueError(
                "disabled behavioral evaluation cannot contain task content"
            )
        return self


class ScenarioSpec(CanonicalModel):
    """One complete editable scenario, free of generated identities and hashes."""

    schema_version: Literal[1] = 1
    scenario: ScenarioIdentity
    attack_family: ScenarioAttackFamily
    subject: ScenarioSubject
    baseline: ScenarioBaseline | None = None
    ingress: ScenarioIngress
    external_attack: ScenarioExternalAttack | None = None
    variants: ScenarioVariants
    evaluation: ScenarioEvaluation
    matrix: ScenarioMatrix
    behavioral: ScenarioBehavioral = Field(default_factory=ScenarioBehavioral)

    @model_validator(mode="after")
    def scenario_is_internally_coherent(self) -> Self:
        baseline = self.baseline
        if self.attack_family == ScenarioAttackFamily.INFORMATION_REPLACEMENT:
            if baseline is None:
                raise ValueError("information replacement requires a trusted baseline")
        if baseline is not None:
            if (
                baseline.seed_after_source_message_index
                >= self.ingress.available_after_source_message_index
            ):
                raise ValueError("baseline must be scheduled before external ingress")
            if any(
                marker not in baseline.content
                for marker in self.evaluation.original_markers
            ):
                raise ValueError(
                    "baseline content must contain every original evaluation marker"
                )

        replacement_markers = self.evaluation.replacement_markers
        if any(marker not in self.variants.poisoned for marker in replacement_markers):
            raise ValueError(
                "poisoned body must contain every replacement evaluation marker"
            )
        controls = (self.variants.clean, self.variants.matched_benign)
        if any(marker in body for body in controls for marker in replacement_markers):
            raise ValueError("control bodies must not contain replacement markers")
        if (
            self.ingress.mode != IngressMode.FORCED_INGESTION
        ):
            raise ValueError(
                "scenario compiler currently supports forced_ingestion only"
            )
        return self


def _matches_profile(value: str) -> bool:
    """Keep condition-profile validation local and dependency-free."""

    import re

    return re.fullmatch(_PROFILE_PATTERN, value) is not None
