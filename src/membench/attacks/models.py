"""Canonical evaluator-only models for PersonaMem attack candidate ranking."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator, model_validator

from membench.memory.models import CanonicalModel


class AttackMethodology(StrEnum):
    """Attack variants supported by the deterministic candidate selectors."""

    ABRUPT_FALSE_CORRECTION = "abrupt_false_correction"
    NEGATED_IDENTIFIER = "negated_identifier"
    OWNERSHIP_CONFUSION = "ownership_confusion"
    GRADUAL_DRIFT = "gradual_drift"
    SEMANTIC_HIJACKING = "semantic_hijacking"
    STORED_INSTRUCTION_INJECTION = "stored_instruction_injection"


class CandidateFitStatus(StrEnum):
    """Whether a ranked candidate can proceed without additional review."""

    ELIGIBLE = "eligible"
    REVIEW_REQUIRED = "review_required"
    INELIGIBLE = "ineligible"


class CandidateAttackFit(CanonicalModel):
    """Privacy-minimized score for one candidate and attack methodology."""

    schema_version: int = Field(default=1, ge=1, le=1)
    candidate_id: str = Field(pattern=r"^pmv2-p[0-9]+-row[0-9]{6}$")
    persona_id: str = Field(pattern=r"^[0-9]+$")
    source_row_index: int = Field(ge=0)
    attack_methodology: AttackMethodology
    contract_family: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    contract_compatible: bool
    status: CandidateFitStatus
    rank: int | None = Field(default=None, ge=1)
    score: int = Field(ge=0, le=100)
    score_components: dict[str, int] = Field(default_factory=dict)

    topic: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    preference_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    conversation_scenario: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    owner: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    sensitive: bool
    updated: bool
    sensitive_field_kinds: tuple[str, ...] = Field(default_factory=tuple)
    evidence_link_kind: str = Field(pattern=r"^[a-z][a-z0-9_]*$")

    false_target_option_index: int | None = Field(default=None, ge=0)
    false_target_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    false_target_lexical_similarity: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    reason_codes: tuple[str, ...] = Field(default_factory=tuple)
    exclusion_codes: tuple[str, ...] = Field(default_factory=tuple)
    warning_codes: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator(
        "sensitive_field_kinds",
        "reason_codes",
        "exclusion_codes",
        "warning_codes",
    )
    @classmethod
    def code_sets_must_be_sorted_and_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(
            not item
            or not item[0].isalpha()
            or not item.replace("_", "a").isalnum()
            or not item.isascii()
            or item != item.lower()
            for item in value
        ):
            raise ValueError("candidate fit codes must be lowercase ASCII slugs")
        if value != tuple(sorted(set(value))):
            raise ValueError("candidate fit codes must be sorted and unique")
        return value

    @field_validator("score_components")
    @classmethod
    def score_components_must_be_valid(
        cls,
        value: dict[str, int],
    ) -> dict[str, int]:
        for name, score in value.items():
            if (
                not name
                or not name[0].isalpha()
                or not name.replace("_", "a").isalnum()
                or not name.isascii()
                or name != name.lower()
            ):
                raise ValueError("score component names must be lowercase slugs")
            if isinstance(score, bool) or not isinstance(score, int) or score < 0:
                raise ValueError("score components must be non-negative integers")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def status_score_and_target_must_be_consistent(self) -> Self:
        if sum(self.score_components.values()) != self.score:
            raise ValueError("candidate score must equal its component sum")
        if self.status == CandidateFitStatus.INELIGIBLE:
            if not self.exclusion_codes:
                raise ValueError("ineligible candidates require exclusion codes")
        elif self.exclusion_codes:
            raise ValueError("non-ineligible candidates cannot have exclusions")
        if self.status == CandidateFitStatus.REVIEW_REQUIRED and not self.warning_codes:
            raise ValueError("review-required candidates require warning codes")
        target_fields = (
            self.false_target_option_index,
            self.false_target_sha256,
            self.false_target_lexical_similarity,
        )
        if any(item is None for item in target_fields) and any(
            item is not None for item in target_fields
        ):
            raise ValueError("false-target evidence must be complete or absent")
        return self
