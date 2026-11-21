"""Editable, reviewable source material for frozen external attack specs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from membench.curation.external_attack_specs import (
    ExternalAttackMethodology,
    ExternalAttackObjective,
    ExternalAttackSpec,
    ExternalAttackSpecSet,
    build_external_attack_spec_set,
)
from membench.curation.projections import AttackReviewStatus
from membench.ingress.models import ExternalSourceKind
from membench.memory.models import CanonicalModel

_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_CASE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"


class ExternalAttackSpecDraftError(ValueError):
    """Raised when editable external attack-spec source material is invalid."""


class ExternalAttackSpecDraft(CanonicalModel):
    """Human-reviewable source text for one frozen external attack spec."""

    schema_version: Literal[1] = 1
    spec_ordinal: int = Field(ge=1)
    attack_spec_id: str = Field(pattern=_ID_PATTERN)
    allocation_case_id: str = Field(pattern=_CASE_ID_PATTERN)
    attack_id: str = Field(pattern=r"^bb-[a-z0-9]+-[0-9]{3}$")
    attack_methodology: ExternalAttackMethodology
    objective: ExternalAttackObjective
    variant: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    knowledge: Literal["black_box"] = "black_box"
    source_kind: ExternalSourceKind
    insert_after_source_message_index: int = Field(ge=0)
    target_reference: str = Field(min_length=1, max_length=100_000)
    public_attacker_view: str = Field(min_length=1, max_length=100_000)
    review_status: AttackReviewStatus = AttackReviewStatus.GENERATED_UNREVIEWED

    @property
    def target_reference_sha256(self) -> str:
        """Hash the exact evaluator-only target reference after review."""

        return _text_sha256(self.target_reference)

    @property
    def public_attacker_view_sha256(self) -> str:
        """Hash the exact declared attacker-visible context after review."""

        return _text_sha256(self.public_attacker_view)

    def freeze(self) -> ExternalAttackSpec:
        """Convert reviewed source text to the immutable hash-only contract."""

        return ExternalAttackSpec(
            spec_ordinal=self.spec_ordinal,
            attack_spec_id=self.attack_spec_id,
            allocation_case_id=self.allocation_case_id,
            attack_id=self.attack_id,
            attack_methodology=self.attack_methodology,
            objective=self.objective,
            variant=self.variant,
            knowledge=self.knowledge,
            source_kind=self.source_kind,
            insert_after_source_message_index=self.insert_after_source_message_index,
            target_reference_sha256=self.target_reference_sha256,
            public_attacker_view_sha256=self.public_attacker_view_sha256,
            review_status=self.review_status,
        )


class ExternalAttackSpecDraftSet(CanonicalModel):
    """Editable source set used to build one external attack-spec artifact."""

    schema_version: Literal[1] = 1
    attack_set_id: str = Field(pattern=_CASE_ID_PATTERN)
    projection_set_id: str = Field(pattern=_CASE_ID_PATTERN)
    specs: tuple[ExternalAttackSpecDraft, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def drafts_must_have_unique_contiguous_identity(self) -> Self:
        if tuple(spec.spec_ordinal for spec in self.specs) != tuple(
            range(1, len(self.specs) + 1)
        ):
            raise ValueError("external attack draft ordinals must be contiguous")
        for attribute in ("attack_spec_id", "allocation_case_id", "attack_id"):
            values = [getattr(spec, attribute) for spec in self.specs]
            if len(values) != len(set(values)):
                raise ValueError(f"external attack drafts require unique {attribute}")
        return self

    def freeze(self) -> ExternalAttackSpecSet:
        """Build the immutable attack-spec set without persisting it."""

        return build_external_attack_spec_set(
            attack_set_id=self.attack_set_id,
            projection_set_id=self.projection_set_id,
            specs=tuple(spec.freeze() for spec in self.specs),
        )


def load_external_attack_spec_draft_set(
    path: str | Path,
) -> ExternalAttackSpecDraftSet:
    """Load editable attack-spec JSON without freezing or exporting it."""

    source = Path(path)
    try:
        return ExternalAttackSpecDraftSet.model_validate_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise ExternalAttackSpecDraftError(
            f"could not load external attack-spec draft: {source}"
        ) from exc


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
