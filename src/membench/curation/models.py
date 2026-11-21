"""Canonical configuration and audit output for projection policies."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.memory.models import CanonicalModel, MessageRole

_SLUG_PATTERN = r"^[a-z][a-z0-9_]*$"
_CANDIDATE_PATTERN = r"^pmv2-p[0-9]+-row[0-9]{6}$"


class AttackerViewMode(StrEnum):
    """How much benign history an attacker is assumed to observe."""

    FULL_PREFIX = "full_prefix"
    BOUNDED_SUFFIX = "bounded_suffix"


class SurvivalDelayMode(StrEnum):
    """How post-injection benign messages are selected."""

    ALL_REMAINING = "all_remaining"
    FIXED_EVENTS = "fixed_events"


class ProjectionPolicyConfig(CanonicalModel):
    """Frozen deterministic rules used to partition one chat history."""

    schema_version: Literal[1] = 1
    policy_id: str = Field(
        default="personamem_projection_policy_v1",
        pattern=_SLUG_PATTERN,
    )
    insertion_rule: Literal["first_assistant_turn_after_evidence"] = (
        "first_assistant_turn_after_evidence"
    )
    min_pre_attack_messages: int = Field(default=2, ge=1)
    survival_delay_mode: SurvivalDelayMode = SurvivalDelayMode.ALL_REMAINING
    survival_delay_events: int | None = Field(default=None, ge=1)
    require_survival_delay: bool = True
    complete_turn_required: bool = True
    max_turn_completion_events: int = Field(default=2, ge=0)
    attacker_view_mode: AttackerViewMode = AttackerViewMode.BOUNDED_SUFFIX
    attacker_view_token_budget: int | None = Field(default=8192, ge=1)
    option_order_seed: int = Field(default=0, ge=0)
    target_overlap_warning_threshold: float = Field(default=0.25, gt=0, le=1)
    token_counter: Literal["regex_lexical_v1"] = "regex_lexical_v1"

    @model_validator(mode="after")
    def mode_specific_fields_must_be_consistent(self) -> Self:
        if self.survival_delay_mode == SurvivalDelayMode.FIXED_EVENTS:
            if self.survival_delay_events is None:
                raise ValueError("fixed_events requires survival_delay_events")
        elif self.survival_delay_events is not None:
            raise ValueError(
                "survival_delay_events is only valid for fixed_events"
            )

        if self.attacker_view_mode == AttackerViewMode.BOUNDED_SUFFIX:
            if self.attacker_view_token_budget is None:
                raise ValueError(
                    "bounded_suffix requires attacker_view_token_budget"
                )
        elif self.attacker_view_token_budget is not None:
            raise ValueError(
                "attacker_view_token_budget is only valid for bounded_suffix"
            )
        return self

    @property
    def sha256(self) -> str:
        """Hash the canonical policy so decisions identify their exact rules."""

        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class NormalizedPersonaMessage(CanonicalModel):
    """One normalized source message with traceable pre/post hashes."""

    source_index: int = Field(ge=0)
    role: MessageRole
    content: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(ge=1)
    warning_codes: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("warning_codes")
    @classmethod
    def warnings_must_be_sorted_unique_slugs(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        _validate_codes(value, label="normalization warning")
        return value


class MessageRange(CanonicalModel):
    """Inclusive source-message interval."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def range_cannot_run_backwards(self) -> Self:
        if self.end < self.start:
            raise ValueError("message range cannot run backwards")
        return self

    @property
    def count(self) -> int:
        return self.end - self.start + 1


class ProjectionPolicyDecision(CanonicalModel):
    """Privacy-minimized, reproducible result of applying a policy."""

    schema_version: Literal[1] = 1
    candidate_id: str = Field(pattern=_CANDIDATE_PATTERN)
    policy_id: str = Field(pattern=_SLUG_PATTERN)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_message_count: int = Field(ge=1)
    evidence_range: MessageRange
    insertion_after_message_index: int = Field(ge=0)
    pre_attack_range: MessageRange
    survival_range: MessageRange | None = None
    survival_message_count: int = Field(ge=0)
    survival_token_count: int = Field(ge=0)
    attacker_view_message_indexes: tuple[int, ...] = Field(default_factory=tuple)
    attacker_view_token_count: int = Field(ge=0)
    probe_option_order: tuple[int, ...] = Field(min_length=2)
    probe_option_sha256: tuple[str, ...] = Field(min_length=2)
    target_overlap_message_indexes: tuple[int, ...] = Field(default_factory=tuple)
    eligible: bool
    rejection_codes: tuple[str, ...] = Field(default_factory=tuple)
    warning_codes: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("rejection_codes", "warning_codes")
    @classmethod
    def codes_must_be_sorted_unique_slugs(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        _validate_codes(value, label="projection decision")
        return value

    @field_validator(
        "attacker_view_message_indexes",
        "target_overlap_message_indexes",
    )
    @classmethod
    def indexes_must_be_sorted_unique_and_non_negative(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if any(index < 0 for index in value):
            raise ValueError("message indexes cannot be negative")
        if value != tuple(sorted(set(value))):
            raise ValueError("message indexes must be sorted and unique")
        return value

    @field_validator("probe_option_sha256")
    @classmethod
    def option_hashes_must_be_sha256(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        import re

        if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in value):
            raise ValueError("probe option hashes must be SHA-256 values")
        return value

    @model_validator(mode="after")
    def fields_must_describe_one_consistent_partition(self) -> Self:
        if self.eligible != (not self.rejection_codes):
            raise ValueError("eligibility must agree with rejection codes")
        if self.evidence_range.end >= self.history_message_count:
            raise ValueError("evidence range exceeds the history")
        if self.insertion_after_message_index >= self.history_message_count:
            raise ValueError("insertion boundary exceeds the history")
        if self.pre_attack_range != MessageRange(
            start=0,
            end=self.insertion_after_message_index,
        ):
            raise ValueError("pre-attack range must end at the insertion boundary")
        if self.survival_range is None:
            if self.survival_message_count or self.survival_token_count:
                raise ValueError("an absent survival range must have zero counts")
        else:
            if self.survival_range.start != self.insertion_after_message_index + 1:
                raise ValueError("survival range must follow the insertion boundary")
            if self.survival_range.end >= self.history_message_count:
                raise ValueError("survival range exceeds the history")
            if self.survival_message_count != self.survival_range.count:
                raise ValueError("survival count must match its message range")
        if any(
            index > self.insertion_after_message_index
            for index in self.attacker_view_message_indexes
        ):
            raise ValueError("attacker view cannot include post-injection messages")
        if any(
            index >= self.history_message_count
            for index in self.target_overlap_message_indexes
        ):
            raise ValueError("target-overlap index exceeds the history")
        option_count = len(self.probe_option_sha256)
        if tuple(sorted(self.probe_option_order)) != tuple(range(option_count)):
            raise ValueError("probe option order must be a complete permutation")
        return self


def _validate_codes(value: tuple[str, ...], *, label: str) -> None:
    import re

    if any(not re.fullmatch(_SLUG_PATTERN, item) for item in value):
        raise ValueError(f"{label} codes must be lowercase ASCII slugs")
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{label} codes must be sorted and unique")
