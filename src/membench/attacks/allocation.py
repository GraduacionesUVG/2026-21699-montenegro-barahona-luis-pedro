"""Deterministically allocate ranked PersonaMem candidates to attack cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.attacks.models import (
    AttackMethodology,
    CandidateAttackFit,
    CandidateFitStatus,
)
from membench.campaign.models import TrialTwin
from membench.data_process.candidate_selection import (
    DEFAULT_SELECTION_DIR,
    CandidateSelection,
    load_candidate_selection,
)
from membench.data_process.data_processing import DATASET_ID
from membench.memory.models import CanonicalModel

ALLOCATION_VERSION = "personamem_attack_allocation_v1"
DEFAULT_ALLOCATION_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "blackbox" / "allocation"
)
_CANDIDATE_PATTERN = r"^pmv2-p[0-9]+-row[0-9]{6}$"
_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
_METHOD_INDEX = {
    methodology: index for index, methodology in enumerate(AttackMethodology)
}
_STATUS_PRIORITY = {
    CandidateFitStatus.ELIGIBLE: 0,
    CandidateFitStatus.REVIEW_REQUIRED: 1,
    CandidateFitStatus.INELIGIBLE: 2,
}
_REQUIRED_TWINS = tuple(TrialTwin)


class AttackAllocationError(RuntimeError):
    """Raised when a trustworthy allocation cannot be produced or loaded."""


class AttackAllocationPolicy(CanonicalModel):
    """Quota and safety gates for one attack methodology."""

    attack_methodology: AttackMethodology
    enabled: bool = True
    target_count: int = Field(ge=0)
    allow_review_required: bool = False
    require_contract_compatible: bool = True

    @model_validator(mode="after")
    def disabled_policies_cannot_request_cases(self) -> Self:
        if not self.enabled and self.target_count:
            raise ValueError("disabled allocation policies must have target_count=0")
        return self


class AttackAllocationConfig(CanonicalModel):
    """Frozen policy controlling deterministic cross-method allocation."""

    schema_version: Literal[1] = 1
    allocation_id: str = Field(pattern=_ID_PATTERN)
    seed: int = Field(default=0, ge=0)
    policies: tuple[AttackAllocationPolicy, ...]
    max_per_persona_per_method: int = Field(default=1, ge=1)
    max_total_per_persona: int | None = Field(default=2, ge=1)
    allow_candidate_reuse_across_methodologies: bool = False
    approved_review_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    require_full_quota: bool = False

    @field_validator("approved_review_candidate_ids")
    @classmethod
    def approved_candidates_are_valid_and_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        import re

        if any(not re.fullmatch(_CANDIDATE_PATTERN, item) for item in value):
            raise ValueError("approved review candidates must be candidate IDs")
        if value != tuple(sorted(set(value))):
            raise ValueError(
                "approved review candidate IDs must be sorted and unique"
            )
        return value

    @model_validator(mode="after")
    def policies_must_cover_every_method_once(self) -> Self:
        observed = [policy.attack_methodology for policy in self.policies]
        if len(observed) != len(set(observed)):
            raise ValueError("allocation policies contain duplicate methodologies")
        if set(observed) != set(AttackMethodology):
            missing = sorted(
                method.value for method in set(AttackMethodology) - set(observed)
            )
            raise ValueError(
                "allocation policies must cover every methodology; missing="
                f"{missing}"
            )
        return self

    @property
    def sha256(self) -> str:
        """Hash the canonical policy for provenance and artifact sealing."""

        return _canonical_sha256(self.model_dump(mode="json"))


class AttackAllocationCase(CanonicalModel):
    """One selected history unit awaiting immutable attack drafting."""

    schema_version: Literal[1] = 1
    allocation_case_id: str = Field(pattern=_ID_PATTERN)
    allocation_ordinal: int = Field(ge=1)
    attack_methodology: AttackMethodology
    contract_family: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    proposed_attack_id: str | None = Field(
        default=None,
        pattern=r"^bb-(pi|opc)-[0-9]{3}$",
    )
    candidate_id: str = Field(pattern=_CANDIDATE_PATTERN)
    persona_id: str = Field(pattern=r"^[0-9]+$")
    source_row_index: int = Field(ge=0)
    candidate_fit_rank: int = Field(ge=1)
    candidate_fit_score: int = Field(ge=0, le=100)
    selection_status: CandidateFitStatus
    contract_compatible: bool
    review_approved: bool
    execution_ready: bool
    blocking_codes: tuple[str, ...] = Field(default_factory=tuple)
    warning_codes: tuple[str, ...] = Field(default_factory=tuple)
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
    required_twins: tuple[TrialTwin, ...] = _REQUIRED_TWINS

    @field_validator("blocking_codes", "warning_codes")
    @classmethod
    def codes_are_sorted_unique_slugs(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not _is_slug(item) for item in value):
            raise ValueError("allocation codes must be lowercase ASCII slugs")
        if value != tuple(sorted(set(value))):
            raise ValueError("allocation codes must be sorted and unique")
        return value

    @model_validator(mode="after")
    def readiness_and_twins_must_be_consistent(self) -> Self:
        if self.selection_status == CandidateFitStatus.INELIGIBLE:
            raise ValueError("ineligible candidates cannot be allocated")
        if self.selection_status == CandidateFitStatus.ELIGIBLE:
            if not self.review_approved:
                raise ValueError("eligible allocations are implicitly approved")
        elif self.review_approved and "review_approval_required" in self.blocking_codes:
            raise ValueError("approved reviews cannot remain blocked by review")
        expected_ready = self.contract_compatible and (
            self.selection_status == CandidateFitStatus.ELIGIBLE
            or self.review_approved
        )
        if self.execution_ready != expected_ready:
            raise ValueError("allocation execution readiness is inconsistent")
        if self.execution_ready and self.blocking_codes:
            raise ValueError("execution-ready allocations cannot have blockers")
        if not self.execution_ready and not self.blocking_codes:
            raise ValueError("blocked allocations require blocking codes")
        if self.required_twins != _REQUIRED_TWINS:
            raise ValueError("allocations require clean, benign, and poisoned twins")
        if self.contract_compatible != (self.proposed_attack_id is not None):
            raise ValueError(
                "contract-compatible allocations require a proposed attack ID"
            )
        target_fields = (
            self.false_target_option_index,
            self.false_target_sha256,
            self.false_target_lexical_similarity,
        )
        if any(item is None for item in target_fields) and any(
            item is not None for item in target_fields
        ):
            raise ValueError("false-target metadata must be complete or absent")
        return self


class AttackMethodAllocationSummary(CanonicalModel):
    """Quota outcome and gate evidence for one attack methodology."""

    attack_methodology: AttackMethodology
    enabled: bool
    target_count: int = Field(ge=0)
    allocated_count: int = Field(ge=0)
    execution_ready_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    shortfall_count: int = Field(ge=0)
    eligible_pool_count: int = Field(ge=0)
    review_required_pool_count: int = Field(ge=0)
    contract_compatible_pool_count: int = Field(ge=0)
    reason_codes: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("reason_codes")
    @classmethod
    def reasons_are_sorted_unique_slugs(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not _is_slug(item) for item in value):
            raise ValueError("allocation summary reasons must be slugs")
        if value != tuple(sorted(set(value))):
            raise ValueError("allocation summary reasons must be sorted and unique")
        return value

    @model_validator(mode="after")
    def counts_must_balance(self) -> Self:
        if self.allocated_count + self.shortfall_count != self.target_count:
            raise ValueError("allocation target must equal allocated plus shortfall")
        if self.execution_ready_count + self.blocked_count != self.allocated_count:
            raise ValueError("allocated count must equal ready plus blocked")
        if self.allocated_count > self.target_count:
            raise ValueError("allocated count cannot exceed target")
        if self.shortfall_count and "quota_shortfall" not in self.reason_codes:
            raise ValueError("allocation shortfalls require a reason code")
        return self


class AttackAllocationManifest(CanonicalModel):
    """Integrity metadata and aggregate counts for an allocation artifact."""

    schema_version: Literal[1] = 1
    dataset_id: Literal["bowen-upenn/PersonaMem-v2"] = DATASET_ID
    dataset_revision: str
    allocation_version: Literal[
        "personamem_attack_allocation_v1"
    ] = ALLOCATION_VERSION
    allocation_id: str = Field(pattern=_ID_PATTERN)
    selection_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_fits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allocation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allocations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_count: int = Field(ge=0)
    allocation_count: int = Field(ge=0)
    execution_ready_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    method_summaries: tuple[AttackMethodAllocationSummary, ...]
    persona_counts: dict[str, int]
    contract_family_counts: dict[str, int]
    required_twins: tuple[TrialTwin, ...] = _REQUIRED_TWINS

    @model_validator(mode="after")
    def aggregate_counts_must_balance(self) -> Self:
        methods = [item.attack_methodology for item in self.method_summaries]
        if len(methods) != len(set(methods)) or set(methods) != set(
            AttackMethodology
        ):
            raise ValueError("allocation manifest requires one summary per method")
        if sum(item.target_count for item in self.method_summaries) != (
            self.requested_count
        ):
            raise ValueError("requested count does not match method summaries")
        if sum(item.allocated_count for item in self.method_summaries) != (
            self.allocation_count
        ):
            raise ValueError("allocation count does not match method summaries")
        if self.execution_ready_count + self.blocked_count != self.allocation_count:
            raise ValueError("manifest allocation readiness counts do not balance")
        if sum(self.persona_counts.values()) != self.allocation_count:
            raise ValueError("persona allocation counts do not balance")
        if sum(self.contract_family_counts.values()) != self.allocation_count:
            raise ValueError("contract-family allocation counts do not balance")
        if self.required_twins != _REQUIRED_TWINS:
            raise ValueError("manifest requires all campaign trial twins")
        return self

    @property
    def sha256(self) -> str:
        """Hash the canonical manifest for downstream campaign provenance."""

        return _canonical_sha256(self.model_dump(mode="json"))


@dataclass(frozen=True)
class AttackAllocationPlan:
    """Selected attack cases plus their frozen policy and selection identity."""

    dataset_revision: str
    selection_manifest_sha256: str
    selection_fits_sha256: str
    config: AttackAllocationConfig
    allocations: tuple[AttackAllocationCase, ...]
    method_summaries: tuple[AttackMethodAllocationSummary, ...]

    def allocations_jsonl(self) -> bytes:
        """Render deterministic evaluator-only allocation JSONL."""

        return _jsonl(self.allocations)

    def config_json(self) -> bytes:
        """Render the canonical allocation policy used for this plan."""

        return _pretty_json(self.config.model_dump(mode="json"))

    @property
    def manifest(self) -> AttackAllocationManifest:
        """Build deterministic integrity and coverage metadata."""

        persona_counts = Counter(item.persona_id for item in self.allocations)
        family_counts = Counter(item.contract_family for item in self.allocations)
        ready_count = sum(item.execution_ready for item in self.allocations)
        return AttackAllocationManifest(
            dataset_revision=self.dataset_revision,
            allocation_id=self.config.allocation_id,
            selection_manifest_sha256=self.selection_manifest_sha256,
            selection_fits_sha256=self.selection_fits_sha256,
            allocation_config_sha256=self.config.sha256,
            allocations_sha256=hashlib.sha256(
                self.allocations_jsonl()
            ).hexdigest(),
            requested_count=sum(
                summary.target_count for summary in self.method_summaries
            ),
            allocation_count=len(self.allocations),
            execution_ready_count=ready_count,
            blocked_count=len(self.allocations) - ready_count,
            method_summaries=self.method_summaries,
            persona_counts=dict(sorted(persona_counts.items(), key=_persona_key)),
            contract_family_counts=dict(sorted(family_counts.items())),
        )


def default_allocation_config(
    *,
    allocation_id: str = "pmv2-pilot-v1",
    seed: int = 0,
) -> AttackAllocationConfig:
    """Return the safety-gated 15-case policy from the black-box contract."""

    targets = {
        AttackMethodology.ABRUPT_FALSE_CORRECTION: 4,
        AttackMethodology.NEGATED_IDENTIFIER: 2,
        AttackMethodology.OWNERSHIP_CONFUSION: 5,
        AttackMethodology.GRADUAL_DRIFT: 2,
        AttackMethodology.SEMANTIC_HIJACKING: 2,
        AttackMethodology.STORED_INSTRUCTION_INJECTION: 0,
    }
    return AttackAllocationConfig(
        allocation_id=allocation_id,
        seed=seed,
        policies=tuple(
            AttackAllocationPolicy(
                attack_methodology=methodology,
                enabled=methodology
                != AttackMethodology.STORED_INSTRUCTION_INJECTION,
                target_count=targets[methodology],
            )
            for methodology in AttackMethodology
        ),
    )


def allocate_attacks(
    selection: CandidateSelection,
    config: AttackAllocationConfig,
) -> AttackAllocationPlan:
    """Allocate balanced cases while preserving review and contract gates."""

    _validate_selection(selection)
    candidate_ids = {fit.candidate_id for fit in selection.fits}
    unknown_approvals = sorted(
        set(config.approved_review_candidate_ids) - candidate_ids
    )
    if unknown_approvals:
        raise AttackAllocationError(
            "review approvals reference unknown candidates: "
            f"{unknown_approvals}"
        )
    policies = {policy.attack_methodology: policy for policy in config.policies}
    fits_by_method = {
        methodology: tuple(
            fit
            for fit in selection.fits
            if fit.attack_methodology == methodology
        )
        for methodology in AttackMethodology
    }
    process_order = sorted(
        (
            policy
            for policy in config.policies
            if policy.enabled and policy.target_count
        ),
        key=lambda policy: (
            _scarcity_ratio(fits_by_method[policy.attack_methodology], policy),
            _METHOD_INDEX[policy.attack_methodology],
        ),
    )

    used_candidates: set[str] = set()
    persona_totals: Counter[str] = Counter()
    selected_by_method: dict[
        AttackMethodology,
        list[CandidateAttackFit],
    ] = {methodology: [] for methodology in AttackMethodology}
    for policy in process_order:
        method = policy.attack_methodology
        method_personas: Counter[str] = Counter()
        while len(selected_by_method[method]) < policy.target_count:
            available = [
                fit
                for fit in fits_by_method[method]
                if _passes_policy_gate(fit, policy)
                and method_personas[fit.persona_id]
                < config.max_per_persona_per_method
                and (
                    config.max_total_per_persona is None
                    or persona_totals[fit.persona_id]
                    < config.max_total_per_persona
                )
                and (
                    config.allow_candidate_reuse_across_methodologies
                    or fit.candidate_id not in used_candidates
                )
            ]
            if not available:
                break
            chosen = min(
                available,
                key=lambda fit: _allocation_rank_key(
                    fit,
                    persona_totals=persona_totals,
                    method_personas=method_personas,
                    seed=config.seed,
                ),
            )
            selected_by_method[method].append(chosen)
            used_candidates.add(chosen.candidate_id)
            persona_totals[chosen.persona_id] += 1
            method_personas[chosen.persona_id] += 1

    allocations = _build_allocation_cases(selected_by_method, config)
    summaries = tuple(
        _method_summary(
            policy=policies[methodology],
            fits=fits_by_method[methodology],
            allocations=tuple(
                item
                for item in allocations
                if item.attack_methodology == methodology
            ),
            config=config,
        )
        for methodology in AttackMethodology
    )
    shortfalls = {
        summary.attack_methodology.value: summary.shortfall_count
        for summary in summaries
        if summary.shortfall_count
    }
    if config.require_full_quota and shortfalls:
        raise AttackAllocationError(
            f"allocation did not satisfy required quotas: {shortfalls}"
        )
    selection_manifest = selection.manifest
    return AttackAllocationPlan(
        dataset_revision=selection.dataset_revision,
        selection_manifest_sha256=selection_manifest.sha256,
        selection_fits_sha256=selection_manifest.fits_sha256,
        config=config,
        allocations=allocations,
        method_summaries=summaries,
    )


def export_attack_allocation(
    plan: AttackAllocationPlan,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Atomically write allocation cases, configuration, and manifest."""

    allocations_path = output_dir / "allocations.jsonl"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.json"
    paths = (allocations_path, config_path, manifest_path)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise AttackAllocationError(
            "refusing to replace allocation files without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(allocations_path, plan.allocations_jsonl())
    _atomic_write(config_path, plan.config_json())
    _atomic_write(
        manifest_path,
        _pretty_json(plan.manifest.model_dump(mode="json")),
    )
    return allocations_path, config_path, manifest_path


def load_attack_allocation(input_dir: Path) -> AttackAllocationPlan:
    """Load and integrity-check an exported attack allocation plan."""

    allocations_path = input_dir / "allocations.jsonl"
    config_path = input_dir / "config.json"
    manifest_path = input_dir / "manifest.json"
    try:
        allocation_bytes = allocations_path.read_bytes()
        config = AttackAllocationConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
        manifest = AttackAllocationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise AttackAllocationError(
            f"could not load attack allocation from {input_dir}: {exc}"
        ) from exc
    observed_hash = hashlib.sha256(allocation_bytes).hexdigest()
    if observed_hash != manifest.allocations_sha256:
        raise AttackAllocationError(
            "attack allocation integrity check failed: "
            f"{observed_hash}; expected {manifest.allocations_sha256}"
        )
    if config.sha256 != manifest.allocation_config_sha256:
        raise AttackAllocationError(
            "attack allocation config does not match the manifest hash"
        )

    allocations = _parse_allocation_jsonl(allocation_bytes)
    plan = AttackAllocationPlan(
        dataset_revision=manifest.dataset_revision,
        selection_manifest_sha256=manifest.selection_manifest_sha256,
        selection_fits_sha256=manifest.selection_fits_sha256,
        config=config,
        allocations=allocations,
        method_summaries=manifest.method_summaries,
    )
    if plan.manifest != manifest:
        raise AttackAllocationError(
            "attack allocation manifest does not match its cases and config"
        )
    return plan


def _validate_selection(selection: CandidateSelection) -> None:
    expected_methods = set(AttackMethodology)
    observed_methods = {fit.attack_methodology for fit in selection.fits}
    if observed_methods != expected_methods:
        raise AttackAllocationError("candidate selection does not cover all methods")
    identities: dict[str, tuple[str, int]] = {}
    expected_candidates: set[str] | None = None
    for methodology in AttackMethodology:
        fits = [
            fit
            for fit in selection.fits
            if fit.attack_methodology == methodology
        ]
        candidate_ids = [fit.candidate_id for fit in fits]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise AttackAllocationError(
                f"candidate selection duplicates {methodology.value} candidates"
            )
        if [fit.rank for fit in fits] != list(range(1, len(fits) + 1)):
            raise AttackAllocationError(
                f"candidate selection ranks are invalid for {methodology.value}"
            )
        current_candidates = set(candidate_ids)
        if expected_candidates is None:
            expected_candidates = current_candidates
        elif current_candidates != expected_candidates:
            raise AttackAllocationError(
                "candidate selection methods cover different candidate sets"
            )
        for fit in fits:
            identity = (fit.persona_id, fit.source_row_index)
            previous = identities.setdefault(fit.candidate_id, identity)
            if previous != identity:
                raise AttackAllocationError(
                    f"candidate identity changes across fits: {fit.candidate_id}"
                )


def _passes_policy_gate(
    fit: CandidateAttackFit,
    policy: AttackAllocationPolicy,
) -> bool:
    if fit.status == CandidateFitStatus.INELIGIBLE:
        return False
    if (
        fit.status == CandidateFitStatus.REVIEW_REQUIRED
        and not policy.allow_review_required
    ):
        return False
    return not (policy.require_contract_compatible and not fit.contract_compatible)


def _scarcity_ratio(
    fits: tuple[CandidateAttackFit, ...],
    policy: AttackAllocationPolicy,
) -> float:
    pool_size = sum(_passes_policy_gate(fit, policy) for fit in fits)
    return pool_size / policy.target_count


def _allocation_rank_key(
    fit: CandidateAttackFit,
    *,
    persona_totals: Counter[str],
    method_personas: Counter[str],
    seed: int,
) -> tuple[int, int, int, int, str, int]:
    tie_breaker = hashlib.sha256(
        f"{seed}:{fit.attack_methodology.value}:{fit.candidate_id}".encode()
    ).hexdigest()
    return (
        _STATUS_PRIORITY[fit.status],
        persona_totals[fit.persona_id],
        method_personas[fit.persona_id],
        -fit.score,
        tie_breaker,
        fit.rank or 0,
    )


def _build_allocation_cases(
    selected: dict[AttackMethodology, list[CandidateAttackFit]],
    config: AttackAllocationConfig,
) -> tuple[AttackAllocationCase, ...]:
    family_ordinals: Counter[str] = Counter()
    cases: list[AttackAllocationCase] = []
    approved = set(config.approved_review_candidate_ids)
    for methodology in AttackMethodology:
        for ordinal, fit in enumerate(selected[methodology], start=1):
            family_ordinals[fit.contract_family] += 1
            proposed_attack_id = _proposed_attack_id(
                fit,
                family_ordinals[fit.contract_family],
            )
            review_approved = (
                fit.status == CandidateFitStatus.ELIGIBLE
                or fit.candidate_id in approved
            )
            blockers: set[str] = set()
            if not fit.contract_compatible:
                blockers.add("contract_extension_required")
            if (
                fit.status == CandidateFitStatus.REVIEW_REQUIRED
                and not review_approved
            ):
                blockers.add("review_approval_required")
            cases.append(
                AttackAllocationCase(
                    allocation_case_id=(
                        f"alloc-{methodology.value.replace('_', '-')}-{ordinal:03d}"
                    ),
                    allocation_ordinal=ordinal,
                    attack_methodology=methodology,
                    contract_family=fit.contract_family,
                    proposed_attack_id=proposed_attack_id,
                    candidate_id=fit.candidate_id,
                    persona_id=fit.persona_id,
                    source_row_index=fit.source_row_index,
                    candidate_fit_rank=fit.rank or 1,
                    candidate_fit_score=fit.score,
                    selection_status=fit.status,
                    contract_compatible=fit.contract_compatible,
                    review_approved=review_approved,
                    execution_ready=not blockers,
                    blocking_codes=tuple(sorted(blockers)),
                    warning_codes=fit.warning_codes,
                    false_target_option_index=fit.false_target_option_index,
                    false_target_sha256=fit.false_target_sha256,
                    false_target_lexical_similarity=(
                        fit.false_target_lexical_similarity
                    ),
                )
            )
    return tuple(cases)


def _proposed_attack_id(
    fit: CandidateAttackFit,
    family_ordinal: int,
) -> str | None:
    if not fit.contract_compatible:
        return None
    prefixes = {
        "preference_inversion": "pi",
        "ownership_provenance_confusion": "opc",
    }
    prefix = prefixes.get(fit.contract_family)
    if prefix is None:
        raise AttackAllocationError(
            f"no attack ID prefix for contract family {fit.contract_family}"
        )
    return f"bb-{prefix}-{family_ordinal:03d}"


def _method_summary(
    *,
    policy: AttackAllocationPolicy,
    fits: tuple[CandidateAttackFit, ...],
    allocations: tuple[AttackAllocationCase, ...],
    config: AttackAllocationConfig,
) -> AttackMethodAllocationSummary:
    eligible_count = sum(fit.status == CandidateFitStatus.ELIGIBLE for fit in fits)
    review_count = sum(
        fit.status == CandidateFitStatus.REVIEW_REQUIRED for fit in fits
    )
    compatible_count = sum(
        fit.status != CandidateFitStatus.INELIGIBLE and fit.contract_compatible
        for fit in fits
    )
    shortfall = policy.target_count - len(allocations)
    reasons: set[str] = set()
    if not policy.enabled:
        reasons.add("method_disabled")
    elif policy.target_count == 0:
        reasons.add("zero_target")
    if shortfall:
        reasons.add("quota_shortfall")
        if review_count and not policy.allow_review_required:
            reasons.add("review_required_excluded")
        if (
            policy.require_contract_compatible
            and any(
                fit.status != CandidateFitStatus.INELIGIBLE
                and not fit.contract_compatible
                for fit in fits
            )
        ):
            reasons.add("contract_incompatible_excluded")
        gated_candidates = tuple(
            fit for fit in fits if _passes_policy_gate(fit, policy)
        )
        if not gated_candidates:
            reasons.add("no_candidates_after_policy_gates")
        else:
            if not config.allow_candidate_reuse_across_methodologies:
                reasons.add("candidate_reuse_constraint_applied")
            if config.max_total_per_persona is not None:
                reasons.add("persona_balance_constraint_applied")
    ready_count = sum(item.execution_ready for item in allocations)
    return AttackMethodAllocationSummary(
        attack_methodology=policy.attack_methodology,
        enabled=policy.enabled,
        target_count=policy.target_count,
        allocated_count=len(allocations),
        execution_ready_count=ready_count,
        blocked_count=len(allocations) - ready_count,
        shortfall_count=shortfall,
        eligible_pool_count=eligible_count,
        review_required_pool_count=review_count,
        contract_compatible_pool_count=compatible_count,
        reason_codes=tuple(sorted(reasons)),
    )


def _parse_allocation_jsonl(payload: bytes) -> tuple[AttackAllocationCase, ...]:
    cases: list[AttackAllocationCase] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise AttackAllocationError("allocation JSONL is not valid UTF-8") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            cases.append(AttackAllocationCase.model_validate_json(line))
        except ValueError as exc:
            raise AttackAllocationError(
                f"invalid allocations.jsonl line {line_number}: {exc}"
            ) from exc
    return tuple(cases)


def _jsonl(values: Sequence[CanonicalModel]) -> bytes:
    text = "".join(
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for value in values
    )
    return text.encode("utf-8")


def _pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_slug(value: str) -> bool:
    return bool(
        value
        and value.isascii()
        and value == value.lower()
        and value[0].isalpha()
        and value.replace("_", "a").isalnum()
    )


def _persona_key(item: tuple[str, int]) -> int:
    return int(item[0])


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError as exc:
        raise AttackAllocationError(f"could not write {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_config(path: Path | None, *, allocation_id: str, seed: int):
    if path is None:
        return default_allocation_config(allocation_id=allocation_id, seed=seed)
    try:
        return AttackAllocationConfig.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise AttackAllocationError(
            f"could not load allocation config from {path}: {exc}"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Allocate ranked PersonaMem candidates to frozen attack cases."
    )
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ALLOCATION_DIR)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--allocation-id", default="pmv2-pilot-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selection = load_candidate_selection(args.selection_dir)
    config = _load_config(
        args.config,
        allocation_id=args.allocation_id,
        seed=args.seed,
    )
    plan = allocate_attacks(selection, config)
    allocations_path, config_path, manifest_path = export_attack_allocation(
        plan,
        args.output_dir,
        overwrite=args.overwrite,
    )
    manifest = plan.manifest
    print(
        f"Allocated {manifest.allocation_count}/{manifest.requested_count} cases "
        f"({manifest.execution_ready_count} execution-ready)."
    )
    print(f"Allocations: {allocations_path}")
    print(f"Config: {config_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
