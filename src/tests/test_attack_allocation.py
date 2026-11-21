import hashlib
import json
from pathlib import Path

import pytest

from membench.attacks.allocation import (
    AttackAllocationConfig,
    AttackAllocationError,
    AttackAllocationPolicy,
    allocate_attacks,
    default_allocation_config,
    export_attack_allocation,
    load_attack_allocation,
)
from membench.attacks.models import (
    AttackMethodology,
    CandidateAttackFit,
    CandidateFitStatus,
)
from membench.campaign.models import TrialTwin
from membench.data_process.candidate_selection import CandidateSelection


def _fit(
    methodology: AttackMethodology,
    *,
    persona: int,
    row: int,
    rank: int,
) -> CandidateAttackFit:
    review_required = methodology in {
        AttackMethodology.NEGATED_IDENTIFIER,
        AttackMethodology.STORED_INSTRUCTION_INJECTION,
    }
    contract_compatible = (
        methodology != AttackMethodology.STORED_INSTRUCTION_INJECTION
    )
    family = (
        "ownership_provenance_confusion"
        if methodology == AttackMethodology.OWNERSHIP_CONFUSION
        else "stored_instruction_injection"
        if methodology == AttackMethodology.STORED_INSTRUCTION_INJECTION
        else "preference_inversion"
    )
    score = 80 - row
    has_false_target = (
        methodology != AttackMethodology.STORED_INSTRUCTION_INJECTION
    )
    warnings = (
        ("contract_extension_required", "human_payload_review_required")
        if methodology == AttackMethodology.STORED_INSTRUCTION_INJECTION
        else ("evidence_link_unverified",)
        if review_required
        else ()
    )
    return CandidateAttackFit(
        candidate_id=f"pmv2-p{persona}-row{row:06d}",
        persona_id=str(persona),
        source_row_index=row,
        attack_methodology=methodology,
        contract_family=family,
        contract_compatible=contract_compatible,
        status=(
            CandidateFitStatus.REVIEW_REQUIRED
            if review_required
            else CandidateFitStatus.ELIGIBLE
        ),
        rank=rank,
        score=score,
        score_components={"synthetic_fit": score},
        topic="travel",
        preference_type="neutral_preferences",
        conversation_scenario="chat_message",
        owner="self",
        sensitive=methodology == AttackMethodology.NEGATED_IDENTIFIER,
        updated=False,
        sensitive_field_kinds=(
            ("email_address",)
            if methodology == AttackMethodology.NEGATED_IDENTIFIER
            else ()
        ),
        evidence_link_kind=(
            "unverified" if review_required else "exact_message"
        ),
        false_target_option_index=0 if has_false_target else None,
        false_target_sha256="a" * 64 if has_false_target else None,
        false_target_lexical_similarity=0.25 if has_false_target else None,
        reason_codes=("synthetic_candidate",),
        warning_codes=warnings,
    )


def _selection() -> CandidateSelection:
    fits: list[CandidateAttackFit] = []
    for methodology in AttackMethodology:
        rank = 0
        for persona in range(1, 11):
            for row in range(3):
                rank += 1
                fits.append(
                    _fit(
                        methodology,
                        persona=persona,
                        row=row,
                        rank=rank,
                    )
                )
    return CandidateSelection(
        dataset_revision="d" * 40,
        inventory_records_sha256="e" * 64,
        top_k=10,
        fits=tuple(fits),
    )


def _summary(plan, methodology: AttackMethodology):
    return next(
        item
        for item in plan.method_summaries
        if item.attack_methodology == methodology
    )


def _stored_only_config() -> AttackAllocationConfig:
    return AttackAllocationConfig(
        allocation_id="stored-planning-v1",
        seed=17,
        policies=tuple(
            AttackAllocationPolicy(
                attack_methodology=methodology,
                enabled=(
                    methodology
                    == AttackMethodology.STORED_INSTRUCTION_INJECTION
                ),
                target_count=(
                    2
                    if methodology
                    == AttackMethodology.STORED_INSTRUCTION_INJECTION
                    else 0
                ),
                allow_review_required=(
                    methodology
                    == AttackMethodology.STORED_INSTRUCTION_INJECTION
                ),
                require_contract_compatible=False,
            )
            for methodology in AttackMethodology
        ),
    )


def test_default_allocation_is_balanced_unique_and_safety_gated() -> None:
    plan = allocate_attacks(_selection(), default_allocation_config(seed=23))

    assert plan.manifest.requested_count == 15
    assert plan.manifest.allocation_count == 13
    assert plan.manifest.execution_ready_count == 13
    assert plan.manifest.blocked_count == 0
    assert len({item.candidate_id for item in plan.allocations}) == 13
    assert max(plan.manifest.persona_counts.values()) <= 2
    assert all(item.required_twins == tuple(TrialTwin) for item in plan.allocations)

    semantic = _summary(plan, AttackMethodology.SEMANTIC_HIJACKING)
    assert semantic.target_count == 2
    assert semantic.allocated_count == 2
    assert semantic.execution_ready_count == 2

    identifier = _summary(plan, AttackMethodology.NEGATED_IDENTIFIER)
    assert identifier.allocated_count == 0
    assert identifier.shortfall_count == 2
    assert "review_required_excluded" in identifier.reason_codes

    stored = _summary(plan, AttackMethodology.STORED_INSTRUCTION_INJECTION)
    assert stored.enabled is False
    assert stored.allocated_count == 0
    assert stored.reason_codes == ("method_disabled",)

    attack_ids = {
        item.proposed_attack_id
        for item in plan.allocations
        if item.proposed_attack_id is not None
    }
    assert len(attack_ids) == 13
    assert all(
        attack_id.startswith(("bb-pi-", "bb-opc-"))
        for attack_id in attack_ids
    )


def test_stored_instruction_can_be_planned_but_remains_blocked() -> None:
    plan = allocate_attacks(_selection(), _stored_only_config())
    summary = _summary(plan, AttackMethodology.STORED_INSTRUCTION_INJECTION)

    assert summary.allocated_count == 2
    assert summary.execution_ready_count == 0
    assert summary.blocked_count == 2
    assert all(item.proposed_attack_id is None for item in plan.allocations)
    assert all(
        item.blocking_codes
        == ("contract_extension_required", "review_approval_required")
        for item in plan.allocations
    )


def test_allocation_export_load_and_integrity_are_deterministic(
    tmp_path: Path,
) -> None:
    selection = _selection()
    config = default_allocation_config(seed=29)
    first = allocate_attacks(selection, config)
    second = allocate_attacks(selection, config)
    output_dir = tmp_path / "allocation"
    allocations_path, config_path, manifest_path = export_attack_allocation(
        first,
        output_dir,
    )
    loaded = load_attack_allocation(output_dir)

    assert first == second
    assert loaded == first
    assert hashlib.sha256(allocations_path.read_bytes()).hexdigest() == (
        first.manifest.allocations_sha256
    )
    assert json.loads(config_path.read_text(encoding="utf-8"))[
        "allocation_id"
    ] == "pmv2-pilot-v1"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))[
        "allocation_count"
    ] == 13
    with pytest.raises(AttackAllocationError, match="without --overwrite"):
        export_attack_allocation(first, output_dir)

    allocations_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(AttackAllocationError, match="integrity check failed"):
        load_attack_allocation(output_dir)


def test_required_quota_and_review_approval_references_are_validated() -> None:
    selection = _selection()
    required = default_allocation_config().model_copy(
        update={"require_full_quota": True}
    )
    with pytest.raises(AttackAllocationError, match="required quotas"):
        allocate_attacks(selection, required)

    invalid_approval = default_allocation_config().model_copy(
        update={
            "approved_review_candidate_ids": (
                "pmv2-p999-row999999",
            )
        }
    )
    with pytest.raises(AttackAllocationError, match="unknown candidates"):
        allocate_attacks(selection, invalid_approval)
