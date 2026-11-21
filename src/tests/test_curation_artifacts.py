from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from membench.attacks.models import AttackMethodology
from membench.curation import (
    ProjectionAuditCriteria,
    ProjectionCandidateContext,
    ProjectionDraft,
    ProjectionDraftConfig,
    ProjectionDraftError,
    ProjectionDraftRecord,
    apply_projection_policy,
    export_projection_draft,
    load_projection_draft,
    normalize_personamem_messages,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _draft() -> ProjectionDraft:
    private_correct = "private correct answer"
    private_incorrect = "private incorrect answer"
    messages = normalize_personamem_messages(
        [
            {"role": "user", "content": "evidence message"},
            {"role": "assistant", "content": "evidence acknowledged"},
            {"role": "user", "content": "benign delay"},
            {"role": "assistant", "content": "benign response"},
        ]
    )
    config = ProjectionDraftConfig(projection_id="synthetic-projection-v1")
    decision = apply_projection_policy(
        ProjectionCandidateContext(
            candidate_id="pmv2-p1-row000001",
            messages=messages,
            evidence_start=0,
            evidence_end=0,
            probe_options=(private_correct, private_incorrect),
            protected_texts=(private_correct,),
        ),
        config.policy,
    )
    record = ProjectionDraftRecord(
        projection_ordinal=1,
        allocation_case_id="alloc-abrupt-false-correction-001",
        allocation_ordinal=1,
        proposed_attack_id="bb-pi-001",
        attack_methodology=AttackMethodology.ABRUPT_FALSE_CORRECTION,
        contract_family="preference_inversion",
        candidate_id="pmv2-p1-row000001",
        persona_id="1",
        source_row_index=1,
        source_export_file="persona_1.json",
        source_export_sha256="a" * 64,
        source_record_sha256="b" * 64,
        raw_persona_sha256="c" * 64,
        chat_history_sha256="d" * 64,
        normalized_history_sha256="e" * 64,
        evidence_link_kind="exact_message",
        evidence_similarity=1.0,
        evidence_message_sha256=(messages[0].normalized_sha256,),
        insertion_message_sha256=messages[1].normalized_sha256,
        pre_attack_replay_message_indexes=(0, 1),
        pre_attack_replay_token_count=sum(
            message.token_count for message in messages[:2]
        ),
        survival_replay_message_indexes=(2, 3),
        survival_replay_token_count=sum(
            message.token_count for message in messages[2:]
        ),
        probe_query_sha256="f" * 64,
        correct_answer_sha256=_sha256(private_correct),
        incorrect_answer_sha256=(_sha256(private_incorrect),),
        false_target_option_index=0,
        false_target_sha256=_sha256(private_incorrect),
        policy_decision=decision,
        audit=ProjectionAuditCriteria(
            evidence_resolved=True,
            evidence_precedes_or_meets_insertion=True,
            insertion_ends_on_assistant=True,
            survival_contiguous_after_insertion=True,
            survival_ends_on_assistant=True,
            attacker_view_pre_insertion_only=True,
            attacker_view_system_free=True,
            replay_system_messages_quarantined=True,
        ),
    )
    draft = ProjectionDraft(
        dataset_revision="1" * 40,
        allocation_id="synthetic-allocation-v1",
        allocation_manifest_sha256="2" * 64,
        inventory_manifest_sha256="3" * 64,
        config=config,
        records=(record,),
    )
    assert private_correct.encode() not in draft.decisions_jsonl()
    assert private_incorrect.encode() not in draft.decisions_jsonl()
    return draft


def test_projection_draft_export_load_and_integrity(tmp_path: Path) -> None:
    draft = _draft()
    output = tmp_path / "projection-draft"
    decisions, config, manifest = export_projection_draft(draft, output)

    assert decisions.exists()
    assert config.exists()
    assert manifest.exists()
    assert load_projection_draft(output) == draft
    assert draft.manifest.record_count == 1
    assert draft.manifest.eligible_count == 1

    decisions.write_bytes(decisions.read_bytes() + b"\n")
    with pytest.raises(ProjectionDraftError, match="integrity"):
        load_projection_draft(output)


def test_projection_export_refuses_implicit_overwrite(tmp_path: Path) -> None:
    draft = _draft()
    output = tmp_path / "projection-draft"
    export_projection_draft(draft, output)

    with pytest.raises(ProjectionDraftError, match="--overwrite"):
        export_projection_draft(draft, output)
