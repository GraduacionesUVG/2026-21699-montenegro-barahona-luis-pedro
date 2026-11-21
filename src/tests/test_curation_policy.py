from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from membench.curation import (
    AttackerViewMode,
    ProjectionCandidateContext,
    ProjectionPolicyConfig,
    ProjectionPolicyError,
    SurvivalDelayMode,
    apply_projection_policy,
    normalize_personamem_message,
    normalize_personamem_messages,
)
from membench.memory.models import MessageRole


def _messages(*items: tuple[str, str]):
    return normalize_personamem_messages(
        [{"role": role, "content": content} for role, content in items]
    )


def _context(messages, **overrides):
    values = {
        "candidate_id": "pmv2-p371-row000586",
        "messages": messages,
        "evidence_start": 1,
        "evidence_end": 2,
        "probe_options": ("correct answer", "false answer", "another answer"),
        "protected_texts": ("correct answer",),
    }
    values.update(overrides)
    return ProjectionCandidateContext(**values)


def test_normalization_accepts_mapping_json_and_python_literal() -> None:
    mapping = normalize_personamem_message(
        {"role": "USER", "content": " Cafe\u0301\r\nvisit "},
        source_index=0,
    )
    json_message = normalize_personamem_message(
        '{"role":"assistant","content":"Done"}',
        source_index=1,
    )
    literal_message = normalize_personamem_message(
        "{'role': 'user', 'content': 'Again'}",
        source_index=2,
    )
    plain = normalize_personamem_message(
        "Plain text",
        source_index=3,
        default_role="assistant",
    )

    assert mapping.role == MessageRole.USER
    assert mapping.content == "Café\nvisit"
    assert mapping.source_sha256 != mapping.normalized_sha256
    assert json_message.role == MessageRole.ASSISTANT
    assert literal_message.content == "Again"
    assert plain.role == MessageRole.ASSISTANT


def test_normalization_rejects_unsafe_shapes_and_flags_replacement_text() -> None:
    with pytest.raises(ProjectionPolicyError, match="role is missing"):
        normalize_personamem_message("plain", source_index=0)
    with pytest.raises(ProjectionPolicyError, match="unsupported message role"):
        normalize_personamem_message(
            {"role": "tool", "content": "value"},
            source_index=0,
        )
    with pytest.raises(ProjectionPolicyError, match="control"):
        normalize_personamem_message(
            {"role": "user", "content": "bad\x00value"},
            source_index=0,
        )

    warned = normalize_personamem_message(
        {"role": "user", "content": "broken \ufffd text"},
        source_index=0,
    )
    assert warned.warning_codes == ("unicode_replacement_character",)


def test_all_remaining_partition_and_option_order_are_deterministic() -> None:
    messages = _messages(
        ("system", "rules"),
        ("user", "I prefer window seats"),
        ("user", "please remember that"),
        ("assistant", "I will remember it"),
        ("user", "unrelated later question"),
        ("assistant", "unrelated later answer"),
    )
    context = _context(messages)

    first = apply_projection_policy(context)
    second = apply_projection_policy(context)

    assert first == second
    assert first.eligible is True
    assert first.insertion_after_message_index == 3
    assert first.pre_attack_range.model_dump() == {"start": 0, "end": 3}
    assert first.survival_range is not None
    assert first.survival_range.model_dump() == {"start": 4, "end": 5}
    assert first.survival_message_count == 2
    assert sorted(first.probe_option_order) == [0, 1, 2]
    assert first.probe_option_sha256[0] == hashlib.sha256(
        b"correct answer"
    ).hexdigest()


def test_full_prefix_attacker_view_excludes_system_messages() -> None:
    messages = _messages(
        ("system", "private system rule"),
        ("user", "preference evidence"),
        ("assistant", "acknowledged"),
        ("user", "delay"),
        ("assistant", "delay reply"),
    )
    config = ProjectionPolicyConfig(
        attacker_view_mode=AttackerViewMode.FULL_PREFIX,
        attacker_view_token_budget=None,
    )
    decision = apply_projection_policy(
        _context(messages, evidence_start=1, evidence_end=1),
        config,
    )

    assert decision.attacker_view_message_indexes == (1, 2)
    assert decision.eligible is True


def test_bounded_attacker_view_keeps_whole_suffix_and_warns_on_omission() -> None:
    messages = _messages(
        ("user", "old evidence token"),
        ("assistant", "old acknowledgement"),
        ("user", "new short"),
        ("assistant", "new reply"),
        ("user", "delay"),
        ("assistant", "done"),
    )
    config = ProjectionPolicyConfig(attacker_view_token_budget=9)
    decision = apply_projection_policy(
        _context(messages, evidence_start=0, evidence_end=0),
        config,
    )

    assert decision.attacker_view_message_indexes
    assert 0 not in decision.attacker_view_message_indexes
    assert "attacker_view_omits_evidence" in decision.warning_codes
    assert decision.attacker_view_token_count <= 9


def test_fixed_survival_extends_to_a_complete_assistant_turn() -> None:
    messages = _messages(
        ("user", "evidence"),
        ("assistant", "ack"),
        ("user", "delay question"),
        ("assistant", "delay answer"),
        ("user", "unused"),
        ("assistant", "unused answer"),
    )
    config = ProjectionPolicyConfig(
        survival_delay_mode=SurvivalDelayMode.FIXED_EVENTS,
        survival_delay_events=1,
        max_turn_completion_events=1,
    )
    decision = apply_projection_policy(
        _context(messages, evidence_start=0, evidence_end=0),
        config,
    )

    assert decision.survival_range is not None
    assert decision.survival_range.model_dump() == {"start": 2, "end": 3}
    assert decision.survival_message_count == 2
    assert decision.eligible is True


def test_all_remaining_trims_only_an_incomplete_trailing_turn() -> None:
    messages = _messages(
        ("user", "evidence"),
        ("assistant", "ack"),
        ("user", "complete delay"),
        ("assistant", "complete delay answer"),
        ("user", "dangling final question"),
    )
    decision = apply_projection_policy(
        _context(messages, evidence_start=0, evidence_end=0)
    )

    assert decision.eligible is True
    assert decision.survival_range is not None
    assert decision.survival_range.model_dump() == {"start": 2, "end": 3}
    assert (
        "trailing_incomplete_survival_messages_omitted"
        in decision.warning_codes
    )


def test_policy_rejects_missing_completion_and_detects_target_overlap() -> None:
    messages = _messages(
        ("user", "evidence"),
        ("assistant", "ack"),
        ("user", "the protected phrase appears later"),
    )
    config = ProjectionPolicyConfig(
        survival_delay_mode=SurvivalDelayMode.FIXED_EVENTS,
        survival_delay_events=1,
        max_turn_completion_events=0,
    )
    decision = apply_projection_policy(
        _context(
            messages,
            evidence_start=0,
            evidence_end=0,
            protected_texts=("protected phrase",),
        ),
        config,
    )

    assert decision.eligible is False
    assert decision.rejection_codes == ("incomplete_survival_turn",)
    assert decision.target_overlap_message_indexes == (2,)
    assert "survival_target_overlap_detected" in decision.warning_codes


def test_malformed_context_is_an_error_not_a_policy_rejection() -> None:
    messages = _messages(("user", "one"), ("assistant", "two"))
    malformed = ProjectionCandidateContext(
        candidate_id="pmv2-p371-row000586",
        messages=(messages[1], messages[0]),
        evidence_start=0,
        evidence_end=0,
        probe_options=("a", "b"),
    )
    with pytest.raises(ProjectionPolicyError, match="consecutive"):
        apply_projection_policy(malformed)


def test_policy_configuration_enforces_mode_specific_fields() -> None:
    with pytest.raises(ValidationError, match="survival_delay_events"):
        ProjectionPolicyConfig(
            survival_delay_mode=SurvivalDelayMode.FIXED_EVENTS,
        )
    with pytest.raises(ValidationError, match="only valid"):
        ProjectionPolicyConfig(
            attacker_view_mode=AttackerViewMode.FULL_PREFIX,
            attacker_view_token_budget=10,
        )
