"""Pure deterministic policies for partitioning PersonaMem histories.

This module deliberately does not render attack text or write artifacts. It
turns already-approved evaluator inputs into a privacy-minimized partition
decision that a later projection builder can consume.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from membench.context_budget import RegexTokenCounter
from membench.curation.models import (
    AttackerViewMode,
    MessageRange,
    NormalizedPersonaMessage,
    ProjectionPolicyConfig,
    ProjectionPolicyDecision,
    SurvivalDelayMode,
)
from membench.memory.models import MessageRole

_TOKEN_COUNTER = RegexTokenCounter()
_SEARCH_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_CANDIDATE_PATTERN = re.compile(r"^pmv2-p[0-9]+-row[0-9]{6}$")


class ProjectionPolicyError(ValueError):
    """Raised when inputs are malformed rather than merely policy-ineligible."""


@dataclass(frozen=True)
class ProjectionCandidateContext:
    """Private inputs needed to derive one public policy decision.

    ``probe_options`` and ``protected_texts`` are used only to derive hashes,
    a deterministic option order, and overlap warnings. Their text is never
    copied into :class:`ProjectionPolicyDecision`.
    """

    candidate_id: str
    messages: tuple[NormalizedPersonaMessage, ...]
    evidence_start: int
    evidence_end: int
    probe_options: tuple[str, ...]
    protected_texts: tuple[str, ...] = ()


def default_projection_policy() -> ProjectionPolicyConfig:
    """Return the recommended full-history replay and bounded-view policy."""

    return ProjectionPolicyConfig()


def normalize_personamem_message(
    value: Mapping[str, Any] | str,
    *,
    source_index: int,
    default_role: MessageRole | str | None = None,
) -> NormalizedPersonaMessage:
    """Normalize one dict, JSON string, or Python-literal PersonaMem message.

    The fallback to ``ast.literal_eval`` supports dataset fields serialized
    with single quotes without executing arbitrary code. Mojibake is not
    guessed or repaired; a replacement character is surfaced for review.
    """

    if source_index < 0:
        raise ProjectionPolicyError("source_index cannot be negative")

    source_bytes: bytes
    parsed: Any = value
    if isinstance(value, str):
        source_bytes = value.encode("utf-8")
        stripped = value.strip()
        if not stripped:
            raise ProjectionPolicyError("message source cannot be blank")
        parsed = _parse_serialized_message(stripped)
    elif isinstance(value, Mapping):
        try:
            source_bytes = _canonical_json_bytes(dict(value))
        except (TypeError, ValueError) as exc:
            raise ProjectionPolicyError(
                "message mapping must contain JSON-serializable values"
            ) from exc
    else:
        raise ProjectionPolicyError("message must be a mapping or string")

    if isinstance(parsed, Mapping):
        role_value = parsed.get("role", default_role)
        content_value = parsed.get("content")
    elif isinstance(parsed, str):
        role_value = default_role
        content_value = parsed
    else:
        raise ProjectionPolicyError(
            "serialized message must decode to a mapping or plain string"
        )

    if role_value is None:
        raise ProjectionPolicyError("message role is missing")
    try:
        role = MessageRole(str(role_value).strip().lower())
    except ValueError as exc:
        raise ProjectionPolicyError(
            f"unsupported message role: {role_value!r}"
        ) from exc
    if not isinstance(content_value, str):
        raise ProjectionPolicyError("message content must be a string")

    content = _normalize_text(content_value)
    if not content:
        raise ProjectionPolicyError("message content cannot be blank")
    if any(ord(character) < 32 and character not in "\n\t" for character in content):
        raise ProjectionPolicyError("message content contains C0 control characters")

    warnings: set[str] = set()
    if "\ufffd" in content:
        warnings.add("unicode_replacement_character")

    normalized_payload = {"content": content, "role": role.value}
    return NormalizedPersonaMessage(
        source_index=source_index,
        role=role,
        content=content,
        source_sha256=_sha256(source_bytes),
        normalized_sha256=_sha256(_canonical_json_bytes(normalized_payload)),
        token_count=_TOKEN_COUNTER.count(content),
        warning_codes=tuple(sorted(warnings)),
    )


def normalize_personamem_messages(
    values: Sequence[Mapping[str, Any] | str],
    *,
    default_role: MessageRole | str | None = None,
) -> tuple[NormalizedPersonaMessage, ...]:
    """Normalize a complete history while assigning stable source indexes."""

    if not values:
        raise ProjectionPolicyError("chat history cannot be empty")
    return tuple(
        normalize_personamem_message(
            value,
            source_index=index,
            default_role=default_role,
        )
        for index, value in enumerate(values)
    )


def apply_projection_policy(
    context: ProjectionCandidateContext,
    config: ProjectionPolicyConfig | None = None,
) -> ProjectionPolicyDecision:
    """Apply immutable partition, attacker-view, and option-order rules."""

    policy = config or default_projection_policy()
    _validate_context(context)

    messages = context.messages
    rejections: set[str] = set()
    warnings = {
        warning for message in messages for warning in message.warning_codes
    }

    insertion_index = _find_insertion_boundary(
        messages,
        evidence_end=context.evidence_end,
    )
    if insertion_index is None:
        insertion_index = context.evidence_end
        rejections.add("no_complete_turn_after_evidence")

    pre_attack_range = MessageRange(start=0, end=insertion_index)
    if pre_attack_range.count < policy.min_pre_attack_messages:
        rejections.add("insufficient_pre_attack_messages")

    survival_range = _select_survival_range(
        messages,
        insertion_index=insertion_index,
        config=policy,
        rejections=rejections,
        warnings=warnings,
    )
    survival_messages = (
        messages[survival_range.start : survival_range.end + 1]
        if survival_range is not None
        else ()
    )
    if policy.require_survival_delay and not survival_messages:
        rejections.add("survival_delay_required")

    attacker_indexes, attacker_token_count = _select_attacker_view(
        messages,
        insertion_index=insertion_index,
        config=policy,
        rejections=rejections,
    )
    evidence_indexes = set(range(context.evidence_start, context.evidence_end + 1))
    if not evidence_indexes.intersection(attacker_indexes):
        warnings.add("attacker_view_omits_evidence")

    overlap_indexes = _find_target_overlap(
        survival_messages,
        protected_texts=context.protected_texts,
        threshold=policy.target_overlap_warning_threshold,
    )
    if overlap_indexes:
        warnings.add("survival_target_overlap_detected")

    option_hashes = tuple(
        _sha256(_normalize_text(option).encode("utf-8"))
        for option in context.probe_options
    )
    option_order = tuple(
        sorted(
            range(len(option_hashes)),
            key=lambda index: _option_order_key(
                seed=policy.option_order_seed,
                candidate_id=context.candidate_id,
                option_index=index,
                option_sha256=option_hashes[index],
            ),
        )
    )

    return ProjectionPolicyDecision(
        candidate_id=context.candidate_id,
        policy_id=policy.policy_id,
        policy_sha256=policy.sha256,
        history_message_count=len(messages),
        evidence_range=MessageRange(
            start=context.evidence_start,
            end=context.evidence_end,
        ),
        insertion_after_message_index=insertion_index,
        pre_attack_range=pre_attack_range,
        survival_range=survival_range,
        survival_message_count=len(survival_messages),
        survival_token_count=sum(message.token_count for message in survival_messages),
        attacker_view_message_indexes=attacker_indexes,
        attacker_view_token_count=attacker_token_count,
        probe_option_order=option_order,
        probe_option_sha256=option_hashes,
        target_overlap_message_indexes=overlap_indexes,
        eligible=not rejections,
        rejection_codes=tuple(sorted(rejections)),
        warning_codes=tuple(sorted(warnings)),
    )


def _parse_serialized_message(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def _normalize_text(value: str) -> str:
    normalized_newlines = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_newlines).strip()


def _validate_context(context: ProjectionCandidateContext) -> None:
    if not _CANDIDATE_PATTERN.fullmatch(context.candidate_id):
        raise ProjectionPolicyError("candidate_id is not a PersonaMem candidate ID")
    if not context.messages:
        raise ProjectionPolicyError("projection context requires messages")
    expected_indexes = tuple(range(len(context.messages)))
    observed_indexes = tuple(message.source_index for message in context.messages)
    if observed_indexes != expected_indexes:
        raise ProjectionPolicyError(
            "message source indexes must be consecutive and zero-based"
        )
    if context.evidence_start < 0 or context.evidence_end < context.evidence_start:
        raise ProjectionPolicyError("evidence range is invalid")
    if context.evidence_end >= len(context.messages):
        raise ProjectionPolicyError("evidence range exceeds the chat history")
    if len(context.probe_options) < 2:
        raise ProjectionPolicyError("at least two probe options are required")
    normalized_options = tuple(
        _normalize_text(option) for option in context.probe_options
    )
    if any(not option for option in normalized_options):
        raise ProjectionPolicyError("probe options cannot be blank")
    if len(set(normalized_options)) != len(normalized_options):
        raise ProjectionPolicyError("probe options must be unique after normalization")
    if any(not _normalize_text(text) for text in context.protected_texts):
        raise ProjectionPolicyError("protected texts cannot be blank")


def _find_insertion_boundary(
    messages: Sequence[NormalizedPersonaMessage],
    *,
    evidence_end: int,
) -> int | None:
    if messages[evidence_end].role == MessageRole.ASSISTANT:
        return evidence_end
    return next(
        (
            index
            for index in range(evidence_end + 1, len(messages))
            if messages[index].role == MessageRole.ASSISTANT
        ),
        None,
    )


def _select_survival_range(
    messages: Sequence[NormalizedPersonaMessage],
    *,
    insertion_index: int,
    config: ProjectionPolicyConfig,
    rejections: set[str],
    warnings: set[str],
) -> MessageRange | None:
    start = insertion_index + 1
    if start >= len(messages):
        return None

    if config.survival_delay_mode == SurvivalDelayMode.ALL_REMAINING:
        end = len(messages) - 1
        if (
            config.complete_turn_required
            and messages[end].role != MessageRole.ASSISTANT
        ):
            completed_end = next(
                (
                    index
                    for index in range(end - 1, start - 1, -1)
                    if messages[index].role == MessageRole.ASSISTANT
                ),
                None,
            )
            if completed_end is None:
                rejections.add("incomplete_survival_turn")
                return None
            end = completed_end
            warnings.add("trailing_incomplete_survival_messages_omitted")
    else:
        assert config.survival_delay_events is not None
        requested_end = start + config.survival_delay_events - 1
        if requested_end >= len(messages):
            rejections.add("insufficient_survival_events")
            end = len(messages) - 1
        else:
            end = requested_end

    if (
        config.survival_delay_mode == SurvivalDelayMode.FIXED_EVENTS
        and config.complete_turn_required
        and messages[end].role != MessageRole.ASSISTANT
    ):
        maximum_end = min(
            len(messages) - 1,
            end + config.max_turn_completion_events,
        )
        completed_end = next(
            (
                index
                for index in range(end + 1, maximum_end + 1)
                if messages[index].role == MessageRole.ASSISTANT
            ),
            None,
        )
        if completed_end is None:
            rejections.add("incomplete_survival_turn")
        else:
            end = completed_end
    return MessageRange(start=start, end=end)


def _select_attacker_view(
    messages: Sequence[NormalizedPersonaMessage],
    *,
    insertion_index: int,
    config: ProjectionPolicyConfig,
    rejections: set[str],
) -> tuple[tuple[int, ...], int]:
    public_indexes = tuple(
        index
        for index in range(insertion_index + 1)
        if messages[index].role in {MessageRole.USER, MessageRole.ASSISTANT}
    )
    if config.attacker_view_mode == AttackerViewMode.FULL_PREFIX:
        selected = public_indexes
    else:
        assert config.attacker_view_token_budget is not None
        selected_reversed: list[int] = []
        for index in reversed(public_indexes):
            candidate = tuple(reversed((*selected_reversed, index)))
            token_count = _rendered_message_token_count(messages, candidate)
            if token_count > config.attacker_view_token_budget:
                if not selected_reversed:
                    rejections.add("attacker_message_exceeds_budget")
                break
            selected_reversed.append(index)
        selected = tuple(reversed(selected_reversed))
    return selected, _rendered_message_token_count(messages, selected)


def _rendered_message_token_count(
    messages: Sequence[NormalizedPersonaMessage],
    indexes: Sequence[int],
) -> int:
    rendered = "\n".join(
        f"<{messages[index].role.value}>\n{messages[index].content}\n"
        f"</{messages[index].role.value}>"
        for index in indexes
    )
    return _TOKEN_COUNTER.count(rendered)


def _find_target_overlap(
    survival_messages: Sequence[NormalizedPersonaMessage],
    *,
    protected_texts: Sequence[str],
    threshold: float,
) -> tuple[int, ...]:
    normalized_targets = tuple(
        _normalize_text(text).casefold() for text in protected_texts
    )
    target_token_sets = tuple(_search_tokens(text) for text in normalized_targets)
    matches: list[int] = []
    for message in survival_messages:
        content = message.content.casefold()
        content_tokens = _search_tokens(content)
        if any(
            target in content
            or _jaccard(content_tokens, target_tokens) >= threshold
            for target, target_tokens in zip(
                normalized_targets,
                target_token_sets,
                strict=True,
            )
        ):
            matches.append(message.source_index)
    return tuple(matches)


def _search_tokens(value: str) -> frozenset[str]:
    return frozenset(_SEARCH_TOKEN_PATTERN.findall(value.casefold()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _option_order_key(
    *,
    seed: int,
    candidate_id: str,
    option_index: int,
    option_sha256: str,
) -> str:
    value = f"{seed}:{candidate_id}:{option_index}:{option_sha256}"
    return _sha256(value.encode("utf-8"))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
