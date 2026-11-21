"""Build and integrity-check privacy-minimized projection decision artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.attacks.allocation import (
    AttackAllocationCase,
    AttackAllocationPlan,
    load_attack_allocation,
)
from membench.attacks.models import AttackMethodology
from membench.curation.models import (
    MessageRange,
    ProjectionPolicyConfig,
    ProjectionPolicyDecision,
)
from membench.curation.policy import (
    ProjectionCandidateContext,
    ProjectionPolicyError,
    apply_projection_policy,
    normalize_personamem_message,
    normalize_personamem_messages,
)
from membench.data_process.candidate_inventory import (
    CandidateInventory,
    CandidateInventoryRecord,
    CandidateInventorySource,
    load_candidate_inventory,
)
from membench.memory.models import CanonicalModel, MessageRole

PROJECTION_DRAFT_VERSION = "personamem_projection_draft_v1"
_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALLOCATION_DIR = _ROOT / "data" / "blackbox" / "allocation-native-v1"
DEFAULT_INVENTORY_DIR = _ROOT / "data" / "blackbox" / "inventory"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "raw_data"
DEFAULT_OUTPUT_DIR = _ROOT / "data" / "blackbox" / "projection-draft"


class ProjectionDraftError(RuntimeError):
    """Raised when a projection draft cannot be reproduced safely."""


class ProjectionDraftConfig(CanonicalModel):
    """Frozen projection policy plus public/private boundary decisions."""

    schema_version: Literal[1] = 1
    projection_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    policy: ProjectionPolicyConfig = Field(default_factory=ProjectionPolicyConfig)
    source_message_indexing: Literal["zero_based"] = "zero_based"
    system_message_policy: Literal["quarantine"] = "quarantine"
    private_content_policy: Literal["hash_only"] = "hash_only"

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))


class ProjectionAuditCriteria(CanonicalModel):
    """Machine-checkable gates associated with one projection decision."""

    evidence_resolved: bool
    evidence_precedes_or_meets_insertion: bool
    insertion_ends_on_assistant: bool
    survival_contiguous_after_insertion: bool
    survival_ends_on_assistant: bool
    attacker_view_pre_insertion_only: bool
    attacker_view_system_free: bool
    replay_system_messages_quarantined: bool
    decision_artifact_contains_private_text: Literal[False] = False


class ProjectionDraftRecord(CanonicalModel):
    """One allocation joined to its reproducible projection policy decision."""

    schema_version: Literal[1] = 1
    projection_ordinal: int = Field(ge=1)
    allocation_case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    allocation_ordinal: int = Field(ge=1)
    proposed_attack_id: str = Field(pattern=r"^bb-(pi|opc)-[0-9]{3}$")
    attack_methodology: AttackMethodology
    contract_family: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    candidate_id: str = Field(pattern=r"^pmv2-p[0-9]+-row[0-9]{6}$")
    persona_id: str = Field(pattern=r"^[0-9]+$")
    source_row_index: int = Field(ge=0)

    source_export_file: str = Field(pattern=r"^persona_[0-9]+\.json$")
    source_export_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_persona_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chat_history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    evidence_link_kind: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    evidence_similarity: float = Field(ge=0, le=1)
    evidence_message_sha256: tuple[str, ...] = Field(min_length=1)
    insertion_message_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization_warning_message_indexes: tuple[int, ...] = Field(
        default_factory=tuple
    )
    quarantined_system_message_indexes: tuple[int, ...] = Field(
        default_factory=tuple
    )
    pre_attack_replay_message_indexes: tuple[int, ...] = Field(
        default_factory=tuple
    )
    pre_attack_replay_token_count: int = Field(ge=0)
    survival_replay_message_indexes: tuple[int, ...] = Field(
        default_factory=tuple
    )
    survival_replay_token_count: int = Field(ge=0)
    omitted_trailing_message_indexes: tuple[int, ...] = Field(
        default_factory=tuple
    )

    probe_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correct_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    incorrect_answer_sha256: tuple[str, ...] = Field(min_length=1)
    false_target_option_index: int | None = Field(default=None, ge=0)
    false_target_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    policy_decision: ProjectionPolicyDecision
    audit: ProjectionAuditCriteria

    @field_validator(
        "normalization_warning_message_indexes",
        "quarantined_system_message_indexes",
        "pre_attack_replay_message_indexes",
        "survival_replay_message_indexes",
        "omitted_trailing_message_indexes",
    )
    @classmethod
    def indexes_must_be_sorted_unique_nonnegative(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if any(index < 0 for index in value):
            raise ValueError("projection indexes cannot be negative")
        if value != tuple(sorted(set(value))):
            raise ValueError("projection indexes must be sorted and unique")
        return value

    @field_validator("evidence_message_sha256", "incorrect_answer_sha256")
    @classmethod
    def hashes_must_be_sha256(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        import re

        if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in value):
            raise ValueError("projection hash collections must contain SHA-256")
        return value

    @model_validator(mode="after")
    def identities_and_ranges_must_be_consistent(self) -> Self:
        decision = self.policy_decision
        if decision.candidate_id != self.candidate_id:
            raise ValueError("policy decision belongs to another candidate")
        if len(self.evidence_message_sha256) != decision.evidence_range.count:
            raise ValueError("evidence hashes must match the evidence range")
        if set(self.quarantined_system_message_indexes) & set(
            self.pre_attack_replay_message_indexes
            + self.survival_replay_message_indexes
        ):
            raise ValueError("quarantined system messages cannot be replayed")
        target_fields = (
            self.false_target_option_index,
            self.false_target_sha256,
        )
        if any(item is None for item in target_fields) and any(
            item is not None for item in target_fields
        ):
            raise ValueError("false-target identity must be complete or absent")
        return self


class ProjectionDraftManifest(CanonicalModel):
    """Integrity and coverage metadata for projection decision JSONL."""

    schema_version: Literal[1] = 1
    projection_draft_version: Literal[
        "personamem_projection_draft_v1"
    ] = PROJECTION_DRAFT_VERSION
    projection_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    dataset_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    allocation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    allocation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(gt=0)
    eligible_count: int = Field(ge=0)
    ineligible_count: int = Field(ge=0)
    warning_counts: dict[str, int] = Field(default_factory=dict)
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    attack_method_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def counts_must_balance(self) -> Self:
        if self.eligible_count + self.ineligible_count != self.record_count:
            raise ValueError("projection eligibility counts must balance")
        if sum(self.attack_method_counts.values()) != self.record_count:
            raise ValueError("projection attack-method counts must balance")
        return self


@dataclass(frozen=True)
class ProjectionDraft:
    """Projection records and the immutable inputs that produced them."""

    dataset_revision: str
    allocation_id: str
    allocation_manifest_sha256: str
    inventory_manifest_sha256: str
    config: ProjectionDraftConfig
    records: tuple[ProjectionDraftRecord, ...]

    def decisions_jsonl(self) -> bytes:
        return _jsonl(self.records)

    def config_json(self) -> bytes:
        return _pretty_json(self.config.model_dump(mode="json"))

    @property
    def manifest(self) -> ProjectionDraftManifest:
        warning_counts = Counter(
            warning
            for record in self.records
            for warning in record.policy_decision.warning_codes
        )
        rejection_counts = Counter(
            rejection
            for record in self.records
            for rejection in record.policy_decision.rejection_codes
        )
        method_counts = Counter(
            record.attack_methodology.value for record in self.records
        )
        eligible_count = sum(
            record.policy_decision.eligible for record in self.records
        )
        return ProjectionDraftManifest(
            projection_id=self.config.projection_id,
            dataset_revision=self.dataset_revision,
            allocation_id=self.allocation_id,
            allocation_manifest_sha256=self.allocation_manifest_sha256,
            inventory_manifest_sha256=self.inventory_manifest_sha256,
            projection_config_sha256=self.config.sha256,
            decisions_sha256=hashlib.sha256(self.decisions_jsonl()).hexdigest(),
            record_count=len(self.records),
            eligible_count=eligible_count,
            ineligible_count=len(self.records) - eligible_count,
            warning_counts=dict(sorted(warning_counts.items())),
            rejection_counts=dict(sorted(rejection_counts.items())),
            attack_method_counts=dict(sorted(method_counts.items())),
        )


def build_projection_draft(
    allocation: AttackAllocationPlan,
    inventory: CandidateInventory,
    raw_dir: Path,
    config: ProjectionDraftConfig,
) -> ProjectionDraft:
    """Join frozen allocations to raw histories and persist policy decisions."""

    if allocation.dataset_revision != inventory.dataset_revision:
        raise ProjectionDraftError(
            "allocation and inventory dataset revisions do not match"
        )
    inventory_by_id = {record.candidate_id: record for record in inventory.records}
    sources_by_persona = {
        source.persona_id: source for source in inventory.source_exports
    }
    raw_cache: dict[str, Mapping[str, Any]] = {}
    records: list[ProjectionDraftRecord] = []
    for projection_ordinal, case in enumerate(allocation.allocations, start=1):
        if not case.execution_ready:
            raise ProjectionDraftError(
                f"allocation is not execution-ready: {case.allocation_case_id}"
            )
        candidate = inventory_by_id.get(case.candidate_id)
        if candidate is None:
            raise ProjectionDraftError(
                f"allocation candidate is absent from inventory: {case.candidate_id}"
            )
        source = sources_by_persona.get(case.persona_id)
        if source is None:
            raise ProjectionDraftError(
                f"inventory has no source export for persona {case.persona_id}"
            )
        document = raw_cache.get(case.persona_id)
        if document is None:
            document = _load_raw_export(
                raw_dir / source.file_name,
                source=source,
                dataset_revision=inventory.dataset_revision,
            )
            raw_cache[case.persona_id] = document
        records.append(
            _build_record(
                projection_ordinal=projection_ordinal,
                case=case,
                candidate=candidate,
                source=source,
                document=document,
                config=config,
            )
        )
    ordered = tuple(sorted(records, key=lambda item: item.projection_ordinal))
    return ProjectionDraft(
        dataset_revision=inventory.dataset_revision,
        allocation_id=allocation.config.allocation_id,
        allocation_manifest_sha256=allocation.manifest.sha256,
        inventory_manifest_sha256=inventory.manifest.sha256,
        config=config,
        records=ordered,
    )


def export_projection_draft(
    draft: ProjectionDraft,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Atomically export decision JSONL, config, and integrity manifest."""

    decisions_path = output_dir / "projection_decisions.jsonl"
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.json"
    paths = (decisions_path, config_path, manifest_path)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise ProjectionDraftError(
            "refusing to replace projection draft without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(decisions_path, draft.decisions_jsonl())
    _atomic_write(config_path, draft.config_json())
    _atomic_write(
        manifest_path,
        _pretty_json(draft.manifest.model_dump(mode="json")),
    )
    return decisions_path, config_path, manifest_path


def load_projection_draft(input_dir: Path) -> ProjectionDraft:
    """Load a projection draft and verify every recorded integrity hash."""

    decisions_path = input_dir / "projection_decisions.jsonl"
    config_path = input_dir / "config.json"
    manifest_path = input_dir / "manifest.json"
    try:
        decision_bytes = decisions_path.read_bytes()
        config = ProjectionDraftConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
        manifest = ProjectionDraftManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ProjectionDraftError(
            f"could not load projection draft from {input_dir}: {exc}"
        ) from exc
    if hashlib.sha256(decision_bytes).hexdigest() != manifest.decisions_sha256:
        raise ProjectionDraftError("projection decision integrity check failed")
    if config.sha256 != manifest.projection_config_sha256:
        raise ProjectionDraftError("projection config does not match its manifest")
    records = _parse_records(decision_bytes)
    draft = ProjectionDraft(
        dataset_revision=manifest.dataset_revision,
        allocation_id=manifest.allocation_id,
        allocation_manifest_sha256=manifest.allocation_manifest_sha256,
        inventory_manifest_sha256=manifest.inventory_manifest_sha256,
        config=config,
        records=records,
    )
    if draft.manifest != manifest:
        raise ProjectionDraftError(
            "projection manifest does not match its records and config"
        )
    return draft


def _build_record(
    *,
    projection_ordinal: int,
    case: AttackAllocationCase,
    candidate: CandidateInventoryRecord,
    source: CandidateInventorySource,
    document: Mapping[str, Any],
    config: ProjectionDraftConfig,
) -> ProjectionDraftRecord:
    _validate_candidate_identity(case, candidate)
    row = _find_benchmark_row(document, case.source_row_index)
    if _canonical_sha256(row) != candidate.source_record_sha256:
        raise ProjectionDraftError(
            f"benchmark row changed after inventory: {case.candidate_id}"
        )
    messages = _raw_messages(document)
    try:
        normalized = normalize_personamem_messages(messages)
    except ProjectionPolicyError as exc:
        raise ProjectionDraftError(
            f"could not normalize history for {case.candidate_id}: {exc}"
        ) from exc
    if len(normalized) != candidate.history_message_count:
        raise ProjectionDraftError(
            f"history message count changed for {case.candidate_id}"
        )
    if (
        candidate.evidence_message_start is None
        or candidate.evidence_message_end is None
    ):
        raise ProjectionDraftError(
            f"candidate has no resolved evidence range: {case.candidate_id}"
        )

    correct_answer = _required_text(row, "correct_answer")
    incorrect_answers = _incorrect_answers(row.get("incorrect_answers"))
    query = _required_text(row, "user_query")
    if _sha256(query.encode("utf-8")) != candidate.query_sha256:
        raise ProjectionDraftError(f"probe query changed for {case.candidate_id}")
    if _sha256(correct_answer.encode("utf-8")) != candidate.answer_sha256:
        raise ProjectionDraftError(f"correct answer changed for {case.candidate_id}")
    incorrect_hashes = tuple(
        _sha256(answer.encode("utf-8")) for answer in incorrect_answers
    )
    if incorrect_hashes != candidate.incorrect_answer_sha256:
        raise ProjectionDraftError(
            f"incorrect answers changed for {case.candidate_id}"
        )

    false_target = None
    if case.false_target_option_index is not None:
        try:
            false_target = incorrect_answers[case.false_target_option_index]
        except IndexError as exc:
            raise ProjectionDraftError(
                f"false-target option is out of range: {case.candidate_id}"
            ) from exc
        if _sha256(false_target.encode("utf-8")) != case.false_target_sha256:
            raise ProjectionDraftError(
                f"false-target hash changed for {case.candidate_id}"
            )

    try:
        probe = normalize_personamem_message(query, source_index=0)
    except ProjectionPolicyError as exc:
        raise ProjectionDraftError(
            f"could not normalize probe for {case.candidate_id}: {exc}"
        ) from exc
    if probe.role != MessageRole.USER:
        raise ProjectionDraftError(f"probe is not a user message: {case.candidate_id}")
    protected = _unique_nonblank(
        correct_answer,
        row.get("preference"),
        row.get("prev_pref"),
        probe.content,
        false_target,
    )
    decision = apply_projection_policy(
        ProjectionCandidateContext(
            candidate_id=case.candidate_id,
            messages=normalized,
            evidence_start=candidate.evidence_message_start,
            evidence_end=candidate.evidence_message_end,
            probe_options=(correct_answer, *incorrect_answers),
            protected_texts=protected,
        ),
        config.policy,
    )

    system_indexes = tuple(
        message.source_index
        for message in normalized
        if message.role == MessageRole.SYSTEM
    )
    pre_indexes = _replay_indexes(normalized, decision.pre_attack_range)
    survival_indexes = (
        _replay_indexes(normalized, decision.survival_range)
        if decision.survival_range is not None
        else ()
    )
    omitted_start = (
        decision.survival_range.end + 1
        if decision.survival_range is not None
        else decision.insertion_after_message_index + 1
    )
    omitted_indexes = tuple(range(omitted_start, len(normalized)))
    evidence_hashes = tuple(
        normalized[index].normalized_sha256
        for index in range(
            decision.evidence_range.start,
            decision.evidence_range.end + 1,
        )
    )
    attacker_roles = (
        normalized[index].role for index in decision.attacker_view_message_indexes
    )
    audit = ProjectionAuditCriteria(
        evidence_resolved=True,
        evidence_precedes_or_meets_insertion=(
            decision.evidence_range.end <= decision.insertion_after_message_index
        ),
        insertion_ends_on_assistant=(
            normalized[decision.insertion_after_message_index].role
            == MessageRole.ASSISTANT
        ),
        survival_contiguous_after_insertion=(
            decision.survival_range is None
            or decision.survival_range.start
            == decision.insertion_after_message_index + 1
        ),
        survival_ends_on_assistant=(
            decision.survival_range is None
            or normalized[decision.survival_range.end].role
            == MessageRole.ASSISTANT
        ),
        attacker_view_pre_insertion_only=all(
            index <= decision.insertion_after_message_index
            for index in decision.attacker_view_message_indexes
        ),
        attacker_view_system_free=all(
            role != MessageRole.SYSTEM for role in attacker_roles
        ),
        replay_system_messages_quarantined=not (
            set(system_indexes) & set((*pre_indexes, *survival_indexes))
        ),
    )
    return ProjectionDraftRecord(
        projection_ordinal=projection_ordinal,
        allocation_case_id=case.allocation_case_id,
        allocation_ordinal=case.allocation_ordinal,
        proposed_attack_id=case.proposed_attack_id,
        attack_methodology=case.attack_methodology,
        contract_family=case.contract_family,
        candidate_id=case.candidate_id,
        persona_id=case.persona_id,
        source_row_index=case.source_row_index,
        source_export_file=source.file_name,
        source_export_sha256=source.sha256,
        source_record_sha256=candidate.source_record_sha256,
        raw_persona_sha256=candidate.raw_persona_sha256,
        chat_history_sha256=candidate.chat_history_sha256,
        normalized_history_sha256=_normalized_history_sha256(normalized),
        evidence_link_kind=candidate.evidence_link_kind,
        evidence_similarity=candidate.evidence_similarity,
        evidence_message_sha256=evidence_hashes,
        insertion_message_sha256=normalized[
            decision.insertion_after_message_index
        ].normalized_sha256,
        normalization_warning_message_indexes=tuple(
            message.source_index
            for message in normalized
            if message.warning_codes
        ),
        quarantined_system_message_indexes=system_indexes,
        pre_attack_replay_message_indexes=pre_indexes,
        pre_attack_replay_token_count=sum(
            normalized[index].token_count for index in pre_indexes
        ),
        survival_replay_message_indexes=survival_indexes,
        survival_replay_token_count=sum(
            normalized[index].token_count for index in survival_indexes
        ),
        omitted_trailing_message_indexes=omitted_indexes,
        probe_query_sha256=candidate.query_sha256,
        correct_answer_sha256=candidate.answer_sha256,
        incorrect_answer_sha256=candidate.incorrect_answer_sha256,
        false_target_option_index=case.false_target_option_index,
        false_target_sha256=case.false_target_sha256,
        policy_decision=decision,
        audit=audit,
    )


def _validate_candidate_identity(
    case: AttackAllocationCase,
    candidate: CandidateInventoryRecord,
) -> None:
    if (candidate.persona_id, candidate.source_row_index) != (
        case.persona_id,
        case.source_row_index,
    ):
        raise ProjectionDraftError(
            f"allocation identity conflicts with inventory: {case.candidate_id}"
        )
    if not candidate.structurally_eligible:
        raise ProjectionDraftError(
            f"allocated candidate is structurally ineligible: {case.candidate_id}"
        )


def _load_raw_export(
    path: Path,
    *,
    source: CandidateInventorySource,
    dataset_revision: str,
) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectionDraftError(f"could not read raw export {path}: {exc}") from exc
    if _sha256(payload) != source.sha256:
        raise ProjectionDraftError(f"raw export changed after inventory: {path}")
    if not isinstance(document, Mapping):
        raise ProjectionDraftError(f"raw export is not an object: {path}")
    dataset = document.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("revision") != dataset_revision:
        raise ProjectionDraftError(f"raw export revision changed: {path}")
    if str(document.get("persona_id")) != source.persona_id:
        raise ProjectionDraftError(f"raw export persona changed: {path}")
    return document


def _raw_messages(document: Mapping[str, Any]) -> Sequence[Mapping[str, Any] | str]:
    chat = document.get("chat_history_32k")
    if not isinstance(chat, Mapping):
        raise ProjectionDraftError("raw export has no chat_history_32k object")
    messages = chat.get("chat_history")
    if not isinstance(messages, list) or not messages:
        raise ProjectionDraftError("raw export has no chat history messages")
    return messages


def _find_benchmark_row(
    document: Mapping[str, Any],
    source_row_index: int,
) -> Mapping[str, Any]:
    rows = document.get("benchmark_records")
    if not isinstance(rows, list):
        raise ProjectionDraftError("raw export has no benchmark records")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("source_row_index") == source_row_index
    ]
    if len(matches) != 1:
        raise ProjectionDraftError(
            "expected one benchmark source row "
            f"{source_row_index}; found {len(matches)}"
        )
    return matches[0]


def _incorrect_answers(value: Any) -> tuple[str, ...]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ProjectionDraftError("incorrect answers are not valid JSON") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ProjectionDraftError("incorrect answers must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise ProjectionDraftError("incorrect answers must contain non-blank text")
    return tuple(item.strip() for item in parsed)


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ProjectionDraftError(f"benchmark field {field} must be non-blank text")
    return value.strip()


def _unique_nonblank(*values: Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        text = value.strip()
        if text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


def _replay_indexes(
    messages: Sequence[Any],
    message_range: MessageRange,
) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(message_range.start, message_range.end + 1)
        if messages[index].role in {MessageRole.USER, MessageRole.ASSISTANT}
    )


def _normalized_history_sha256(messages: Sequence[Any]) -> str:
    return _canonical_sha256(
        [
            {
                "index": message.source_index,
                "role": message.role.value,
                "normalized_sha256": message.normalized_sha256,
            }
            for message in messages
        ]
    )


def _parse_records(payload: bytes) -> tuple[ProjectionDraftRecord, ...]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ProjectionDraftError("projection decisions are not UTF-8") from exc
    records: list[ProjectionDraftRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(ProjectionDraftRecord.model_validate_json(line))
        except ValueError as exc:
            raise ProjectionDraftError(
                f"invalid projection decision line {line_number}: {exc}"
            ) from exc
    if not records:
        raise ProjectionDraftError("projection draft contains no decisions")
    ordinals = [record.projection_ordinal for record in records]
    if ordinals != list(range(1, len(records) + 1)):
        raise ProjectionDraftError("projection decision ordinals are invalid")
    return tuple(records)


def _jsonl(values: Sequence[CanonicalModel]) -> bytes:
    return (
        "".join(
            json.dumps(
                value.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for value in values
        )
    ).encode("utf-8")


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
    return _sha256(payload)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError as exc:
        raise ProjectionDraftError(f"could not write {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize deterministic PersonaMem projection decisions."
    )
    parser.add_argument("--allocation-dir", type=Path, default=DEFAULT_ALLOCATION_DIR)
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--projection-id",
        default="pmv2-native-projection-draft-v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    allocation = load_attack_allocation(args.allocation_dir)
    inventory = load_candidate_inventory(args.inventory_dir)
    config = ProjectionDraftConfig(projection_id=args.projection_id)
    draft = build_projection_draft(allocation, inventory, args.raw_dir, config)
    decisions_path, config_path, manifest_path = export_projection_draft(
        draft,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        f"Materialized {draft.manifest.record_count} projection decisions "
        f"({draft.manifest.eligible_count} eligible)."
    )
    print(f"Decisions: {decisions_path}")
    print(f"Config: {config_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
