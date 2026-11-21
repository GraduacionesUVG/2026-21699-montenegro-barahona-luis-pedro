"""Normalize evaluator-only PersonaMem exports into a safe candidate inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.context_budget import RegexTokenCounter
from membench.data_process.data_processing import DATASET_ID
from membench.memory.models import CanonicalModel

INVENTORY_VERSION = "personamem_candidate_inventory_v1"
DEFAULT_RAW_EXPORT_DIR = Path(__file__).resolve().parents[1] / "raw_data"
DEFAULT_INVENTORY_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "blackbox" / "inventory"
)

_REVISION_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SLUG_PATTERN = r"^[a-z][a-z0-9_]*$"
_CANDIDATE_ID_PATTERN = r"^pmv2-p[0-9]+-row[0-9]{6}$"
_TOKEN_COUNTER = RegexTokenCounter()
_SEARCH_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)

_SENSITIVE_KIND_ALIASES = {
    "driver_license_number": "drivers_license_number",
    "drivers_license_number": "drivers_license_number",
    "social_security_number": "social_security_number",
    "ssn": "social_security_number",
}


class CandidateInventoryError(RuntimeError):
    """Raised when raw exports cannot form a trustworthy inventory."""


class CandidateOwner(StrEnum):
    """Normalized benchmark subject ownership constants."""

    SELF = "self"
    OTHERS = "others"
    UNKNOWN = "unknown"


class EvidenceLinkKind(StrEnum):
    """Strength of an automatic snippet-to-history association."""

    EXACT_MESSAGE = "exact_message"
    CONTAINED_WINDOW = "contained_window"
    STRONG_LEXICAL = "strong_lexical"
    WEAK_LEXICAL = "weak_lexical"
    UNVERIFIED = "unverified"


class CandidateInventoryRecord(CanonicalModel):
    """One privacy-minimized benchmark row used for candidate selection."""

    schema_version: Literal[1] = 1
    candidate_id: str = Field(pattern=_CANDIDATE_ID_PATTERN)
    dataset_revision: str
    persona_id: str = Field(pattern=r"^[0-9]+$")
    source_row_index: int = Field(ge=0)
    source_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_persona_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chat_history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    topic: str = Field(pattern=_SLUG_PATTERN)
    preference_type: str = Field(pattern=_SLUG_PATTERN)
    conversation_scenario: str = Field(pattern=_SLUG_PATTERN)
    owner: Literal["self", "others", "unknown"]
    updated: bool
    sensitive: bool
    sensitive_field_kinds: tuple[str, ...] = Field(default_factory=tuple)

    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_character_count: int = Field(ge=0)
    query_token_count: int = Field(ge=0)
    answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_character_count: int = Field(ge=0)
    answer_token_count: int = Field(ge=0)
    incorrect_answer_sha256: tuple[str, ...] = Field(default_factory=tuple)
    incorrect_option_count: int = Field(ge=0)
    related_snippet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    related_snippet_token_count: int = Field(ge=0)

    evidence_distance_tokens: int = Field(ge=0)
    relevant_history_tokens: int = Field(ge=0)
    irrelevant_history_tokens: int = Field(ge=0)
    history_total_tokens: int = Field(ge=0)
    history_message_count: int = Field(ge=1)
    history_user_message_count: int = Field(ge=0)
    history_assistant_message_count: int = Field(ge=0)
    history_system_message_count: int = Field(ge=0)

    evidence_link_kind: Literal[
        "exact_message",
        "contained_window",
        "strong_lexical",
        "weak_lexical",
        "unverified",
    ]
    evidence_similarity: float = Field(ge=0, le=1)
    evidence_message_start: int | None = Field(default=None, ge=0)
    evidence_message_end: int | None = Field(default=None, ge=0)

    structurally_eligible: bool
    rejection_reasons: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator(
        "sensitive_field_kinds",
        "rejection_reasons",
        "warnings",
    )
    @classmethod
    def string_sets_must_be_sorted_and_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not re.fullmatch(_SLUG_PATTERN, item) for item in value):
            raise ValueError("inventory reason and category values must be slugs")
        if value != tuple(sorted(set(value))):
            raise ValueError("inventory reason and category values must be sorted")
        return value

    @field_validator("incorrect_answer_sha256")
    @classmethod
    def incorrect_hashes_must_be_valid(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not _SHA256_PATTERN.fullmatch(item) for item in value):
            raise ValueError("incorrect answer hashes must be SHA-256 values")
        return value

    @model_validator(mode="after")
    def eligibility_and_evidence_must_be_consistent(self) -> Self:
        if self.structurally_eligible != (not self.rejection_reasons):
            raise ValueError("structural eligibility must agree with rejection reasons")
        if self.incorrect_option_count != len(self.incorrect_answer_sha256):
            raise ValueError("incorrect option count must match its hashes")
        evidence_indexes = (
            self.evidence_message_start,
            self.evidence_message_end,
        )
        if any(item is None for item in evidence_indexes) and any(
            item is not None for item in evidence_indexes
        ):
            raise ValueError("evidence message indexes must both be present or absent")
        if (
            self.evidence_message_start is not None
            and self.evidence_message_end is not None
            and self.evidence_message_end < self.evidence_message_start
        ):
            raise ValueError("evidence message range cannot run backwards")
        counted_roles = (
            self.history_user_message_count
            + self.history_assistant_message_count
            + self.history_system_message_count
        )
        if counted_roles > self.history_message_count:
            raise ValueError("history role counts cannot exceed total messages")
        return self


class CandidateInventorySource(CanonicalModel):
    """Hash and row count for one evaluator-only raw export."""

    file_name: str = Field(pattern=r"^persona_[0-9]+\.json$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    persona_id: str = Field(pattern=r"^[0-9]+$")
    benchmark_record_count: int = Field(gt=0)


class CandidateInventoryManifest(CanonicalModel):
    """Integrity and summary metadata for normalized candidate JSONL."""

    schema_version: Literal[1] = 1
    dataset_id: Literal["bowen-upenn/PersonaMem-v2"] = DATASET_ID
    dataset_revision: str
    inventory_version: Literal[
        "personamem_candidate_inventory_v1"
    ] = INVENTORY_VERSION
    token_counter: Literal["regex_lexical_v1"] = "regex_lexical_v1"
    record_count: int = Field(gt=0)
    structurally_eligible_count: int = Field(ge=0)
    structurally_ineligible_count: int = Field(ge=0)
    persona_count: int = Field(gt=0)
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_exports: tuple[CandidateInventorySource, ...] = Field(min_length=1)
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    warning_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("dataset_revision")
    @classmethod
    def revision_must_be_immutable(cls, value: str) -> str:
        if not _REVISION_PATTERN.fullmatch(value):
            raise ValueError("dataset revision must be a 40- or 64-character hash")
        return value

    @model_validator(mode="after")
    def counts_must_balance(self) -> Self:
        if (
            self.structurally_eligible_count
            + self.structurally_ineligible_count
            != self.record_count
        ):
            raise ValueError("inventory structural eligibility counts must balance")
        personas = {item.persona_id for item in self.source_exports}
        if len(personas) != self.persona_count:
            raise ValueError("inventory persona count must match its sources")
        if len(personas) != len(self.source_exports):
            raise ValueError("inventory sources must contain one export per persona")
        return self

    @property
    def sha256(self) -> str:
        """Hash the canonical manifest for campaign artifact registration."""

        return _canonical_sha256(self.model_dump(mode="json"))


@dataclass(frozen=True)
class CandidateInventory:
    """Normalized records and the exact source exports from which they came."""

    dataset_revision: str
    records: tuple[CandidateInventoryRecord, ...]
    source_exports: tuple[CandidateInventorySource, ...]

    def records_jsonl(self) -> bytes:
        """Render deterministic UTF-8 JSONL without private source text."""

        text = "".join(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in self.records
        )
        return text.encode("utf-8")

    @property
    def manifest(self) -> CandidateInventoryManifest:
        """Build deterministic counts and integrity metadata for the inventory."""

        rejection_counts = Counter(
            reason for record in self.records for reason in record.rejection_reasons
        )
        warning_counts = Counter(
            warning for record in self.records for warning in record.warnings
        )
        eligible_count = sum(record.structurally_eligible for record in self.records)
        records_bytes = self.records_jsonl()
        return CandidateInventoryManifest(
            dataset_revision=self.dataset_revision,
            record_count=len(self.records),
            structurally_eligible_count=eligible_count,
            structurally_ineligible_count=len(self.records) - eligible_count,
            persona_count=len({item.persona_id for item in self.source_exports}),
            records_sha256=hashlib.sha256(records_bytes).hexdigest(),
            source_exports=self.source_exports,
            rejection_counts=dict(sorted(rejection_counts.items())),
            warning_counts=dict(sorted(warning_counts.items())),
        )


def normalize_candidate_inventory(input_dir: Path) -> CandidateInventory:
    """Load all raw persona exports and normalize every benchmark row."""

    paths = tuple(sorted(input_dir.glob("persona_*.json"), key=lambda path: path.name))
    if not paths:
        raise CandidateInventoryError(
            f"candidate inventory input has no persona_*.json files: {input_dir}"
        )

    revisions: set[str] = set()
    records: list[CandidateInventoryRecord] = []
    sources: list[CandidateInventorySource] = []
    for path in paths:
        document, source_sha256 = _read_export(path)
        revision, source, normalized = _normalize_export_document(
            document,
            file_name=path.name,
            source_sha256=source_sha256,
        )
        revisions.add(revision)
        sources.append(source)
        records.extend(normalized)

    if len(revisions) != 1:
        raise CandidateInventoryError(
            "raw persona exports contain multiple dataset revisions: "
            + ", ".join(sorted(revisions))
        )
    ordered_records = tuple(
        sorted(
            records,
            key=lambda item: (int(item.persona_id), item.source_row_index),
        )
    )
    candidate_ids = [item.candidate_id for item in ordered_records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CandidateInventoryError("normalized candidate IDs are not unique")
    ordered_sources = tuple(sorted(sources, key=lambda item: int(item.persona_id)))
    source_personas = [item.persona_id for item in ordered_sources]
    if len(source_personas) != len(set(source_personas)):
        raise CandidateInventoryError("raw exports contain duplicate personas")
    return CandidateInventory(
        dataset_revision=next(iter(revisions)),
        records=ordered_records,
        source_exports=ordered_sources,
    )


def export_candidate_inventory(
    inventory: CandidateInventory,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Atomically write candidate JSONL and its integrity manifest."""

    records_path = output_dir / "candidates.jsonl"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (records_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise CandidateInventoryError(
            "refusing to replace inventory files without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    records_bytes = inventory.records_jsonl()
    manifest_bytes = (
        json.dumps(
            inventory.manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(records_path, records_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    return records_path, manifest_path


def load_candidate_inventory(input_dir: Path) -> CandidateInventory:
    """Load and integrity-check a previously exported normalized inventory."""

    records_path = input_dir / "candidates.jsonl"
    manifest_path = input_dir / "manifest.json"
    try:
        records_bytes = records_path.read_bytes()
        manifest = CandidateInventoryManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise CandidateInventoryError(
            f"could not load candidate inventory from {input_dir}: {exc}"
        ) from exc
    observed_hash = hashlib.sha256(records_bytes).hexdigest()
    if observed_hash != manifest.records_sha256:
        raise CandidateInventoryError(
            "candidate inventory integrity check failed: "
            f"{observed_hash}; expected {manifest.records_sha256}"
        )

    records: list[CandidateInventoryRecord] = []
    try:
        for line_number, line in enumerate(
            records_bytes.decode("utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                records.append(CandidateInventoryRecord.model_validate_json(line))
            except ValueError as exc:
                raise CandidateInventoryError(
                    f"invalid candidates.jsonl line {line_number}: {exc}"
                ) from exc
    except UnicodeDecodeError as exc:
        raise CandidateInventoryError(
            "candidate inventory records are not valid UTF-8"
        ) from exc
    if not records:
        raise CandidateInventoryError("candidate inventory contains no records")

    inventory = CandidateInventory(
        dataset_revision=manifest.dataset_revision,
        records=tuple(records),
        source_exports=manifest.source_exports,
    )
    if inventory.manifest != manifest:
        raise CandidateInventoryError(
            "candidate inventory manifest does not match its normalized records"
        )
    return inventory


def _normalize_export_document(
    document: Mapping[str, Any],
    *,
    file_name: str,
    source_sha256: str,
) -> tuple[
    str,
    CandidateInventorySource,
    tuple[CandidateInventoryRecord, ...],
]:
    dataset = _require_mapping(document.get("dataset"), "dataset")
    if dataset.get("id") != DATASET_ID:
        raise CandidateInventoryError(
            f"{file_name} has unexpected dataset ID: {dataset.get('id')!r}"
        )
    revision = _require_text(dataset.get("revision"), "dataset.revision")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise CandidateInventoryError(
            f"{file_name} has a mutable or invalid dataset revision"
        )

    persona_id = _persona_id(document.get("persona_id"), file_name)
    if file_name != f"persona_{persona_id}.json":
        raise CandidateInventoryError(
            f"{file_name} does not match contained persona {persona_id}"
        )
    source_files = _require_mapping(document.get("source_files"), "source_files")
    raw_persona_sha256 = _require_sha256(
        source_files.get("raw_persona_sha256"),
        "source_files.raw_persona_sha256",
    )
    chat_history_sha256 = _require_sha256(
        source_files.get("chat_history_32k_sha256"),
        "source_files.chat_history_32k_sha256",
    )
    raw_persona = _require_mapping(document.get("raw_persona"), "raw_persona")
    persona = _require_mapping(
        raw_persona.get(persona_id),
        f"raw_persona.{persona_id}",
    )
    sensitive_information = _optional_mapping(persona.get("sensitive_information"))

    chat = _require_mapping(document.get("chat_history_32k"), "chat_history_32k")
    metadata = _require_mapping(chat.get("metadata"), "chat_history_32k.metadata")
    if _persona_id(metadata.get("persona_id"), file_name) != persona_id:
        raise CandidateInventoryError(
            f"{file_name} chat history belongs to another persona"
        )
    messages = _validated_messages(chat.get("chat_history"), file_name)
    history_counts = Counter(item["role"] for item in messages)
    evidence_index = _build_evidence_index(messages)

    benchmark_records = document.get("benchmark_records")
    if not isinstance(benchmark_records, list) or not benchmark_records:
        raise CandidateInventoryError(
            f"{file_name} has no benchmark_records array"
        )

    normalized: list[CandidateInventoryRecord] = []
    for row in benchmark_records:
        if not isinstance(row, Mapping):
            raise CandidateInventoryError(
                f"{file_name} contains a non-object benchmark record"
            )
        normalized.append(
            _normalize_row(
                row,
                revision=revision,
                persona_id=persona_id,
                raw_persona_sha256=raw_persona_sha256,
                chat_history_sha256=chat_history_sha256,
                messages=messages,
                evidence_index=evidence_index,
                history_counts=history_counts,
                sensitive_information=sensitive_information,
            )
        )

    source = CandidateInventorySource(
        file_name=file_name,
        sha256=source_sha256,
        persona_id=persona_id,
        benchmark_record_count=len(normalized),
    )
    return revision, source, tuple(normalized)


def _normalize_row(
    row: Mapping[str, Any],
    *,
    revision: str,
    persona_id: str,
    raw_persona_sha256: str,
    chat_history_sha256: str,
    messages: tuple[dict[str, str], ...],
    evidence_index: _EvidenceIndex,
    history_counts: Counter[str],
    sensitive_information: Mapping[str, Any],
) -> CandidateInventoryRecord:
    row_persona_id = _persona_id(row.get("persona_id"), "benchmark record")
    if row_persona_id != persona_id:
        raise CandidateInventoryError(
            f"benchmark record persona {row_persona_id} does not match {persona_id}"
        )
    source_row_index = _nonnegative_int(
        row.get("source_row_index"),
        "source_row_index",
    )
    rejection_reasons: set[str] = set()
    warnings: set[str] = set()

    query = _candidate_text(
        row.get("user_query"),
        "missing_user_query",
        rejection_reasons,
    )
    answer = _candidate_text(
        row.get("correct_answer"),
        "missing_correct_answer",
        rejection_reasons,
    )
    snippet = _candidate_text(
        row.get("related_conversation_snippet"),
        "missing_related_snippet",
        rejection_reasons,
    )
    incorrect_answers = _incorrect_answers(
        row.get("incorrect_answers"),
        answer,
        rejection_reasons,
    )

    topic = _category(row.get("topic_preference"))
    preference_type = _category(row.get("pref_type"))
    scenario = _category(row.get("conversation_scenario"))
    if topic == "unknown":
        rejection_reasons.add("missing_topic")
    if preference_type == "unknown":
        rejection_reasons.add("missing_preference_type")
    if scenario == "unknown":
        rejection_reasons.add("missing_conversation_scenario")

    raw_owner = _category(row.get("who"))
    owner = (
        raw_owner
        if raw_owner in {CandidateOwner.SELF, CandidateOwner.OTHERS}
        else CandidateOwner.UNKNOWN
    )
    if owner == CandidateOwner.UNKNOWN:
        rejection_reasons.add("unknown_owner")
    updated = _strict_bool(row.get("updated"), "updated")
    sensitive = _strict_bool(row.get("sensitive_info"), "sensitive_info")

    sensitive_kinds = _sensitive_field_kinds(
        row,
        sensitive_information,
    )
    if sensitive and not sensitive_kinds:
        warnings.add("sensitive_field_unmatched")
    if len(sensitive_kinds) > 1:
        warnings.add("multiple_sensitive_fields")

    evidence = _link_evidence(snippet, evidence_index)
    if evidence.kind == EvidenceLinkKind.UNVERIFIED:
        warnings.add("evidence_link_unverified")
    elif evidence.kind == EvidenceLinkKind.WEAK_LEXICAL:
        warnings.add("evidence_link_weak")

    incorrect_hashes = tuple(_text_sha256(item) for item in incorrect_answers)
    return CandidateInventoryRecord(
        candidate_id=f"pmv2-p{persona_id}-row{source_row_index:06d}",
        dataset_revision=revision,
        persona_id=persona_id,
        source_row_index=source_row_index,
        source_record_sha256=_canonical_sha256(row),
        raw_persona_sha256=raw_persona_sha256,
        chat_history_sha256=chat_history_sha256,
        topic=topic,
        preference_type=preference_type,
        conversation_scenario=scenario,
        owner=owner,
        updated=updated,
        sensitive=sensitive,
        sensitive_field_kinds=sensitive_kinds,
        query_sha256=_text_sha256(query),
        query_character_count=len(query),
        query_token_count=_TOKEN_COUNTER.count(query),
        answer_sha256=_text_sha256(answer),
        answer_character_count=len(answer),
        answer_token_count=_TOKEN_COUNTER.count(answer),
        incorrect_answer_sha256=incorrect_hashes,
        incorrect_option_count=len(incorrect_hashes),
        related_snippet_sha256=_text_sha256(snippet),
        related_snippet_token_count=_TOKEN_COUNTER.count(snippet),
        evidence_distance_tokens=_nonnegative_int(
            row.get("distance_from_related_snippet_to_query_32k"),
            "distance_from_related_snippet_to_query_32k",
        ),
        relevant_history_tokens=_nonnegative_int(
            row.get("num_persona_relevant_tokens_32k"),
            "num_persona_relevant_tokens_32k",
        ),
        irrelevant_history_tokens=_nonnegative_int(
            row.get("num_persona_irrelevant_tokens_32k"),
            "num_persona_irrelevant_tokens_32k",
        ),
        history_total_tokens=_nonnegative_int(
            row.get("total_tokens_in_chat_history_32k"),
            "total_tokens_in_chat_history_32k",
        ),
        history_message_count=len(messages),
        history_user_message_count=history_counts["user"],
        history_assistant_message_count=history_counts["assistant"],
        history_system_message_count=history_counts["system"],
        evidence_link_kind=evidence.kind,
        evidence_similarity=evidence.similarity,
        evidence_message_start=evidence.start,
        evidence_message_end=evidence.end,
        structurally_eligible=not rejection_reasons,
        rejection_reasons=tuple(sorted(rejection_reasons)),
        warnings=tuple(sorted(warnings)),
    )


@dataclass(frozen=True)
class _EvidenceLink:
    kind: str
    similarity: float
    start: int | None
    end: int | None


@dataclass(frozen=True)
class _EvidenceWindow:
    text: str
    tokens: frozenset[str]
    start: int
    end: int


@dataclass(frozen=True)
class _EvidenceIndex:
    messages: tuple[str, ...]
    windows: tuple[_EvidenceWindow, ...]


def _build_evidence_index(
    messages: tuple[dict[str, str], ...],
) -> _EvidenceIndex:
    normalized_messages = tuple(
        _normalized_search_text(item["content"]) for item in messages
    )
    windows: list[_EvidenceWindow] = []
    for start in range(len(normalized_messages)):
        for end in range(start, min(start + 6, len(normalized_messages))):
            text = " ".join(normalized_messages[start : end + 1]).strip()
            if not text:
                continue
            windows.append(
                _EvidenceWindow(
                    text=text,
                    tokens=frozenset(_SEARCH_TOKEN_PATTERN.findall(text)),
                    start=start,
                    end=end,
                )
            )
    return _EvidenceIndex(messages=normalized_messages, windows=tuple(windows))


def _link_evidence(
    snippet: str,
    evidence_index: _EvidenceIndex,
) -> _EvidenceLink:
    normalized_snippet = _normalized_search_text(snippet)
    if not normalized_snippet:
        return _EvidenceLink(EvidenceLinkKind.UNVERIFIED, 0.0, None, None)
    for index, content in enumerate(evidence_index.messages):
        if content and content == normalized_snippet:
            return _EvidenceLink(EvidenceLinkKind.EXACT_MESSAGE, 1.0, index, index)

    best_similarity = 0.0
    best_start: int | None = None
    best_end: int | None = None
    best_contained = False
    snippet_tokens = set(_SEARCH_TOKEN_PATTERN.findall(normalized_snippet))
    for window in evidence_index.windows:
        smaller_size = min(len(snippet_tokens), len(window.tokens))
        contained = smaller_size >= 5 and (
            normalized_snippet in window.text or window.text in normalized_snippet
        )
        union = snippet_tokens | window.tokens
        similarity = (
            0.0 if not union else len(snippet_tokens & window.tokens) / len(union)
        )
        if contained:
            containment = len(snippet_tokens & window.tokens) / smaller_size
            similarity = max(similarity, containment)
        if similarity > best_similarity:
            best_similarity = similarity
            best_start = window.start
            best_end = window.end
            best_contained = contained

    rounded = round(best_similarity, 6)
    if best_contained:
        kind = EvidenceLinkKind.CONTAINED_WINDOW
    elif rounded >= 0.75:
        kind = EvidenceLinkKind.STRONG_LEXICAL
    elif rounded >= 0.4:
        kind = EvidenceLinkKind.WEAK_LEXICAL
    else:
        return _EvidenceLink(EvidenceLinkKind.UNVERIFIED, rounded, None, None)
    return _EvidenceLink(kind, rounded, best_start, best_end)


def _sensitive_field_kinds(
    row: Mapping[str, Any],
    sensitive_information: Mapping[str, Any],
) -> tuple[str, ...]:
    searchable = _normalized_search_text(
        " ".join(
            str(row.get(field) or "")
            for field in (
                "correct_answer",
                "preference",
                "prev_pref",
                "related_conversation_snippet",
            )
        )
    )
    matched: set[str] = set()
    for raw_kind, raw_value in sensitive_information.items():
        value = _normalized_search_text(str(raw_value or ""))
        if not value or value not in searchable:
            continue
        kind = _category(str(raw_kind))
        if kind.startswith("code_file_path") or kind.startswith("user_code_file"):
            kind = "code_file_path"
        else:
            kind = _SENSITIVE_KIND_ALIASES.get(kind, kind)
        matched.add(kind)
    return tuple(sorted(matched))


def _incorrect_answers(
    value: Any,
    correct_answer: str,
    rejection_reasons: set[str],
) -> tuple[str, ...]:
    parsed: Any = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            rejection_reasons.add("invalid_incorrect_answers_json")
            return ()
    if not isinstance(parsed, list):
        rejection_reasons.add("invalid_incorrect_answers")
        return ()

    result: list[str] = []
    for item in parsed:
        if not isinstance(item, str) or not item.strip():
            rejection_reasons.add("invalid_incorrect_option")
            continue
        result.append(item.strip())
    normalized = [item.casefold() for item in result]
    if len(normalized) != len(set(normalized)):
        rejection_reasons.add("duplicate_incorrect_options")
    if correct_answer and correct_answer.strip().casefold() in set(normalized):
        rejection_reasons.add("correct_answer_in_incorrect_options")
    if len(result) < 1:
        rejection_reasons.add("missing_incorrect_options")
    return tuple(result)


def _candidate_text(value: Any, reason: str, issues: set[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        issues.add(reason)
        return ""
    return value.strip()


def _validated_messages(value: Any, file_name: str) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise CandidateInventoryError(f"{file_name} has an empty chat history")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise CandidateInventoryError(
                f"{file_name} chat message {index} is not an object"
            )
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise CandidateInventoryError(
                f"{file_name} chat message {index} has unsupported role {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise CandidateInventoryError(
                f"{file_name} chat message {index} has empty content"
            )
        messages.append({"role": role, "content": content})
    return tuple(messages)


def _read_export(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateInventoryError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CandidateInventoryError(f"raw export is not a JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateInventoryError(f"raw export field {field} must be an object")
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateInventoryError(f"raw export field {field} must be text")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if not _SHA256_PATTERN.fullmatch(text):
        raise CandidateInventoryError(f"raw export field {field} must be SHA-256")
    return text


def _persona_id(value: Any, source: str) -> str:
    if isinstance(value, bool):
        raise CandidateInventoryError(f"{source} has an invalid persona ID")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateInventoryError(f"{source} has an invalid persona ID") from exc
    if parsed < 0 or str(value).strip() != str(parsed):
        raise CandidateInventoryError(f"{source} has an invalid persona ID")
    return str(parsed)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandidateInventoryError(
            f"benchmark field {field} must be a non-negative integer"
        )
    return value


def _strict_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise CandidateInventoryError(f"benchmark field {field} must be Boolean")
    return value


def _category(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = unicodedata.normalize("NFKD", value.strip()).encode(
        "ascii", "ignore"
    ).decode()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    if not normalized:
        return "unknown"
    if not normalized[0].isalpha():
        normalized = f"value_{normalized}"
    return normalized


def _normalized_search_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError as exc:
        raise CandidateInventoryError(f"could not write {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize PersonaMem-v2 raw exports for candidate selection."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RAW_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INVENTORY_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inventory = normalize_candidate_inventory(args.input_dir)
    records_path, manifest_path = export_candidate_inventory(
        inventory,
        args.output_dir,
        overwrite=args.overwrite,
    )
    manifest = inventory.manifest
    print(
        f"Normalized {manifest.record_count} candidates across "
        f"{manifest.persona_count} personas "
        f"({manifest.structurally_eligible_count} structurally eligible)."
    )
    print(f"Records: {records_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
