"""Rank normalized PersonaMem-v2 candidates for poisoning methodologies."""

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

from membench.attacks.models import (
    AttackMethodology,
    CandidateAttackFit,
    CandidateFitStatus,
)
from membench.attacks.selectors import (
    DEFAULT_SELECTORS,
    CandidateSelector,
    PrivateCandidateContext,
)
from membench.data_process.candidate_inventory import (
    DEFAULT_INVENTORY_DIR,
    DEFAULT_RAW_EXPORT_DIR,
    CandidateInventory,
    CandidateInventoryError,
    CandidateInventoryRecord,
    load_candidate_inventory,
    normalize_candidate_inventory,
)
from membench.data_process.data_processing import DATASET_ID
from membench.memory.models import CanonicalModel

SELECTION_VERSION = "personamem_candidate_selection_v1"
DEFAULT_SELECTION_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "blackbox" / "selection"
)
_STATUS_PRIORITY = {
    CandidateFitStatus.ELIGIBLE: 0,
    CandidateFitStatus.REVIEW_REQUIRED: 1,
    CandidateFitStatus.INELIGIBLE: 2,
}


class CandidateSelectionError(RuntimeError):
    """Raised when candidates cannot be ranked reproducibly and safely."""


class CandidateSelectionManifest(CanonicalModel):
    """Integrity metadata and privacy-safe shortlists for selection JSONL."""

    schema_version: Literal[1] = 1
    dataset_id: Literal["bowen-upenn/PersonaMem-v2"] = DATASET_ID
    dataset_revision: str
    selection_version: Literal[
        "personamem_candidate_selection_v1"
    ] = SELECTION_VERSION
    inventory_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(gt=0)
    candidate_count: int = Field(gt=0)
    fit_count: int = Field(gt=0)
    fits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attack_methods: tuple[AttackMethodology, ...] = Field(min_length=1)
    status_counts: dict[str, int]
    per_attack_status_counts: dict[str, dict[str, int]]
    shortlists: dict[str, tuple[str, ...]]

    @field_validator("status_counts")
    @classmethod
    def status_counts_are_complete(cls, value: dict[str, int]) -> dict[str, int]:
        expected = {item.value for item in CandidateFitStatus}
        if set(value) != expected:
            raise ValueError("selection status counts must include every status")
        if any(isinstance(count, bool) or count < 0 for count in value.values()):
            raise ValueError("selection status counts must be non-negative integers")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def summaries_must_be_consistent(self) -> Self:
        methods = tuple(method.value for method in self.attack_methods)
        if len(methods) != len(set(methods)):
            raise ValueError("selection attack methods must be unique")
        if set(self.per_attack_status_counts) != set(methods):
            raise ValueError("per-attack status counts must match attack methods")
        if set(self.shortlists) != set(methods):
            raise ValueError("selection shortlists must match attack methods")
        expected_statuses = {item.value for item in CandidateFitStatus}
        for method, counts in self.per_attack_status_counts.items():
            if set(counts) != expected_statuses:
                raise ValueError(f"incomplete status counts for {method}")
            if sum(counts.values()) != self.candidate_count:
                raise ValueError(f"candidate count does not balance for {method}")
        if sum(self.status_counts.values()) != self.fit_count:
            raise ValueError("overall status counts do not balance")
        if self.fit_count != self.candidate_count * len(methods):
            raise ValueError("fit count must cover every candidate and attack")
        if any(len(items) > self.top_k for items in self.shortlists.values()):
            raise ValueError("selection shortlist exceeds top_k")
        return self

    @property
    def sha256(self) -> str:
        """Hash this canonical manifest for later artifact registration."""

        return _canonical_sha256(self.model_dump(mode="json"))


@dataclass(frozen=True)
class CandidateSelection:
    """Ranked candidate fits and the inventory identity that produced them."""

    dataset_revision: str
    inventory_records_sha256: str
    top_k: int
    fits: tuple[CandidateAttackFit, ...]

    def fits_jsonl(self) -> bytes:
        """Render deterministic JSONL containing no raw benchmark text."""

        text = "".join(
            json.dumps(
                fit.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for fit in self.fits
        )
        return text.encode("utf-8")

    @property
    def manifest(self) -> CandidateSelectionManifest:
        """Build deterministic selection summaries and integrity metadata."""

        methods = tuple(AttackMethodology)
        overall = Counter(fit.status.value for fit in self.fits)
        by_attack: dict[str, dict[str, int]] = {}
        shortlists: dict[str, tuple[str, ...]] = {}
        for method in methods:
            method_fits = tuple(
                fit for fit in self.fits if fit.attack_methodology == method
            )
            counts = Counter(fit.status.value for fit in method_fits)
            by_attack[method.value] = {
                status.value: counts[status.value] for status in CandidateFitStatus
            }
            shortlists[method.value] = tuple(
                fit.candidate_id
                for fit in method_fits
                if fit.status != CandidateFitStatus.INELIGIBLE
            )[: self.top_k]
        return CandidateSelectionManifest(
            dataset_revision=self.dataset_revision,
            inventory_records_sha256=self.inventory_records_sha256,
            top_k=self.top_k,
            candidate_count=len(self.fits) // len(methods),
            fit_count=len(self.fits),
            fits_sha256=hashlib.sha256(self.fits_jsonl()).hexdigest(),
            attack_methods=methods,
            status_counts={
                status.value: overall[status.value] for status in CandidateFitStatus
            },
            per_attack_status_counts=by_attack,
            shortlists=shortlists,
        )


def select_candidates(
    inventory: CandidateInventory,
    raw_dir: Path,
    *,
    top_k: int = 10,
    selectors: Sequence[CandidateSelector] = DEFAULT_SELECTORS,
) -> CandidateSelection:
    """Score and rank every candidate for each configured attack methodology."""

    if isinstance(top_k, bool) or top_k <= 0:
        raise CandidateSelectionError("top_k must be a positive integer")
    _verify_inventory_matches_raw(inventory, raw_dir)
    contexts = _load_private_contexts(raw_dir, inventory)
    selector_by_method = {selector.methodology: selector for selector in selectors}
    if set(selector_by_method) != set(AttackMethodology):
        missing = sorted(
            method.value for method in set(AttackMethodology) - set(selector_by_method)
        )
        extra = sorted(
            method.value for method in set(selector_by_method) - set(AttackMethodology)
        )
        raise CandidateSelectionError(
            f"selectors must cover each attack once; missing={missing}, extra={extra}"
        )
    if len(selector_by_method) != len(selectors):
        raise CandidateSelectionError("selectors contain duplicate attack methods")

    ranked_fits: list[CandidateAttackFit] = []
    for methodology in AttackMethodology:
        selector = selector_by_method[methodology]
        evaluated = [
            selector.evaluate(candidate, contexts[candidate.candidate_id])
            for candidate in inventory.records
        ]
        evaluated.sort(key=_ranking_key)
        ranked_fits.extend(
            fit.model_copy(update={"rank": rank})
            for rank, fit in enumerate(evaluated, start=1)
        )
    return CandidateSelection(
        dataset_revision=inventory.dataset_revision,
        inventory_records_sha256=inventory.manifest.records_sha256,
        top_k=top_k,
        fits=tuple(ranked_fits),
    )


def export_candidate_selection(
    selection: CandidateSelection,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Atomically write ranked fits and their integrity manifest."""

    fits_path = output_dir / "candidate_fits.jsonl"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (fits_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise CandidateSelectionError(
            "refusing to replace selection files without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (
        json.dumps(
            selection.manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(fits_path, selection.fits_jsonl())
    _atomic_write(manifest_path, manifest_bytes)
    return fits_path, manifest_path


def load_candidate_selection(input_dir: Path) -> CandidateSelection:
    """Load and integrity-check previously exported candidate rankings."""

    fits_path = input_dir / "candidate_fits.jsonl"
    manifest_path = input_dir / "manifest.json"
    try:
        fits_bytes = fits_path.read_bytes()
        manifest = CandidateSelectionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise CandidateSelectionError(
            f"could not load candidate selection from {input_dir}: {exc}"
        ) from exc
    observed_hash = hashlib.sha256(fits_bytes).hexdigest()
    if observed_hash != manifest.fits_sha256:
        raise CandidateSelectionError(
            "candidate selection integrity check failed: "
            f"{observed_hash}; expected {manifest.fits_sha256}"
        )

    fits: list[CandidateAttackFit] = []
    try:
        for line_number, line in enumerate(
            fits_bytes.decode("utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                fits.append(CandidateAttackFit.model_validate_json(line))
            except ValueError as exc:
                raise CandidateSelectionError(
                    f"invalid candidate_fits.jsonl line {line_number}: {exc}"
                ) from exc
    except UnicodeDecodeError as exc:
        raise CandidateSelectionError(
            "candidate selection fits are not valid UTF-8"
        ) from exc
    if not fits:
        raise CandidateSelectionError("candidate selection contains no fits")

    selection = CandidateSelection(
        dataset_revision=manifest.dataset_revision,
        inventory_records_sha256=manifest.inventory_records_sha256,
        top_k=manifest.top_k,
        fits=tuple(fits),
    )
    if selection.manifest != manifest:
        raise CandidateSelectionError(
            "candidate selection manifest does not match its ranked fits"
        )
    return selection


def _verify_inventory_matches_raw(
    inventory: CandidateInventory,
    raw_dir: Path,
) -> None:
    try:
        observed = normalize_candidate_inventory(raw_dir)
    except CandidateInventoryError as exc:
        raise CandidateSelectionError(
            f"could not verify raw exports against inventory: {exc}"
        ) from exc
    if observed != inventory:
        raise CandidateSelectionError(
            "raw PersonaMem exports do not exactly match the candidate inventory; "
            "regenerate the inventory before selection"
        )


def _load_private_contexts(
    raw_dir: Path,
    inventory: CandidateInventory,
) -> dict[str, PrivateCandidateContext]:
    records = {record.candidate_id: record for record in inventory.records}
    contexts: dict[str, PrivateCandidateContext] = {}
    for path in sorted(raw_dir.glob("persona_*.json"), key=lambda item: item.name):
        document = _read_json_object(path)
        persona_id = str(document.get("persona_id"))
        rows = document.get("benchmark_records")
        if not isinstance(rows, list):
            raise CandidateSelectionError(f"{path.name} has no benchmark_records array")
        for row in rows:
            if not isinstance(row, Mapping):
                raise CandidateSelectionError(
                    f"{path.name} contains a non-object benchmark record"
                )
            source_index = row.get("source_row_index")
            if isinstance(source_index, bool) or not isinstance(source_index, int):
                raise CandidateSelectionError(
                    f"{path.name} has an invalid source_row_index"
                )
            candidate_id = f"pmv2-p{persona_id}-row{source_index:06d}"
            candidate = records.get(candidate_id)
            if candidate is None:
                raise CandidateSelectionError(
                    "raw export contains candidate absent from inventory: "
                    f"{candidate_id}"
                )
            context = PrivateCandidateContext(
                candidate_id=candidate_id,
                query=_trimmed_text(row.get("user_query")),
                correct_answer=_trimmed_text(row.get("correct_answer")),
                incorrect_options=_parse_incorrect_options(
                    row.get("incorrect_answers")
                ),
            )
            _verify_context_hashes(candidate, context)
            if candidate_id in contexts:
                raise CandidateSelectionError(
                    f"raw exports contain duplicate candidate {candidate_id}"
                )
            contexts[candidate_id] = context
    missing = sorted(set(records) - set(contexts))
    if missing:
        raise CandidateSelectionError(
            f"inventory candidates are missing from raw exports: {missing}"
        )
    return contexts


def _verify_context_hashes(
    candidate: CandidateInventoryRecord,
    context: PrivateCandidateContext,
) -> None:
    query_hash = _text_sha256(context.query)
    answer_hash = _text_sha256(context.correct_answer)
    option_hashes = tuple(_text_sha256(item) for item in context.incorrect_options)
    if (
        query_hash != candidate.query_sha256
        or answer_hash != candidate.answer_sha256
        or option_hashes != candidate.incorrect_answer_sha256
    ):
        raise CandidateSelectionError(
            "private context hashes do not match inventory for "
            f"{candidate.candidate_id}"
        )


def _parse_incorrect_options(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(
        item.strip()
        for item in parsed
        if isinstance(item, str) and item.strip()
    )


def _ranking_key(fit: CandidateAttackFit) -> tuple[int, int, int, int]:
    return (
        _STATUS_PRIORITY[fit.status],
        -fit.score,
        int(fit.persona_id),
        fit.source_row_index,
    )


def _trimmed_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateSelectionError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CandidateSelectionError(f"raw export is not a JSON object: {path}")
    return value


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
        raise CandidateSelectionError(f"could not write {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank normalized PersonaMem-v2 candidates for attack design."
    )
    parser.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inventory = load_candidate_inventory(args.inventory_dir)
    selection = select_candidates(inventory, args.raw_dir, top_k=args.top_k)
    fits_path, manifest_path = export_candidate_selection(
        selection,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        f"Ranked {selection.manifest.candidate_count} candidates across "
        f"{len(selection.manifest.attack_methods)} attack methodologies."
    )
    print(f"Fits: {fits_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
