"""Select and export a small, reproducible PersonaMem-v2 raw-data sample.

The exported documents are evaluator-only research data. They intentionally
contain upstream profile data, benchmark labels, and synthetic sensitive
information, so they must never be passed wholesale to a tested agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DATASET_ID = "bowen-upenn/PersonaMem-v2"
# Pin the source instead of following the moving ``main`` branch.
DEFAULT_REVISION = "ed956dea41521fc4499acbc63f966e0fd3c053ba"
BENCHMARK_FILE = "benchmark/text/benchmark.csv"
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "raw_data"


class PersonaMemProcessingError(RuntimeError):
    """Raised when the upstream data cannot produce a valid, linked export."""


@dataclass(frozen=True)
class PersonaSelection:
    """All benchmark data and source links associated with one selected persona."""

    persona_id: int
    raw_persona_file: str
    chat_history_32k_file: str
    benchmark_records: tuple[dict[str, Any], ...]
    has_sensitive_info: bool
    has_security_topic_preference: bool

    @property
    def selection_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.has_sensitive_info:
            reasons.append("sensitive_info")
        if self.has_security_topic_preference:
            reasons.append("security_topic_preference")
        return tuple(reasons)


@dataclass(frozen=True)
class DownloadedJson:
    """Parsed source JSON together with the hash of its original bytes."""

    payload: Any
    sha256: str


JsonDownloader = Callable[[str, str], DownloadedJson]


def _parse_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", "", "none", "null"}:
            return False
    raise PersonaMemProcessingError(f"invalid sensitive_info value: {value!r}")


def _is_security_topic(value: Any) -> bool:
    return isinstance(value, str) and "security" in value.casefold()


def _persona_id(value: Any) -> int:
    if isinstance(value, bool):
        raise PersonaMemProcessingError("persona_id cannot be Boolean")
    try:
        persona_id = int(value)
    except (TypeError, ValueError) as exc:
        raise PersonaMemProcessingError(f"invalid persona_id: {value!r}") from exc
    if persona_id < 0 or str(value).strip() != str(persona_id):
        raise PersonaMemProcessingError(f"invalid persona_id: {value!r}")
    return persona_id


def _validated_source_path(value: Any, expected_directory: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PersonaMemProcessingError(
            f"missing PersonaMem-v2 source path for {expected_directory}"
        )
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    expected_prefix = ("data", expected_directory)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:2] != expected_prefix
        or path.suffix.casefold() != ".json"
    ):
        raise PersonaMemProcessingError(
            f"unexpected PersonaMem-v2 source path: {value!r}"
        )
    return path.as_posix()


def _single_source_path(
    persona_id: int,
    records: Sequence[Mapping[str, Any]],
    field: str,
    expected_directory: str,
) -> str:
    paths = {
        _validated_source_path(record.get(field), expected_directory)
        for record in records
    }
    if len(paths) != 1:
        raise PersonaMemProcessingError(
            f"persona {persona_id} has inconsistent {field} values: {sorted(paths)}"
        )
    return next(iter(paths))


def _seeded_order(persona_ids: Iterable[int], seed: int) -> list[int]:
    ordered = sorted(persona_ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def select_personas(
    records: Iterable[Mapping[str, Any]],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[PersonaSelection, ...]:
    """Select distinct qualifying personas from the text benchmark.

    A persona qualifies when at least one of its rows has ``sensitive_info`` set
    or a ``topic_preference`` containing ``security`` (case-insensitive). The
    seed only determines ordering after eligibility is calculated; it never
    weakens the filter.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PersonaMemProcessingError(
                f"benchmark row {source_index} is not an object"
            )
        if "persona_id" not in record:
            raise PersonaMemProcessingError(
                f"benchmark row {source_index} is missing persona_id"
            )
        persona_id = _persona_id(record["persona_id"])
        copied = dict(record)
        copied["source_row_index"] = source_index
        grouped[persona_id].append(copied)

    candidates: dict[int, PersonaSelection] = {}
    for persona_id, persona_records in grouped.items():
        has_sensitive_info = any(
            _parse_boolean(record.get("sensitive_info"))
            for record in persona_records
        )
        has_security_topic = any(
            _is_security_topic(record.get("topic_preference"))
            for record in persona_records
        )
        if not (has_sensitive_info or has_security_topic):
            continue

        candidates[persona_id] = PersonaSelection(
            persona_id=persona_id,
            raw_persona_file=_single_source_path(
                persona_id, persona_records, "raw_persona_file", "raw_data"
            ),
            chat_history_32k_file=_single_source_path(
                persona_id,
                persona_records,
                "chat_history_32k_link",
                "chat_history_32k",
            ),
            benchmark_records=tuple(persona_records),
            has_sensitive_info=has_sensitive_info,
            has_security_topic_preference=has_security_topic,
        )

    if len(candidates) < sample_size:
        raise PersonaMemProcessingError(
            f"requested {sample_size} personas, but only {len(candidates)} distinct "
            "personas matched sensitive_info or a security topic preference"
        )

    ordered_ids = _seeded_order(candidates, seed)
    return tuple(candidates[persona_id] for persona_id in ordered_ids[:sample_size])


def load_benchmark_records(
    *, revision: str = DEFAULT_REVISION
) -> Iterable[Mapping[str, Any]]:
    """Stream only the PersonaMem-v2 text benchmark at an immutable revision."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise PersonaMemProcessingError(
            "Hugging Face datasets is not installed; install the project's "
            "data-processing dependency group"
        ) from exc

    dataset = load_dataset(
        DATASET_ID,
        data_files={"benchmark_text": BENCHMARK_FILE},
        split="benchmark_text",
        revision=revision,
        streaming=True,
    )
    return dataset


def download_json(source_path: str, revision: str) -> DownloadedJson:
    """Download and parse one pinned JSON file from the dataset repository."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PersonaMemProcessingError(
            "huggingface_hub is unavailable; install the project's "
            "data-processing dependency group"
        ) from exc

    cached_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=source_path,
            repo_type="dataset",
            revision=revision,
        )
    )
    try:
        source_bytes = cached_path.read_bytes()
        return DownloadedJson(
            payload=json.loads(source_bytes),
            sha256=hashlib.sha256(source_bytes).hexdigest(),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PersonaMemProcessingError(
            f"could not read downloaded JSON {source_path}: {exc}"
        ) from exc


def _validate_downloaded_sources(
    selection: PersonaSelection,
    raw_persona: DownloadedJson,
    chat_history: DownloadedJson,
) -> None:
    persona_key = str(selection.persona_id)
    if (
        not isinstance(raw_persona.payload, Mapping)
        or persona_key not in raw_persona.payload
    ):
        raise PersonaMemProcessingError(
            f"raw persona file does not contain persona {selection.persona_id}"
        )

    if not isinstance(chat_history.payload, Mapping):
        raise PersonaMemProcessingError("32k chat history is not a JSON object")
    metadata = chat_history.payload.get("metadata")
    messages = chat_history.payload.get("chat_history")
    if not isinstance(metadata, Mapping):
        raise PersonaMemProcessingError("32k chat history has no metadata object")
    if _persona_id(metadata.get("persona_id")) != selection.persona_id:
        raise PersonaMemProcessingError(
            f"32k chat history does not belong to persona {selection.persona_id}"
        )
    if not isinstance(messages, list) or not messages:
        raise PersonaMemProcessingError("32k chat history has no messages")


def build_export_document(
    selection: PersonaSelection,
    *,
    revision: str,
    seed: int,
    raw_persona: DownloadedJson,
    chat_history: DownloadedJson,
) -> dict[str, Any]:
    """Build one self-contained persona document after checking its linkage."""

    _validate_downloaded_sources(selection, raw_persona, chat_history)
    matching_count = sum(
        _parse_boolean(record.get("sensitive_info"))
        or _is_security_topic(record.get("topic_preference"))
        for record in selection.benchmark_records
    )
    return {
        "schema_version": 1,
        "dataset": {
            "id": DATASET_ID,
            "revision": revision,
            "benchmark_file": BENCHMARK_FILE,
        },
        "persona_id": selection.persona_id,
        "selection": {
            "seed": seed,
            "criteria": "sensitive_info OR security in topic_preference",
            "reasons": list(selection.selection_reasons),
            "matching_benchmark_record_count": matching_count,
            "total_benchmark_record_count": len(selection.benchmark_records),
        },
        "source_files": {
            "raw_persona_file": selection.raw_persona_file,
            "raw_persona_sha256": raw_persona.sha256,
            "chat_history_32k_file": selection.chat_history_32k_file,
            "chat_history_32k_sha256": chat_history.sha256,
        },
        "usage_warning": (
            "Evaluator-only synthetic research data; do not provide raw profiles, "
            "sensitive fields, or answer labels directly to the tested agent."
        ),
        "raw_persona": raw_persona.payload,
        "chat_history_32k": chat_history.payload,
        "benchmark_records": list(selection.benchmark_records),
    }


def export_personas(
    selections: Sequence[PersonaSelection],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    revision: str = DEFAULT_REVISION,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
    downloader: JsonDownloader = download_json,
) -> tuple[Path, ...]:
    """Download linked sources and write one independent JSON per persona."""

    if len({item.persona_id for item in selections}) != len(selections):
        raise PersonaMemProcessingError("persona selections must be distinct")

    targets = [output_dir / f"persona_{item.persona_id}.json" for item in selections]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise PersonaMemProcessingError(
            "refusing to replace existing exports without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for selection, target in zip(selections, targets, strict=True):
        raw_persona = downloader(selection.raw_persona_file, revision)
        chat_history = downloader(selection.chat_history_32k_file, revision)
        document = build_export_document(
            selection,
            revision=revision,
            seed=seed,
            raw_persona=raw_persona,
            chat_history=chat_history,
        )
        temporary = target.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        except OSError as exc:
            raise PersonaMemProcessingError(f"could not write {target}: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        written.append(target)
    return tuple(written)


def process_personamem_v2(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    revision: str = DEFAULT_REVISION,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Run the pinned selection and export pipeline."""

    records = load_benchmark_records(revision=revision)
    selections = select_personas(records, sample_size=sample_size, seed=seed)
    return export_personas(
        selections,
        output_dir=output_dir,
        revision=revision,
        seed=seed,
        overwrite=overwrite,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export linked PersonaMem-v2 profile and 32k-context documents."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = process_personamem_v2(
        output_dir=args.output_dir,
        revision=args.revision,
        sample_size=args.sample_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Exported {len(paths)} distinct PersonaMem-v2 personas:")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
