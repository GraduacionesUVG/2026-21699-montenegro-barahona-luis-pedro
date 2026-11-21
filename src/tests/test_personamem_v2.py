import hashlib
import json
from pathlib import Path

import pytest

from membench.personamem_v2 import (
    PersonaMemDataError,
    PersonaMemV2Dataset,
    PersonaMemV2Settings,
)

REVISION = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_projection(
    root: Path, *, public_metadata: dict[str, object] | None = None
) -> PersonaMemV2Settings:
    subset = "blackbox_battery_v1"
    subset_dir = root / "subsets" / subset
    subset_dir.mkdir(parents=True)
    events_path = subset_dir / "replay_events.jsonl"
    probes_path = subset_dir / "probes.jsonl"
    event = {
        "schema_version": 1,
        "dataset_revision": REVISION,
        "subset": subset,
        "source_record_id": "benchmark-row-1",
        "history_unit_id": "history-001",
        "persona_id": "persona-001",
        "event_id": "event-002",
        "sequence": 2,
        "messages": [
            {"role": "user", "content": "Could you suggest a notebook?"},
            {"role": "assistant", "content": "Certainly."},
        ],
        "allowed_for_memory": True,
        "public_metadata": public_metadata or {"source": "text-benchmark"},
    }
    earlier = {
        **event,
        "event_id": "event-001",
        "sequence": 1,
        "messages": [
            {"role": "user", "content": "Blue works well with my desk."},
            {"role": "assistant", "content": "That sounds coordinated."},
        ],
    }
    probe = {
        "schema_version": 1,
        "dataset_revision": REVISION,
        "subset": subset,
        "source_record_id": "benchmark-row-1",
        "history_unit_id": "history-001",
        "persona_id": "persona-001",
        "event_id": "probe-001",
        "query": "Which notebook best suits me?",
        "options": ["A blue notebook", "A red notebook"],
        "allowed_for_memory": False,
    }
    events_path.write_text(
        "\n".join(json.dumps(item) for item in (event, earlier)) + "\n",
        encoding="utf-8",
    )
    probes_path.write_text(json.dumps(probe) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset": "personamem_v2",
        "revision": REVISION,
        "subset": subset,
        "replay_events_sha256": _sha256(events_path),
        "probes_sha256": _sha256(probes_path),
    }
    (subset_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return PersonaMemV2Settings(
        root=root,
        subset=subset,
        expected_revision=REVISION,
    )


def test_loader_verifies_projection_and_restores_event_order(tmp_path: Path) -> None:
    settings = _write_projection(tmp_path)

    dataset = PersonaMemV2Dataset.load(settings)
    case = dataset.case("history-001")

    assert dataset.manifest.revision == REVISION
    assert [event.event_id for event in case.replay_events] == [
        "event-001",
        "event-002",
    ]
    assert not case.probe.allowed_for_memory
    assert case.probe.options == ("A blue notebook", "A red notebook")


def test_loader_rejects_private_label_leakage(tmp_path: Path) -> None:
    settings = _write_projection(
        tmp_path, public_metadata={"nested": {"correct_answer": "private"}}
    )

    with pytest.raises(PersonaMemDataError, match="evaluator-only"):
        PersonaMemV2Dataset.load(settings)


def test_loader_rejects_tampered_projection(tmp_path: Path) -> None:
    settings = _write_projection(tmp_path)
    settings.replay_events_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(PersonaMemDataError, match="integrity check failed"):
        PersonaMemV2Dataset.load(settings)
