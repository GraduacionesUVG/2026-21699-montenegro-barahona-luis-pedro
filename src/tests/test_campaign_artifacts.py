import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from membench.campaign.artifacts import (
    ArtifactExistsError,
    ArtifactIntegrityError,
    ArtifactPathError,
    ArtifactSealedError,
    CampaignArtifactError,
    CampaignArtifactStore,
)
from membench.campaign.models import ArtifactKind, ArtifactVisibility


def test_store_writes_canonical_json_and_verifies_its_reference(
    tmp_path: Path,
) -> None:
    store = CampaignArtifactStore.prepare(tmp_path / "campaign-001")

    reference = store.write_json(
        artifact_id="manifest:run-001",
        kind=ArtifactKind.RESOLVED_MANIFEST,
        relative_path="children/run-001/resolved_manifest.json",
        value={"z": 2, "a": 1},
    )

    path = store.verify(reference)
    assert path.read_bytes() == b'{"a":1,"z":2}\n'
    assert reference.relative_path == (
        "children/run-001/resolved_manifest.json"
    )
    assert reference.size_bytes == len(path.read_bytes())
    assert reference.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.references() == (reference,)


def test_jsonl_writer_is_append_only_and_seal_writes_complete_inventory(
    tmp_path: Path,
) -> None:
    store = CampaignArtifactStore.prepare(tmp_path / "campaign-002")
    writer = store.open_jsonl(
        artifact_id="events:run-001",
        kind=ArtifactKind.EVENTS,
        relative_path="children/run-001/events.jsonl",
    )
    writer.append({"sequence": 1, "at": datetime(2026, 9, 28, tzinfo=UTC)})
    writer.append({"sequence": 2})
    events_reference = writer.close()

    assert writer.close() == events_reference
    with pytest.raises(ArtifactSealedError, match="closed"):
        writer.append({"sequence": 3})

    index = store.seal()

    assert index.checksums.kind == ArtifactKind.CHECKSUMS
    assert index.artifacts == store.references()
    assert index.reference_for("events:run-001") == events_reference
    checksum_document = json.loads(store.verify(index.checksums).read_text())
    assert checksum_document["algorithm"] == "sha256"
    assert checksum_document["artifacts"] == [
        events_reference.model_dump(mode="json")
    ]
    with pytest.raises(ArtifactSealedError, match="sealed"):
        store.write_json(
            artifact_id="too-late",
            kind=ArtifactKind.OTHER,
            relative_path="too-late.json",
            value={},
        )


def test_store_detects_tampering_open_writers_and_unregistered_files(
    tmp_path: Path,
) -> None:
    store = CampaignArtifactStore.prepare(tmp_path / "campaign-003")
    reference = store.write_json(
        artifact_id="result:run-001",
        kind=ArtifactKind.CHILD_RESULT,
        relative_path="children/run-001/result.json",
        value={"status": "succeeded"},
    )
    store.verify(reference).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        store.verify(reference)
    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        store.seal()

    other = CampaignArtifactStore.prepare(tmp_path / "campaign-004")
    writer = other.open_jsonl(
        artifact_id="events:run-002",
        kind=ArtifactKind.EVENTS,
        relative_path="children/run-002/events.jsonl",
    )
    with pytest.raises(CampaignArtifactError, match="remain open"):
        other.seal()
    writer.abort()
    (other.root / "unregistered.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(CampaignArtifactError, match="unregistered"):
        other.seal()


def test_copy_and_existing_registration_preserve_exact_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b'{"persona_id":"p-001"}\n')
    store = CampaignArtifactStore.prepare(tmp_path / "campaign-005")

    copied = store.copy_file(
        artifact_id="dataset-001",
        kind=ArtifactKind.DATASET_PROJECTION,
        relative_path="inputs/personas.jsonl",
        source=source,
        media_type="application/x-ndjson",
        visibility=ArtifactVisibility.EVALUATOR_ONLY,
    )
    external_path = store.prepare_directory("children/run-001") / "result.json"
    external_path.write_bytes(b'{"status":"failed"}\n')
    registered = store.register_existing(
        artifact_id="child-result:run-001",
        kind=ArtifactKind.CHILD_RESULT,
        relative_path="children/run-001/result.json",
        media_type="application/json",
        expected_sha256=hashlib.sha256(external_path.read_bytes()).hexdigest(),
        expected_size_bytes=external_path.stat().st_size,
    )

    assert store.verify(copied).read_bytes() == source.read_bytes()
    assert store.verify(registered) == external_path.resolve()

    with pytest.raises(ArtifactExistsError, match="already registered"):
        store.write_json(
            artifact_id="dataset-001",
            kind=ArtifactKind.OTHER,
            relative_path="other.json",
            value={},
        )


@pytest.mark.parametrize(
    "relative_path",
    ("../escape.json", "/absolute.json", "C:/drive.json", "a//b.json"),
)
def test_store_rejects_nonportable_or_noncanonical_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    store = CampaignArtifactStore.prepare(
        tmp_path / f"campaign-{abs(hash(relative_path))}"
    )

    with pytest.raises(ArtifactPathError):
        store.write_json(
            artifact_id="unsafe",
            kind=ArtifactKind.OTHER,
            relative_path=relative_path,
            value={},
        )


def test_atomic_writes_use_short_destination_independent_temp_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str]] = []
    original = tempfile.NamedTemporaryFile

    def recording_named_temporary_file(*args, **kwargs):
        observed.append((kwargs["prefix"], kwargs["suffix"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "membench.campaign.artifacts.tempfile.NamedTemporaryFile",
        recording_named_temporary_file,
    )
    store = CampaignArtifactStore.prepare(tmp_path / "short-temp-names")
    long_name = f"{'marker-matches-' + ('x' * 96)}.json"
    reference = store.write_json(
        artifact_id="evaluation:marker-matches",
        kind=ArtifactKind.EVALUATION_RESULT,
        relative_path=f"evaluation/probe/{long_name}",
        value={"matched": False},
    )
    source = tmp_path / "source.json"
    source.write_bytes(b"{}\n")
    copied = store.copy_file(
        artifact_id="evaluation:copied",
        kind=ArtifactKind.OTHER,
        relative_path="evaluation/probe/copied.json",
        source=source,
        media_type="application/json",
    )

    assert store.verify(reference).is_file()
    assert store.verify(copied).read_bytes() == source.read_bytes()
    assert observed == [(".tmp-", ".part"), (".tmp-", ".part")]
    assert all(long_name not in prefix for prefix, _ in observed)
