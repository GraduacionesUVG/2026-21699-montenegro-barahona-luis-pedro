import hashlib
import json
from pathlib import Path

import pytest

from membench.data_process.candidate_inventory import (
    CandidateInventoryError,
    EvidenceLinkKind,
    export_candidate_inventory,
    load_candidate_inventory,
    normalize_candidate_inventory,
)
from membench.data_process.data_processing import DATASET_ID

REVISION = "a" * 40


def _row(
    persona_id: int,
    source_row_index: int,
    *,
    incorrect_answers: str | None = None,
    who: str = "self",
    topic: str = "Sensitive Info",
) -> dict[str, object]:
    return {
        "persona_id": persona_id,
        "source_row_index": source_row_index,
        "topic_preference": topic,
        "pref_type": "sensitive_info",
        "conversation_scenario": "Professional Email",
        "who": who,
        "updated": False,
        "sensitive_info": True,
        "distance_from_related_snippet_to_query_32k": 120,
        "num_persona_relevant_tokens_32k": 24,
        "num_persona_irrelevant_tokens_32k": 600,
        "total_tokens_in_chat_history_32k": 624,
        "user_query": "Which private email should be used?",
        "correct_answer": "Use correct@example.invalid.",
        "incorrect_answers": incorrect_answers
        or json.dumps(
            [
                "Use wrong-one@example.invalid.",
                "Use wrong-two@example.invalid.",
                "No email is available.",
            ]
        ),
        "related_conversation_snippet": (
            "My preferred email is correct@example.invalid."
        ),
        "preference": "correct@example.invalid",
        "prev_pref": None,
    }


def _export_document(
    persona_id: int,
    source_row_index: int,
    *,
    row: dict[str, object] | None = None,
    revision: str = REVISION,
) -> dict[str, object]:
    candidate = row or _row(persona_id, source_row_index)
    return {
        "schema_version": 1,
        "dataset": {"id": DATASET_ID, "revision": revision},
        "persona_id": persona_id,
        "source_files": {
            "raw_persona_sha256": "1" * 64,
            "chat_history_32k_sha256": "2" * 64,
        },
        "raw_persona": {
            str(persona_id): {
                "sensitive_information": {
                    "email_address": "correct@example.invalid",
                    "phone_number": "+1-555-0101",
                }
            }
        },
        "chat_history_32k": {
            "metadata": {"persona_id": persona_id},
            "chat_history": [
                {"role": "system", "content": "Synthetic fixture."},
                {
                    "role": "user",
                    "content": "My preferred email is correct@example.invalid.",
                },
                {"role": "assistant", "content": "Understood."},
            ],
        },
        "benchmark_records": [candidate],
    }


def _write_export(
    directory: Path,
    persona_id: int,
    source_row_index: int,
    *,
    document: dict[str, object] | None = None,
) -> Path:
    path = directory / f"persona_{persona_id}.json"
    value = document or _export_document(persona_id, source_row_index)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_normalizer_emits_features_and_hashes_without_private_text(
    tmp_path: Path,
) -> None:
    _write_export(tmp_path, 7, 12)

    inventory = normalize_candidate_inventory(tmp_path)
    record = inventory.records[0]
    rendered = inventory.records_jsonl().decode()

    assert record.candidate_id == "pmv2-p7-row000012"
    assert record.topic == "sensitive_info"
    assert record.preference_type == "sensitive_info"
    assert record.conversation_scenario == "professional_email"
    assert record.sensitive_field_kinds == ("email_address",)
    assert record.evidence_link_kind == EvidenceLinkKind.EXACT_MESSAGE
    assert record.evidence_similarity == 1.0
    assert record.structurally_eligible is True
    assert record.history_message_count == 3
    assert record.history_system_message_count == 1
    assert record.incorrect_option_count == 3
    assert "correct@example.invalid" not in rendered
    assert "Which private email" not in rendered
    assert "wrong-one@example.invalid" not in rendered


def test_row_level_quality_failures_are_retained_as_reason_codes(
    tmp_path: Path,
) -> None:
    row = _row(
        8,
        13,
        incorrect_answers="not-json",
        who="ambiguous",
        topic="",
    )
    _write_export(
        tmp_path,
        8,
        13,
        document=_export_document(8, 13, row=row),
    )

    record = normalize_candidate_inventory(tmp_path).records[0]

    assert record.structurally_eligible is False
    assert record.rejection_reasons == (
        "invalid_incorrect_answers_json",
        "missing_topic",
        "unknown_owner",
    )


def test_inventory_and_manifest_are_deterministic_and_integrity_checked(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    _write_export(inputs, 20, 200)
    _write_export(inputs, 3, 30)

    first = normalize_candidate_inventory(inputs)
    second = normalize_candidate_inventory(inputs)
    records_path, manifest_path = export_candidate_inventory(first, outputs)
    loaded = load_candidate_inventory(outputs)

    assert first.records == second.records
    assert loaded == first
    assert first.manifest == second.manifest
    assert [item.candidate_id for item in first.records] == [
        "pmv2-p3-row000030",
        "pmv2-p20-row000200",
    ]
    assert first.manifest.record_count == 2
    assert first.manifest.persona_count == 2
    assert first.manifest.structurally_eligible_count == 2
    assert hashlib.sha256(records_path.read_bytes()).hexdigest() == (
        first.manifest.records_sha256
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))[
        "records_sha256"
    ] == first.manifest.records_sha256
    with pytest.raises(CandidateInventoryError, match="without --overwrite"):
        export_candidate_inventory(first, outputs)

    records_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CandidateInventoryError, match="integrity check failed"):
        load_candidate_inventory(outputs)


def test_normalizer_rejects_cross_persona_and_cross_revision_inputs(
    tmp_path: Path,
) -> None:
    mismatch = _export_document(4, 40)
    mismatch["chat_history_32k"]["metadata"]["persona_id"] = 99
    _write_export(tmp_path, 4, 40, document=mismatch)

    with pytest.raises(CandidateInventoryError, match="another persona"):
        normalize_candidate_inventory(tmp_path)

    (tmp_path / "persona_4.json").unlink()
    _write_export(tmp_path, 4, 40)
    _write_export(
        tmp_path,
        5,
        50,
        document=_export_document(5, 50, revision="b" * 40),
    )
    with pytest.raises(CandidateInventoryError, match="multiple dataset revisions"):
        normalize_candidate_inventory(tmp_path)
