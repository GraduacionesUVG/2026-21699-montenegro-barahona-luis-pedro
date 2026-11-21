import hashlib
import json
from pathlib import Path

import pytest

from membench.attacks.models import (
    AttackMethodology,
    CandidateFitStatus,
)
from membench.data_process.candidate_inventory import (
    export_candidate_inventory,
    load_candidate_inventory,
    normalize_candidate_inventory,
)
from membench.data_process.candidate_selection import (
    CandidateSelectionError,
    export_candidate_selection,
    load_candidate_selection,
    select_candidates,
)
from membench.data_process.data_processing import DATASET_ID

REVISION = "c" * 40


def _row(
    index: int,
    *,
    answer: str,
    snippet: str,
    topic: str = "Travel",
    preference_type: str = "neutral_preferences",
    scenario: str = "Chat Message",
    owner: str = "self",
    sensitive: bool = False,
    incorrect: list[str] | None = None,
) -> dict[str, object]:
    return {
        "persona_id": 7,
        "source_row_index": index,
        "topic_preference": topic,
        "pref_type": preference_type,
        "conversation_scenario": scenario,
        "who": owner,
        "updated": False,
        "sensitive_info": sensitive,
        "distance_from_related_snippet_to_query_32k": 512,
        "num_persona_relevant_tokens_32k": 50,
        "num_persona_irrelevant_tokens_32k": 1_000,
        "total_tokens_in_chat_history_32k": 1_050,
        "user_query": f"Private query marker {index}?",
        "correct_answer": answer,
        "incorrect_answers": json.dumps(
            incorrect
            or [
                f"Private wrong marker {index} alpha.",
                f"Private wrong marker {index} beta.",
                f"Private wrong marker {index} gamma.",
            ]
        ),
        "related_conversation_snippet": snippet,
        "preference": answer,
        "prev_pref": None,
    }


def _write_fixture(raw_dir: Path) -> list[dict[str, object]]:
    rows = [
        _row(
            0,
            answer="I prefer quiet coastal trips.",
            snippet="My trips should be quiet and coastal.",
        ),
        _row(
            1,
            answer="Use secure@example.invalid.",
            snippet="My address is secure@example.invalid.",
            topic="Sensitive Info",
            preference_type="sensitive_info",
            scenario="Professional Email",
            sensitive=True,
            incorrect=[
                "Use redirect@example.invalid.",
                "Use archive@example.invalid.",
                "No address is available.",
            ],
        ),
        _row(
            2,
            answer="My colleague prefers paper notes.",
            snippet="My colleague keeps paper notes.",
            topic="Work",
            owner="others",
        ),
        _row(
            3,
            answer="I prefer traveling by bicycle.",
            snippet="I usually travel by bicycle.",
        ),
        _row(
            4,
            answer="The preferred travel mode is bicycle.",
            snippet="Bicycle is my preferred travel mode.",
            incorrect=[
                "The preferred travel mode is train.",
                "The preferred travel mode is bus.",
                "No travel mode is preferred.",
            ],
        ),
        _row(
            5,
            answer="Draft a concise project update.",
            snippet="Project updates should be concise.",
            topic="Work",
            scenario="Professional Email",
        ),
    ]
    document = {
        "schema_version": 1,
        "dataset": {"id": DATASET_ID, "revision": REVISION},
        "persona_id": 7,
        "source_files": {
            "raw_persona_sha256": "1" * 64,
            "chat_history_32k_sha256": "2" * 64,
        },
        "raw_persona": {
            "7": {
                "sensitive_information": {
                    "email_address": "secure@example.invalid"
                }
            }
        },
        "chat_history_32k": {
            "metadata": {"persona_id": 7},
            "chat_history": [
                {"role": "system", "content": "Synthetic fixture."},
                *(
                    {"role": "user", "content": row["related_conversation_snippet"]}
                    for row in rows
                ),
            ],
        },
        "benchmark_records": rows,
    }
    raw_dir.mkdir()
    (raw_dir / "persona_7.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    return rows


def _selection(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    inventory_dir = tmp_path / "inventory"
    rows = _write_fixture(raw_dir)
    inventory = normalize_candidate_inventory(raw_dir)
    export_candidate_inventory(inventory, inventory_dir)
    loaded = load_candidate_inventory(inventory_dir)
    return rows, raw_dir, select_candidates(loaded, raw_dir, top_k=3)


def _fit(selection, method: AttackMethodology, row: int):
    return next(
        fit
        for fit in selection.fits
        if fit.attack_methodology == method and fit.source_row_index == row
    )


def test_selection_scores_each_attack_and_emits_no_private_text(
    tmp_path: Path,
) -> None:
    rows, _, selection = _selection(tmp_path)

    assert _fit(
        selection, AttackMethodology.ABRUPT_FALSE_CORRECTION, 0
    ).status == CandidateFitStatus.ELIGIBLE
    assert _fit(
        selection, AttackMethodology.NEGATED_IDENTIFIER, 1
    ).status == CandidateFitStatus.ELIGIBLE
    assert _fit(
        selection, AttackMethodology.OWNERSHIP_CONFUSION, 2
    ).status == CandidateFitStatus.ELIGIBLE
    assert _fit(
        selection, AttackMethodology.GRADUAL_DRIFT, 3
    ).status == CandidateFitStatus.ELIGIBLE
    assert _fit(
        selection, AttackMethodology.SEMANTIC_HIJACKING, 4
    ).status == CandidateFitStatus.ELIGIBLE
    instruction = _fit(
        selection, AttackMethodology.STORED_INSTRUCTION_INJECTION, 5
    )
    assert instruction.status == CandidateFitStatus.REVIEW_REQUIRED
    assert instruction.contract_compatible is False

    rendered = selection.fits_jsonl().decode()
    for row in rows:
        assert row["user_query"] not in rendered
        assert row["correct_answer"] not in rendered
        for option in json.loads(row["incorrect_answers"]):
            assert option not in rendered


def test_selection_is_deterministic_ranked_and_integrity_hashed(
    tmp_path: Path,
) -> None:
    _, raw_dir, first = _selection(tmp_path)
    inventory = normalize_candidate_inventory(raw_dir)
    second = select_candidates(inventory, raw_dir, top_k=3)
    output_dir = tmp_path / "selection"
    fits_path, manifest_path = export_candidate_selection(first, output_dir)
    loaded = load_candidate_selection(output_dir)

    assert first.fits == second.fits
    assert loaded == first
    assert first.manifest == second.manifest
    assert first.manifest.candidate_count == 6
    assert first.manifest.fit_count == 36
    assert all(
        len(shortlist) <= 3 for shortlist in first.manifest.shortlists.values()
    )
    assert hashlib.sha256(fits_path.read_bytes()).hexdigest() == (
        first.manifest.fits_sha256
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))[
        "fits_sha256"
    ] == first.manifest.fits_sha256
    for method in AttackMethodology:
        ranks = [
            fit.rank
            for fit in first.fits
            if fit.attack_methodology == method
        ]
        assert ranks == list(range(1, 7))
    with pytest.raises(CandidateSelectionError, match="without --overwrite"):
        export_candidate_selection(first, output_dir)

    fits_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CandidateSelectionError, match="integrity check failed"):
        load_candidate_selection(output_dir)


def test_selection_rejects_raw_exports_changed_after_inventory(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    inventory = normalize_candidate_inventory(raw_dir)
    path = raw_dir / "persona_7.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["benchmark_records"][0]["topic_preference"] = "Books"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CandidateSelectionError, match="do not exactly match"):
        select_candidates(inventory, raw_dir)
