import json
from pathlib import Path

import pytest

from membench.data_process.data_processing import (
    DATASET_ID,
    DownloadedJson,
    PersonaMemProcessingError,
    export_personas,
    select_personas,
)


def _row(
    persona_id: int,
    *,
    sensitive: bool = False,
    topic: str = "Travel",
) -> dict[str, object]:
    return {
        "persona_id": persona_id,
        "raw_persona_file": f"data/raw_data/profile_persona{persona_id}.json",
        "chat_history_32k_link": (
            f"data/chat_history_32k/history_persona{persona_id}.json"
        ),
        "sensitive_info": sensitive,
        "topic_preference": topic,
        "user_query": f"query for {persona_id}",
        "correct_answer": f"answer for {persona_id}",
    }


def test_select_personas_is_distinct_filtered_and_seeded() -> None:
    rows = [
        _row(0),
        _row(1, sensitive=True),
        _row(1, topic="Cybersecurity"),
        _row(2, topic="Account Security"),
        _row(3, sensitive=True),
        _row(4),
    ]

    first = select_personas(rows, sample_size=3, seed=9)
    again = select_personas(rows, sample_size=3, seed=9)

    assert first == again
    assert {selection.persona_id for selection in first} == {1, 2, 3}
    assert all(selection.selection_reasons for selection in first)
    assert next(item for item in first if item.persona_id == 1).benchmark_records == (
        {**rows[1], "source_row_index": 1},
        {**rows[2], "source_row_index": 2},
    )


def test_select_personas_rejects_an_insufficient_eligible_sample() -> None:
    with pytest.raises(PersonaMemProcessingError, match="only 1 distinct"):
        select_personas([_row(0), _row(1, sensitive=True)], sample_size=2)


def test_export_personas_writes_linked_self_contained_documents(
    tmp_path: Path,
) -> None:
    selection = select_personas(
        [_row(7, sensitive=True), _row(7, topic="Security")],
        sample_size=1,
    )[0]

    def downloader(source_path: str, revision: str) -> DownloadedJson:
        assert revision == "a" * 40
        if "raw_data" in source_path:
            return DownloadedJson(
                payload={"7": {"name": "Synthetic User"}}, sha256="1" * 64
            )
        return DownloadedJson(
            payload={
                "metadata": {"persona_id": 7},
                "chat_history": [{"role": "user", "content": "hello"}],
            },
            sha256="2" * 64,
        )

    paths = export_personas(
        [selection],
        output_dir=tmp_path,
        revision="a" * 40,
        seed=42,
        downloader=downloader,
    )

    assert paths == (tmp_path / "persona_7.json",)
    document = json.loads(paths[0].read_text(encoding="utf-8"))
    assert document["dataset"]["id"] == DATASET_ID
    assert document["persona_id"] == 7
    assert document["selection"]["reasons"] == [
        "sensitive_info",
        "security_topic_preference",
    ]
    assert document["raw_persona"]["7"]["name"] == "Synthetic User"
    assert document["chat_history_32k"]["metadata"]["persona_id"] == 7
    assert len(document["benchmark_records"]) == 2


def test_export_rejects_mismatched_chat_history(tmp_path: Path) -> None:
    selection = select_personas([_row(3, sensitive=True)], sample_size=1)[0]

    def downloader(source_path: str, revision: str) -> DownloadedJson:
        if "raw_data" in source_path:
            return DownloadedJson(payload={"3": {}}, sha256="1" * 64)
        return DownloadedJson(
            payload={
                "metadata": {"persona_id": 99},
                "chat_history": [{"role": "user", "content": "wrong"}],
            },
            sha256="2" * 64,
        )

    with pytest.raises(PersonaMemProcessingError, match="does not belong"):
        export_personas([selection], output_dir=tmp_path, downloader=downloader)
