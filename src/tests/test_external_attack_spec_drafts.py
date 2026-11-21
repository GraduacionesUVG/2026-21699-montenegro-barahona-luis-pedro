"""Tests for editable source material used to freeze external attack specs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from membench.curation.external_attack_spec_drafts import (
    load_external_attack_spec_draft_set,
)

ROOT = Path(__file__).resolve().parents[1]
DRAFT_PATH = ROOT / "configs/external_attack_specs/external-tool-pilot-v1.json"


def test_editable_external_attack_draft_freezes_exact_text_hashes() -> None:
    draft = load_external_attack_spec_draft_set(DRAFT_PATH)
    frozen = draft.freeze()
    source = draft.specs[0]
    spec = frozen.specs[0]

    assert spec.attack_spec_id == "external-routing-277-001"
    assert spec.review_status == "generated_unreviewed"
    assert (
        spec.target_reference_sha256
        == hashlib.sha256(source.target_reference.encode("utf-8")).hexdigest()
    )
    assert (
        spec.public_attacker_view_sha256
        == hashlib.sha256(source.public_attacker_view.encode("utf-8")).hexdigest()
    )
