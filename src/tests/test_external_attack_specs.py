"""Tests for independent, integrity-checked external attack specifications."""

from __future__ import annotations

import pytest

from membench.curation.external_attack_specs import (
    EXTERNAL_ATTACK_SPECS_FILE,
    ExternalAttackMethodology,
    ExternalAttackObjective,
    ExternalAttackSpec,
    ExternalAttackSpecError,
    build_external_attack_spec_set,
    export_external_attack_spec_set,
    load_external_attack_spec_set,
)
from membench.ingress.models import ExternalSourceKind


def _attack_set():  # type: ignore[no-untyped-def]
    spec = ExternalAttackSpec(
        spec_ordinal=1,
        attack_spec_id="external-spec-document-001",
        allocation_case_id="alloc-document-001",
        attack_id="bb-ext-002",
        attack_methodology=ExternalAttackMethodology.FALSE_SECURITY_PROCEDURE,
        objective=ExternalAttackObjective.SECURITY_PROCEDURE_SUBVERSION,
        variant="untrusted_procedure_update",
        source_kind=ExternalSourceKind.DOCUMENT,
        insert_after_source_message_index=8,
        target_reference_sha256="c" * 64,
        public_attacker_view_sha256="d" * 64,
        review_status="generated_unreviewed",
    )
    return build_external_attack_spec_set(
        attack_set_id="pilot-external-attacks-002",
        projection_set_id="pmv2-projections-v1",
        specs=(spec,),
    )


def test_external_attack_specs_round_trip_with_their_exact_hashes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    attack_set = _attack_set()

    manifest = export_external_attack_spec_set(attack_set, tmp_path)
    loaded = load_external_attack_spec_set(tmp_path)

    assert loaded == attack_set
    assert loaded.specs[0].sha256 == attack_set.specs[0].sha256
    assert manifest == attack_set.manifest()


def test_tampered_external_attack_specs_are_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    export_external_attack_spec_set(_attack_set(), tmp_path)
    path = tmp_path / EXTERNAL_ATTACK_SPECS_FILE
    path.write_bytes(path.read_bytes().replace(b"procedure", b"tampered"))

    with pytest.raises(ExternalAttackSpecError, match="hash"):
        load_external_attack_spec_set(tmp_path)
