"""Immutable attack contracts for email and document ingress experiments.

These specifications define what an external-source attack attempts to do.
They deliberately contain no twin-specific source body; those exact clean,
matched-benign, and poisoned payloads live in delivery projections.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from membench.campaign.artifacts import canonical_json_bytes
from membench.curation.projections import AttackReviewStatus
from membench.ingress.models import ExternalSourceKind
from membench.memory.models import CanonicalModel

EXTERNAL_ATTACK_SPEC_VERSION = "external_attack_specs_v1"
EXTERNAL_ATTACK_SPECS_FILE = "external_attack_specs.jsonl"
EXTERNAL_ATTACK_MANIFEST_FILE = "external_attack_specs_manifest.json"
_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_CASE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ExternalAttackMethodology(StrEnum):
    """Mechanisms tested through an external email or document source."""

    ABRUPT_FALSE_CORRECTION = "abrupt_false_correction"
    OWNERSHIP_CONFUSION = "ownership_confusion"
    SEMANTIC_HIJACKING = "semantic_hijacking"
    STORED_INSTRUCTION_INJECTION = "stored_instruction_injection"
    DESTINATION_REDIRECTION = "destination_redirection"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    PROVENANCE_LAUNDERING = "provenance_laundering"
    FALSE_SECURITY_PROCEDURE = "false_security_procedure"


class ExternalAttackObjective(StrEnum):
    """Evaluator-only intended effect of an external-source attack."""

    FACT_CORRUPTION = "fact_corruption"
    PREFERENCE_INVERSION = "preference_inversion"
    DESTINATION_REDIRECTION = "destination_redirection"
    OWNERSHIP_MISATTRIBUTION = "ownership_misattribution"
    SEMANTIC_REDEFINITION = "semantic_redefinition"
    INSTRUCTION_PERSISTENCE = "instruction_persistence"
    SECURITY_PROCEDURE_SUBVERSION = "security_procedure_subversion"


class ExternalAttackSpecError(RuntimeError):
    """Raised when an external attack-specification artifact is unsafe."""


class ExternalAttackSpec(CanonicalModel):
    """One source-independent external attack definition and target reference."""

    schema_version: Literal[1] = 1
    spec_ordinal: int = Field(ge=1)
    attack_spec_id: str = Field(pattern=_ID_PATTERN)
    allocation_case_id: str = Field(pattern=_CASE_ID_PATTERN)
    attack_id: str = Field(pattern=r"^bb-[a-z0-9]+-[0-9]{3}$")
    attack_methodology: ExternalAttackMethodology
    objective: ExternalAttackObjective
    variant: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    knowledge: Literal["black_box"] = "black_box"
    source_kind: ExternalSourceKind
    insert_after_source_message_index: int = Field(ge=0)
    target_reference_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_attacker_view_sha256: str = Field(pattern=_SHA256_PATTERN)
    review_status: AttackReviewStatus

    @property
    def sha256(self) -> str:
        """Return the stable identity used by delivery-family references."""

        return hashlib.sha256(canonical_json_bytes(self)).hexdigest()


class ExternalAttackSpecManifest(CanonicalModel):
    """Integrity metadata for one frozen external attack-specification set."""

    schema_version: Literal[1] = 1
    version: Literal["external_attack_specs_v1"] = EXTERNAL_ATTACK_SPEC_VERSION
    attack_set_id: str = Field(pattern=_CASE_ID_PATTERN)
    projection_set_id: str = Field(pattern=_CASE_ID_PATTERN)
    spec_count: int = Field(ge=1)
    specs_sha256: str = Field(pattern=_SHA256_PATTERN)


class ExternalAttackSpecSet(CanonicalModel):
    """Deterministic external attack battery, separate from conversation attacks."""

    schema_version: Literal[1] = 1
    attack_set_id: str = Field(pattern=_CASE_ID_PATTERN)
    projection_set_id: str = Field(pattern=_CASE_ID_PATTERN)
    specs: tuple[ExternalAttackSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def specs_must_have_unique_contiguous_identity(self) -> Self:
        if tuple(spec.spec_ordinal for spec in self.specs) != tuple(
            range(1, len(self.specs) + 1)
        ):
            raise ValueError("external attack-spec ordinals must be contiguous")
        for attribute in ("attack_spec_id", "allocation_case_id", "attack_id"):
            values = [getattr(spec, attribute) for spec in self.specs]
            if len(values) != len(set(values)):
                raise ValueError(f"external attack specs require unique {attribute}")
        return self

    def spec_for(self, attack_spec_id: str) -> ExternalAttackSpec:
        """Resolve one exact specification by its stable external ID."""

        for spec in self.specs:
            if spec.attack_spec_id == attack_spec_id:
                return spec
        raise KeyError(attack_spec_id)

    def jsonl_bytes(self) -> bytes:
        """Return canonical line-delimited specification records."""

        return b"".join(
            canonical_json_bytes(spec.model_dump(mode="json")).rstrip(b"\n") + b"\n"
            for spec in self.specs
        )

    def manifest(self) -> ExternalAttackSpecManifest:
        """Build integrity metadata for the exact JSONL representation."""

        return ExternalAttackSpecManifest(
            attack_set_id=self.attack_set_id,
            projection_set_id=self.projection_set_id,
            spec_count=len(self.specs),
            specs_sha256=hashlib.sha256(self.jsonl_bytes()).hexdigest(),
        )


def build_external_attack_spec_set(
    *,
    attack_set_id: str,
    projection_set_id: str,
    specs: tuple[ExternalAttackSpec, ...],
) -> ExternalAttackSpecSet:
    """Validate and return one immutable external attack battery."""

    return ExternalAttackSpecSet(
        attack_set_id=attack_set_id,
        projection_set_id=projection_set_id,
        specs=specs,
    )


def export_external_attack_spec_set(
    attack_set: ExternalAttackSpecSet,
    destination: str | Path,
) -> ExternalAttackSpecManifest:
    """Persist exact external attack specifications without overwriting evidence."""

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    specs_path = output / EXTERNAL_ATTACK_SPECS_FILE
    manifest_path = output / EXTERNAL_ATTACK_MANIFEST_FILE
    if specs_path.exists() or manifest_path.exists():
        raise ExternalAttackSpecError("external attack-spec output already exists")
    specs_path.write_bytes(attack_set.jsonl_bytes())
    manifest = attack_set.manifest()
    manifest_path.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    return manifest


def load_external_attack_spec_set(destination: str | Path) -> ExternalAttackSpecSet:
    """Load an external attack battery and verify its exact frozen bytes."""

    root = Path(destination)
    specs_path = root / EXTERNAL_ATTACK_SPECS_FILE
    manifest_path = root / EXTERNAL_ATTACK_MANIFEST_FILE
    try:
        content = specs_path.read_bytes()
        manifest = ExternalAttackSpecManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise ExternalAttackSpecError(
            f"could not load external attack specifications: {exc}"
        ) from exc
    if hashlib.sha256(content).hexdigest() != manifest.specs_sha256:
        raise ExternalAttackSpecError(
            "external attack-spec JSONL hash does not match manifest"
        )
    try:
        specs = tuple(
            ExternalAttackSpec.model_validate(json.loads(line))
            for line in content.splitlines()
            if line.strip()
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise ExternalAttackSpecError(
            f"external attack-spec JSONL is invalid: {exc}"
        ) from exc
    if len(specs) != manifest.spec_count:
        raise ExternalAttackSpecError(
            "external attack-spec count does not match manifest"
        )
    attack_set = ExternalAttackSpecSet(
        attack_set_id=manifest.attack_set_id,
        projection_set_id=manifest.projection_set_id,
        specs=specs,
    )
    if attack_set.manifest() != manifest:
        raise ExternalAttackSpecError(
            "external attack-spec manifest does not match decoded content"
        )
    return attack_set
