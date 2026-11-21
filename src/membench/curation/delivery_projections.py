"""Freeze twin-aligned external email and document delivery projections.

This module intentionally does not decide whether an external source is read by
the agent.  It provides the common, immutable input for both forced ingestion
and future agent-selected tool-use conditions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from membench.campaign.artifacts import canonical_json_bytes
from membench.campaign.models import TrialTwin
from membench.curation.external_attack_specs import (
    ExternalAttackMethodology,
    ExternalAttackObjective,
    ExternalAttackSpec,
    ExternalAttackSpecSet,
)
from membench.ingress.models import (
    ExternalDelivery,
    ExternalSourceKind,
    SourceSelectionTask,
)
from membench.memory.models import CanonicalModel

DELIVERY_PROJECTION_VERSION = "external_delivery_projections_v1"
DELIVERY_PROJECTIONS_FILE = "external_deliveries.jsonl"
DELIVERY_MANIFEST_FILE = "external_deliveries_manifest.json"
_TWIN_ORDER = (TrialTwin.CLEAN, TrialTwin.MATCHED_BENIGN, TrialTwin.POISONED)


class DeliveryProjectionError(RuntimeError):
    """Raised when frozen external delivery evidence is incomplete or altered."""


class DeliveryProjectionFamily(CanonicalModel):
    """One source family, with exactly one frozen delivery for each twin."""

    schema_version: Literal[1] = 1
    family_ordinal: int = Field(ge=1)
    delivery_family_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    allocation_case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    attack_spec_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    attack_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attack_methodology: ExternalAttackMethodology
    attack_objective: ExternalAttackObjective
    attack_variant: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: ExternalSourceKind
    persona_id: str | None = Field(default=None, pattern=r"^[0-9]+$")
    fixture_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_.-]*$")
    fixture_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    deliveries: tuple[ExternalDelivery, ...] = Field(min_length=3, max_length=3)
    selection_task: SourceSelectionTask | None = None

    @model_validator(mode="after")
    def deliveries_must_be_aligned_twins(self) -> Self:
        fixture_binding = (self.persona_id, self.fixture_id, self.fixture_sha256)
        if any(value is not None for value in fixture_binding) and any(
            value is None for value in fixture_binding
        ):
            raise ValueError(
                "delivery fixture provenance requires persona, fixture ID, and hash"
            )
        if tuple(delivery.twin for delivery in self.deliveries) != _TWIN_ORDER:
            raise ValueError(
                "deliveries must appear once in clean, matched-benign, poisoned order"
            )
        shared_fields = (
            "delivery_family_id",
            "allocation_case_id",
            "source_kind",
            "source_id",
            "title",
            "sender",
            "available_after_source_message_index",
        )
        reference = self.deliveries[0]
        for delivery in self.deliveries:
            if delivery.delivery_family_id != self.delivery_family_id:
                raise ValueError("delivery family ID does not match family")
            if delivery.allocation_case_id != self.allocation_case_id:
                raise ValueError("delivery allocation case does not match family")
            if delivery.source_kind != self.source_kind:
                raise ValueError("delivery source kind does not match family")
            for field_name in shared_fields:
                if getattr(delivery, field_name) != getattr(reference, field_name):
                    raise ValueError(
                        f"delivery field {field_name!r} must be twin-aligned"
                    )
        if len({delivery.delivery_id for delivery in self.deliveries}) != len(
            self.deliveries
        ):
            raise ValueError("delivery IDs must be unique within a family")
        if len({delivery.trial_id for delivery in self.deliveries}) != len(
            self.deliveries
        ):
            raise ValueError("trial IDs must be unique within a family")
        if self.selection_task is not None and (
            self.selection_task.delivery_family_id != self.delivery_family_id
            or self.selection_task.allocation_case_id != self.allocation_case_id
        ):
            raise ValueError("selection task does not belong to its delivery family")
        return self

    def validate_attack_spec(self, spec: ExternalAttackSpec) -> None:
        """Prove this family is a rendered delivery of one exact attack spec."""

        expected = {
            "attack_spec_id": spec.attack_spec_id,
            "attack_spec_sha256": spec.sha256,
            "allocation_case_id": spec.allocation_case_id,
            "attack_methodology": spec.attack_methodology,
            "attack_objective": spec.objective,
            "attack_variant": spec.variant,
            "target_reference_sha256": spec.target_reference_sha256,
            "source_kind": spec.source_kind,
        }
        observed = {
            "attack_spec_id": self.attack_spec_id,
            "attack_spec_sha256": self.attack_spec_sha256,
            "allocation_case_id": self.allocation_case_id,
            "attack_methodology": self.attack_methodology,
            "attack_objective": self.attack_objective,
            "attack_variant": self.attack_variant,
            "target_reference_sha256": self.target_reference_sha256,
            "source_kind": self.source_kind,
        }
        if observed != expected:
            mismatched = sorted(
                name for name in expected if observed[name] != expected[name]
            )
            raise DeliveryProjectionError(
                "delivery family does not match external attack spec: "
                + ", ".join(mismatched)
            )


class DeliveryProjectionManifest(CanonicalModel):
    """Integrity metadata for one JSONL set of external delivery families."""

    schema_version: Literal[1] = 1
    version: Literal["external_delivery_projections_v1"] = DELIVERY_PROJECTION_VERSION
    delivery_set_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    attack_set_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    projection_set_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    family_count: int = Field(ge=1)
    families_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DeliveryProjectionSet(CanonicalModel):
    """Deterministic set of external source deliveries for one projection set."""

    schema_version: Literal[1] = 1
    delivery_set_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    attack_set_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    projection_set_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    families: tuple[DeliveryProjectionFamily, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def families_must_be_deterministic(self) -> Self:
        if tuple(family.family_ordinal for family in self.families) != tuple(
            range(1, len(self.families) + 1)
        ):
            raise ValueError("family ordinals must be contiguous and start at one")
        if len({family.delivery_family_id for family in self.families}) != len(
            self.families
        ):
            raise ValueError("delivery family IDs must be unique")
        return self

    def jsonl_bytes(self) -> bytes:
        """Return canonical, line-delimited family records for freezing."""

        return b"".join(
            canonical_json_bytes(family.model_dump(mode="json")).rstrip(b"\n") + b"\n"
            for family in self.families
        )

    def manifest(self) -> DeliveryProjectionManifest:
        """Build the exact manifest corresponding to the current JSONL bytes."""

        return DeliveryProjectionManifest(
            delivery_set_id=self.delivery_set_id,
            attack_set_id=self.attack_set_id,
            projection_set_id=self.projection_set_id,
            family_count=len(self.families),
            families_sha256=hashlib.sha256(self.jsonl_bytes()).hexdigest(),
        )


def build_delivery_projection_set(
    *,
    delivery_set_id: str,
    attack_set_id: str,
    projection_set_id: str,
    families: tuple[DeliveryProjectionFamily, ...],
) -> DeliveryProjectionSet:
    """Validate and return one deterministic delivery-projection set."""

    return DeliveryProjectionSet(
        delivery_set_id=delivery_set_id,
        attack_set_id=attack_set_id,
        projection_set_id=projection_set_id,
        families=families,
    )


def validate_delivery_attack_links(
    delivery_set: DeliveryProjectionSet,
    attack_set: ExternalAttackSpecSet,
) -> None:
    """Verify every frozen delivery family renders one exact external attack.

    The set IDs and shared projection-set ID prevent a valid delivery artifact
    from being accidentally paired with a different external attack battery.
    """

    if delivery_set.attack_set_id != attack_set.attack_set_id:
        raise DeliveryProjectionError("delivery and external attack set IDs differ")
    if delivery_set.projection_set_id != attack_set.projection_set_id:
        raise DeliveryProjectionError(
            "delivery and external attack projection-set IDs differ"
        )
    for family in delivery_set.families:
        try:
            spec = attack_set.spec_for(family.attack_spec_id)
        except KeyError as exc:
            raise DeliveryProjectionError(
                f"delivery family references an unknown external attack spec: "
                f"{family.attack_spec_id}"
            ) from exc
        family.validate_attack_spec(spec)


def export_delivery_projection_set(
    projection_set: DeliveryProjectionSet,
    destination: str | Path,
) -> DeliveryProjectionManifest:
    """Persist external deliveries and their integrity manifest without overwrite."""

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    deliveries_path = output / DELIVERY_PROJECTIONS_FILE
    manifest_path = output / DELIVERY_MANIFEST_FILE
    if deliveries_path.exists() or manifest_path.exists():
        raise DeliveryProjectionError("delivery projection output already exists")
    content = projection_set.jsonl_bytes()
    manifest = projection_set.manifest()
    deliveries_path.write_bytes(content)
    manifest_path.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    return manifest


def load_delivery_projection_set(destination: str | Path) -> DeliveryProjectionSet:
    """Load and verify a previously frozen external delivery-projection set."""

    root = Path(destination)
    deliveries_path = root / DELIVERY_PROJECTIONS_FILE
    manifest_path = root / DELIVERY_MANIFEST_FILE
    try:
        content = deliveries_path.read_bytes()
        manifest = DeliveryProjectionManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise DeliveryProjectionError(
            f"could not load delivery projections: {exc}"
        ) from exc
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != manifest.families_sha256:
        raise DeliveryProjectionError(
            "delivery-projection JSONL hash does not match manifest"
        )
    try:
        families = tuple(
            DeliveryProjectionFamily.model_validate(json.loads(line))
            for line in content.splitlines()
            if line.strip()
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise DeliveryProjectionError(
            f"delivery-projection JSONL is invalid: {exc}"
        ) from exc
    if len(families) != manifest.family_count:
        raise DeliveryProjectionError(
            "delivery-projection family count does not match manifest"
        )
    projection_set = DeliveryProjectionSet(
        delivery_set_id=manifest.delivery_set_id,
        attack_set_id=manifest.attack_set_id,
        projection_set_id=manifest.projection_set_id,
        families=families,
    )
    if projection_set.manifest() != manifest:
        raise DeliveryProjectionError(
            "delivery-projection manifest does not match decoded content"
        )
    return projection_set
