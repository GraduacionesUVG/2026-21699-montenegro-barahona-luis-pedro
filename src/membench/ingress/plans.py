"""Frozen per-twin ingress plans derived from external delivery artifacts.

An ingress plan contains only compact identifiers, hashes, and execution-mode
choices.  The external delivery projection set remains the authority for source
content, headers, and the full source-selection task.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.agent.models import AgentRunConfiguration
from membench.campaign.artifacts import canonical_json_bytes
from membench.campaign.models import TrialTwin
from membench.curation.delivery_projections import (
    DeliveryProjectionError,
    DeliveryProjectionSet,
    validate_delivery_attack_links,
)
from membench.curation.external_attack_specs import ExternalAttackSpecSet
from membench.ingress.models import ExternalSourceKind, IngressMode, SourceSelectionTask
from membench.memory.models import CanonicalModel
from membench.tools.fixtures import ToolFixtureBundle

INGRESS_PLAN_VERSION = "trial_ingress_plans_v1"
INGRESS_PLANS_FILE = "trial_ingress_plans.jsonl"
INGRESS_PLAN_MANIFEST_FILE = "trial_ingress_plans_manifest.json"
_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_SET_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class TrialIngressPlanError(RuntimeError):
    """Raised when a compact ingress plan cannot be safely reconstructed."""


class TrialIngressPlan(CanonicalModel):
    """One compact, immutable ingress decision for one external-delivery twin."""

    schema_version: Literal[1] = 1
    plan_ordinal: int = Field(ge=1)
    ingress_plan_id: str = Field(pattern=_ID_PATTERN)
    trial_id: str = Field(pattern=_ID_PATTERN)
    host_trial_id: str = Field(pattern=_ID_PATTERN)
    allocation_case_id: str = Field(pattern=_SET_ID_PATTERN)
    twin: TrialTwin
    persona_id: str = Field(pattern=r"^[0-9]+$")
    ingress_mode: IngressMode
    delivery_family_id: str = Field(pattern=_ID_PATTERN)
    delivery_id: str = Field(pattern=_ID_PATTERN)
    delivery_content_sha256: str = Field(pattern=_SHA256_PATTERN)
    attack_spec_id: str = Field(pattern=_ID_PATTERN)
    attack_spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_kind: ExternalSourceKind
    fixture_id: str = Field(pattern=_ID_PATTERN)
    fixture_sha256: str = Field(pattern=_SHA256_PATTERN)
    selection_task_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    selection_task_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    allowed_tools: tuple[str, ...] = Field(default_factory=tuple)
    selection_agent_configuration: AgentRunConfiguration | None = None
    probe_writeback_allowed: Literal[False] = False
    response_writeback_allowed: Literal[False] = False

    @field_validator("allowed_tools")
    @classmethod
    def allowed_tools_must_be_sorted_and_unique(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not item or not item.replace("_", "a").isalnum() for item in value):
            raise ValueError("ingress plan tool names are malformed")
        if value != tuple(sorted(set(value))):
            raise ValueError("ingress plan tools must be sorted and unique")
        return value

    @model_validator(mode="after")
    def selection_contract_must_match_mode(self) -> Self:
        selection = (
            self.selection_task_id,
            self.selection_task_sha256,
            self.selection_agent_configuration,
        )
        if self.ingress_mode == IngressMode.AGENT_SELECTED:
            if any(value is None for value in selection) or not self.allowed_tools:
                raise ValueError(
                    "agent-selected ingress requires a selection task and tools"
                )
            assert self.selection_agent_configuration is not None
            if self.selection_agent_configuration.allowed_tools != self.allowed_tools:
                raise ValueError(
                    "selection configuration tools must match ingress plan tools"
                )
            if not self.selection_agent_configuration.memory_read_only:
                raise ValueError("selection configuration must be memory read-only")
            if self.selection_agent_configuration.allow_external_side_effects:
                raise ValueError(
                    "selection configuration cannot allow external side effects"
                )
        elif any(value is not None for value in selection) or self.allowed_tools:
            raise ValueError("forced ingestion cannot carry source-selection data")
        return self

    @property
    def sha256(self) -> str:
        """Return the stable identity later stored in a compact run binding."""

        return hashlib.sha256(canonical_json_bytes(self)).hexdigest()


class TrialIngressPlanManifest(CanonicalModel):
    """Integrity and source-artifact bindings for one frozen ingress-plan set."""

    schema_version: Literal[1] = 1
    version: Literal["trial_ingress_plans_v1"] = INGRESS_PLAN_VERSION
    ingress_plan_set_id: str = Field(pattern=_SET_ID_PATTERN)
    ingress_mode: IngressMode
    delivery_set_id: str = Field(pattern=_SET_ID_PATTERN)
    delivery_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    attack_set_id: str = Field(pattern=_SET_ID_PATTERN)
    attack_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    fixture_id: str = Field(pattern=_ID_PATTERN)
    fixture_sha256: str = Field(pattern=_SHA256_PATTERN)
    plan_count: int = Field(ge=1)
    plans_sha256: str = Field(pattern=_SHA256_PATTERN)


class TrialIngressPlanSet(CanonicalModel):
    """Derived compact plans for every twin in one external-delivery set."""

    schema_version: Literal[1] = 1
    ingress_plan_set_id: str = Field(pattern=_SET_ID_PATTERN)
    ingress_mode: IngressMode
    delivery_set_id: str = Field(pattern=_SET_ID_PATTERN)
    delivery_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    attack_set_id: str = Field(pattern=_SET_ID_PATTERN)
    attack_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    fixture_id: str = Field(pattern=_ID_PATTERN)
    fixture_sha256: str = Field(pattern=_SHA256_PATTERN)
    plans: tuple[TrialIngressPlan, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def plans_must_be_complete_and_deterministic(self) -> Self:
        if tuple(plan.plan_ordinal for plan in self.plans) != tuple(
            range(1, len(self.plans) + 1)
        ):
            raise ValueError(
                "ingress plan ordinals must be contiguous and start at one"
            )
        if any(plan.ingress_mode != self.ingress_mode for plan in self.plans):
            raise ValueError("ingress plans must use the set ingress mode")
        if any(
            plan.fixture_id != self.fixture_id
            or plan.fixture_sha256 != self.fixture_sha256
            for plan in self.plans
        ):
            raise ValueError("ingress plans must use the set fixture binding")
        for attribute in (
            "ingress_plan_id",
            "trial_id",
            "host_trial_id",
            "delivery_id",
        ):
            values = [getattr(plan, attribute) for plan in self.plans]
            if len(values) != len(set(values)):
                raise ValueError(f"ingress plans require unique {attribute}")
        return self

    def jsonl_bytes(self) -> bytes:
        """Return canonical line-delimited plan records for immutable export."""

        return b"".join(
            canonical_json_bytes(plan.model_dump(mode="json")).rstrip(b"\n") + b"\n"
            for plan in self.plans
        )

    def manifest(self) -> TrialIngressPlanManifest:
        """Return integrity metadata for the exact JSONL representation."""

        return TrialIngressPlanManifest(
            ingress_plan_set_id=self.ingress_plan_set_id,
            ingress_mode=self.ingress_mode,
            delivery_set_id=self.delivery_set_id,
            delivery_manifest_sha256=self.delivery_manifest_sha256,
            attack_set_id=self.attack_set_id,
            attack_manifest_sha256=self.attack_manifest_sha256,
            fixture_id=self.fixture_id,
            fixture_sha256=self.fixture_sha256,
            plan_count=len(self.plans),
            plans_sha256=hashlib.sha256(self.jsonl_bytes()).hexdigest(),
        )

    def plan_for_trial(self, trial_id: str) -> TrialIngressPlan:
        """Return a plan by its external delivery trial identity."""

        for plan in self.plans:
            if plan.trial_id == trial_id:
                return plan.model_copy(deep=True)
        raise KeyError(f"no ingress plan for trial {trial_id!r}")

    def plan_for_host_trial(self, host_trial_id: str) -> TrialIngressPlan:
        """Return the external plan bound to one PersonaMem replay/probe trial."""

        for plan in self.plans:
            if plan.host_trial_id == host_trial_id:
                return plan.model_copy(deep=True)
        raise KeyError(f"no ingress plan for host trial {host_trial_id!r}")


def build_trial_ingress_plan_set(
    *,
    ingress_plan_set_id: str,
    ingress_mode: IngressMode,
    delivery_set: DeliveryProjectionSet,
    attack_set: ExternalAttackSpecSet,
    fixture: ToolFixtureBundle,
    host_trial_ids: Mapping[str, str],
    selection_agent_configuration: AgentRunConfiguration | None = None,
) -> TrialIngressPlanSet:
    """Derive compact per-twin plans from verified external source artifacts."""

    try:
        validate_delivery_attack_links(delivery_set, attack_set)
    except DeliveryProjectionError as exc:
        raise TrialIngressPlanError(
            "delivery and attack artifacts do not match"
        ) from exc
    delivery_manifest = delivery_set.manifest()
    attack_manifest = attack_set.manifest()
    _validate_host_trial_ids(delivery_set, host_trial_ids)
    plans: list[TrialIngressPlan] = []
    for family in delivery_set.families:
        _validate_family_fixture_binding(family, fixture)
        selection = _selection_for_mode(family.selection_task, ingress_mode)
        for delivery in family.deliveries:
            plans.append(
                TrialIngressPlan(
                    plan_ordinal=len(plans) + 1,
                    ingress_plan_id=(
                        f"ingress-{ingress_plan_set_id}-{delivery.delivery_id}"
                    ),
                    trial_id=delivery.trial_id,
                    host_trial_id=host_trial_ids[delivery.trial_id],
                    allocation_case_id=delivery.allocation_case_id,
                    twin=delivery.twin,
                    persona_id=family.persona_id,
                    ingress_mode=ingress_mode,
                    delivery_family_id=family.delivery_family_id,
                    delivery_id=delivery.delivery_id,
                    delivery_content_sha256=delivery.content_sha256,
                    attack_spec_id=family.attack_spec_id,
                    attack_spec_sha256=family.attack_spec_sha256,
                    source_kind=delivery.source_kind,
                    fixture_id=fixture.fixture_id,
                    fixture_sha256=fixture.sha256,
                    selection_task_id=selection[0],
                    selection_task_sha256=selection[1],
                    allowed_tools=selection[2],
                    selection_agent_configuration=(
                        selection_agent_configuration
                        if ingress_mode == IngressMode.AGENT_SELECTED
                        else None
                    ),
                )
            )
    return TrialIngressPlanSet(
        ingress_plan_set_id=ingress_plan_set_id,
        ingress_mode=ingress_mode,
        delivery_set_id=delivery_set.delivery_set_id,
        delivery_manifest_sha256=delivery_manifest.families_sha256,
        attack_set_id=attack_set.attack_set_id,
        attack_manifest_sha256=attack_manifest.specs_sha256,
        fixture_id=fixture.fixture_id,
        fixture_sha256=fixture.sha256,
        plans=tuple(plans),
    )


def export_trial_ingress_plan_set(
    plan_set: TrialIngressPlanSet,
    destination: str | Path,
) -> TrialIngressPlanManifest:
    """Persist plans and manifest without overwriting a prior ingress artifact."""

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    plans_path = output / INGRESS_PLANS_FILE
    manifest_path = output / INGRESS_PLAN_MANIFEST_FILE
    if plans_path.exists() or manifest_path.exists():
        raise TrialIngressPlanError("ingress-plan output already exists")
    plans_path.write_bytes(plan_set.jsonl_bytes())
    manifest = plan_set.manifest()
    manifest_path.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    return manifest


def load_trial_ingress_plan_set(destination: str | Path) -> TrialIngressPlanSet:
    """Load frozen plan bytes and verify their manifest before use."""

    root = Path(destination)
    plans_path = root / INGRESS_PLANS_FILE
    manifest_path = root / INGRESS_PLAN_MANIFEST_FILE
    try:
        content = plans_path.read_bytes()
        manifest = TrialIngressPlanManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise TrialIngressPlanError(
            f"could not load trial ingress plans: {exc}"
        ) from exc
    if hashlib.sha256(content).hexdigest() != manifest.plans_sha256:
        raise TrialIngressPlanError("ingress-plan JSONL hash does not match manifest")
    try:
        plans = tuple(
            TrialIngressPlan.model_validate(json.loads(line))
            for line in content.splitlines()
            if line.strip()
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise TrialIngressPlanError(f"ingress-plan JSONL is invalid: {exc}") from exc
    if len(plans) != manifest.plan_count:
        raise TrialIngressPlanError("ingress-plan count does not match manifest")
    plan_set = TrialIngressPlanSet(
        ingress_plan_set_id=manifest.ingress_plan_set_id,
        ingress_mode=manifest.ingress_mode,
        delivery_set_id=manifest.delivery_set_id,
        delivery_manifest_sha256=manifest.delivery_manifest_sha256,
        attack_set_id=manifest.attack_set_id,
        attack_manifest_sha256=manifest.attack_manifest_sha256,
        fixture_id=manifest.fixture_id,
        fixture_sha256=manifest.fixture_sha256,
        plans=plans,
    )
    if plan_set.manifest() != manifest:
        raise TrialIngressPlanError(
            "ingress-plan manifest does not match decoded bytes"
        )
    return plan_set


def _validate_host_trial_ids(
    delivery_set: DeliveryProjectionSet,
    host_trial_ids: Mapping[str, str],
) -> None:
    external_trial_ids = {
        delivery.trial_id
        for family in delivery_set.families
        for delivery in family.deliveries
    }
    supplied = set(host_trial_ids)
    if supplied != external_trial_ids:
        missing = sorted(external_trial_ids - supplied)
        unexpected = sorted(supplied - external_trial_ids)
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        raise TrialIngressPlanError(
            "host trial mapping must cover every external trial exactly once"
            + (": " + "; ".join(details) if details else "")
        )
    if any(
        not isinstance(value, str) or not value for value in host_trial_ids.values()
    ):
        raise TrialIngressPlanError("host trial IDs must be non-blank strings")
    if len(set(host_trial_ids.values())) != len(host_trial_ids):
        raise TrialIngressPlanError("host trial IDs must be unique within a plan set")


def _validate_family_fixture_binding(
    family: object,
    fixture: ToolFixtureBundle,
) -> None:
    persona_id = family.persona_id
    fixture_id = family.fixture_id
    fixture_sha256 = family.fixture_sha256
    if persona_id is None or fixture_id is None or fixture_sha256 is None:
        raise TrialIngressPlanError("delivery family has no frozen fixture binding")
    if fixture_id != fixture.fixture_id or fixture_sha256 != fixture.sha256:
        raise TrialIngressPlanError("delivery family fixture binding does not match")
    if persona_id not in fixture.persona_ids:
        raise TrialIngressPlanError(
            "delivery family persona is not declared by fixture"
        )


def _selection_for_mode(
    task: SourceSelectionTask | None,
    mode: IngressMode,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    if mode == IngressMode.FORCED_INGESTION:
        return None, None, ()
    if task is None:
        raise TrialIngressPlanError(
            "agent-selected ingress requires a delivery-family selection task"
        )
    return (
        task.selection_task_id,
        hashlib.sha256(canonical_json_bytes(task)).hexdigest(),
        task.allowed_tools,
    )
