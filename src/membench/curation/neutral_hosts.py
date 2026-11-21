"""Derive neutral external-ingress hosts from verified PersonaMem projections.

The builder deliberately reuses only source-derived events from a clean
conversation trial.  It never copies the generated conversation overlays, so
the resulting clean, matched-benign, and poisoned twins differ only when an
external delivery is bound later by the ingress plan.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from membench.campaign.models import TrialTwin
from membench.curation.delivery_projections import (
    DeliveryProjectionError,
    DeliveryProjectionSet,
    load_delivery_projection_set,
)
from membench.curation.projections import (
    AttackReviewStatus,
    AttackSpec,
    EventStage,
    PrivateLabel,
    ProjectionCase,
    ProjectionEvent,
    ProjectionKind,
    ProjectionSet,
    ProjectionSetConfig,
    ProjectionSetError,
    PublicProbe,
    TrialProjection,
    _canonical_sha256,
    load_projection_set,
    memory_visible_replay_signature,
)
from membench.memory.models import CanonicalModel

_ID_PATTERN = r"^[a-z][a-z0-9_-]*$"


class NeutralHostError(RuntimeError):
    """Raised when a neutral-host configuration cannot be safely materialized."""


class NeutralHostDefinition(CanonicalModel):
    """One source conversation case and its derived neutral-host identity."""

    schema_version: Literal[1] = 1
    persona_id: str = Field(pattern=r"^[0-9]+$")
    source_case_id: str = Field(pattern=_ID_PATTERN)
    host_case_id: str = Field(pattern=_ID_PATTERN)


class NeutralHostConfig(CanonicalModel):
    """Editable selection of source cases used to construct neutral hosts."""

    schema_version: Literal[1] = 1
    projection_set_id: str = Field(
        default="pmv2-neutral-external-hosts-v1",
        pattern=_ID_PATTERN,
    )
    hosts: tuple[NeutralHostDefinition, ...] = Field(min_length=1)

    @field_validator("hosts")
    @classmethod
    def hosts_must_be_deterministic_and_unique(
        cls, value: tuple[NeutralHostDefinition, ...]
    ) -> tuple[NeutralHostDefinition, ...]:
        for attribute in ("persona_id", "source_case_id", "host_case_id"):
            values = tuple(getattr(item, attribute) for item in value)
            if len(values) != len(set(values)):
                raise ValueError(f"neutral hosts require unique {attribute}")
        return value


def load_neutral_host_config(path: str | Path) -> NeutralHostConfig:
    """Load one editable neutral-host selection document."""

    source = Path(path)
    try:
        return NeutralHostConfig.model_validate_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise NeutralHostError(
            f"could not load neutral-host configuration: {source}"
        ) from exc


def build_neutral_host_projection_set(
    config: NeutralHostConfig,
    *,
    source_projection_directory: str | Path,
    delivery_projection_directories: Sequence[str | Path] = (),
) -> ProjectionSet:
    """Build verified neutral hosts and validate supplied external deliveries.

    Each supplied delivery directory is integrity-loaded before its availability
    boundary is checked against the neutral host for that delivery's persona.
    An empty directory sequence is supported while authoring hosts; callers
    should supply every frozen delivery set before using the output in a
    campaign.
    """

    try:
        source = load_projection_set(Path(source_projection_directory))
    except ProjectionSetError as exc:
        raise NeutralHostError("could not load source projection set") from exc
    try:
        delivery_sets = tuple(
            load_delivery_projection_set(directory)
            for directory in delivery_projection_directories
        )
    except (OSError, ValueError, DeliveryProjectionError) as exc:
        raise NeutralHostError("could not load external delivery projections") from exc

    built = _build_neutral_host_projection_set(
        config,
        source_projection_set=source,
    )
    _validate_delivery_availability(
        config,
        projection_set=built,
        delivery_sets=delivery_sets,
    )
    return built


def _build_neutral_host_projection_set(
    config: NeutralHostConfig,
    *,
    source_projection_set: ProjectionSet,
) -> ProjectionSet:
    source_cases = {
        case.allocation_case_id: case for case in source_projection_set.cases
    }
    source_specs = {
        spec.allocation_case_id: spec for spec in source_projection_set.attack_specs
    }
    source_probes = {
        probe.allocation_case_id: probe for probe in source_projection_set.probes
    }
    source_labels = {
        label.allocation_case_id: label
        for label in source_projection_set.private_labels
    }
    source_trials = {
        (trial.allocation_case_id, trial.twin): trial
        for trial in source_projection_set.trials
    }

    cases: list[ProjectionCase] = []
    specs: list[AttackSpec] = []
    probes: list[PublicProbe] = []
    labels: list[PrivateLabel] = []
    trials: list[TrialProjection] = []
    for ordinal, host in enumerate(config.hosts, start=1):
        try:
            source_case = source_cases[host.source_case_id]
            source_spec = source_specs[host.source_case_id]
            source_probe = source_probes[host.source_case_id]
            source_label = source_labels[host.source_case_id]
            clean_trial = source_trials[(host.source_case_id, TrialTwin.CLEAN)]
        except KeyError as exc:
            raise NeutralHostError(
                f"neutral-host source is incomplete: {host.source_case_id}"
            ) from exc
        if source_case.persona_id != host.persona_id:
            raise NeutralHostError(
                f"source case {host.source_case_id} belongs to persona "
                f"{source_case.persona_id}, not {host.persona_id}"
            )
        if source_case.projection_kind != ProjectionKind.CONVERSATION_ATTACK:
            raise NeutralHostError(
                "neutral hosts must be derived from conversation-attack sources: "
                f"{host.source_case_id}"
            )

        host_case, host_spec, host_probe, host_label, host_trials = _build_host_family(
            host,
            ordinal=ordinal,
            source_case=source_case,
            source_spec=source_spec,
            source_probe=source_probe,
            source_label=source_label,
            clean_trial=clean_trial,
        )
        cases.append(host_case)
        specs.append(host_spec)
        probes.append(host_probe)
        labels.append(host_label)
        trials.extend(host_trials)

    config_payload = source_projection_set.config.model_dump(mode="json")
    config_payload.update(
        {
            "projection_set_id": config.projection_set_id,
            "excluded_allocation_case_ids": [],
            "attack_review_status": AttackReviewStatus.APPROVED,
        }
    )
    projection_set = ProjectionSet(
        source_projection_draft_sha256=(
            source_projection_set.source_projection_draft_sha256
        ),
        source_allocation_count=len(cases),
        config=ProjectionSetConfig.model_validate(config_payload),
        cases=tuple(cases),
        attack_specs=tuple(specs),
        trials=tuple(trials),
        probes=tuple(probes),
        private_labels=tuple(labels),
        exclusions=(),
    )
    try:
        _ = projection_set.manifest
    except ProjectionSetError as exc:
        raise NeutralHostError("neutral-host projection set is invalid") from exc
    return projection_set


def _build_host_family(
    host: NeutralHostDefinition,
    *,
    ordinal: int,
    source_case: ProjectionCase,
    source_spec: AttackSpec,
    source_probe: PublicProbe,
    source_label: PrivateLabel,
    clean_trial: TrialProjection,
) -> tuple[
    ProjectionCase,
    AttackSpec,
    PublicProbe,
    PrivateLabel,
    tuple[TrialProjection, ...],
]:
    host_spec_id = f"host-spec-{host.host_case_id}"
    host_probe_id = f"host-probe-{host.host_case_id}"
    host_label_id = f"host-label-{host.host_case_id}"
    source_events = tuple(
        event
        for event in clean_trial.replay_events
        if event.source_message_index is not None
    )
    expected_indexes = (
        source_case.pre_attack_source_message_indexes
        + source_case.survival_source_message_indexes
    )
    observed_indexes = tuple(event.source_message_index for event in source_events)
    if observed_indexes != expected_indexes or any(
        event.stage not in {EventStage.BENIGN_PREFIX, EventStage.BENIGN_SURVIVAL}
        for event in source_events
    ):
        raise NeutralHostError(
            "clean source trial does not contain the expected benign replay: "
            f"{host.source_case_id}"
        )

    host_case = ProjectionCase(
        projection_kind=ProjectionKind.NEUTRAL_EXTERNAL_HOST,
        source_allocation_case_id=source_case.allocation_case_id,
        projection_ordinal=ordinal,
        allocation_case_id=host.host_case_id,
        candidate_id=source_case.candidate_id,
        persona_id=source_case.persona_id,
        source_row_index=source_case.source_row_index,
        attack_id=source_case.attack_id,
        attack_methodology=source_case.attack_methodology,
        projection_decision_sha256=source_case.projection_decision_sha256,
        attack_spec_id=host_spec_id,
        probe_id=host_probe_id,
        private_label_id=host_label_id,
        evidence_range_start=source_case.evidence_range_start,
        evidence_range_end=source_case.evidence_range_end,
        insertion_after_source_message_index=(
            source_case.insertion_after_source_message_index
        ),
        survival_range_start=source_case.survival_range_start,
        survival_range_end=source_case.survival_range_end,
        pre_attack_source_message_indexes=(
            source_case.pre_attack_source_message_indexes
        ),
        survival_source_message_indexes=(
            source_case.survival_source_message_indexes
        ),
        quarantined_system_message_indexes=(
            source_case.quarantined_system_message_indexes
        ),
    )
    host_spec = AttackSpec(
        projection_kind=ProjectionKind.NEUTRAL_EXTERNAL_HOST,
        attack_spec_id=host_spec_id,
        allocation_case_id=host.host_case_id,
        attack_id=source_spec.attack_id,
        attack_methodology=source_spec.attack_methodology,
        variant="neutral_external_host",
        insert_after_source_message_index=(
            source_spec.insert_after_source_message_index
        ),
        public_attacker_view_sha256=source_spec.public_attacker_view_sha256,
        target_reference_sha256=source_spec.target_reference_sha256,
        token_delta_by_event=(),
        maximum_event_tokens=source_spec.maximum_event_tokens,
        review_status=AttackReviewStatus.APPROVED,
        payload_sha256=_canonical_sha256(
            {"matched_benign": [], "poisoned": []}
        ),
    )
    host_probe = PublicProbe.model_validate(
        {
            **source_probe.model_dump(mode="json"),
            "probe_id": host_probe_id,
            "allocation_case_id": host.host_case_id,
        }
    )
    host_label = PrivateLabel.model_validate(
        {
            **source_label.model_dump(mode="json"),
            "private_label_id": host_label_id,
            "allocation_case_id": host.host_case_id,
        }
    )
    host_trials = tuple(
        _build_host_trial(
            host_case=host_case,
            twin=twin,
            source_events=source_events,
        )
        for twin in TrialTwin
    )
    if len({memory_visible_replay_signature(trial) for trial in host_trials}) != 1:
        raise NeutralHostError(
            "neutral host twins must have identical memory-visible replay: "
            f"{host.host_case_id}"
        )
    return host_case, host_spec, host_probe, host_label, host_trials


def _build_host_trial(
    *,
    host_case: ProjectionCase,
    twin: TrialTwin,
    source_events: Sequence[ProjectionEvent],
) -> TrialProjection:
    trial_id = f"{host_case.allocation_case_id}--{twin.value.replace('_', '-')}"
    events = tuple(
        ProjectionEvent(
            event_id=f"{trial_id}.event-{ordinal:04d}",
            ordinal=ordinal,
            stage=event.stage,
            role=event.role,
            content=event.content,
            source_message_index=event.source_message_index,
            token_count=event.token_count,
            content_sha256=event.content_sha256,
        )
        for ordinal, event in enumerate(source_events, start=1)
    )
    return TrialProjection(
        projection_kind=ProjectionKind.NEUTRAL_EXTERNAL_HOST,
        trial_id=trial_id,
        allocation_case_id=host_case.allocation_case_id,
        twin=twin,
        attack_spec_id=host_case.attack_spec_id,
        probe_id=host_case.probe_id,
        private_label_id=host_case.private_label_id,
        replay_events=events,
        replay_sha256=_canonical_sha256(
            [event.model_dump(mode="json") for event in events]
        ),
    )


def _validate_delivery_availability(
    config: NeutralHostConfig,
    *,
    projection_set: ProjectionSet,
    delivery_sets: Sequence[DeliveryProjectionSet],
) -> None:
    """Require every supplied delivery to become available in its host replay."""

    host_trials = {
        case.persona_id: next(
            trial
            for trial in projection_set.trials
            if (
                trial.allocation_case_id == case.allocation_case_id
                and trial.twin == TrialTwin.CLEAN
            )
        )
        for case in projection_set.cases
    }
    for delivery_set in delivery_sets:
        if delivery_set.projection_set_id != config.projection_set_id:
            raise NeutralHostError(
                "delivery projection set targets a different projection set: "
                f"{delivery_set.projection_set_id}"
            )
        for family in delivery_set.families:
            if family.persona_id is None:
                raise NeutralHostError(
                    "external delivery family has no persona provenance: "
                    f"{family.delivery_family_id}"
                )
            try:
                host_trial = host_trials[family.persona_id]
            except KeyError as exc:
                raise NeutralHostError(
                    "no neutral host exists for external delivery persona "
                    f"{family.persona_id}: {family.delivery_family_id}"
                ) from exc
            replay_indexes = {
                event.source_message_index
                for event in host_trial.replay_events
                if event.source_message_index is not None
            }
            for delivery in family.deliveries:
                if delivery.available_after_source_message_index not in replay_indexes:
                    raise NeutralHostError(
                        "external delivery availability index is absent from "
                        f"neutral host replay: {delivery.delivery_id}"
                    )
