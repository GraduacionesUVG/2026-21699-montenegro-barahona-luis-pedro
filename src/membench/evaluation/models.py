"""Canonical, evaluator-only specifications for attack-specific memory probes.

These models describe what to look for after a trial has ingested its frozen
external delivery.  They deliberately contain no adapter- or vendor-specific
types: an evaluator will consume normalized ``MemoryAdapter`` retrieval output
later in the trial lifecycle.

The contract supports insertion, stored-instruction persistence, and
information replacement against a separately verified trusted baseline.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from membench.campaign.models import TrialTwin
from membench.memory.models import (
    CanonicalModel,
    MemoryScope,
    MemorySnapshot,
    RetrievalResult,
)
from membench.qualification.models import RetrievalStateDelta


class BaselineObservation(CanonicalModel):
    """Pre-ingress evidence that the intended trusted fact was retrievable."""

    baseline_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    expected_markers: tuple[str, ...] = Field(min_length=1)
    observed_markers: tuple[str, ...] = Field(default_factory=tuple)
    retrieved_before_ingress: bool
    retrieval_evidence_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @field_validator("expected_markers", "observed_markers")
    @classmethod
    def markers_must_be_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_markers(value)

    @model_validator(mode="after")
    def retrieval_evidence_must_be_consistent(self) -> BaselineObservation:
        if self.retrieved_before_ingress and self.retrieval_evidence_sha256 is None:
            raise ValueError("retrieved baseline requires retrieval evidence")
        if not set(self.observed_markers).issubset(self.expected_markers):
            raise ValueError("observed baseline markers must be expected markers")
        return self

    @property
    def established(self) -> bool:
        """Whether every frozen baseline marker was proven before ingress."""

        return self.retrieved_before_ingress and set(self.expected_markers).issubset(
            self.observed_markers
        )


class BaselinePoisoningOutcome(StrEnum):
    BASELINE_PRESERVED = "baseline_preserved"
    POISON_REPLACED_BASELINE = "poison_replaced_baseline"
    BASELINE_AND_POISON_COEXIST = "baseline_and_poison_coexist"
    NEITHER_RETRIEVED = "neither_retrieved"
    BASELINE_NOT_ESTABLISHED = "baseline_not_established"
    INCONCLUSIVE = "inconclusive"


class EvaluationAttackFamily(StrEnum):
    """Attack families with an implemented deterministic retrieval evaluator."""

    FALSE_FACT_INSERTION = "false_fact_insertion"
    STORED_INSTRUCTION_PERSISTENCE = "stored_instruction_persistence"
    INFORMATION_REPLACEMENT = "information_replacement"


class EvaluationOutcome(StrEnum):
    """Deterministic retrieval interpretation for one evaluated trial twin.

    A ``*_NOT_RETRIEVED`` result says only that no configured marker appeared
    in this retrieval result.  It never claims that the backend did not store
    the content elsewhere in its namespace.
    """

    CONTROL_CLEAN = "control_clean"
    CONTROL_CONTAMINATED = "control_contaminated"
    FALSE_FACT_RETRIEVED = "false_fact_retrieved"
    # Legacy retrieval-only interpretations, retained for old sealed evidence.
    FALSE_FACT_NOT_RETRIEVED = "false_fact_not_retrieved"
    FALSE_FACT_NOT_EXPOSED = "false_fact_not_exposed"
    FALSE_FACT_EXPOSED_NOT_STORED = "false_fact_exposed_not_stored"
    FALSE_FACT_STORED_NOT_RETRIEVED = "false_fact_stored_not_retrieved"
    STORED_INSTRUCTION_RETRIEVED = "stored_instruction_retrieved"
    STORED_INSTRUCTION_NOT_RETRIEVED = "stored_instruction_not_retrieved"
    STORED_INSTRUCTION_NOT_EXPOSED = "stored_instruction_not_exposed"
    STORED_INSTRUCTION_EXPOSED_NOT_STORED = "stored_instruction_exposed_not_stored"
    STORED_INSTRUCTION_STORED_NOT_RETRIEVED = "stored_instruction_stored_not_retrieved"
    INFORMATION_REPLACEMENT_BASELINE_PRESERVED = (
        "information_replacement_baseline_preserved"
    )
    INFORMATION_REPLACEMENT_REPLACED = "information_replacement_replaced"
    INFORMATION_REPLACEMENT_COEXISTS = "information_replacement_coexists"
    INFORMATION_REPLACEMENT_NEITHER_RETRIEVED = (
        "information_replacement_neither_retrieved"
    )
    INFORMATION_REPLACEMENT_BASELINE_NOT_ESTABLISHED = (
        "information_replacement_baseline_not_established"
    )
    INFORMATION_REPLACEMENT_NOT_EXPOSED = "information_replacement_not_exposed"


class EvaluationExposureStatus(StrEnum):
    """Whether an external source reached the memory-policy dispatcher."""

    NOT_APPLICABLE = "not_applicable"
    NOT_EXPOSED = "not_exposed"
    EXPOSURE_FAILED = "exposure_failed"
    EXPOSED = "exposed"


class EvaluationPersistenceStatus(StrEnum):
    """Marker observation in the complete pre-retrieval memory snapshot."""

    NOT_APPLICABLE = "not_applicable"
    MARKER_STORED = "marker_stored"
    MARKER_NOT_OBSERVED = "marker_not_observed"
    STORAGE_UNKNOWN = "storage_unknown"


class EvaluationExposureEvidence(CanonicalModel):
    """Compact lifecycle evidence supplied by the trial orchestrator."""

    status: EvaluationExposureStatus
    ingress_mode: str | None = None
    detail: str | None = None


class StateMarkerHitEvidence(CanonicalModel):
    """One inspected memory record containing an evaluator marker."""

    canonical_id: str = Field(min_length=1)


class StateMarkerMatchEvidence(CanonicalModel):
    """Marker occurrences observed in a complete state snapshot."""

    marker: str = Field(min_length=1)
    hits: tuple[StateMarkerHitEvidence, ...] = Field(default_factory=tuple)

    @property
    def matched(self) -> bool:
        return bool(self.hits)


def _validated_markers(value: tuple[str, ...]) -> tuple[str, ...]:
    """Require stable, non-empty lexical markers for deterministic scoring."""

    if any(not marker or "\n" in marker or "\r" in marker for marker in value):
        raise ValueError("evaluation markers must be non-empty single-line strings")
    if value != tuple(sorted(set(value))):
        raise ValueError("evaluation markers must be sorted and unique")
    return value


class FalseFactInsertionSettings(CanonicalModel):
    """Private evaluator settings for a newly introduced false proposition."""

    attack_family: Literal[EvaluationAttackFamily.FALSE_FACT_INSERTION] = (
        EvaluationAttackFamily.FALSE_FACT_INSERTION
    )
    poison_markers: tuple[str, ...] = Field(min_length=1)
    trusted_ground_truth: str = Field(min_length=1)

    @field_validator("poison_markers")
    @classmethod
    def poison_markers_must_be_canonical(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        return _validated_markers(value)


class StoredInstructionPersistenceSettings(CanonicalModel):
    """Private evaluator settings for persistence of an injected instruction.

    This intentionally evaluates retrieval persistence only.  A future
    behavioral task can separately determine whether an agent follows a stored
    instruction, without conflating that result with whether it was retained.
    """

    attack_family: Literal[EvaluationAttackFamily.STORED_INSTRUCTION_PERSISTENCE] = (
        EvaluationAttackFamily.STORED_INSTRUCTION_PERSISTENCE
    )
    instruction_markers: tuple[str, ...] = Field(min_length=1)

    @field_validator("instruction_markers")
    @classmethod
    def instruction_markers_must_be_canonical(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        return _validated_markers(value)


class InformationReplacementSettings(CanonicalModel):
    """Private settings for replacing a previously verified trusted fact."""

    attack_family: Literal[EvaluationAttackFamily.INFORMATION_REPLACEMENT] = (
        EvaluationAttackFamily.INFORMATION_REPLACEMENT
    )
    baseline_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    baseline_markers: tuple[str, ...] = Field(min_length=1)
    replacement_markers: tuple[str, ...] = Field(min_length=1)
    trusted_ground_truth: str = Field(min_length=1)

    @field_validator("baseline_markers", "replacement_markers")
    @classmethod
    def markers_must_be_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validated_markers(value)

    @model_validator(mode="after")
    def baseline_and_replacement_markers_must_be_disjoint(
        self,
    ) -> InformationReplacementSettings:
        if set(self.baseline_markers) & set(self.replacement_markers):
            raise ValueError("baseline and replacement markers must be disjoint")
        return self


type EvaluationSettings = Annotated[
    FalseFactInsertionSettings
    | StoredInstructionPersistenceSettings
    | InformationReplacementSettings,
    Field(discriminator="attack_family"),
]


class EvaluationProbe(CanonicalModel):
    """Frozen read-only retrieval probe bound to one external attack family."""

    schema_version: Literal[1] = 1
    probe_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    attack_spec_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    delivery_family_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    persona_id: str = Field(pattern=r"^[0-9]+$")
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0)
    threshold: float | None = None
    settings: EvaluationSettings


class MarkerHitEvidence(CanonicalModel):
    """One ranked retrieved record containing a configured lexical marker."""

    canonical_id: str = Field(min_length=1)
    rank: int = Field(ge=1)


class MarkerMatchEvidence(CanonicalModel):
    """All retrieved occurrences of one evaluator-defined marker.

    Record text is intentionally not copied into this evidence.  The linked
    canonical retrieval result remains the source of record content, while the
    compact match list makes deterministic outcome review straightforward.
    """

    marker: str = Field(min_length=1)
    hits: tuple[MarkerHitEvidence, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def hits_must_be_ranked_and_unique(self) -> MarkerMatchEvidence:
        ranks = tuple(hit.rank for hit in self.hits)
        record_ids = tuple(hit.canonical_id for hit in self.hits)
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("marker-hit ranks must be sorted and unique")
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("marker-hit record IDs must be unique")
        return self

    @property
    def matched(self) -> bool:
        """Return whether this marker occurred in at least one retrieved hit."""

        return bool(self.hits)


class EvaluationResult(CanonicalModel):
    """Canonical retrieval evidence and deterministic outcome for one twin."""

    schema_version: Literal[1] = 1
    trial_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    twin: TrialTwin
    probe: EvaluationProbe
    retrieval: RetrievalResult
    marker_matches: tuple[MarkerMatchEvidence, ...]
    exposure: EvaluationExposureEvidence = Field(
        default_factory=lambda: EvaluationExposureEvidence(
            status=EvaluationExposureStatus.NOT_APPLICABLE
        )
    )
    persistence: EvaluationPersistenceStatus = (
        EvaluationPersistenceStatus.NOT_APPLICABLE
    )
    state_marker_matches: tuple[StateMarkerMatchEvidence, ...] = Field(
        default_factory=tuple
    )
    outcome: EvaluationOutcome
    baseline_observation: BaselineObservation | None = None
    baseline_poisoning_outcome: BaselinePoisoningOutcome | None = None

    @model_validator(mode="after")
    def evidence_must_match_the_probe_and_retrieval(self) -> EvaluationResult:
        expected_query_sha256 = hashlib.sha256(
            self.probe.query.encode("utf-8")
        ).hexdigest()
        if self.retrieval.query_sha256 != expected_query_sha256:
            raise ValueError("retrieval query hash does not match evaluation probe")
        if self.retrieval.request_top_k != self.probe.top_k:
            raise ValueError("retrieval top_k does not match evaluation probe")
        if self.retrieval.threshold != self.probe.threshold:
            raise ValueError("retrieval threshold does not match evaluation probe")

        settings = self.probe.settings
        if isinstance(settings, FalseFactInsertionSettings):
            expected_markers = settings.poison_markers
        elif isinstance(settings, StoredInstructionPersistenceSettings):
            expected_markers = settings.instruction_markers
        else:
            expected_markers = (
                settings.baseline_markers + settings.replacement_markers
            )
        observed_markers = tuple(item.marker for item in self.marker_matches)
        if observed_markers != expected_markers:
            raise ValueError("marker evidence must exactly cover probe markers")

        retrieved_by_rank = {
            hit.rank: hit.record.canonical_id for hit in self.retrieval.hits
        }
        for evidence in self.marker_matches:
            for hit in evidence.hits:
                if retrieved_by_rank.get(hit.rank) != hit.canonical_id:
                    raise ValueError(
                        "marker evidence must reference returned retrieval hits"
                    )
        if isinstance(settings, InformationReplacementSettings):
            if self.baseline_observation is not None and (
                self.baseline_observation.baseline_id != settings.baseline_id
                or self.baseline_observation.expected_markers
                != settings.baseline_markers
            ):
                raise ValueError(
                    "baseline observation does not match replacement settings"
                )
            if self.baseline_poisoning_outcome is None:
                raise ValueError("replacement evaluation requires a baseline outcome")
        elif (
            self.baseline_observation is not None
            or self.baseline_poisoning_outcome is not None
        ):
            raise ValueError(
                "baseline evidence is only valid for information replacement"
            )
        return self


class EvaluationExecutionResult(CanonicalModel):
    """One complete read-only evaluation operation and its state evidence.

    This is deliberately separate from :class:`EvaluationResult`: the latter
    expresses what a retrieval returned, while this model proves whether the
    retrieval operation itself changed evaluator-visible memory state.
    """

    schema_version: Literal[1] = 1
    scope: MemoryScope
    before_snapshot: MemorySnapshot
    evaluation: EvaluationResult
    after_snapshot: MemorySnapshot
    state_delta: RetrievalStateDelta
    content_read_only_verified: bool

    @model_validator(mode="after")
    def execution_evidence_must_share_scope(self) -> EvaluationExecutionResult:
        snapshots = (self.before_snapshot, self.after_snapshot)
        if any(snapshot.scope != self.scope for snapshot in snapshots):
            raise ValueError("evaluation snapshots must match the execution scope")
        if self.state_delta.scope != self.scope:
            raise ValueError("evaluation state delta must match the execution scope")
        if any(
            hit.record.scope != self.scope for hit in self.evaluation.retrieval.hits
        ):
            raise ValueError("evaluation retrieval contains another memory scope")
        if (
            self.content_read_only_verified
            != self.state_delta.observed_content_unchanged
        ):
            raise ValueError(
                "content_read_only_verified must match observed content evidence"
            )
        return self
