"""Immutable contracts for external email and document ingress.

These models intentionally describe an external source independently from an
agent tool.  The same frozen delivery can therefore be supplied directly to a
writer or first exposed through a mock inbox/document tool.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.memory.models import CanonicalModel, JsonObject, MemoryScope
from membench.memory.policy import PolicyDispatchResult

_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class BaselineSeedPolicy(StrEnum):
    """The immutable route used to establish trusted ground truth."""

    DIRECT = "direct"


class TrustedBaselineSeed(CanonicalModel):
    """Frozen trusted fact scheduled for memory insertion before an attack."""

    schema_version: Literal[1] = 1
    baseline_id: str = Field(pattern=_ID_PATTERN)
    persona_id: int = Field(ge=0)
    content: str = Field(min_length=1, max_length=100_000)
    retrieval_query: str = Field(min_length=1)
    required_markers: tuple[str, ...] = Field(min_length=1)
    forbidden_pre_ingress_markers: tuple[str, ...] = Field(default_factory=tuple)

    available_after_source_message_index: int = Field(ge=0)

    # Trusted ground truth is intentionally not subject to a mechanism's
    # native extraction policy.  This lets the replay and attack remain the
    # independently selected experimental write path.
    seed_policy: BaselineSeedPolicy = BaselineSeedPolicy.DIRECT

    source_kind: Literal["synthetic_trusted_record"] = "synthetic_trusted_record"
    delivery_kind: Literal["trusted_baseline"] = "trusted_baseline"

    fixture_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    fixture_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    content_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("required_markers", "forbidden_pre_ingress_markers")
    @classmethod
    def markers_must_be_unique_and_nonblank(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("baseline markers must be nonblank and unique")
        return value

    @model_validator(mode="after")
    def baseline_content_and_markers_must_match(self) -> Self:
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_sha256 != expected:
            raise ValueError("baseline content_sha256 does not match content")
        if any(marker not in self.content for marker in self.required_markers):
            raise ValueError("baseline content must contain every required marker")
        if set(self.required_markers) & set(self.forbidden_pre_ingress_markers):
            raise ValueError("trusted and forbidden baseline markers must differ")
        if (self.fixture_id is None) != (self.fixture_sha256 is None):
            raise ValueError("baseline fixture ID and SHA-256 must be paired")
        return self

class TrustedBaselineManifest(CanonicalModel):
    baseline_set_id: str = Field(pattern=_ID_PATTERN)
    baselines: tuple[TrustedBaselineSeed, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def baseline_ids_must_be_unique(self) -> Self:
        ids = tuple(item.baseline_id for item in self.baselines)
        if len(ids) != len(set(ids)):
            raise ValueError("trusted baseline IDs must be unique")
        return self


class ExternalSourceKind(StrEnum):
    """Kinds of source supplied through controlled external ingress."""

    EMAIL = "email"
    DOCUMENT = "document"


class IngressMode(StrEnum):
    """Whether a model selects the source before memory ingestion."""

    AGENT_SELECTED = "agent_selected"
    FORCED_INGESTION = "forced_ingestion"


class ExternalDelivery(CanonicalModel):
    """One exact twin-specific external source, frozen before execution."""

    schema_version: Literal[1] = 1
    delivery_id: str = Field(pattern=_ID_PATTERN)
    delivery_family_id: str = Field(pattern=_ID_PATTERN)
    trial_id: str = Field(pattern=_ID_PATTERN)
    allocation_case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    twin: Literal["clean", "matched_benign", "poisoned"]
    source_kind: ExternalSourceKind
    source_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=1, max_length=300)
    sender: str | None = Field(default=None, min_length=3, max_length=320)
    content: str = Field(min_length=1, max_length=100_000)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    available_after_source_message_index: int = Field(ge=0)

    @field_validator("sender")
    @classmethod
    def sender_must_be_nonblank_when_present(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("delivery sender cannot be blank")
        return value

    @model_validator(mode="after")
    def source_metadata_must_match_kind_and_content(self) -> Self:
        if self.source_kind == ExternalSourceKind.EMAIL and self.sender is None:
            raise ValueError("email deliveries require a sender")
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_sha256 != expected:
            raise ValueError("delivery content_sha256 does not match content")
        return self


class SourceSelectionTask(CanonicalModel):
    """Public task used only by the agent-selected ingress stage."""

    schema_version: Literal[1] = 1
    selection_task_id: str = Field(pattern=_ID_PATTERN)
    delivery_family_id: str = Field(pattern=_ID_PATTERN)
    allocation_case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    instruction: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    allowed_tools: tuple[str, ...] = Field(min_length=1)

    @field_validator("allowed_tools")
    @classmethod
    def tools_must_be_sorted_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or not item.replace("_", "a").isalnum() for item in value):
            raise ValueError("selection task tool names are malformed")
        if value != tuple(sorted(set(value))):
            raise ValueError("selection task tool names must be sorted and unique")
        return value


class NormalizedExternalContent(CanonicalModel):
    """Versioned exact representation supplied to a memory write policy."""

    normalizer: Literal["external_source_normalizer_v1"]
    source_kind: ExternalSourceKind
    source_id: str = Field(pattern=_ID_PATTERN)
    source_content_sha256: str = Field(pattern=_SHA256_PATTERN)
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def text_hash_must_match(self) -> Self:
        expected = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_sha256 != expected:
            raise ValueError("normalized external-content hash does not match text")
        return self


class ExternalIngestionRequest(CanonicalModel):
    """Trusted input to the common memory-policy dispatcher."""

    operation_id: str = Field(min_length=1)
    scope: MemoryScope
    mode: IngressMode
    delivery: ExternalDelivery
    normalized: NormalizedExternalContent
    source_event_id: str = Field(min_length=1)
    native_options: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalized_source_must_match_delivery(self) -> Self:
        if (
            self.normalized.source_kind != self.delivery.source_kind
            or self.normalized.source_id != self.delivery.source_id
            or self.normalized.source_content_sha256 != self.delivery.content_sha256
        ):
            raise ValueError("normalized content does not belong to its delivery")
        return self


class ExternalIngestionResult(CanonicalModel):
    """Auditable result of one source becoming eligible for memory storage."""

    request: ExternalIngestionRequest
    dispatch: PolicyDispatchResult
    selected_by_agent: bool

    @model_validator(mode="after")
    def selection_flag_must_match_mode(self) -> Self:
        expected = self.request.mode == IngressMode.AGENT_SELECTED
        if self.selected_by_agent != expected:
            raise ValueError("selected_by_agent does not match ingress mode")
        if self.dispatch.operation_id != self.request.operation_id:
            raise ValueError("dispatch operation_id does not match ingress request")
        return self
