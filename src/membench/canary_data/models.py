"""Strict contracts for synthetic canaries and common qualifications."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from membench.contract import WritePolicy
from membench.memory.capabilities import AdapterCapabilities
from membench.memory.models import (
    CanonicalModel,
    JsonObject,
    MemoryMessage,
    MemoryType,
)


class CanaryCategory(StrEnum):
    """Purpose of one synthetic record within qualification."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    ISOLATION = "isolation"
    GUARD_REJECTION = "guard_rejection"


class GuardExpectation(StrEnum):
    """Expected shared-guard treatment, independent of storage behavior."""

    WRITE = "write"
    REJECT = "reject"


class CanaryRetrievalExpectation(CanonicalModel):
    """Query and text evidence required after a successful canary write."""

    query: str = Field(min_length=1)
    required_fragments: tuple[str, ...] = Field(min_length=1)

    @field_validator("required_fragments")
    @classmethod
    def fragments_must_be_unique_and_nonblank(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not fragment.strip() for fragment in value):
            raise ValueError("required fragments must not be blank")
        normalized = [fragment.strip().casefold() for fragment in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("required fragments must be case-insensitively unique")
        return value


class SyntheticCanary(CanonicalModel):
    """Safe, versioned input and expected evidence for adapter qualification."""

    schema_version: Literal[1] = 1
    canary_id: str = Field(pattern=r"^[a-z][a-z0-9_]*_v\d+$")
    category: CanaryCategory
    description: str = Field(min_length=1)
    synthetic: Literal[True] = True
    safe_to_log_raw: Literal[True] = True
    marker: str = Field(min_length=6, pattern=r"^[A-Z0-9-]+$")
    memory_type: MemoryType
    source_event_id: str = Field(pattern=r"^canary-event-[a-z0-9-]+$")
    messages: tuple[MemoryMessage, ...] = Field(min_length=1)
    retrieval_expectations: tuple[CanaryRetrievalExpectation, ...] = Field(
        min_length=1
    )
    guard_expectation: GuardExpectation
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def marker_and_category_must_match_payload(self) -> Self:
        conversation = "\n".join(message.content for message in self.messages)
        if self.marker.casefold() not in conversation.casefold():
            raise ValueError("canary marker must appear in its conversation")
        if not any(
            self.marker.casefold() in fragment.casefold()
            for expectation in self.retrieval_expectations
            for fragment in expectation.required_fragments
        ):
            raise ValueError("canary marker must be required retrieval evidence")

        expected_type = {
            CanaryCategory.SEMANTIC: MemoryType.SEMANTIC,
            CanaryCategory.EPISODIC: MemoryType.EPISODIC,
            CanaryCategory.PROCEDURAL: MemoryType.PROCEDURAL,
        }.get(self.category)
        if expected_type is not None and self.memory_type != expected_type:
            raise ValueError(
                f"{self.category.value} canaries require "
                f"memory_type={expected_type.value}"
            )
        if (
            self.category == CanaryCategory.GUARD_REJECTION
            and self.guard_expectation != GuardExpectation.REJECT
        ):
            raise ValueError("guard-rejection canaries must expect rejection")
        return self


class CanaryFile(CanonicalModel):
    """Integrity-pinned canary document referenced by the suite manifest."""

    canary_id: str = Field(pattern=r"^[a-z][a-z0-9_]*_v\d+$")
    path: str = Field(pattern=r"^canaries/[a-z][a-z0-9_]*_v\d+\.json$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def path_stem_must_match_id(self) -> Self:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("canary paths must be relative and traversal-free")
        if path.stem != self.canary_id:
            raise ValueError("canary path stem must match canary_id")
        return self


class CommonQualification(CanonicalModel):
    """Reusable selection of canaries and lifecycle prerequisites."""

    qualification_id: str = Field(pattern=r"^[a-z][a-z0-9_]*_v\d+$")
    description: str = Field(min_length=1)
    canary_ids: tuple[str, ...] = Field(min_length=1)
    supported_policies: tuple[WritePolicy, ...] = Field(min_length=1)
    required_capabilities: tuple[str, ...] = Field(default_factory=tuple)
    requires_secondary_scope: bool = False
    requires_guard: bool = False
    requires_responder: bool = False

    @field_validator(
        "canary_ids",
        "supported_policies",
        "required_capabilities",
    )
    @classmethod
    def tuple_values_must_be_unique(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(value) != len(set(value)):
            raise ValueError("qualification tuple values must be unique")
        return value

    @field_validator("required_capabilities")
    @classmethod
    def capabilities_must_be_known(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        known = frozenset(AdapterCapabilities.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError("unknown capabilities: " + ", ".join(unknown))
        return value

    @model_validator(mode="after")
    def guard_dependency_must_match_policies(self) -> Self:
        has_guarded_policy = WritePolicy.SHARED_GUARDED in self.supported_policies
        if self.requires_guard and self.supported_policies != (
            WritePolicy.SHARED_GUARDED,
        ):
            raise ValueError(
                "guard qualifications must support only shared_guarded"
            )
        if has_guarded_policy and not self.requires_guard:
            raise ValueError(
                "qualifications supporting shared_guarded must require a guard"
            )
        return self


class CanaryManifest(CanonicalModel):
    """Integrity manifest and reusable qualification catalogue."""

    schema_version: Literal[1] = 1
    suite_id: Literal["common_adapter_qualification"]
    suite_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    canaries: tuple[CanaryFile, ...] = Field(min_length=1)
    qualifications: tuple[CommonQualification, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_and_references_must_be_valid(self) -> Self:
        canary_ids = [item.canary_id for item in self.canaries]
        if len(canary_ids) != len(set(canary_ids)):
            raise ValueError("manifest canary IDs must be unique")
        paths = [item.path for item in self.canaries]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest canary paths must be unique")
        qualification_ids = [
            item.qualification_id for item in self.qualifications
        ]
        if len(qualification_ids) != len(set(qualification_ids)):
            raise ValueError("manifest qualification IDs must be unique")

        known = set(canary_ids)
        for qualification in self.qualifications:
            unknown = sorted(set(qualification.canary_ids) - known)
            if unknown:
                raise ValueError(
                    f"qualification {qualification.qualification_id!r} references "
                    "unknown canaries: " + ", ".join(unknown)
                )
        return self
