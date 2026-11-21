"""Named, reusable mechanism/write-policy profiles for scenario matrices."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, ValidationError, field_validator, model_validator

from membench.campaign.models import ChildEnvironmentSpec
from membench.campaign.preparation_models import CampaignConditionSpec
from membench.contract import ConditionConfig
from membench.memory.models import CanonicalModel

_PROFILE_ID_PATTERN = r"^[a-z][a-z0-9_.-]*$"


class ScenarioProfileError(ValueError):
    """A condition-profile registry is malformed or cannot be loaded."""


class ScenarioConditionProfile(CanonicalModel):
    """One pinned mechanism and writing-policy condition."""

    profile_id: str = Field(pattern=_PROFILE_ID_PATTERN)
    condition: ConditionConfig
    execution_condition: CampaignConditionSpec | None = None
    environment: ChildEnvironmentSpec | None = None

    @model_validator(mode="after")
    def execution_details_must_match_the_named_condition(
        self,
    ) -> ScenarioConditionProfile:
        if (self.execution_condition is None) != (self.environment is None):
            raise ValueError(
                "execution_condition and environment must be supplied together"
            )
        if (
            self.execution_condition is not None
            and self.execution_condition.condition_id != self.profile_id
        ):
            raise ValueError("execution_condition.condition_id must equal profile_id")
        if (
            self.execution_condition is not None
            and self.execution_condition.manifest_template.condition != self.condition
        ):
            raise ValueError(
                "execution_condition manifest condition must match profile"
            )
        if (
            self.environment is not None
            and self.environment.mechanism != self.condition.mechanism
        ):
            raise ValueError("execution environment mechanism must match profile")
        return self


class ScenarioConditionProfileRegistry(CanonicalModel):
    """A small human-maintained registry referenced by ScenarioSpec.matrix."""

    schema_version: int = Field(default=1, ge=1)
    profiles: tuple[ScenarioConditionProfile, ...] = Field(min_length=1)

    @field_validator("profiles")
    @classmethod
    def profile_ids_must_be_unique(
        cls, value: tuple[ScenarioConditionProfile, ...]
    ) -> tuple[ScenarioConditionProfile, ...]:
        identifiers = tuple(profile.profile_id for profile in value)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("condition profile IDs must be unique")
        return value

    def resolve(self, profile_id: str) -> ScenarioConditionProfile:
        """Return one profile or fail before any artifact is generated."""

        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise ScenarioProfileError(f"unknown condition profile: {profile_id}")


def load_condition_profile_registry(
    path: str | Path,
) -> ScenarioConditionProfileRegistry:
    """Load the editable JSON registry without freezing or copying it."""

    source = Path(path)
    try:
        return ScenarioConditionProfileRegistry.model_validate_json(source.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise ScenarioProfileError(
            f"could not load condition profile registry {source}: {exc}"
        ) from exc
