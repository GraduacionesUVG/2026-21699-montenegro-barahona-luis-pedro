"""Early, dependency-free validation for authored scenario specifications."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from membench.scenario.ids import ScenarioIds, derive_scenario_ids
from membench.scenario.models import ScenarioSpec


class ScenarioValidationError(ValueError):
    """A scenario cannot safely be compiled into frozen campaign inputs."""


@dataclass(frozen=True)
class ScenarioValidationReport:
    """Validated compiler inputs, without writing any artifact."""

    scenario_key: str
    ids: ScenarioIds
    validated_source_message_indexes: tuple[int, ...]


def validate_scenario_spec(
    spec: ScenarioSpec,
    *,
    host_source_message_indexes: Iterable[int] | None = None,
    known_condition_profiles: Iterable[str] | None = None,
) -> ScenarioValidationReport:
    """Validate references that require compiler-owned registries.

    Pydantic validates intrinsic content when ``ScenarioSpec`` is constructed.
    This function validates optional host/profile registry evidence supplied by
    the caller, without opening files or producing artifacts.
    """

    spec = ScenarioSpec.model_validate(spec.model_dump(mode="json"))
    ids = derive_scenario_ids(spec)
    _validate_condition_profiles(spec, known_condition_profiles)
    indexes = _validate_host_indexes(spec, host_source_message_indexes)
    _validate_generated_ids(ids)
    return ScenarioValidationReport(
        scenario_key=ids.scenario_key,
        ids=ids,
        validated_source_message_indexes=indexes,
    )


def _validate_condition_profiles(
    spec: ScenarioSpec, known: Iterable[str] | None
) -> None:
    if known is None:
        return
    available = set(known)
    missing = sorted(set(spec.matrix.condition_profiles) - available)
    if missing:
        raise ScenarioValidationError(
            "unknown condition profile(s): " + ", ".join(missing)
        )


def _validate_host_indexes(
    spec: ScenarioSpec, indexes: Iterable[int] | None
) -> tuple[int, ...]:
    if indexes is None:
        return ()
    observed = tuple(sorted(set(indexes)))
    if any(index < 0 for index in observed):
        raise ScenarioValidationError("host source-message indexes cannot be negative")
    required = {spec.ingress.available_after_source_message_index}
    if spec.baseline is not None:
        required.add(spec.baseline.seed_after_source_message_index)
    missing = sorted(required - set(observed))
    if missing:
        raise ScenarioValidationError(
            "neutral host is missing required source-message index(es): "
            + ", ".join(str(value) for value in missing)
        )
    return observed


def _validate_generated_ids(ids: ScenarioIds) -> None:
    values = (
        ids.attack_set_id,
        ids.delivery_set_id,
        ids.recipe_set_id,
        ids.attack_spec_id,
        ids.delivery_family_id,
        ids.baseline_id,
        ids.evaluation_probe_id,
        ids.selection_task_id,
        ids.behavioral_task_id,
        ids.behavioral_oracle_id,
        ids.ingress_plan_set_id,
    )
    if len(values) != len(set(values)):
        raise ScenarioValidationError("derived scenario identities must be unique")
