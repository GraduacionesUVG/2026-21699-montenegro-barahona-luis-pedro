"""Load editable behavioral inputs and verify standalone frozen artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from membench.behavioral.models import (
    BehavioralTaskTemplate,
    CaseSummaryBehavioralOracle,
)
from membench.campaign.artifacts import canonical_json_bytes
from membench.memory.models import CanonicalModel

BEHAVIORAL_TASK_FILE = "behavioral_task.json"
BEHAVIORAL_ORACLE_FILE = "behavioral_oracle.json"
BEHAVIORAL_MANIFEST_FILE = "behavioral_manifest.json"
BEHAVIORAL_VERSION = "behavioral_task_v1"


class BehavioralInputLoadError(RuntimeError):
    """Raised when behavioral task input cannot be read or trusted."""


class BehavioralManifest(CanonicalModel):
    """Integrity metadata for a frozen task/oracle pair."""

    schema_version: Literal[1] = 1
    version: Literal["behavioral_task_v1"] = BEHAVIORAL_VERSION
    behavioral_task_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    oracle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_behavioral_task(path: str | Path) -> BehavioralTaskTemplate:
    """Load an editable public behavioral-task template."""

    source = Path(path)
    try:
        return BehavioralTaskTemplate.model_validate_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise BehavioralInputLoadError(
            f"could not load behavioral task: {source}"
        ) from exc


def load_behavioral_oracle(path: str | Path) -> CaseSummaryBehavioralOracle:
    """Load an editable evaluator-only behavioral oracle."""

    source = Path(path)
    try:
        return CaseSummaryBehavioralOracle.model_validate_json(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise BehavioralInputLoadError(
            f"could not load behavioral oracle: {source}"
        ) from exc


def export_frozen_behavioral_inputs(
    task: BehavioralTaskTemplate,
    oracle: CaseSummaryBehavioralOracle,
    destination: str | Path,
) -> BehavioralManifest:
    """Freeze one validated task/oracle pair without overwriting evidence."""

    _validate_pair(task, oracle)
    output = Path(destination)
    task_path = output / BEHAVIORAL_TASK_FILE
    oracle_path = output / BEHAVIORAL_ORACLE_FILE
    manifest_path = output / BEHAVIORAL_MANIFEST_FILE
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BehavioralInputLoadError(
            f"could not create frozen behavioral input directory: {output}"
        ) from exc
    if any(path.exists() for path in (task_path, oracle_path, manifest_path)):
        raise BehavioralInputLoadError("frozen behavioral output already exists")

    task_bytes = canonical_json_bytes(task)
    oracle_bytes = canonical_json_bytes(oracle)
    manifest = BehavioralManifest(
        behavioral_task_id=task.behavioral_task_id,
        task_sha256=hashlib.sha256(task_bytes).hexdigest(),
        oracle_id=oracle.oracle_id,
        oracle_sha256=hashlib.sha256(oracle_bytes).hexdigest(),
    )
    try:
        task_path.write_bytes(task_bytes)
        oracle_path.write_bytes(oracle_bytes)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
    except OSError as exc:
        raise BehavioralInputLoadError(
            f"could not write frozen behavioral inputs: {output}"
        ) from exc
    return manifest


def load_frozen_behavioral_inputs(
    destination: str | Path,
) -> tuple[BehavioralTaskTemplate, CaseSummaryBehavioralOracle]:
    """Load a canonical frozen pair and prove it matches its manifest."""

    root = Path(destination)
    try:
        task_bytes = (root / BEHAVIORAL_TASK_FILE).read_bytes()
        oracle_bytes = (root / BEHAVIORAL_ORACLE_FILE).read_bytes()
        manifest = BehavioralManifest.model_validate_json(
            (root / BEHAVIORAL_MANIFEST_FILE).read_bytes()
        )
        task = BehavioralTaskTemplate.model_validate_json(task_bytes)
        oracle = CaseSummaryBehavioralOracle.model_validate_json(oracle_bytes)
    except (OSError, ValueError) as exc:
        raise BehavioralInputLoadError(
            f"could not load frozen behavioral inputs: {root}"
        ) from exc
    if (
        canonical_json_bytes(task) != task_bytes
        or canonical_json_bytes(oracle) != oracle_bytes
    ):
        raise BehavioralInputLoadError("frozen behavioral JSON is not canonical")
    expected = BehavioralManifest(
        behavioral_task_id=task.behavioral_task_id,
        task_sha256=hashlib.sha256(task_bytes).hexdigest(),
        oracle_id=oracle.oracle_id,
        oracle_sha256=hashlib.sha256(oracle_bytes).hexdigest(),
    )
    if manifest != expected:
        raise BehavioralInputLoadError(
            "behavioral task/oracle content does not match manifest"
        )
    _validate_pair(task, oracle)
    return task, oracle


def validate_behavioral_pair(
    task: BehavioralTaskTemplate,
    oracle: CaseSummaryBehavioralOracle,
) -> None:
    """Public wrapper used by campaign preparation before freezing a pair."""

    _validate_pair(task, oracle)


def _validate_pair(
    task: BehavioralTaskTemplate,
    oracle: CaseSummaryBehavioralOracle,
) -> None:
    expected = {
        "behavioral_task_id": task.behavioral_task_id,
        "attack_spec_id": task.attack_spec_id,
        "delivery_family_id": task.delivery_family_id,
        "persona_id": task.persona_id,
    }
    observed = {
        "behavioral_task_id": oracle.behavioral_task_id,
        "attack_spec_id": oracle.attack_spec_id,
        "delivery_family_id": oracle.delivery_family_id,
        "persona_id": oracle.persona_id,
    }
    if observed != expected:
        mismatches = ", ".join(
            key for key in expected if expected[key] != observed[key]
        )
        raise BehavioralInputLoadError(
            "behavioral task and oracle identities differ: " + mismatches
        )
