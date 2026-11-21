"""Integrity-checking loader for bundled synthetic qualification data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from membench.canary_data.models import (
    CanaryManifest,
    CommonQualification,
    SyntheticCanary,
)


class CanaryDataError(RuntimeError):
    """Raised when bundled qualification data is invalid or has changed."""


@dataclass(frozen=True, slots=True)
class CanarySuite:
    """Validated manifest and canaries for one immutable suite version."""

    manifest: CanaryManifest
    canaries: tuple[SyntheticCanary, ...]

    @classmethod
    def load(cls, root: Path | None = None) -> CanarySuite:
        """Load bundled data, verify hashes, and validate cross-file identity."""
        suite_root = root or Path(__file__).resolve().parent
        manifest_path = suite_root / "manifest.json"
        manifest = _read_model(manifest_path, CanaryManifest)
        canaries: list[SyntheticCanary] = []
        for entry in manifest.canaries:
            path = suite_root.joinpath(*entry.path.split("/"))
            actual_hash = _sha256(path)
            if actual_hash != entry.sha256:
                raise CanaryDataError(
                    f"integrity check failed for {path}: {actual_hash}; "
                    f"expected {entry.sha256}"
                )
            canary = _read_model(path, SyntheticCanary)
            if canary.canary_id != entry.canary_id:
                raise CanaryDataError(
                    f"canary ID mismatch for {path}: {canary.canary_id!r}; "
                    f"expected {entry.canary_id!r}"
                )
            canaries.append(canary)
        return cls(manifest=manifest, canaries=tuple(canaries))

    def canary(self, canary_id: str) -> SyntheticCanary:
        """Return one canary by stable ID."""
        try:
            return next(item for item in self.canaries if item.canary_id == canary_id)
        except StopIteration as exc:
            raise KeyError(f"unknown synthetic canary: {canary_id}") from exc

    def qualification(self, qualification_id: str) -> CommonQualification:
        """Return one common qualification by stable ID."""
        try:
            return next(
                item
                for item in self.manifest.qualifications
                if item.qualification_id == qualification_id
            )
        except StopIteration as exc:
            raise KeyError(f"unknown qualification: {qualification_id}") from exc

    def qualification_canaries(
        self,
        qualification_id: str,
    ) -> tuple[SyntheticCanary, ...]:
        """Resolve a qualification's ordered canary references."""
        qualification = self.qualification(qualification_id)
        return tuple(self.canary(item) for item in qualification.canary_ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CanaryDataError(f"could not read canary data {path}: {exc}") from exc
    return digest.hexdigest()


def _read_model[ModelT: (CanaryManifest, SyntheticCanary)](
    path: Path,
    model: type[ModelT],
) -> ModelT:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise CanaryDataError(f"invalid canary data {path}: {exc}") from exc
