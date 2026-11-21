"""Durable, append-only artifacts for one adapter qualification run."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from membench.contract import ExperimentManifest
from membench.memory.requirements import CapabilityValidation
from membench.qualification.events import QualificationEventSink
from membench.qualification.models import (
    QualificationEvent,
    QualificationPlan,
    QualificationReport,
)


class QualificationArtifactError(RuntimeError):
    """Raised when qualification evidence cannot be written safely."""


@dataclass(frozen=True, slots=True)
class QualificationArtifactIndex:
    """Paths and hashes produced after a qualification directory is sealed."""

    run_dir: Path
    files: tuple[Path, ...]
    sha256_by_name: Mapping[str, str]


class JsonlQualificationEventSink(QualificationEventSink):
    """Persist validated events in order, flushing each record to disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._last_sequence = 0
        self._closed = False
        self._lock = Lock()

    def emit(self, event: QualificationEvent) -> None:
        """Append exactly the next event sequence or fail instrumentation."""
        with self._lock:
            if self._closed:
                raise QualificationArtifactError("qualification event sink is closed")
            expected = self._last_sequence + 1
            if event.sequence != expected:
                raise QualificationArtifactError(
                    "qualification event sequence must be contiguous: "
                    f"expected {expected}, got {event.sequence}"
                )
            line = json.dumps(
                event.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            try:
                with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(line)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise QualificationArtifactError(
                    f"could not append qualification event to {self.path}: {exc}"
                ) from exc
            self._last_sequence = event.sequence

    def close(self) -> None:
        """Prevent further appends before checksums are computed."""
        with self._lock:
            self._closed = True


class QualificationArtifactStore:
    """Own one new run directory and atomically persist its final evidence."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._sealed = False
        self._event_sink = JsonlQualificationEventSink(run_dir / "events.jsonl")

    @classmethod
    def prepare(
        cls,
        artifact_root: Path,
        plan: QualificationPlan,
    ) -> QualificationArtifactStore:
        """Create a unique mechanism/run directory without overwriting evidence."""
        run_dir = artifact_root / plan.mechanism.value / plan.scope.run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise QualificationArtifactError(
                f"qualification artifact directory already exists: {run_dir}"
            ) from exc
        except OSError as exc:
            raise QualificationArtifactError(
                f"could not create qualification artifact directory {run_dir}: {exc}"
            ) from exc
        return cls(run_dir)

    @property
    def event_sink(self) -> JsonlQualificationEventSink:
        """Return the single append-only event sink owned by this store."""
        return self._event_sink

    def write_manifest(self, manifest: ExperimentManifest) -> Path:
        """Persist the fully validated manifest and its canonical hash."""
        return self._write_json(
            "resolved_manifest.json",
            {
                "sha256": manifest.sha256,
                "manifest": manifest.model_dump(mode="json"),
            },
        )

    def write_plan(self, plan: QualificationPlan) -> Path:
        """Persist the mechanism-neutral instructions executed by the runner."""
        return self._write_json("plan.json", plan.model_dump(mode="json"))

    def write_resolved_settings(self, settings: Mapping[str, Any]) -> Path:
        """Persist public adapter settings after rejecting likely secret fields."""
        _assert_public_payload(settings)
        return self._write_json("resolved_settings.json", dict(settings))

    def write_capability_validation(
        self,
        validation: CapabilityValidation,
    ) -> Path:
        """Persist the manifest-to-adapter capability preflight result."""
        return self._write_json("capabilities.json", validation.as_dict())

    def write_responder_settings(self, settings: Mapping[str, Any]) -> Path:
        """Persist public responder settings without credential values."""
        _assert_public_payload(settings)
        return self._write_json("responder_settings.json", dict(settings))

    def write_responder_tokenizer_settings(
        self,
        settings: Mapping[str, Any],
    ) -> Path:
        """Persist the tokenizer identity and non-secret cache policy."""
        _assert_public_payload(settings)
        return self._write_json(
            "responder_tokenizer_settings.json",
            dict(settings),
        )

    def write_report(self, report: QualificationReport) -> tuple[Path, ...]:
        """Persist the complete report plus convenient evaluator-facing views."""
        paths = [
            self._write_json("report.json", report.model_dump(mode="json")),
            self._write_json(
                "summary.json", report.summary.model_dump(mode="json")
            ),
        ]
        if report.retrieval_read_only:
            paths.append(
                self._write_json(
                    "retrieval_read_only.json",
                    [
                        item.model_dump(mode="json")
                        for item in report.retrieval_read_only
                    ],
                )
            )
        if report.quiescence:
            paths.append(
                self._write_json(
                    "quiescence.json",
                    [
                        item.model_dump(mode="json")
                        for item in report.quiescence
                    ],
                )
            )
        if report.responder_stages:
            paths.append(
                self._write_json(
                    "responder_stages.json",
                    [
                        item.model_dump(mode="json")
                        for item in report.responder_stages
                    ],
                )
            )
        optional_documents = (
            ("snapshot_before.json", report.snapshot_before),
            ("snapshot_after.json", report.snapshot_after),
            ("secondary_snapshot_before.json", report.secondary_snapshot_before),
            ("secondary_snapshot_after.json", report.secondary_snapshot_after),
            ("secondary_flush_result.json", report.secondary_flush_result),
            ("provenance.json", report.provenance),
            ("secondary_provenance.json", report.secondary_provenance),
            ("reset_result.json", report.reset_result),
            ("secondary_reset_result.json", report.secondary_reset_result),
        )
        for name, model in optional_documents:
            if model is not None:
                paths.append(self._write_json(name, model.model_dump(mode="json")))
        return tuple(paths)

    def seal(self, report: QualificationReport) -> QualificationArtifactIndex:
        """Write final evidence, close events, and hash every immutable artifact."""
        if self._sealed:
            raise QualificationArtifactError(
                "qualification artifacts are already sealed"
            )
        self.write_report(report)
        self._event_sink.close()

        artifact_files = tuple(
            sorted(
                (
                    path
                    for path in self.run_dir.iterdir()
                    if path.is_file() and path.name != "checksums.json"
                ),
                key=lambda path: path.name,
            )
        )
        checksums = {path.name: _sha256(path) for path in artifact_files}
        checksum_path = self._write_json(
            "checksums.json",
            {"algorithm": "sha256", "files": checksums},
        )
        self._sealed = True
        return QualificationArtifactIndex(
            run_dir=self.run_dir,
            files=(*artifact_files, checksum_path),
            sha256_by_name=checksums,
        )

    def _write_json(self, name: str, payload: Any) -> Path:
        if self._sealed:
            raise QualificationArtifactError("qualification artifacts are sealed")
        if Path(name).name != name or not name.endswith(".json"):
            raise QualificationArtifactError(f"invalid artifact file name: {name!r}")
        destination = self.run_dir / name
        if destination.exists():
            raise QualificationArtifactError(
                f"qualification artifact already exists: {destination}"
            )

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.run_dir,
                prefix=f".{name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        except (OSError, TypeError, ValueError) as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise QualificationArtifactError(
                f"could not write qualification artifact {destination}: {exc}"
            ) from exc
        return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise QualificationArtifactError(
            f"could not hash qualification artifact {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _assert_public_payload(payload: Mapping[str, Any], prefix: str = "") -> None:
    forbidden_fragments = (
        "api_key",
        "authorization",
        "password",
        "postgres_dsn",
        "refresh_token",
        "access_token",
        "secret",
    )
    for raw_key, value in payload.items():
        key = str(raw_key)
        normalized = key.casefold()
        location = f"{prefix}.{key}" if prefix else key
        if not normalized.endswith("_configured") and any(
            fragment in normalized for fragment in forbidden_fragments
        ):
            raise QualificationArtifactError(
                f"resolved settings contain a secret-bearing field: {location}"
            )
        if isinstance(value, Mapping):
            _assert_public_payload(value, location)
