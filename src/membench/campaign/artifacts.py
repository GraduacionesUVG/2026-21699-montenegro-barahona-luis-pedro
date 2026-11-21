"""Integrity-checked persistence for launcher and child-run evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Self

from pydantic import BaseModel

from membench.campaign.models import (
    ArtifactKind,
    ArtifactReference,
    ArtifactVisibility,
)


class CampaignArtifactError(RuntimeError):
    """Base error for unsafe or unsuccessful artifact operations."""


class ArtifactPathError(CampaignArtifactError):
    """Raised when an artifact path escapes or ambiguously names the root."""


class ArtifactExistsError(CampaignArtifactError):
    """Raised when immutable evidence would be overwritten or duplicated."""


class ArtifactIntegrityError(CampaignArtifactError):
    """Raised when persisted bytes do not match their artifact reference."""


class ArtifactSealedError(CampaignArtifactError):
    """Raised when a sealed artifact store receives another mutation."""


_ATOMIC_TEMP_PREFIX = ".tmp-"
_ATOMIC_TEMP_SUFFIX = ".part"


@dataclass(frozen=True, slots=True)
class CampaignArtifactIndex:
    """Complete reference inventory returned when a store is sealed."""

    root: Path
    artifacts: tuple[ArtifactReference, ...]
    checksums: ArtifactReference

    def reference_for(self, artifact_id: str) -> ArtifactReference:
        """Resolve one artifact by its stable ID."""

        for reference in self.artifacts:
            if reference.artifact_id == artifact_id:
                return reference
        raise KeyError(artifact_id)


class CampaignArtifactStore:
    """Own one campaign root and persist immutable evidence beneath it."""

    def __init__(self, root: Path) -> None:
        resolved = root.resolve()
        if not resolved.exists():
            raise CampaignArtifactError(
                f"campaign artifact root does not exist: {resolved}"
            )
        if not resolved.is_dir():
            raise CampaignArtifactError(
                f"campaign artifact root is not a directory: {resolved}"
            )
        self.root = resolved
        self._sealed = False
        self._references_by_id: dict[str, ArtifactReference] = {}
        self._artifact_ids_by_path: dict[str, str] = {}
        self._reservations: dict[str, str] = {}
        self._lock = Lock()

    @classmethod
    def prepare(cls, root: str | Path) -> CampaignArtifactStore:
        """Create a new campaign root without overwriting prior evidence."""

        destination = Path(root).resolve()
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactExistsError(
                f"campaign artifact root already exists: {destination}"
            ) from exc
        except OSError as exc:
            raise CampaignArtifactError(
                f"could not create campaign artifact root {destination}: {exc}"
            ) from exc
        return cls(destination)

    @classmethod
    def open_existing(cls, root: str | Path) -> CampaignArtifactStore:
        """Open an existing root so verified child artifacts can be registered."""

        return cls(Path(root))

    @property
    def sealed(self) -> bool:
        """Report whether the checksum inventory has been finalized."""

        return self._sealed

    def references(self) -> tuple[ArtifactReference, ...]:
        """Return registered references in deterministic path order."""

        with self._lock:
            return tuple(
                sorted(
                    self._references_by_id.values(),
                    key=lambda item: item.relative_path,
                )
            )

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve one portable path and prove it remains below the root."""

        portable = _portable_relative_path(relative_path)
        destination = (self.root / Path(*portable.parts)).resolve()
        if not destination.is_relative_to(self.root):
            raise ArtifactPathError(
                f"artifact path escapes campaign root: {relative_path!r}"
            )
        return destination

    def prepare_directory(
        self,
        relative_path: str,
        *,
        exist_ok: bool = False,
    ) -> Path:
        """Create a safe child directory for a launcher request."""

        self._assert_mutable()
        destination = self.resolve_path(relative_path)
        try:
            destination.mkdir(parents=True, exist_ok=exist_ok)
        except FileExistsError as exc:
            raise ArtifactExistsError(
                f"artifact directory already exists: {destination}"
            ) from exc
        except OSError as exc:
            raise CampaignArtifactError(
                f"could not create artifact directory {destination}: {exc}"
            ) from exc
        return destination

    def write_json(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        relative_path: str,
        value: Any,
        visibility: ArtifactVisibility = ArtifactVisibility.EVALUATOR_ONLY,
        media_type: str = "application/json",
    ) -> ArtifactReference:
        """Atomically persist canonical JSON and return its exact reference."""

        try:
            content = canonical_json_bytes(value)
        except (TypeError, ValueError) as exc:
            raise CampaignArtifactError(
                f"artifact {artifact_id!r} is not canonical JSON: {exc}"
            ) from exc
        return self.write_bytes(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative_path,
            content=content,
            media_type=media_type,
            visibility=visibility,
        )

    def write_jsonl(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        relative_path: str,
        records: Iterable[Any],
        visibility: ArtifactVisibility = ArtifactVisibility.EVALUATOR_ONLY,
        media_type: str = "application/x-ndjson",
    ) -> ArtifactReference:
        """Atomically write a complete sequence of canonical JSON lines."""

        try:
            content = b"".join(canonical_json_bytes(item) for item in records)
        except (TypeError, ValueError) as exc:
            raise CampaignArtifactError(
                f"artifact {artifact_id!r} contains invalid JSONL: {exc}"
            ) from exc
        return self.write_bytes(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative_path,
            content=content,
            media_type=media_type,
            visibility=visibility,
        )

    def write_bytes(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        relative_path: str,
        content: bytes,
        media_type: str,
        visibility: ArtifactVisibility = ArtifactVisibility.EVALUATOR_ONLY,
    ) -> ArtifactReference:
        """Atomically persist immutable bytes under a validated identity."""

        identity = _validated_identity(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative_path,
            media_type=media_type,
            visibility=visibility,
        )
        destination = self._reserve(identity.artifact_id, identity.relative_path)
        try:
            _atomic_write(destination, content)
            reference = _reference_for_path(identity, destination)
        except Exception:
            self._release(identity.artifact_id, identity.relative_path)
            raise
        return self._complete(reference)

    def copy_file(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        relative_path: str,
        source: str | Path,
        media_type: str,
        visibility: ArtifactVisibility = ArtifactVisibility.EVALUATOR_ONLY,
    ) -> ArtifactReference:
        """Atomically copy a frozen input while preserving its exact bytes."""

        identity = _validated_identity(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative_path,
            media_type=media_type,
            visibility=visibility,
        )
        source_path = Path(source).resolve()
        if not source_path.exists() or not source_path.is_file():
            raise CampaignArtifactError(
                f"artifact source is not a readable file: {source_path}"
            )
        destination = self._reserve(identity.artifact_id, identity.relative_path)
        if source_path == destination:
            self._release(identity.artifact_id, identity.relative_path)
            raise CampaignArtifactError("artifact source and destination are identical")
        try:
            _atomic_copy(source_path, destination)
            reference = _reference_for_path(identity, destination)
        except Exception:
            self._release(identity.artifact_id, identity.relative_path)
            raise
        return self._complete(reference)

    def open_jsonl(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        relative_path: str,
        visibility: ArtifactVisibility = ArtifactVisibility.EVALUATOR_ONLY,
        media_type: str = "application/x-ndjson",
    ) -> JsonlArtifactWriter:
        """Create an append-only JSONL writer finalized into one reference."""

        identity = _validated_identity(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative_path,
            media_type=media_type,
            visibility=visibility,
        )
        destination = self._reserve(identity.artifact_id, identity.relative_path)
        try:
            _atomic_write(destination, b"")
        except Exception:
            self._release(identity.artifact_id, identity.relative_path)
            raise
        return JsonlArtifactWriter(self, identity, destination)

    def register_existing(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        relative_path: str,
        media_type: str,
        visibility: ArtifactVisibility = ArtifactVisibility.EVALUATOR_ONLY,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
    ) -> ArtifactReference:
        """Register child-produced evidence after optional integrity checks."""

        identity = _validated_identity(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative_path,
            media_type=media_type,
            visibility=visibility,
        )
        destination = self._reserve(
            identity.artifact_id,
            identity.relative_path,
            allow_existing=True,
        )
        try:
            if not destination.exists() or not destination.is_file():
                raise CampaignArtifactError(
                    f"existing artifact is not a file: {destination}"
                )
            reference = _reference_for_path(identity, destination)
            if expected_sha256 is not None and reference.sha256 != expected_sha256:
                raise ArtifactIntegrityError(
                    f"artifact hash mismatch for {identity.relative_path}: "
                    f"expected {expected_sha256}, got {reference.sha256}"
                )
            if (
                expected_size_bytes is not None
                and reference.size_bytes != expected_size_bytes
            ):
                raise ArtifactIntegrityError(
                    f"artifact size mismatch for {identity.relative_path}: "
                    f"expected {expected_size_bytes}, got {reference.size_bytes}"
                )
        except Exception:
            self._release(identity.artifact_id, identity.relative_path)
            raise
        return self._complete(reference)

    def verify(self, reference: ArtifactReference) -> Path:
        """Verify path, byte length, and SHA-256 for one artifact reference."""

        path = self.resolve_path(reference.relative_path)
        if not path.exists() or not path.is_file():
            raise ArtifactIntegrityError(
                f"artifact file is missing: {reference.relative_path}"
            )
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ArtifactIntegrityError(
                f"could not inspect artifact {reference.relative_path}: {exc}"
            ) from exc
        if size != reference.size_bytes:
            raise ArtifactIntegrityError(
                f"artifact size mismatch for {reference.relative_path}: "
                f"expected {reference.size_bytes}, got {size}"
            )
        digest = sha256_file(path)
        if digest != reference.sha256:
            raise ArtifactIntegrityError(
                f"artifact hash mismatch for {reference.relative_path}: "
                f"expected {reference.sha256}, got {digest}"
            )
        return path

    def seal(
        self,
        *,
        artifact_id: str = "campaign-checksums",
        relative_path: str = "checksums.json",
        require_all_files_registered: bool = True,
    ) -> CampaignArtifactIndex:
        """Verify the inventory, write its checksum document, and freeze writes."""

        self._assert_mutable()
        with self._lock:
            if self._reservations:
                open_paths = ", ".join(sorted(self._reservations.values()))
                raise CampaignArtifactError(
                    f"cannot seal while artifacts remain open: {open_paths}"
                )
            references = tuple(
                sorted(
                    self._references_by_id.values(),
                    key=lambda item: item.relative_path,
                )
            )

        for reference in references:
            self.verify(reference)
        if require_all_files_registered:
            self._assert_no_unregistered_files(references)

        checksums = self.write_json(
            artifact_id=artifact_id,
            kind=ArtifactKind.CHECKSUMS,
            relative_path=relative_path,
            value={
                "schema_version": 1,
                "algorithm": "sha256",
                "artifacts": [
                    reference.model_dump(mode="json")
                    for reference in references
                ],
            },
        )
        self._sealed = True
        return CampaignArtifactIndex(
            root=self.root,
            artifacts=self.references(),
            checksums=checksums,
        )

    def _reserve(
        self,
        artifact_id: str,
        relative_path: str,
        *,
        allow_existing: bool = False,
    ) -> Path:
        self._assert_mutable()
        destination = self.resolve_path(relative_path)
        with self._lock:
            if (
                artifact_id in self._references_by_id
                or artifact_id in self._reservations
            ):
                raise ArtifactExistsError(
                    f"artifact ID is already registered: {artifact_id}"
                )
            if relative_path in self._artifact_ids_by_path or relative_path in (
                self._reservations.values()
            ):
                raise ArtifactExistsError(
                    f"artifact path is already registered: {relative_path}"
                )
            if destination.exists() and not allow_existing:
                raise ArtifactExistsError(
                    f"artifact file already exists: {destination}"
                )
            self._reservations[artifact_id] = relative_path
        return destination

    def _complete(self, reference: ArtifactReference) -> ArtifactReference:
        with self._lock:
            expected = self._reservations.pop(reference.artifact_id, None)
            if expected != reference.relative_path:
                raise CampaignArtifactError(
                    f"artifact reservation was lost: {reference.artifact_id}"
                )
            self._references_by_id[reference.artifact_id] = reference
            self._artifact_ids_by_path[reference.relative_path] = (
                reference.artifact_id
            )
        return reference

    def _release(self, artifact_id: str, relative_path: str) -> None:
        with self._lock:
            if self._reservations.get(artifact_id) == relative_path:
                del self._reservations[artifact_id]

    def _assert_mutable(self) -> None:
        if self._sealed:
            raise ArtifactSealedError("campaign artifact store is sealed")

    def _assert_no_unregistered_files(
        self,
        references: tuple[ArtifactReference, ...],
    ) -> None:
        registered = {reference.relative_path for reference in references}
        observed = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        unregistered = sorted(observed - registered)
        if unregistered:
            raise CampaignArtifactError(
                "campaign artifact root contains unregistered files: "
                + ", ".join(unregistered)
            )


class JsonlArtifactWriter:
    """Append canonical records and register the immutable file on close."""

    def __init__(
        self,
        store: CampaignArtifactStore,
        identity: ArtifactReference,
        path: Path,
    ) -> None:
        self._store = store
        self._identity = identity
        self.path = path
        self._closed = False
        self._reference: ArtifactReference | None = None
        self._lock = Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    def append(self, value: Any) -> None:
        """Append and fsync one canonical JSON record."""

        try:
            line = canonical_json_bytes(value)
        except (TypeError, ValueError) as exc:
            raise CampaignArtifactError(f"invalid JSONL record: {exc}") from exc
        with self._lock:
            if self._closed:
                raise ArtifactSealedError("JSONL artifact writer is closed")
            try:
                with self.path.open("ab") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                raise CampaignArtifactError(
                    f"could not append JSONL artifact {self.path}: {exc}"
                ) from exc

    def close(self) -> ArtifactReference:
        """Hash and register the completed JSONL artifact exactly once."""

        with self._lock:
            if self._reference is not None:
                return self._reference
            if self._closed:
                raise ArtifactSealedError("JSONL artifact writer is closed")
            self._closed = True
            reference = _reference_for_path(self._identity, self.path)
            self._reference = self._store._complete(reference)
            return self._reference

    def abort(self) -> None:
        """Discard an incomplete writer-owned file and release its reservation."""

        with self._lock:
            if self._reference is not None:
                raise ArtifactSealedError(
                    "registered JSONL artifacts cannot be aborted"
                )
            if self._closed:
                return
            self._closed = True
            try:
                self.path.unlink(missing_ok=True)
            except OSError as exc:
                raise CampaignArtifactError(
                    f"could not remove incomplete JSONL artifact {self.path}: {exc}"
                ) from exc
            finally:
                self._store._release(
                    self._identity.artifact_id,
                    self._identity.relative_path,
                )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exception, traceback
        if exception_type is None:
            self.close()
        else:
            self.abort()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one value deterministically as UTF-8 JSON terminated by LF."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    text = json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{text}\n".encode()


def sha256_file(path: str | Path) -> str:
    """Calculate SHA-256 without loading a potentially large artifact at once."""

    artifact_path = Path(path)
    digest = hashlib.sha256()
    try:
        with artifact_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CampaignArtifactError(
            f"could not hash artifact {artifact_path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _portable_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactPathError("artifact path must be a non-blank string")
    if "\\" in value:
        raise ArtifactPathError("artifact paths must use portable '/' separators")
    path = PurePosixPath(value)
    drive_like = bool(path.parts and path.parts[0].endswith(":"))
    canonical = path.as_posix()
    if (
        path.is_absolute()
        or drive_like
        or value in {".", ".."}
        or ".." in path.parts
        or canonical != value
    ):
        raise ArtifactPathError(
            "artifact path must be canonical, relative, and cannot traverse"
        )
    return path


def _validated_identity(
    *,
    artifact_id: str,
    kind: ArtifactKind,
    relative_path: str,
    media_type: str,
    visibility: ArtifactVisibility,
) -> ArtifactReference:
    _portable_relative_path(relative_path)
    try:
        return ArtifactReference(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative_path,
            sha256="0" * 64,
            media_type=media_type,
            size_bytes=0,
            visibility=visibility,
        )
    except (TypeError, ValueError) as exc:
        raise CampaignArtifactError(
            f"invalid artifact identity for {artifact_id!r}: {exc}"
        ) from exc


def _reference_for_path(
    identity: ArtifactReference,
    path: Path,
) -> ArtifactReference:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CampaignArtifactError(
            f"could not inspect artifact {path}: {exc}"
        ) from exc
    return identity.model_copy(
        update={
            "sha256": sha256_file(path),
            "size_bytes": size,
        }
    )


def _atomic_write(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            # Do not repeat the destination name here. Deep child/evaluation
            # paths can have valid final names while a destination-derived
            # temporary name crosses the traditional Windows MAX_PATH limit.
            # NamedTemporaryFile already provides collision-resistant names.
            prefix=_ATOMIC_TEMP_PREFIX,
            suffix=_ATOMIC_TEMP_SUFFIX,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise CampaignArtifactError(
            f"could not atomically write artifact {destination}: {exc}"
        ) from exc


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with source.open("rb") as input_stream, tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=_ATOMIC_TEMP_PREFIX,
            suffix=_ATOMIC_TEMP_SUFFIX,
            delete=False,
        ) as output_stream:
            temporary = Path(output_stream.name)
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        temporary.replace(destination)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise CampaignArtifactError(
            f"could not atomically copy artifact to {destination}: {exc}"
        ) from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
