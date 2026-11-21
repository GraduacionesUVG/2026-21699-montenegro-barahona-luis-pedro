"""Resolve campaign mechanisms to validated, isolated child environments.

The serialized :class:`~membench.campaign.models.ChildEnvironmentSpec` records
what a campaign intends to execute.  This module performs the host-specific
part of that contract: resolving paths, verifying that the interpreter and
working directory exist, and constructing an allowlisted subprocess
environment without leaking unrelated parent-process secrets.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from membench.campaign.models import (
    CampaignConfiguration,
    ChildEnvironmentSpec,
)
from membench.contract import Mechanism

_CONTROLLED_PYTHON_ENVIRONMENT = {
    # Child settings must use only the variables deliberately inherited below;
    # otherwise their load_dotenv() calls could repopulate the entire repo .env.
    "PYTHON_DOTENV_DISABLED": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
    "PYTHONUTF8": "1",
}

# Windows requires a valid SystemRoot for some side-by-side assemblies.  These
# keys support process creation; experiment and provider settings still have to
# be explicitly named by ChildEnvironmentSpec.inherited_env_keys.
_WINDOWS_PROCESS_ENV_KEYS = (
    "COMSPEC",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
)


class ChildEnvironmentError(RuntimeError):
    """Base error for mechanism-to-child-environment resolution."""


class EnvironmentNotConfiguredError(ChildEnvironmentError):
    """Raised when a campaign did not configure the requested mechanism."""


class EnvironmentUnavailableError(ChildEnvironmentError):
    """Raised when a configured interpreter or working directory is unusable."""


class MissingEnvironmentVariablesError(ChildEnvironmentError):
    """Raised when strict resolution cannot inherit every declared variable."""

    def __init__(self, mechanism: Mechanism, missing: tuple[str, ...]) -> None:
        self.mechanism = mechanism
        self.missing = missing
        names = ", ".join(missing)
        super().__init__(
            f"{mechanism.value} child environment is missing declared "
            f"variables: {names}"
        )


@dataclass(frozen=True, slots=True)
class ResolvedChildEnvironment:
    """Host-ready command, directory, and environment for one child process."""

    spec: ChildEnvironmentSpec
    python_executable: Path
    working_directory: Path
    inherited_environment: Mapping[str, str]
    missing_inherited_env_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Prevent a launcher or caller from mutating the resolved environment
        # after preflight and silently changing the experimental condition.
        immutable = MappingProxyType(dict(self.inherited_environment))
        object.__setattr__(self, "inherited_environment", immutable)

    @property
    def mechanism(self) -> Mechanism:
        """Return the mechanism routed through this child environment."""

        return self.spec.mechanism

    @property
    def timeout_seconds(self) -> float:
        """Return the configured wall-clock limit for the child process."""

        return self.spec.timeout_seconds

    @property
    def command(self) -> tuple[str, ...]:
        """Return the shell-free command used to invoke the child module."""

        return (str(self.python_executable), "-m", self.spec.module)

    def subprocess_environment(self) -> dict[str, str]:
        """Return a fresh mutable environment mapping for ``subprocess``."""

        return dict(self.inherited_environment)


class ChildEnvironmentRegistry:
    """Resolve immutable mechanism registrations against one host snapshot."""

    def __init__(
        self,
        specs: Iterable[ChildEnvironmentSpec],
        *,
        project_root: str | Path | None = None,
        source_environment: Mapping[str, str] | None = None,
    ) -> None:
        root = Path.cwd() if project_root is None else Path(project_root)
        self._project_root = root.resolve()
        if not self._project_root.exists():
            raise EnvironmentUnavailableError(
                f"campaign project root does not exist: {self._project_root}"
            )
        if not self._project_root.is_dir():
            raise EnvironmentUnavailableError(
                f"campaign project root is not a directory: {self._project_root}"
            )

        registrations: dict[Mechanism, ChildEnvironmentSpec] = {}
        for spec in specs:
            if spec.mechanism in registrations:
                raise ChildEnvironmentError(
                    "duplicate child environment for mechanism: "
                    f"{spec.mechanism.value}"
                )
            registrations[spec.mechanism] = spec
        if not registrations:
            raise ChildEnvironmentError(
                "at least one child environment must be registered"
            )

        source = os.environ if source_environment is None else source_environment
        self._specs = MappingProxyType(registrations)
        self._source_environment = MappingProxyType(dict(source))

    @classmethod
    def from_configuration(
        cls,
        configuration: CampaignConfiguration,
        *,
        project_root: str | Path | None = None,
        source_environment: Mapping[str, str] | None = None,
    ) -> ChildEnvironmentRegistry:
        """Create a registry from a validated campaign configuration."""

        return cls(
            configuration.environments,
            project_root=project_root,
            source_environment=source_environment,
        )

    @property
    def project_root(self) -> Path:
        """Return the absolute root used for resolving relative paths."""

        return self._project_root

    def available_mechanisms(self) -> tuple[Mechanism, ...]:
        """List registered mechanisms in deterministic value order."""

        return tuple(sorted(self._specs, key=lambda item: item.value))

    def spec_for(self, mechanism: Mechanism | str) -> ChildEnvironmentSpec:
        """Return the registered serialized specification for ``mechanism``."""

        normalized = _normalize_mechanism(mechanism)
        try:
            return self._specs[normalized]
        except KeyError as exc:
            available = ", ".join(
                item.value for item in self.available_mechanisms()
            )
            raise EnvironmentNotConfiguredError(
                f"no child environment is configured for {normalized.value}; "
                f"available: {available or 'none'}"
            ) from exc

    def resolve(
        self,
        mechanism: Mechanism | str,
        *,
        require_all_env_keys: bool = True,
    ) -> ResolvedChildEnvironment:
        """Resolve and preflight one mechanism-specific child environment.

        Declared environment keys are copied from the source snapshot.  In
        strict mode (the launcher default), any absent declared key fails
        preflight.  Callers can use non-strict mode for setup diagnostics and
        inspect ``missing_inherited_env_keys`` on the returned object.
        """

        spec = self.spec_for(mechanism)
        python_executable = self._resolve_path(spec.python_executable)
        if not python_executable.exists():
            raise EnvironmentUnavailableError(
                f"{spec.mechanism.value} Python interpreter does not exist: "
                f"{python_executable}"
            )
        if not python_executable.is_file():
            raise EnvironmentUnavailableError(
                f"{spec.mechanism.value} Python interpreter is not a file: "
                f"{python_executable}"
            )
        if os.name != "nt" and not os.access(python_executable, os.X_OK):
            raise EnvironmentUnavailableError(
                f"{spec.mechanism.value} Python interpreter is not executable: "
                f"{python_executable}"
            )

        working_directory = (
            self._project_root
            if spec.working_directory is None
            else self._resolve_path(spec.working_directory)
        )
        if not working_directory.exists():
            raise EnvironmentUnavailableError(
                f"{spec.mechanism.value} working directory does not exist: "
                f"{working_directory}"
            )
        if not working_directory.is_dir():
            raise EnvironmentUnavailableError(
                f"{spec.mechanism.value} working directory is not a directory: "
                f"{working_directory}"
            )

        child_environment, missing = self._build_environment(spec)
        if require_all_env_keys and missing:
            raise MissingEnvironmentVariablesError(spec.mechanism, missing)
        return ResolvedChildEnvironment(
            spec=spec,
            python_executable=python_executable,
            working_directory=working_directory,
            inherited_environment=child_environment,
            missing_inherited_env_keys=missing,
        )

    def resolve_all(
        self,
        *,
        require_all_env_keys: bool = True,
    ) -> tuple[ResolvedChildEnvironment, ...]:
        """Preflight every registration in deterministic mechanism order."""

        return tuple(
            self.resolve(
                mechanism,
                require_all_env_keys=require_all_env_keys,
            )
            for mechanism in self.available_mechanisms()
        )

    def _resolve_path(self, configured: str) -> Path:
        path = Path(configured)
        if not path.is_absolute():
            path = self._project_root / path
        return path.resolve()

    def _build_environment(
        self,
        spec: ChildEnvironmentSpec,
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        result: dict[str, str] = {}

        if os.name == "nt":
            for key in _WINDOWS_PROCESS_ENV_KEYS:
                value = self._source_environment.get(key)
                if value is not None:
                    result[key] = value

        missing: list[str] = []
        for key in spec.inherited_env_keys:
            value = self._source_environment.get(key)
            if value is None:
                missing.append(key)
            else:
                result[key] = value

        # These values are controlled by the harness and therefore cannot be
        # changed accidentally through the declared inheritance allowlist.
        result.update(_CONTROLLED_PYTHON_ENVIRONMENT)
        return result, tuple(missing)


def _normalize_mechanism(mechanism: Mechanism | str) -> Mechanism:
    if isinstance(mechanism, Mechanism):
        return mechanism
    try:
        return Mechanism(mechanism.strip().lower())
    except (AttributeError, ValueError) as exc:
        raise EnvironmentNotConfiguredError(
            f"unknown memory mechanism: {mechanism!r}"
        ) from exc
