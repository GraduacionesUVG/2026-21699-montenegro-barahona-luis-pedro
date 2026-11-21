import os
import sys
from pathlib import Path

import pytest

from membench.campaign.environments import (
    ChildEnvironmentError,
    ChildEnvironmentRegistry,
    EnvironmentNotConfiguredError,
    EnvironmentUnavailableError,
    MissingEnvironmentVariablesError,
)
from membench.campaign.models import ChildEnvironmentSpec
from membench.contract import Mechanism


def _spec(
    mechanism: Mechanism = Mechanism.TELEMEM,
    **changes: object,
) -> ChildEnvironmentSpec:
    values: dict[str, object] = {
        "mechanism": mechanism,
        "python_executable": sys.executable,
        "module": "membench.agent",
        "inherited_env_keys": ("OLLAMA_BASE_URL", "OLLAMA_CHAT_MODEL"),
    }
    values.update(changes)
    return ChildEnvironmentSpec.model_validate(values)


def test_registry_resolves_a_shell_free_allowlisted_child_environment(
    tmp_path: Path,
) -> None:
    registry = ChildEnvironmentRegistry(
        (_spec(),),
        project_root=tmp_path,
        source_environment={
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "OLLAMA_CHAT_MODEL": "test-model",
            "UNRELATED_SECRET": "must-not-leak",
        },
    )

    resolved = registry.resolve(Mechanism.TELEMEM)

    assert resolved.command == (
        str(Path(sys.executable).resolve()),
        "-m",
        "membench.agent",
    )
    assert resolved.working_directory == tmp_path.resolve()
    assert resolved.timeout_seconds == 900.0
    assert resolved.missing_inherited_env_keys == ()
    child_environment = resolved.subprocess_environment()
    assert child_environment["OLLAMA_CHAT_MODEL"] == "test-model"
    assert child_environment["PYTHON_DOTENV_DISABLED"] == "1"
    assert child_environment["PYTHONIOENCODING"] == "utf-8"
    assert "UNRELATED_SECRET" not in child_environment

    child_environment["OLLAMA_CHAT_MODEL"] = "mutated"
    assert resolved.inherited_environment["OLLAMA_CHAT_MODEL"] == "test-model"
    with pytest.raises(TypeError):
        resolved.inherited_environment["NEW_KEY"] = "not-allowed"  # type: ignore[index]


def test_registry_resolves_relative_paths_and_orders_mechanisms(
    tmp_path: Path,
) -> None:
    relative_python = Path("runtime") / Path(sys.executable).name
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (tmp_path / "work").mkdir()
    executable = runtime / Path(sys.executable).name
    executable.write_bytes(Path(sys.executable).read_bytes())
    if os.name != "nt":
        executable.chmod(0o755)

    registry = ChildEnvironmentRegistry(
        (
            _spec(
                Mechanism.LANGMEM,
                python_executable=str(relative_python),
                working_directory="work",
                inherited_env_keys=(),
            ),
            _spec(
                Mechanism.DAKERA,
                inherited_env_keys=(),
            ),
        ),
        project_root=tmp_path,
        source_environment={},
    )

    resolved = registry.resolve("langmem")

    assert resolved.python_executable == executable.resolve()
    assert resolved.working_directory == (tmp_path / "work").resolve()
    assert registry.available_mechanisms() == (
        Mechanism.DAKERA,
        Mechanism.LANGMEM,
    )
    assert len(registry.resolve_all()) == 2


def test_registry_reports_missing_declared_variables_in_strict_and_probe_modes(
    tmp_path: Path,
) -> None:
    registry = ChildEnvironmentRegistry(
        (_spec(),),
        project_root=tmp_path,
        source_environment={"OLLAMA_BASE_URL": "http://127.0.0.1:11434"},
    )

    with pytest.raises(MissingEnvironmentVariablesError) as captured:
        registry.resolve(Mechanism.TELEMEM)
    assert captured.value.missing == ("OLLAMA_CHAT_MODEL",)

    resolved = registry.resolve(
        Mechanism.TELEMEM,
        require_all_env_keys=False,
    )
    assert resolved.missing_inherited_env_keys == ("OLLAMA_CHAT_MODEL",)
    assert "OLLAMA_CHAT_MODEL" not in resolved.inherited_environment


def test_registry_rejects_duplicates_unknown_mechanisms_and_unusable_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ChildEnvironmentError, match="duplicate"):
        ChildEnvironmentRegistry(
            (_spec(), _spec()),
            project_root=tmp_path,
        )

    registry = ChildEnvironmentRegistry(
        (_spec(inherited_env_keys=()),),
        project_root=tmp_path,
        source_environment={},
    )
    with pytest.raises(EnvironmentNotConfiguredError, match="langmem"):
        registry.resolve(Mechanism.LANGMEM)
    with pytest.raises(EnvironmentNotConfiguredError, match="unknown"):
        registry.resolve("not-a-mechanism")

    missing_interpreter = ChildEnvironmentRegistry(
        (
            _spec(
                python_executable="missing/python",
                inherited_env_keys=(),
            ),
        ),
        project_root=tmp_path,
        source_environment={},
    )
    with pytest.raises(EnvironmentUnavailableError, match="does not exist"):
        missing_interpreter.resolve(Mechanism.TELEMEM)

    missing_working_directory = ChildEnvironmentRegistry(
        (
            _spec(
                working_directory="missing-workdir",
                inherited_env_keys=(),
            ),
        ),
        project_root=tmp_path,
        source_environment={},
    )
    with pytest.raises(EnvironmentUnavailableError, match="working directory"):
        missing_working_directory.resolve(Mechanism.TELEMEM)
