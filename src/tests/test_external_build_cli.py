"""End-to-end checks for the external artifact-build command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from membench.cli import app
from membench.curation.delivery_projections import load_delivery_projection_set
from membench.curation.external_attack_specs import load_external_attack_spec_set

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def test_external_build_freezes_an_explicit_unreviewed_pipeline_check(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "external-artifacts"

    result = RUNNER.invoke(
        app,
        [
            "external",
            "build",
            "--attack-draft",
            str(ROOT / "configs/external_attack_specs/external-tool-pilot-v1.json"),
            "--recipe",
            str(
                ROOT
                / "configs/external_delivery_recipes"
                / "external-tool-pilot-v1.example.json"
            ),
            "--fixture",
            str(ROOT / "configs/fixtures/external-tool-pilot-v1.json"),
            "--output-dir",
            str(output),
            "--allow-unreviewed",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    attack_set = load_external_attack_spec_set(output)
    delivery_set = load_delivery_projection_set(output)
    assert attack_set.specs[0].review_status == "generated_unreviewed"
    assert delivery_set.families[0].attack_spec_id == attack_set.specs[0].attack_spec_id


def test_external_build_requires_review_by_default(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "rejected-artifacts"

    result = RUNNER.invoke(
        app,
        [
            "external",
            "build",
            "--attack-draft",
            str(ROOT / "configs/external_attack_specs/external-tool-pilot-v1.json"),
            "--recipe",
            str(
                ROOT
                / "configs/external_delivery_recipes"
                / "external-tool-pilot-v1.example.json"
            ),
            "--fixture",
            str(ROOT / "configs/fixtures/external-tool-pilot-v1.json"),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "not approved" in result.output
    assert not output.exists()
