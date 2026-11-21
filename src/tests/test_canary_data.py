import json
import shutil
from pathlib import Path

import pytest

from membench.canary_data import (
    CanaryCategory,
    CanarySuite,
    GuardExpectation,
)
from membench.canary_data.loader import CanaryDataError
from membench.contract import WritePolicy
from membench.memory.models import MemoryType


def test_bundled_suite_loads_integrity_checked_qualifications() -> None:
    suite = CanarySuite.load()

    assert suite.manifest.suite_version == "1.1.0"
    assert len(suite.canaries) == 5
    assert {
        item.qualification_id for item in suite.manifest.qualifications
    } == {
        "baseline_semantic_v1",
        "namespace_isolation_v1",
        "shared_guard_admission_v1",
        "memory_family_surface_v1",
        "responder_read_only_v1",
    }

    baseline = suite.qualification("baseline_semantic_v1")
    assert baseline.supported_policies == (
        WritePolicy.DIRECT,
        WritePolicy.NATIVE_SELECTIVE,
    )
    assert [item.canary_id for item in suite.qualification_canaries(
        baseline.qualification_id
    )] == ["semantic_preference_v1"]


def test_canaries_are_synthetic_loggable_and_self_identifying() -> None:
    suite = CanarySuite.load()

    for canary in suite.canaries:
        assert canary.synthetic
        assert canary.safe_to_log_raw
        conversation = "\n".join(item.content for item in canary.messages)
        assert canary.marker.casefold() in conversation.casefold()
        evidence = {
            fragment.casefold()
            for expectation in canary.retrieval_expectations
            for fragment in expectation.required_fragments
        }
        assert canary.marker.casefold() in evidence

    assert suite.canary("semantic_preference_v1").memory_type == MemoryType.SEMANTIC
    assert suite.canary("episodic_event_v1").category == CanaryCategory.EPISODIC
    assert (
        suite.canary("procedural_rule_v1").memory_type
        == MemoryType.PROCEDURAL
    )
    assert (
        suite.canary("guard_injection_v1").guard_expectation
        == GuardExpectation.REJECT
    )


def test_loader_rejects_tampered_canary(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "src" / "membench" / "canary_data"
    target = tmp_path / "canary_data"
    shutil.copytree(source, target)
    canary_path = target / "canaries" / "semantic_preference_v1.json"
    value = json.loads(canary_path.read_text(encoding="utf-8"))
    value["description"] = "tampered after manifest creation"
    canary_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(CanaryDataError, match="integrity check failed"):
        CanarySuite.load(target)


def test_unknown_canary_and_qualification_raise_key_error() -> None:
    suite = CanarySuite.load()

    with pytest.raises(KeyError, match="unknown synthetic canary"):
        suite.canary("unknown_v1")
    with pytest.raises(KeyError, match="unknown qualification"):
        suite.qualification("unknown_v1")
