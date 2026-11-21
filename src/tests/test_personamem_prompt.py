from membench.personamem_prompt import (
    PERSONAMEM_RESPONDER_PROMPT_VERSION,
    PersonaMemPromptBuilder,
)
from membench.personamem_v2 import PersonaMemProbe

REVISION = "a" * 40


def test_prompt_is_deterministic_and_keeps_options_unlabeled() -> None:
    probe = PersonaMemProbe(
        dataset_revision=REVISION,
        subset="blackbox_battery_v1",
        source_record_id="row-001",
        history_unit_id="history-001",
        persona_id="persona-001",
        event_id="probe-001",
        query="Which notebook best suits me?",
        options=("A blue notebook", "A red notebook"),
    )
    builder = PersonaMemPromptBuilder()

    first = builder.build(probe, "[memory] The user prefers blue notebooks.")
    second = builder.build(probe, "[memory] The user prefers blue notebooks.")

    assert first == second
    assert first.version == PERSONAMEM_RESPONDER_PROMPT_VERSION
    assert "A. A blue notebook" in first.messages[1].content
    assert "B. A red notebook" in first.messages[1].content
    assert "Final Answer: [Letter]" in first.messages[1].content
