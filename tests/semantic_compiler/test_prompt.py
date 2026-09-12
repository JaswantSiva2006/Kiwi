from __future__ import annotations

from kivi_memory.semantic_compiler.prompt import SYSTEM_PROMPT


def test_prompt_contains_core_semantic_rules_without_schema_dump() -> None:
    assert "source_spans.text must copy" in SYSTEM_PROMPT
    assert "DO NOT normalize" in SYSTEM_PROMPT
    assert "Emit one assertion for each independently updateable proposition." in SYSTEM_PROMPT
    assert "Use UPPER_SNAKE_CASE." in SYSTEM_PROMPT
    assert "COMMITMENT_OPEN_LOOP = unresolved promise or obligation" in SYSTEM_PROMPT
    assert "CALENDAR_EVENT" in SYSTEM_PROMPT
    assert "calendar_event" in SYSTEM_PROMPT
    assert "A promise or agreement to perform a future action is an OBLIGATION," in SYSTEM_PROMPT
    assert '"properties"' not in SYSTEM_PROMPT
    assert '"$defs"' not in SYSTEM_PROMPT
