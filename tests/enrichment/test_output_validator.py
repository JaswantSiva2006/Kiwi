from __future__ import annotations

import pytest
from pydantic import ValidationError

from kivi_memory.enrichment.validator import validate_compiler_output
from kivi_memory.common.schemas import CompilerOutput, MemoryEpisode
from tests.common.test_input_schema import valid_episode_data


def base_assertion() -> dict:
    return {
        "canonical_text": "Priya is handling Atlas.",
        "memory_type": "PROJECT_GOAL_TOPIC",
        "subject": {"text": "Priya", "entity_type": "PERSON"},
        "semantic_arguments": [{"role": "project", "text": "Atlas", "is_entity": True, "entity_type": "PROJECT"}],
        "predicate_type": "RESPONSIBLE_FOR",
        "modality": "FACT",
        "polarity": "POSITIVE",
        "certainty": "CERTAIN",
        "explicitness": "EXPLICIT",
        "attributed_to": "USER",
        "source_spans": [{"message_id": "m1", "text": "Priya's basically handling Atlas now."}],
    }


def valid_episode() -> MemoryEpisode:
    return MemoryEpisode.model_validate(valid_episode_data())


def test_valid_empty_compiler_output_passes() -> None:
    report = validate_compiler_output(CompilerOutput.model_validate({"assertions": []}), valid_episode())
    assert report.valid
    assert report.issues == []


def test_source_span_referencing_nonexistent_message_fails() -> None:
    assertion = base_assertion()
    assertion["source_spans"] = [{"message_id": "missing", "text": "Priya"}]
    output = CompilerOutput.model_validate({"assertions": [assertion]})
    report = validate_compiler_output(output, valid_episode())
    assert not report.valid
    assert any(issue.code == "SOURCE_MESSAGE_NOT_FOUND" for issue in report.issues)


def test_source_text_that_does_not_occur_verbatim_fails() -> None:
    assertion = base_assertion()
    assertion["source_spans"] = [{"message_id": "m1", "text": "Priya handles Atlas."}]
    output = CompilerOutput.model_validate({"assertions": [assertion]})
    report = validate_compiler_output(output, valid_episode())
    assert not report.valid
    assert any(issue.code == "SOURCE_SPAN_NOT_VERBATIM" for issue in report.issues)


def test_multi_message_source_spans_succeed() -> None:
    assertion = base_assertion()
    assertion["source_spans"] = [
        {"message_id": "m1", "text": "Priya's basically handling Atlas now."},
        {"message_id": "m2", "text": "Since Rohit moved over to payments?"},
        {"message_id": "m3", "text": "Yeah."},
    ]
    output = CompilerOutput.model_validate({"assertions": [assertion]})
    report = validate_compiler_output(output, valid_episode())
    assert report.valid


def test_duplicate_canonical_assertions_are_detected() -> None:
    first = base_assertion()
    second = base_assertion()
    second["canonical_text"] = " priya   is handling atlas. "
    output = CompilerOutput.model_validate({"assertions": [first, second]})
    report = validate_compiler_output(output, valid_episode())
    assert report.valid
    assert any(issue.code == "DUPLICATE_ASSERTION" for issue in report.issues)


def test_unclean_predicate_type_is_caught() -> None:
    assertion = base_assertion()
    assertion["predicate_type"] = "handles project"
    output = CompilerOutput.model_validate({"assertions": [assertion]})
    report = validate_compiler_output(output, valid_episode())
    assert not report.valid
    assert any(issue.code == "MALFORMED_PREDICATE_TYPE" for issue in report.issues)


def test_empty_source_spans_fails() -> None:
    assertion = base_assertion()
    assertion["source_spans"] = []
    output = CompilerOutput.model_validate({"assertions": [assertion]})
    report = validate_compiler_output(output, valid_episode())
    assert not report.valid
    assert any(issue.code == "MISSING_SOURCE_SPAN" for issue in report.issues)


def test_removed_schema_fields_are_rejected() -> None:
    assertion = base_assertion()
    assertion["semantic" + "_type"] = "RELATION"
    assertion["subject" + "_mentions"] = [{"text": "Priya", "entity_type": "PERSON"}]
    assertion["object" + "_mentions"] = [{"text": "Atlas", "entity_type": "PROJECT"}]
    assertion["relation" + "_type"] = "RESPONSIBLE_FOR"
    assertion["event" + "_type"] = None

    with pytest.raises(ValidationError):
        CompilerOutput.model_validate({"assertions": [assertion]})
