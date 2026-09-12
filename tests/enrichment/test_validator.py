from __future__ import annotations

from copy import deepcopy

import pytest

from kivi_memory.common.schemas import CalendarEventExtraction, CandidateSemanticAssertion, MemoryEpisode
from kivi_memory.enrichment.validator import validate_assertions
from tests.common.test_input_schema import valid_episode_data


def valid_episode() -> MemoryEpisode:
    return MemoryEpisode.model_validate(valid_episode_data())


def base_assertion() -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": "Priya handles Atlas.",
            "memory_type": "PROJECT_GOAL_TOPIC",
            "subject": {"text": "Priya", "entity_type": "PERSON"},
            "semantic_arguments": [{"role": "project", "text": "Atlas", "is_entity": True, "entity_type": "PROJECT"}],
            "predicate_type": "RESPONSIBLE_FOR",
            "modality": "FACT",
            "polarity": "POSITIVE",
            "certainty": "CERTAIN",
            "explicitness": "EXPLICIT",
            "attributed_to": "user",
            "source_spans": [{"message_id": "m1", "text": "Priya's basically handling Atlas now."}],
        }
    )


def codes(assertion: CandidateSemanticAssertion) -> list[str]:
    result = validate_assertions([assertion], valid_episode())[0]
    return [issue.code for issue in result.validation_report.issues]


def test_valid_grounded_assertion() -> None:
    result = validate_assertions([base_assertion()], valid_episode())[0]
    assert result.validation_report.valid
    assert result.validation_report.issues == []


def test_nonexistent_source_message_id() -> None:
    data = base_assertion().model_dump()
    data["source_spans"] = [{"message_id": "missing", "text": "Priya"}]
    assert "SOURCE_MESSAGE_NOT_FOUND" in codes(CandidateSemanticAssertion.model_validate(data))


def test_non_verbatim_source_span() -> None:
    data = base_assertion().model_dump()
    data["source_spans"] = [{"message_id": "m1", "text": "Priya handles Atlas."}]
    assert "SOURCE_SPAN_NOT_VERBATIM" in codes(CandidateSemanticAssertion.model_validate(data))


def test_missing_source_span() -> None:
    assertion = base_assertion().model_copy(update={"source_spans": []})
    assert "MISSING_SOURCE_SPAN" in codes(assertion)


def test_malformed_predicate_type() -> None:
    assertion = base_assertion().model_copy(update={"predicate_type": "responsible for"})
    result = validate_assertions([assertion], valid_episode())[0]
    assert not result.validation_report.valid
    assert "MALFORMED_PREDICATE_TYPE" in [issue.code for issue in result.validation_report.issues]


def test_duplicated_subject_argument() -> None:
    data = base_assertion().model_dump()
    data["semantic_arguments"].append({"role": "person", "text": "Priya", "is_entity": True, "entity_type": "PERSON"})
    result = validate_assertions([CandidateSemanticAssertion.model_validate(data)], valid_episode())[0]
    assert result.validation_report.valid
    assert "SUBJECT_DUPLICATED_IN_ARGUMENTS" in [issue.code for issue in result.validation_report.issues]


def test_duplicate_semantic_argument() -> None:
    data = base_assertion().model_dump()
    data["semantic_arguments"].append({"role": "project", "text": "Atlas", "is_entity": True, "entity_type": "PROJECT"})
    result = validate_assertions([CandidateSemanticAssertion.model_validate(data)], valid_episode())[0]
    assert result.validation_report.valid
    assert "DUPLICATE_ARGUMENT" in [issue.code for issue in result.validation_report.issues]


def test_duplicate_source_span() -> None:
    data = base_assertion().model_dump()
    data["source_spans"].append({"message_id": "m1", "text": "Priya's basically handling Atlas now."})
    result = validate_assertions([CandidateSemanticAssertion.model_validate(data)], valid_episode())[0]
    assert result.validation_report.valid
    assert "DUPLICATE_SOURCE_SPAN" in [issue.code for issue in result.validation_report.issues]


def test_duplicate_assertion_warns_on_both() -> None:
    first = base_assertion()
    second = CandidateSemanticAssertion.model_validate({**base_assertion().model_dump(), "canonical_text": " priya handles atlas. "})
    results = validate_assertions([first, second], valid_episode())
    assert all("DUPLICATE_ASSERTION" in [issue.code for issue in result.validation_report.issues] for result in results)


def test_arbitrary_clean_entity_type_allowed() -> None:
    data = base_assertion().model_dump()
    data["subject"]["entity_type"] = "CUSTOM_ROLE"
    result = validate_assertions([CandidateSemanticAssertion.model_validate(data)], valid_episode())[0]
    assert result.validation_report.valid


def test_calendar_event_allows_temporal_stage_to_supply_structured_calendar_event() -> None:
    assertion = CandidateSemanticAssertion.model_validate(
        {**base_assertion().model_dump(), "memory_type": "CALENDAR_EVENT", "calendar_event": None}
    )

    assert assertion.calendar_event is None
    assert validate_assertions([assertion], valid_episode())[0].validation_report.valid


def test_legacy_calendar_event_payload_still_validates_when_present() -> None:
    assertion = CandidateSemanticAssertion.model_validate(
            {
                **base_assertion().model_dump(),
                "memory_type": "CALENDAR_EVENT",
                "calendar_event": CalendarEventExtraction(
                    title="Atlas review",
                    recurrence_text="every Tuesday",
                ).model_dump(),
            }
        )

    assert validate_assertions([assertion], valid_episode())[0].validation_report.valid


def test_non_calendar_assertion_allows_no_calendar_event() -> None:
    assertion = base_assertion()

    result = validate_assertions([assertion], valid_episode())[0]

    assert result.validation_report.valid
    assert assertion.calendar_event is None


def test_canonical_text_need_not_occur_verbatim() -> None:
    assertion = base_assertion().model_copy(update={"canonical_text": "Priya is responsible for Atlas."})
    result = validate_assertions([assertion], valid_episode())[0]
    assert result.validation_report.valid


def test_validator_never_mutates_assertion() -> None:
    assertion = base_assertion()
    before = deepcopy(assertion.model_dump())
    validate_assertions([assertion], valid_episode())
    assert assertion.model_dump() == before
