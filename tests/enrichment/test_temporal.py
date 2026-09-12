from __future__ import annotations

import os

import pytest

from kivi_memory.common.config import KiviCompilerConfig
from kivi_memory.common.schemas import CandidateSemanticAssertion, MemoryEpisode
from kivi_memory.enrichment.temporal import (
    TemporalNormalizer,
    TemporalNormalizerError,
    format_temporal_input,
)
from tests.common.test_input_schema import valid_episode_data


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def chat_structured(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def episode_with_text(text: str, timestamp: str = "2026-09-03T10:00:00") -> MemoryEpisode:
    data = valid_episode_data()
    data["messages"] = [{"message_id": "m1", "role": "USER", "timestamp": timestamp, "text": text}]
    return MemoryEpisode.model_validate(data)


def assertion(text: str, source_text: str | None = None) -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": text,
            "memory_type": "OTHER",
            "subject": {"text": "user", "entity_type": "PERSON"},
            "semantic_arguments": [],
            "predicate_type": "TEST_TEMPORAL",
            "modality": "FACT",
            "polarity": "POSITIVE",
            "certainty": "CERTAIN",
            "explicitness": "EXPLICIT",
            "attributed_to": "user",
            "source_spans": [{"message_id": "m1", "text": source_text or text}],
        }
    )


def normalizer_with(responses: list[dict]) -> TemporalNormalizer:
    return TemporalNormalizer(
        client=FakeClient(responses),
        config=KiviCompilerConfig(
            temporal_reasoning_model="qwen-thinking-test",
            temporal_normalization_model="qwen-instruct-test",
            max_attempts=2,
        ),
    )


@pytest.mark.parametrize(
    ("text", "response", "expected_precision"),
    [
        ("Priya handles Atlas.", {"temporal_kind": "NONE", "temporal_precision": "NONE"}, "NONE"),
        ("The user will send it today.", {"temporal_kind": "DISCRETE_EVENT", "event_time": "2026-09-03", "temporal_precision": "DAY"}, "DAY"),
        ("The user will send it tomorrow.", {"temporal_kind": "DISCRETE_EVENT", "event_time": "2026-09-04", "temporal_precision": "DAY"}, "DAY"),
        ("The user will send it tonight.", {"temporal_kind": "DISCRETE_EVENT", "event_time": "2026-09-03", "temporal_precision": "DAY"}, "DAY"),
        ("The user sent it yesterday.", {"temporal_kind": "DISCRETE_EVENT", "event_time": "2026-09-02", "temporal_precision": "DAY"}, "DAY"),
        ("The user will send it next Friday.", {"temporal_kind": "DISCRETE_EVENT", "event_time": "2026-09-04", "temporal_precision": "DAY"}, "DAY"),
        ("The user has a deadline on 2026-10-15.", {"temporal_kind": "DISCRETE_EVENT", "event_time": "2026-10-15", "temporal_precision": "DAY"}, "DAY"),
        ("The user worked there in June 2025.", {"temporal_kind": "STATE_INTERVAL", "event_time": "2025-06", "temporal_precision": "MONTH"}, "MONTH"),
        ("The user worked there from June to August 2025.", {"temporal_kind": "STATE_INTERVAL", "valid_from_hint": "2025-06", "valid_to_hint": "2025-08", "temporal_precision": "MONTH"}, "MONTH"),
        ("Atlas targets around October.", {"temporal_kind": "DISCRETE_EVENT", "event_time": "2026-10", "temporal_precision": "APPROXIMATE"}, "APPROXIMATE"),
        ("The user swims every Tuesday.", {"temporal_kind": "RECURRENCE", "temporal_precision": "DAY", "recurrence": "WEEKLY", "recurrence_specifics": "TUESDAY"}, "DAY"),
        ("Priya is currently responsible for Atlas.", {"temporal_kind": "NONE", "temporal_precision": "NONE"}, "NONE"),
        ("The deck is due tonight.", {"temporal_kind": "DISCRETE_EVENT", "event_time": "2026-09-03", "temporal_precision": "DAY"}, "DAY"),
        ("The user will do it then.", {"temporal_kind": "NONE", "temporal_precision": "NONE"}, "NONE"),
    ],
)
def test_temporal_normalizer_accepts_structured_model_outputs(text: str, response: dict, expected_precision: str) -> None:
    reasoning_payload = {
        "temporal_kind": response.get("temporal_kind", "NONE"),
        "temporal_precision": expected_precision,
        "recurrence": response.get("recurrence", "NONE"),
        "recurrence_specifics": response.get("recurrence_specifics"),
    }
    normalization_payload = {
        "valid_from_hint": response.get("valid_from_hint"),
        "valid_to_hint": response.get("valid_to_hint"),
        "event_time": response.get("event_time"),
    }
    result = normalizer_with([reasoning_payload, normalization_payload]).normalize(assertion(text), episode_with_text(text))

    assert result.temporal_metadata.temporal_precision == expected_precision
    assert result.model_name == "qwen-thinking-test+qwen-instruct-test"


def test_temporal_normalizer_sanitizes_long_recurrence_specifics() -> None:
    dirty_specifics = (
        "Wednesday at 14:00 (2 PM), next week relative to the reference timestamp "
        "resolves to a Wednesday, but this sentence is explanatory reasoning and "
        "must never be stored in calendar recurrence_specifics."
    )

    result = normalizer_with(
        [
            {
                "temporal_kind": "RECURRENCE",
                "temporal_precision": "EXACT",
                "recurrence": "WEEKLY",
                "recurrence_specifics": dirty_specifics,
            },
            {"valid_from_hint": "2026-04-08T14:00:00+05:30", "valid_to_hint": None, "event_time": None},
        ]
    ).normalize(
        assertion("The Payments review goes back to Wednesday at 2 PM next week."),
        episode_with_text("Next week it goes back to Wednesday at 2 PM.", "2026-04-04T10:31:00+05:30"),
    )

    assert result.temporal_metadata.recurrence_specifics == "WEDNESDAY 2 PM"


def test_temporal_normalizer_prefers_daypart_over_reference_time_in_recurrence_specifics() -> None:
    dirty_specifics = "THURSDAY:EVENING+1WEEKS_FROM_REFERENCE_DATE_2026-02-05T23:48:00+05:30"

    result = normalizer_with(
        [
            {
                "temporal_kind": "RECURRENCE",
                "temporal_precision": "APPROXIMATE",
                "recurrence": "WEEKLY",
                "recurrence_specifics": dirty_specifics,
            },
            {"valid_from_hint": "2026-02-12T18:00:00+05:30", "valid_to_hint": None, "event_time": None},
        ]
    ).normalize(
        assertion("Badminton is scheduled for Thursday evenings."),
        episode_with_text("I am moving badminton to Thursday evenings from next week.", "2026-02-05T23:48:00+05:30"),
    )

    assert result.temporal_metadata.recurrence_specifics == "THURSDAY EVENING"


def test_temporal_input_contains_only_assertion_evidence_and_reference() -> None:
    episode = episode_with_text("I told Rahul I'd send it tonight.", "2026-09-03T10:00:55")
    formatted = format_temporal_input(assertion("The user promised to send it tonight.", "I told Rahul I'd send it tonight."), episode)

    assert "<assertion>" in formatted
    assert '<source_text message_id="m1">' in formatted
    assert "<reference_timestamp" in formatted
    assert "2026-09-03T10:00:55" in formatted
    assert "<reference_timezone>\nAsia/Kolkata\n</reference_timezone>" in formatted
    assert "messages" not in formatted


def test_temporal_client_uses_temporal_model_schema_and_temperature_zero() -> None:
    fake = FakeClient(
        [
            {"temporal_kind": "NONE", "temporal_precision": "NONE", "recurrence": "NONE", "recurrence_specifics": None},
            {"valid_from_hint": None, "valid_to_hint": None, "event_time": None},
        ]
    )
    normalizer = TemporalNormalizer(
        client=fake,
        config=KiviCompilerConfig(
            temporal_reasoning_model="qwen-thinking-test",
            temporal_normalization_model="qwen-instruct-test",
        ),
    )

    normalizer.normalize(assertion("Priya handles Atlas."), episode_with_text("Priya handles Atlas."))

    reasoning_call = fake.calls[0]
    normalization_call = fake.calls[1]
    assert reasoning_call["model"] == "qwen-thinking-test"
    assert reasoning_call["temperature"] == 0
    assert reasoning_call["think"] is False
    assert reasoning_call["json_schema"]["type"] == "object"
    assert normalization_call["model"] == "qwen-instruct-test"
    assert normalization_call["temperature"] == 0
    assert normalization_call["think"] is False
    assert normalization_call["json_schema"]["type"] == "object"


def test_temporal_normalizer_retries_once_on_invalid_metadata() -> None:
    result = normalizer_with(
        [
            {"temporal_kind": "DISCRETE_EVENT", "temporal_precision": "DAY", "recurrence": "NONE", "recurrence_specifics": None},
            {"valid_from_hint": None, "valid_to_hint": None, "event_time": "not a date"},
            {"valid_from_hint": None, "valid_to_hint": None, "event_time": "2026-09-03"},
        ]
    ).normalize(assertion("Priya handles Atlas."), episode_with_text("Priya handles Atlas."))

    assert result.temporal_metadata.event_time == "2026-09-03"


def test_temporal_normalizer_respects_max_attempts() -> None:
    normalizer = normalizer_with(
        [
            {"temporal_kind": "STATE_INTERVAL", "temporal_precision": "MONTH", "recurrence": "NONE", "recurrence_specifics": None},
            {"valid_from_hint": "2026-10", "valid_to_hint": "2026-09", "event_time": None},
            {"valid_from_hint": "2026-10", "valid_to_hint": "2026-09", "event_time": None},
        ]
    )

    with pytest.raises(TemporalNormalizerError):
        normalizer.normalize(assertion("The user worked there from October to September."), episode_with_text("The user worked there from October to September."))


def test_multiple_source_message_timestamps_are_included() -> None:
    data = valid_episode_data()
    data["messages"] = [
        {"message_id": "m1", "role": "USER", "timestamp": "2026-09-03T10:00:00", "text": "I need to send it."},
        {"message_id": "m2", "role": "ASSISTANT", "timestamp": "2026-09-03T10:00:10", "text": "Tonight?"},
        {"message_id": "m3", "role": "USER", "timestamp": "2026-09-03T10:00:20", "text": "Yes, tonight."},
    ]
    episode = MemoryEpisode.model_validate(data)
    data_assertion = assertion("The user needs to send it tonight.", "I need to send it.").model_dump()
    data_assertion["source_spans"] = [
        {"message_id": "m1", "text": "I need to send it."},
        {"message_id": "m3", "text": "Yes, tonight."},
    ]

    formatted = format_temporal_input(CandidateSemanticAssertion.model_validate(data_assertion), episode)

    assert '<reference_timestamp message_id="m1">\n2026-09-03T10:00:00\n</reference_timestamp>' in formatted
    assert '<reference_timestamp message_id="m3">\n2026-09-03T10:00:20\n</reference_timestamp>' in formatted


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("KIVI_RUN_OLLAMA_INTEGRATION") != "1",
    reason="set KIVI_RUN_OLLAMA_INTEGRATION=1 to call local Ollama",
)
def test_local_qwen_temporal_integration() -> None:
    episode = episode_with_text("I told Rahul I'd send the deck tonight.", "2026-09-03T10:00:55")
    result = TemporalNormalizer().normalize(
        assertion("The user promised to send Rahul the deck tonight.", "I told Rahul I'd send the deck tonight."),
        episode,
    )

    assert result.temporal_metadata.event_time is not None
    assert result.temporal_metadata.temporal_precision in {"NONE", "DAY", "EXACT", "APPROXIMATE"}
    assert result.reasoning_metadata is not None
    assert result.normalization_metadata is not None
