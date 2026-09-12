from __future__ import annotations

from pathlib import Path

import pytest

from kivi_memory.semantic_compiler.compiler import SemanticCompiler, SemanticCompilerError, compiler_output_json_schema
from kivi_memory.common.config import KiviCompilerConfig
from kivi_memory.common.io import load_compiler_output, write_compiler_output
from kivi_memory.semantic_compiler.prompt import RETRY_INSTRUCTION
from kivi_memory.common.schemas import MemoryEpisode
from kivi_memory.common.config import load_config_from_env
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


def valid_raw_output() -> dict:
    return {
        "assertions": [
            {
                "canonical_text": "Priya is handling Atlas now.",
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
        ]
    }


def episode() -> MemoryEpisode:
    return MemoryEpisode.model_validate(valid_episode_data())


def compiler_with(client: FakeClient) -> SemanticCompiler:
    return SemanticCompiler(
        client=client,
        config=KiviCompilerConfig(model="qwen-test", max_attempts=2, temperature=0),
    )


def test_compiler_accepts_valid_empty_assertions() -> None:
    client = FakeClient([{"assertions": []}])
    result = compiler_with(client).compile(episode())

    assert result.output.assertions == []
    assert result.validation_report.valid
    assert result.model_name == "qwen-test"
    assert result.attempt_count == 1
    assert client.calls[0]["json_schema"]["type"] == "object"
    assert client.calls[0]["temperature"] == 0


def test_semantic_compiler_default_model_is_qwen_3_5_9b_think_true(monkeypatch) -> None:
    monkeypatch.delenv("KIVI_MODEL", raising=False)
    monkeypatch.delenv("KIVI_THINK", raising=False)
    monkeypatch.delenv("KIVI_SEMANTIC_COMPILER_MODEL", raising=False)
    monkeypatch.delenv("KIVI_SEMANTIC_COMPILER_THINK", raising=False)

    config = load_config_from_env()

    assert config.model == "qwen3.5:9b"
    assert config.think is True


def test_compiler_validates_pydantic_output_and_retries_once() -> None:
    client = FakeClient([
        {"assertions": [{"bad": "shape"}]},
        valid_raw_output(),
    ])
    result = compiler_with(client).compile(episode())

    assert result.validation_report.valid
    assert result.attempt_count == 2
    assert RETRY_INSTRUCTION in client.calls[1]["user_content"]


def test_compiler_schema_accepts_calendar_event_field() -> None:
    raw = valid_raw_output()
    raw["assertions"][0]["memory_type"] = "CALENDAR_EVENT"
    raw["assertions"][0]["calendar_event"] = {
        "title": "Atlas design review",
        "event_kind": "MEETING",
        "location_text": None,
        "start_time_text": "tomorrow at 3 PM",
        "end_time_text": None,
        "duration_text": None,
        "recurrence_text": None,
        "timezone_text": None,
        "all_day_hint": False,
    }
    client = FakeClient([raw])

    result = compiler_with(client).compile(episode())

    assert result.output.assertions[0].memory_type == "CALENDAR_EVENT"
    assert result.output.assertions[0].calendar_event.title == "Atlas design review"


def test_compiler_generation_schema_does_not_require_calendar_event_field() -> None:
    assertion_schema = compiler_output_json_schema()["$defs"]["CandidateSemanticAssertion"]

    assert "calendar_event" not in assertion_schema["required"]


def test_compiler_retries_once_on_source_grounding_error() -> None:
    invalid = valid_raw_output()
    invalid["assertions"][0]["source_spans"] = [{"message_id": "m1", "text": "not verbatim"}]
    client = FakeClient([invalid, valid_raw_output()])

    result = compiler_with(client).compile(episode())

    assert result.validation_report.valid
    assert result.attempt_count == 2


def test_compiler_respects_max_attempts() -> None:
    invalid = valid_raw_output()
    invalid["assertions"][0]["source_spans"] = [{"message_id": "missing", "text": "not in episode"}]
    client = FakeClient([invalid, invalid])

    with pytest.raises(SemanticCompilerError):
        compiler_with(client).compile(episode())

    assert len(client.calls) == 2


def test_output_gets_written_correctly(tmp_path: Path) -> None:
    client = FakeClient([{"assertions": []}])
    result = compiler_with(client).compile(episode())
    output_path = tmp_path / "output.json"

    write_compiler_output(result.output, output_path)

    loaded = load_compiler_output(output_path)
    assert loaded.assertions == []
