from __future__ import annotations

from kivi_memory.ingestion.episode_formatter import format_memory_episode
from kivi_memory.common.schemas import MemoryEpisode
from tests.common.test_input_schema import valid_episode_data


def test_formatter_produces_expected_role_and_message_formatting() -> None:
    episode = MemoryEpisode.model_validate(valid_episode_data())
    formatted = format_memory_episode(episode)

    assert '<episode id="ep_1" timezone="Asia/Kolkata" locale="en-IN">' in formatted
    assert "[m1][user][2026-09-03T10:14:00]" in formatted
    assert "Priya's basically handling Atlas now." in formatted
    assert "[m2][assistant][2026-09-03T10:14:20]" in formatted


def test_tool_context_only_appears_when_provided() -> None:
    episode = MemoryEpisode.model_validate(valid_episode_data())
    assert "<tool_context>" not in format_memory_episode(episode)

    data = valid_episode_data()
    data["tool_context"] = [
        {
            "tool_name": "calendar",
            "content": "Available slots: 08:00, 11:30, 16:00",
            "source_message_id": "m1",
        }
    ]
    formatted = format_memory_episode(MemoryEpisode.model_validate(data))
    assert "<tool_context>" in formatted
    assert "[calendar][source=m1]" in formatted


def test_grounding_context_only_appears_when_provided() -> None:
    episode = MemoryEpisode.model_validate(valid_episode_data())
    assert "<grounding>" not in format_memory_episode(episode)

    data = valid_episode_data()
    data["grounding_context"] = {
        "current_app": "Slack",
        "active_document": "Atlas Launch Plan",
        "active_participant": None,
        "selected_text": None,
    }
    formatted = format_memory_episode(MemoryEpisode.model_validate(data))
    assert "<grounding>" in formatted
    assert "current_app=Slack" in formatted
    assert "active_document=Atlas Launch Plan" in formatted
    assert "selected_text" not in formatted
