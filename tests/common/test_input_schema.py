from __future__ import annotations

import pytest
from pydantic import ValidationError

from kivi_memory.common.schemas import MemoryEpisode


def valid_episode_data() -> dict:
    return {
        "episode_id": "ep_1",
        "session_id": None,
        "timezone": "Asia/Kolkata",
        "locale": "en-IN",
        "messages": [
            {
                "message_id": "m1",
                "role": "USER",
                "timestamp": "2026-09-03T10:14:00",
                "text": "Priya's basically handling Atlas now.",
            },
            {
                "message_id": "m2",
                "role": "ASSISTANT",
                "timestamp": "2026-09-03T10:14:20",
                "text": "Since Rohit moved over to payments?",
            },
            {
                "message_id": "m3",
                "role": "USER",
                "timestamp": "2026-09-03T10:15:00",
                "text": "Yeah. We're hoping to ship it around the second week of October.",
            },
        ],
        "tool_context": None,
        "grounding_context": None,
    }


def test_valid_memory_episode() -> None:
    episode = MemoryEpisode.model_validate(valid_episode_data())
    assert episode.episode_id == "ep_1"
    assert len(episode.messages) == 3


def test_duplicate_message_ids_fail() -> None:
    data = valid_episode_data()
    data["messages"][1]["message_id"] = "m1"
    with pytest.raises(ValidationError):
        MemoryEpisode.model_validate(data)


def test_episode_with_no_user_messages_fails() -> None:
    data = valid_episode_data()
    for message in data["messages"]:
        message["role"] = "ASSISTANT"
    with pytest.raises(ValidationError):
        MemoryEpisode.model_validate(data)


def test_empty_text_fails() -> None:
    data = valid_episode_data()
    data["messages"][0]["text"] = " "
    with pytest.raises(ValidationError):
        MemoryEpisode.model_validate(data)
