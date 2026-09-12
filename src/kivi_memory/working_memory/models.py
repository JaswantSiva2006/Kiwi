"""Data structures for transient thread messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

VALID_THREAD_MESSAGE_ROLES = {"user", "assistant", "tool", "system"}
VALID_THREAD_EPISODE_ROLES = {"USER", "ASSISTANT"}


@dataclass(frozen=True)
class ThreadMessage:
    message_id: str
    role: str
    text: str
    timestamp: str
    raw_asr: str | None = None
    formatted_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input(cls, message: "ThreadMessage | dict[str, Any]") -> "ThreadMessage":
        if isinstance(message, ThreadMessage):
            data = message.to_dict()
        else:
            data = dict(message)

        message_id = str(data.get("message_id") or uuid4())
        role = str(data.get("role") or "").strip().lower()
        text = data.get("text")
        timestamp = data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        metadata = data.get("metadata")

        if not message_id.strip():
            raise ValueError("message_id must not be empty")
        if role not in VALID_THREAD_MESSAGE_ROLES:
            raise ValueError("role must be one of user, assistant, tool, system")
        if text is None or not str(text).strip():
            raise ValueError("text must not be empty")
        if not str(timestamp).strip():
            raise ValueError("timestamp must not be empty")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")

        return cls(
            message_id=message_id,
            role=role,
            text=str(text),
            timestamp=str(timestamp),
            raw_asr=data.get("raw_asr"),
            formatted_text=data.get("formatted_text"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "text": self.text,
            "raw_asr": self.raw_asr,
            "formatted_text": self.formatted_text,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ThreadEpisodeMessage:
    message_id: str
    role: str
    timestamp: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input(cls, message: "ThreadEpisodeMessage | dict[str, Any]") -> "ThreadEpisodeMessage":
        data = message.to_dict() if isinstance(message, ThreadEpisodeMessage) else dict(message)
        message_id = str(data.get("message_id") or uuid4())
        role = str(data.get("role") or "").strip().upper()
        timestamp = data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        text = data.get("text")
        metadata = data.get("metadata") or {}
        if not message_id.strip():
            raise ValueError("message_id must not be empty")
        if role not in VALID_THREAD_EPISODE_ROLES:
            raise ValueError("role must be USER or ASSISTANT")
        if text is None or not str(text).strip():
            raise ValueError("text must not be empty")
        if not str(timestamp).strip():
            raise ValueError("timestamp must not be empty")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        return cls(message_id=message_id, role=role, timestamp=str(timestamp), text=str(text), metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "timestamp": self.timestamp,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ThreadEpisode:
    episode_id: str
    thread_id: str
    turn_index: int
    timezone: str
    locale: str
    started_at: str
    completed_at: str
    messages: list[ThreadEpisodeMessage]

    @classmethod
    def from_input(cls, episode: "ThreadEpisode | dict[str, Any]") -> "ThreadEpisode":
        data = episode.to_dict() if isinstance(episode, ThreadEpisode) else dict(episode)
        messages = [ThreadEpisodeMessage.from_input(item) for item in data.get("messages") or []]
        started_at = data.get("started_at")
        completed_at = data.get("completed_at")
        if isinstance(started_at, datetime):
            started_at = started_at.isoformat()
        if isinstance(completed_at, datetime):
            completed_at = completed_at.isoformat()
        value = cls(
            episode_id=str(data.get("episode_id") or uuid4()),
            thread_id=str(data.get("thread_id") or "").strip(),
            turn_index=int(data.get("turn_index")),
            timezone=str(data.get("timezone") or "").strip(),
            locale=str(data.get("locale") or "").strip(),
            started_at=str(started_at or "").strip(),
            completed_at=str(completed_at or "").strip(),
            messages=messages,
        )
        value.validate()
        return value

    def validate(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if not self.thread_id.strip():
            raise ValueError("thread_id must not be empty")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        if not self.timezone:
            raise ValueError("timezone must not be empty")
        if not self.locale:
            raise ValueError("locale must not be empty")
        if not self.started_at:
            raise ValueError("started_at must not be empty")
        if not self.completed_at:
            raise ValueError("completed_at must not be empty")
        if len(self.messages) != 2:
            raise ValueError("ThreadEpisode must contain exactly user and assistant messages")
        if [message.role for message in self.messages] != ["USER", "ASSISTANT"]:
            raise ValueError("ThreadEpisode messages must be ordered USER then ASSISTANT")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "thread_id": self.thread_id,
            "turn_index": self.turn_index,
            "timezone": self.timezone,
            "locale": self.locale,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "messages": [message.to_dict() for message in self.messages],
        }
