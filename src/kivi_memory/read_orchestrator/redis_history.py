"""Read-only retrieval over older Redis ThreadEpisodes from the same thread."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kivi_memory.common.config import KiviThreadMemoryConfig, load_thread_memory_config_from_env
from kivi_memory.working_memory import ThreadEpisodeStore
from kivi_memory.working_memory.active_window import select_active_thread_window


@dataclass(frozen=True)
class RedisThreadHistoryResult:
    episodes: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RedisThreadHistoryTool:
    """Search inactive retained ThreadEpisodes without touching DB or writeback."""

    def __init__(
        self,
        *,
        episode_store: ThreadEpisodeStore | None = None,
        config: KiviThreadMemoryConfig | None = None,
    ) -> None:
        self.config = config or load_thread_memory_config_from_env()
        self.episode_store = episode_store or ThreadEpisodeStore(config=self.config)

    def search(
        self,
        *,
        thread_id: str,
        query: str | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        limit: int = 8,
    ) -> RedisThreadHistoryResult:
        if not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")

        entries = self.episode_store.get_thread_episode_entries(thread_id)
        window = select_active_thread_window(
            entries,
            max_active_episodes=self.config.active_window_episodes,
            max_active_tokens=self.config.active_window_tokens,
        )
        older_entries = [
            entry for entry in window.inactive_entries
            if _within_time_filter(entry.episode.completed_at, start, end)
        ]
        ranked = sorted(
            older_entries,
            key=lambda entry: (_score_episode(query or "", entry.episode), _parse_datetime(entry.episode.completed_at) or datetime.min),
            reverse=True,
        )
        selected = ranked[:limit]
        return RedisThreadHistoryResult(
            episodes=[_entry_payload(entry) for entry in selected],
            diagnostics={
                "older_episode_count": len(older_entries),
                "returned_count": len(selected),
                "active_boundary_stream_id": window.boundary_stream_id,
            },
        )


def search_redis_thread_history(
    *,
    thread_id: str,
    query: str | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    limit: int = 8,
    tool: RedisThreadHistoryTool | None = None,
) -> RedisThreadHistoryResult:
    return (tool or RedisThreadHistoryTool()).search(
        thread_id=thread_id,
        query=query,
        start=start,
        end=end,
        limit=limit,
    )


def _score_episode(query: str, episode) -> float:
    terms = {term for term in _tokens(query) if len(term) > 2}
    if not terms:
        return 0.0
    text = " ".join(message.text for message in episode.messages)
    haystack = set(_tokens(text))
    return len(terms & haystack) / max(1, len(terms))


def _tokens(text: str) -> list[str]:
    return ["".join(ch for ch in token.casefold() if ch.isalnum()) for token in text.split()]


def _within_time_filter(value: str, start: str | datetime | None, end: str | datetime | None) -> bool:
    timestamp = _parse_datetime(value)
    if timestamp is None:
        return True
    start_dt = _coerce_datetime(start)
    end_dt = _coerce_datetime(end)
    if start_dt is not None and timestamp < start_dt:
        return False
    if end_dt is not None and timestamp >= end_dt:
        return False
    return True


def _coerce_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return _parse_datetime(value)


def _parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _entry_payload(entry) -> dict[str, Any]:
    return {
        "stream_id": entry.stream_id,
        "episode_id": entry.episode.episode_id,
        "turn_index": entry.episode.turn_index,
        "started_at": entry.episode.started_at,
        "completed_at": entry.episode.completed_at,
        "messages": [message.to_dict() for message in entry.episode.messages],
    }
