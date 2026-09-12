"""Redis Streams storage for completed visible chat turns."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from redis.exceptions import RedisError

from kivi_memory.common.config import (
    DEFAULT_LOCALE,
    DEFAULT_TIMEZONE,
    KiviThreadMemoryConfig,
    load_thread_memory_config_from_env,
)
from kivi_memory.working_memory.models import ThreadEpisode, ThreadEpisodeMessage
from kivi_memory.working_memory.redis_client import get_redis_client
from kivi_memory.working_memory.store import RedisWorkingMemoryError
from kivi_memory.working_memory.active_window import ThreadEpisodeEntry, estimate_episode_tokens

logger = logging.getLogger(__name__)


class ThreadEpisodeBuilder:
    """Build the short-term Redis turn record from visible chat messages only."""

    def build(
        self,
        *,
        thread_id: str,
        turn_index: int,
        user_text: str,
        assistant_text: str,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        episode_id: str | None = None,
        timezone_name: str = DEFAULT_TIMEZONE,
        locale: str = DEFAULT_LOCALE,
        started_at: datetime | str | None = None,
        completed_at: datetime | str | None = None,
        user_timestamp: datetime | str | None = None,
        assistant_timestamp: datetime | str | None = None,
        user_metadata: dict[str, Any] | None = None,
        assistant_metadata: dict[str, Any] | None = None,
    ) -> ThreadEpisode:
        started = _iso(started_at) or _now_iso()
        completed = _iso(completed_at) or _now_iso()
        user_ts = _iso(user_timestamp) or started
        assistant_ts = _iso(assistant_timestamp) or completed
        return ThreadEpisode.from_input(
            {
                "episode_id": episode_id or str(uuid4()),
                "thread_id": thread_id,
                "turn_index": turn_index,
                "timezone": timezone_name,
                "locale": locale,
                "started_at": started,
                "completed_at": completed,
                "messages": [
                    {
                        "message_id": user_message_id or str(uuid4()),
                        "role": "USER",
                        "timestamp": user_ts,
                        "text": user_text,
                        "metadata": user_metadata or {},
                    },
                    {
                        "message_id": assistant_message_id or str(uuid4()),
                        "role": "ASSISTANT",
                        "timestamp": assistant_ts,
                        "text": assistant_text,
                        "metadata": assistant_metadata or {},
                    },
                ],
            }
        )


class ThreadEpisodeStore:
    """Append-only Redis Streams facade for completed thread episodes."""

    def __init__(
        self,
        redis_client: Any | None = None,
        config: KiviThreadMemoryConfig | None = None,
        *,
        redis_url: str | None = None,
    ) -> None:
        self.config = config or load_thread_memory_config_from_env()
        self.redis = redis_client or get_redis_client(redis_url or self.config.redis_url)

    def append_thread_episode(self, thread_id: str, episode: ThreadEpisode | dict[str, Any]) -> ThreadEpisode:
        thread_id = _validate_thread_id(thread_id)
        stored = ThreadEpisode.from_input(episode)
        if stored.thread_id != thread_id:
            raise ValueError("episode.thread_id must match thread_id")
        payload = json.dumps(stored.to_dict(), ensure_ascii=False, separators=(",", ":"))
        started = time.perf_counter()
        try:
            self.redis.xadd(self._episodes_key(thread_id), {"episode": payload})
            self.redis.expire(self._episodes_key(thread_id), self.config.ttl_seconds)
            self._remember_thread(thread_id)
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis thread episode append failed") from exc
        logger.debug("redis_append_episode_ms=%.2f thread_id=%s", _elapsed_ms(started), thread_id)
        return stored

    def get_recent_thread_episodes(self, thread_id: str, max_episodes: int | None = None) -> list[ThreadEpisode]:
        thread_id = _validate_thread_id(thread_id)
        limit = _positive_limit(max_episodes or self.config.context_episodes, "max_episodes")
        started = time.perf_counter()
        try:
            raw_items = self.redis.xrevrange(self._episodes_key(thread_id), count=limit)
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis thread episode read failed") from exc
        episodes = _parse_stream_episodes(raw_items, thread_id)
        episodes.reverse()
        logger.debug("redis_get_episodes_ms=%.2f thread_id=%s episode_count=%s", _elapsed_ms(started), thread_id, len(episodes))
        return episodes

    def get_thread_episode_entries(self, thread_id: str, limit: int | None = None) -> list[ThreadEpisodeEntry]:
        """Return completed episode stream entries in chronological order, preserving stream IDs."""

        thread_id = _validate_thread_id(thread_id)
        started = time.perf_counter()
        try:
            raw_items = self.redis.xrange(self._episodes_key(thread_id), count=limit)
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis thread episode entry read failed") from exc
        entries = _parse_stream_episode_entries(raw_items, thread_id)
        logger.debug("redis_get_episode_entries_ms=%.2f thread_id=%s entry_count=%s", _elapsed_ms(started), thread_id, len(entries))
        return entries

    def load_recent_thread_context(
        self,
        thread_id: str,
        max_episodes: int | None = None,
        max_tokens: int | None = None,
    ) -> list[ThreadEpisode]:
        limit = _positive_limit(max_episodes or self.config.context_episodes, "max_episodes")
        token_budget = _positive_limit(max_tokens or self.config.context_max_tokens, "max_tokens")
        episodes = self.get_recent_thread_episodes(thread_id, limit)
        selected: list[ThreadEpisode] = []
        used = 0
        for episode in reversed(episodes):
            cost = _episode_token_estimate(episode)
            if selected and used + cost > token_budget:
                break
            if not selected and cost > token_budget:
                selected.append(episode)
                break
            selected.append(episode)
            used += cost
        selected.reverse()
        return selected

    def list_thread_ids(self, limit: int = 50) -> list[str]:
        """Return recently active thread IDs, newest first."""

        limit = _positive_limit(limit, "limit")
        try:
            return [str(item) for item in self.redis.zrevrange(self._thread_index_key(), 0, limit - 1)]
        except AttributeError:
            return []
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis thread index read failed") from exc

    def _episodes_key(self, thread_id: str) -> str:
        return thread_episode_key(thread_id)

    def _remember_thread(self, thread_id: str) -> None:
        try:
            self.redis.zadd(self._thread_index_key(), {thread_id: time.time()})
            self.redis.expire(self._thread_index_key(), self.config.ttl_seconds)
        except AttributeError:
            return

    def _thread_index_key(self) -> str:
        return thread_episode_index_key()


def append_thread_episode(
    thread_id: str,
    episode: ThreadEpisode | dict[str, Any],
    *,
    store: ThreadEpisodeStore | None = None,
) -> ThreadEpisode:
    return (store or ThreadEpisodeStore()).append_thread_episode(thread_id, episode)


def get_recent_thread_episodes(
    thread_id: str,
    max_episodes: int,
    *,
    store: ThreadEpisodeStore | None = None,
) -> list[ThreadEpisode]:
    return (store or ThreadEpisodeStore()).get_recent_thread_episodes(thread_id, max_episodes)


def load_recent_thread_context(
    thread_id: str,
    max_episodes: int = 6,
    max_tokens: int = 2500,
    *,
    store: ThreadEpisodeStore | None = None,
) -> list[ThreadEpisode]:
    return (store or ThreadEpisodeStore()).load_recent_thread_context(thread_id, max_episodes, max_tokens)


def thread_episode_key(thread_id: str) -> str:
    return f"kivi:thread:{_validate_thread_id(thread_id)}:episodes"


def thread_episode_index_key() -> str:
    return "kivi:threads:episode_index"


def _parse_stream_episodes(raw_items: list[Any], thread_id: str) -> list[ThreadEpisode]:
    episodes = []
    for _entry_id, fields in raw_items:
        try:
            raw_episode = fields.get("episode") if isinstance(fields, dict) else None
            episodes.append(ThreadEpisode.from_input(json.loads(raw_episode)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Skipping malformed Redis thread episode for thread_id=%s: %s", thread_id, exc)
    return episodes


def _parse_stream_episode_entries(raw_items: list[Any], thread_id: str) -> list[ThreadEpisodeEntry]:
    entries = []
    for entry_id, fields in raw_items:
        try:
            raw_episode = fields.get("episode") if isinstance(fields, dict) else None
            episode = ThreadEpisode.from_input(json.loads(raw_episode))
            entries.append(
                ThreadEpisodeEntry(
                    stream_id=str(entry_id),
                    episode=episode,
                    token_count=estimate_episode_tokens(episode),
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Skipping malformed Redis thread episode entry for thread_id=%s: %s", thread_id, exc)
    return entries


def _episode_token_estimate(episode: ThreadEpisode) -> int:
    text = " ".join(message.text for message in episode.messages)
    return max(1, (len(text) + 3) // 4)


def _validate_thread_id(thread_id: str) -> str:
    value = str(thread_id or "").strip()
    if not value:
        raise ValueError("thread_id must not be empty")
    return value


def _positive_limit(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _iso(value: datetime | str | None) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
