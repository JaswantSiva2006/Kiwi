from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from kivi_memory.common.config import KiviThreadMemoryConfig
from kivi_memory.read_orchestrator.redis_history import RedisThreadHistoryTool
from kivi_memory.working_memory import ThreadEpisodeBuilder, ThreadEpisodeStore


class FakeRedis:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.expiries: dict[str, float] = {}
        self.next_id = 1

    def xadd(self, key, fields):
        stream_id = f"{self.next_id}-0"
        self.next_id += 1
        self.streams.setdefault(key, []).append((stream_id, dict(fields)))
        return stream_id

    def xrange(self, key, min="-", max="+", count=None):
        del min, max
        items = list(self.streams.get(key, []))
        return items[:count] if count is not None else items

    def xrevrange(self, key, max="+", min="-", count=None):
        del max, min
        items = list(reversed(self.streams.get(key, [])))
        return items[:count] if count is not None else items

    def expire(self, key, seconds):
        self.expiries[key] = time.time() + int(seconds)
        return True


def test_redis_thread_history_searches_only_older_than_active_window() -> None:
    store = ThreadEpisodeStore(redis_client=FakeRedis(), config=config(active_episodes=2))
    append(store, "thread-a", 0, "We discussed simulated annealing for route optimization.")
    append(store, "thread-a", 1, "We talked about lunch.")
    append(store, "thread-a", 2, "Recent Atlas question.")
    append(store, "thread-a", 3, "Current recursion explanation.")

    result = RedisThreadHistoryTool(episode_store=store, config=config(active_episodes=2)).search(
        thread_id="thread-a",
        query="optimization method",
        limit=8,
    )

    assert [episode["turn_index"] for episode in result.episodes] == [0, 1]
    assert "simulated annealing" in result.episodes[0]["messages"][0]["text"]
    assert all(episode["turn_index"] not in {2, 3} for episode in result.episodes)
    assert result.diagnostics["active_boundary_stream_id"] == "3-0"


def config(active_episodes: int = 2) -> KiviThreadMemoryConfig:
    return KiviThreadMemoryConfig(
        redis_url="redis://test",
        ttl_seconds=60,
        active_window_episodes=active_episodes,
        active_window_tokens=8000,
    )


def append(store: ThreadEpisodeStore, thread_id: str, index: int, text: str) -> None:
    timestamp = now() + timedelta(minutes=index)
    store.append_thread_episode(
        thread_id,
        ThreadEpisodeBuilder().build(
            episode_id=f"episode-{index}",
            thread_id=thread_id,
            turn_index=index,
            user_text=text,
            assistant_text="Noted.",
            started_at=timestamp,
            completed_at=timestamp,
        ),
    )


def now() -> datetime:
    return datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
