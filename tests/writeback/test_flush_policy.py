from __future__ import annotations

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from kivi_memory.common.config import KiviThreadMemoryConfig
from kivi_memory.working_memory import ThreadEpisodeBuilder, ThreadEpisodeStore
from kivi_memory.writeback import FlushPolicy, WritebackStateStore, get_writeback_candidates, writeback_state_key


class FakeRedis:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
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

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update({str(k): str(v) for k, v in mapping.items()})
        return len(mapping)


def test_fewer_than_twenty_episodes_all_active_no_candidates() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 19)

    candidates = get_writeback_candidates("thread-a", episode_store=episode_store, state_store=state_store)

    assert candidates == []


def test_twenty_three_small_episodes_newest_twenty_active_oldest_three_candidates() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 23)

    candidates = get_writeback_candidates("thread-a", episode_store=episode_store, state_store=state_store)

    assert [entry.episode.turn_index for entry in candidates] == [0, 1, 2]


def test_token_limit_creates_boundary_before_episode_count_limit() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 5, user_text="small", assistant_text="small")
    append_one(episode_store, 5, user_text="latest " + ("word " * 80), assistant_text="answer")
    policy = FlushPolicy(max_active_episodes=20, max_active_tokens=30)

    candidates = get_writeback_candidates(
        "thread-a",
        episode_store=episode_store,
        state_store=state_store,
        policy=policy,
    )

    assert [entry.episode.turn_index for entry in candidates] == [0, 1, 2, 3, 4]


def test_committed_cursor_excludes_already_processed_old_episodes() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 23)
    state_store.set_last_committed_stream_id("thread-a", "2-0")

    candidates = get_writeback_candidates("thread-a", episode_store=episode_store, state_store=state_store)

    assert [entry.stream_id for entry in candidates] == ["3-0"]
    assert [entry.episode.turn_index for entry in candidates] == [2]


def test_nonexistent_thread_and_state_behave_cleanly() -> None:
    redis, episode_store, state_store = stores()

    assert state_store.get_state("missing").last_committed_stream_id is None
    assert get_writeback_candidates("missing", episode_store=episode_store, state_store=state_store) == []
    assert writeback_state_key("missing") == "kivi:thread:missing:writeback_state"


def stores():
    redis = FakeRedis()
    config = KiviThreadMemoryConfig(
        redis_url="redis://test",
        ttl_seconds=60,
        max_messages=100,
        context_messages=24,
        context_max_chars=12000,
        context_episodes=6,
        context_max_tokens=2500,
        active_window_episodes=20,
        active_window_tokens=8000,
    )
    return redis, ThreadEpisodeStore(redis_client=redis, config=config), WritebackStateStore(redis_client=redis, config=config)


def append_episodes(store: ThreadEpisodeStore, count: int, user_text: str = "user", assistant_text: str = "assistant") -> None:
    for index in range(count):
        append_one(store, index, user_text=user_text, assistant_text=assistant_text)


def append_one(store: ThreadEpisodeStore, index: int, user_text: str, assistant_text: str) -> None:
    store.append_thread_episode(
        "thread-a",
        ThreadEpisodeBuilder().build(
            episode_id=f"episode-{index}",
            thread_id="thread-a",
            turn_index=index,
            user_text=f"{user_text} {index}",
            assistant_text=f"{assistant_text} {index}",
            user_message_id=f"u{index}",
            assistant_message_id=f"a{index}",
            started_at=now(),
            completed_at=now(),
        ),
    )


def now() -> datetime:
    return datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
