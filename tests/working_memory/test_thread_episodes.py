from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

from kivi_memory.common.config import KiviThreadMemoryConfig
from kivi_memory.working_memory import (
    ThreadEpisodeBuilder,
    ThreadEpisodeStore,
    get_recent_thread_episodes,
    load_recent_thread_context,
    thread_episode_key,
)


class FakeRedisStreams:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.expiries: dict[str, float] = {}
        self.next_id = 1

    def xadd(self, key, fields):
        entry_id = f"{self.next_id}-0"
        self.next_id += 1
        self.streams.setdefault(key, []).append((entry_id, dict(fields)))
        return entry_id

    def xrevrange(self, key, max="+", min="-", count=None):
        del max, min
        items = list(reversed(self.streams.get(key, [])))
        return items[:count] if count is not None else items

    def expire(self, key, seconds):
        if key in self.streams:
            self.expiries[key] = time.time() + int(seconds)
        return True

    def ttl(self, key):
        expires_at = self.expiries.get(key)
        if expires_at is None:
            return -2
        return max(int(expires_at - time.time()), 0)


def test_build_one_thread_episode() -> None:
    episode = builder().build(
        thread_id="thread-a",
        turn_index=1,
        user_text="Who handles Atlas?",
        assistant_text="Priya handles Atlas.",
        user_message_id="u1",
        assistant_message_id="a1",
        started_at=now(),
        completed_at=now(),
    )

    assert episode.thread_id == "thread-a"
    assert episode.turn_index == 1
    assert [message.role for message in episode.messages] == ["USER", "ASSISTANT"]
    assert [message.text for message in episode.messages] == ["Who handles Atlas?", "Priya handles Atlas."]


def test_append_three_episodes_and_read_latest_n_chronological() -> None:
    store = episode_store()
    for index in range(3):
        store.append_thread_episode("thread-a", make_episode(index))

    latest = get_recent_thread_episodes("thread-a", 2, store=store)

    assert [episode.turn_index for episode in latest] == [1, 2]


def test_empty_nonexistent_thread_returns_empty_list() -> None:
    assert episode_store().get_recent_thread_episodes("missing", 3) == []


def test_context_loader_respects_episode_and_token_limits() -> None:
    store = episode_store(context_episodes=3, context_max_tokens=8)
    store.append_thread_episode("thread-a", make_episode(0, user_text="old tiny", assistant_text="old tiny"))
    store.append_thread_episode("thread-a", make_episode(1, user_text="middle tiny", assistant_text="middle tiny"))
    store.append_thread_episode("thread-a", make_episode(2, user_text="newer message with many many characters", assistant_text="newer answer"))
    store.append_thread_episode("thread-a", make_episode(3, user_text="latest tiny", assistant_text="latest tiny"))

    context = load_recent_thread_context("thread-a", max_episodes=3, max_tokens=8, store=store)

    assert [episode.turn_index for episode in context] == [3]


def test_ttl_is_set_and_refreshed() -> None:
    redis = FakeRedisStreams()
    store = episode_store(redis_client=redis, ttl_seconds=60)
    key = thread_episode_key("thread-a")

    store.append_thread_episode("thread-a", make_episode(0))
    ttl_before = redis.ttl(key)
    redis.expiries[key] -= 10
    store.append_thread_episode("thread-a", make_episode(1))

    assert ttl_before > 0
    assert redis.ttl(key) >= ttl_before - 1


def builder() -> ThreadEpisodeBuilder:
    return ThreadEpisodeBuilder()


def episode_store(
    redis_client=None,
    ttl_seconds: int = 60,
    context_episodes: int = 6,
    context_max_tokens: int = 2500,
) -> ThreadEpisodeStore:
    return ThreadEpisodeStore(
        redis_client=redis_client or FakeRedisStreams(),
        config=KiviThreadMemoryConfig(
            redis_url="redis://test",
            ttl_seconds=ttl_seconds,
            max_messages=100,
            context_messages=24,
            context_max_chars=12000,
            context_episodes=context_episodes,
            context_max_tokens=context_max_tokens,
        ),
    )


def make_episode(index: int, user_text: str | None = None, assistant_text: str | None = None):
    return builder().build(
        episode_id=f"episode-{index}",
        thread_id="thread-a",
        turn_index=index,
        user_text=user_text or f"user {index}",
        assistant_text=assistant_text or f"assistant {index}",
        user_message_id=f"u{index}",
        assistant_message_id=f"a{index}",
        started_at=now(),
        completed_at=now(),
    )


def now() -> datetime:
    return datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
