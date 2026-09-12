from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from kivi_memory.common.config import KiviThreadMemoryConfig
from kivi_memory.ingestion.episode_formatter import format_memory_episode
from kivi_memory.working_memory import ThreadEpisodeBuilder, ThreadEpisodeStore
from kivi_memory.writeback import FlushCoordinator, FlushPolicy, FlushTrigger, MemoryEpisodeBuilder, WritebackBatchPlan
from kivi_memory.writeback.state import WritebackStateStore


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


def test_three_eligible_episodes_create_one_window_eviction_batch() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 23)
    flush_coordinator = coordinator()

    plan = flush_coordinator.plan_from_entries(
        thread_id="thread-a",
        entries=episode_store.get_thread_episode_entries("thread-a"),
        last_committed_stream_id=state_store.get_state("thread-a").last_committed_stream_id,
        now=now(),
    )

    assert plan is not None
    assert plan.trigger == FlushTrigger.WINDOW_EVICTION
    assert [entry.episode.turn_index for entry in plan.target_entries] == [0, 1, 2]
    assert plan.first_target_stream_id == "1-0"
    assert plan.last_target_stream_id == "3-0"


def test_fewer_than_batch_size_has_no_normal_flush() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 22)

    plan = coordinator().plan_from_entries(
        thread_id="thread-a",
        entries=episode_store.get_thread_episode_entries("thread-a"),
        last_committed_stream_id=None,
        now=now(),
    )

    assert plan is None


def test_idle_thread_flushes_remaining_uncommitted_inside_active_window() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 2, completed_at=now() - timedelta(minutes=30))

    plan = coordinator().plan_from_entries(
        thread_id="thread-a",
        entries=episode_store.get_thread_episode_entries("thread-a"),
        last_committed_stream_id=None,
        now=now(),
    )

    assert plan is not None
    assert plan.trigger == FlushTrigger.IDLE
    assert [entry.episode.turn_index for entry in plan.target_entries] == [0, 1]


def test_oldest_eligible_episodes_are_selected_first() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 25)

    plan = coordinator().plan_from_entries(
        thread_id="thread-a",
        entries=episode_store.get_thread_episode_entries("thread-a"),
        last_committed_stream_id=None,
        now=now(),
    )

    assert plan is not None
    assert [entry.stream_id for entry in plan.target_entries] == ["1-0", "2-0", "3-0"]


def test_memory_episode_builder_preserves_exact_messages_and_order() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 5)
    entries = episode_store.get_thread_episode_entries("thread-a")
    plan = coordinator(active_episodes=2).plan_from_entries(
        thread_id="thread-a",
        entries=entries,
        last_committed_stream_id=None,
        now=now(),
    )

    built = MemoryEpisodeBuilder().build(plan=plan, all_entries=entries)

    assert [message.message_id for message in built.memory_episode.messages] == ["u0", "a0", "u1", "a1", "u2", "a2", "u3", "a3"]
    assert [message.text for message in built.memory_episode.messages] == [
        "user 0",
        "assistant 0",
        "user 1",
        "assistant 1",
        "user 2",
        "assistant 2",
        "user 3",
        "assistant 3",
    ]
    assert built.memory_episode.session_id == "thread-a"
    assert built.memory_episode.tool_context is not None
    assert built.memory_episode.tool_context[0].tool_name == "writeback_selection"


def test_preceding_and_following_overlap_are_context_only() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 6)
    entries = episode_store.get_thread_episode_entries("thread-a")
    plan = coordinator(active_episodes=2).plan_from_entries(
        thread_id="thread-a",
        entries=entries,
        last_committed_stream_id="1-0",
        now=now(),
    )

    built = MemoryEpisodeBuilder().build(plan=plan, all_entries=entries)
    by_id = {message.message_id: message for message in built.memory_episode.messages}

    assert by_id["u0"].context_only is True
    assert by_id["a0"].context_only is True
    assert by_id["u4"].context_only is True
    assert by_id["a4"].context_only is True
    assert by_id["u1"].context_only is False
    assert by_id["u3"].context_only is False


def test_target_user_messages_are_memory_eligible_and_assistants_never_are() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 5)
    entries = episode_store.get_thread_episode_entries("thread-a")
    plan = coordinator(active_episodes=2).plan_from_entries(
        thread_id="thread-a",
        entries=entries,
        last_committed_stream_id=None,
        now=now(),
    )

    built = MemoryEpisodeBuilder().build(plan=plan, all_entries=entries)

    for message in built.memory_episode.messages:
        if message.context_only:
            assert message.memory_eligible is False
        elif message.role == "USER":
            assert message.memory_eligible is True
        else:
            assert message.memory_eligible is False


def test_memory_control_handled_user_message_is_not_memory_eligible() -> None:
    redis, episode_store, _state_store = stores()
    del redis
    episode_store.append_thread_episode(
        "thread-a",
        ThreadEpisodeBuilder().build(
            episode_id="episode-0",
            thread_id="thread-a",
            turn_index=0,
            user_text="Forget that I prefer morning meetings.",
            assistant_text="Okay, I won't use that anymore.",
            user_message_id="u0",
            assistant_message_id="a0",
            user_metadata={"memory_control_handled": True, "skip_semantic_writeback": True},
            started_at=now(),
            completed_at=now(),
        ),
    )
    entries = episode_store.get_thread_episode_entries("thread-a")
    plan = WritebackBatchPlan(
        thread_id="thread-a",
        trigger=FlushTrigger.IDLE,
        target_entries=entries,
        first_target_stream_id=entries[0].stream_id,
        last_target_stream_id=entries[-1].stream_id,
    )

    episode = MemoryEpisodeBuilder().build(plan=plan, all_entries=entries).memory_episode

    assert episode.messages[0].message_id == "u0"
    assert episode.messages[0].memory_eligible is False
    assert episode.messages[0].context_only is True


def test_same_target_range_produces_same_ingestion_id_and_formatter_uses_normal_episode_shape() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 5)
    entries = episode_store.get_thread_episode_entries("thread-a")
    plan = coordinator(active_episodes=2).plan_from_entries(
        thread_id="thread-a",
        entries=entries,
        last_committed_stream_id=None,
        now=now(),
    )

    first = MemoryEpisodeBuilder().build(plan=plan, all_entries=entries)
    second = MemoryEpisodeBuilder().build(plan=plan, all_entries=entries)
    formatted = format_memory_episode(first.memory_episode)

    assert first.ingestion_id == second.ingestion_id
    assert first.ingestion_id.startswith("thread-writeback-")
    assert 'memory_eligible="' not in formatted
    assert 'context_only="' not in formatted
    assert 'source_stream_id="' not in formatted
    assert "[u0][user][" in formatted
    assert "<tool_context>" in formatted
    assert "target USER message IDs" in formatted


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
        writeback_batch_size=3,
        idle_flush_after_seconds=900,
    )
    return redis, ThreadEpisodeStore(redis_client=redis, config=config), WritebackStateStore(redis_client=redis, config=config)


def coordinator(active_episodes: int = 20) -> FlushCoordinator:
    return FlushCoordinator(
        policy=FlushPolicy(max_active_episodes=active_episodes, max_active_tokens=8000),
        writeback_batch_size=3,
        idle_flush_after_seconds=900,
    )


def append_episodes(store: ThreadEpisodeStore, count: int, completed_at: datetime | None = None) -> None:
    for index in range(count):
        timestamp = completed_at or now()
        store.append_thread_episode(
            "thread-a",
            ThreadEpisodeBuilder().build(
                episode_id=f"episode-{index}",
                thread_id="thread-a",
                turn_index=index,
                user_text=f"user {index}",
                assistant_text=f"assistant {index}",
                user_message_id=f"u{index}",
                assistant_message_id=f"a{index}",
                started_at=timestamp,
                completed_at=timestamp,
            ),
        )


def now() -> datetime:
    return datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
