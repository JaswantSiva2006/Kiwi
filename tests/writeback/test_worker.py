from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from kivi_memory.common.config import KiviThreadMemoryConfig
from kivi_memory.common.schemas import CandidateSemanticAssertion, CompilerOutput
from kivi_memory.enrichment.validator import validate_compiler_output
from kivi_memory.working_memory import ThreadEpisodeBuilder, ThreadEpisodeStore
from kivi_memory.writeback import FlushCoordinator, FlushPolicy, MemoryWritebackWorker, WritebackStatus
from kivi_memory.writeback.state import WritebackStateStore


class FakeRedis:
    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.values: dict[str, str] = {}
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

    def set(self, key, value, nx=False, ex=None):
        del ex
        if nx and key in self.values:
            return False
        self.values[key] = str(value)
        return True

    def get(self, key):
        return self.values.get(key)

    def delete(self, key):
        existed = key in self.values
        self.values.pop(key, None)
        return int(existed)


class FakePipeline:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    def process(self, episode):
        self.calls.append(episode)
        if self.fail:
            raise RuntimeError("pipeline exploded")
        return {"episode_id": episode.episode_id, "committed": True}


def test_window_writeback_runs_pipeline_and_advances_cursor() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 23)
    pipeline = FakePipeline()

    result = worker(episode_store, state_store, pipeline).process_thread_once("thread-a", now=now())

    assert result.status == WritebackStatus.SUCCESS
    assert len(pipeline.calls) == 1
    assert state_store.get_state("thread-a").last_committed_stream_id == "3-0"
    assert state_store.is_ingestion_completed("thread-a", result.ingestion_id)
    assert len(episode_store.get_thread_episode_entries("thread-a")) == 23


def test_idle_writeback_processes_short_inactive_thread_and_advances_cursor() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 2, completed_at=now() - timedelta(minutes=30))
    pipeline = FakePipeline()

    result = worker(episode_store, state_store, pipeline).process_thread_once("thread-a", now=now())

    assert result.status == WritebackStatus.SUCCESS
    assert result.plan.trigger == "IDLE"
    assert state_store.get_state("thread-a").last_committed_stream_id == "2-0"
    assert len(pipeline.calls) == 1


def test_failure_does_not_advance_cursor_or_mark_ingestion_and_redis_remains() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 23)
    pipeline = FakePipeline(fail=True)

    result = worker(episode_store, state_store, pipeline).process_thread_once("thread-a", now=now())

    assert result.status == WritebackStatus.FAILED
    assert state_store.get_state("thread-a").last_committed_stream_id is None
    assert state_store.get_state("thread-a").ingestion_ids == {}
    assert len(episode_store.get_thread_episode_entries("thread-a")) == 23


def test_idempotency_skips_completed_ingestion_without_duplicate_pipeline_call() -> None:
    redis, episode_store, state_store = stores()
    append_episodes(episode_store, 23)
    first_pipeline = FakePipeline()
    first = worker(episode_store, state_store, first_pipeline).process_thread_once("thread-a", now=now())
    state_store.set_last_committed_stream_id("thread-a", "0-0")
    retry_pipeline = FakePipeline()

    retry = worker(episode_store, state_store, retry_pipeline).process_thread_once("thread-a", now=now())

    assert first.status == WritebackStatus.SUCCESS
    assert retry.status == WritebackStatus.SKIPPED_ALREADY_COMMITTED
    assert retry.ingestion_id == first.ingestion_id
    assert retry_pipeline.calls == []
    assert state_store.get_state("thread-a").last_committed_stream_id == "3-0"


def test_assistant_feedback_protection_rejects_assistant_only_grounding() -> None:
    redis, episode_store, state_store = stores()
    append_custom_episode(
        episode_store,
        0,
        user_text="Okay.",
        assistant_text="Priya handles Atlas.",
        completed_at=now() - timedelta(minutes=30),
    )
    plan = coordinator(active_episodes=20).plan_from_entries(
        thread_id="thread-a",
        entries=episode_store.get_thread_episode_entries("thread-a"),
        now=now(),
    )
    episode = __import__("kivi_memory.writeback", fromlist=["MemoryEpisodeBuilder"]).MemoryEpisodeBuilder().build(
        plan=plan,
        all_entries=episode_store.get_thread_episode_entries("thread-a"),
    ).memory_episode
    assertion = assertion_with_span("Priya handles Atlas.", "a0", "Priya handles Atlas.")

    report = validate_compiler_output(CompilerOutput(assertions=[assertion]), episode)

    assert not report.valid
    assert any(issue.code == "MISSING_MEMORY_ELIGIBLE_USER_GROUNDING" for issue in report.issues)


def test_context_coreference_can_use_assistant_span_but_requires_target_user_grounding() -> None:
    redis, episode_store, state_store = stores()
    append_custom_episode(
        episode_store,
        0,
        user_text="Tell me more.",
        assistant_text="Are you applying to Jane Street SEE?",
        completed_at=now() - timedelta(minutes=31),
    )
    append_custom_episode(
        episode_store,
        1,
        user_text="Yes, I am.",
        assistant_text="Got it.",
        completed_at=now() - timedelta(minutes=30),
    )
    plan = coordinator(active_episodes=20).plan_from_entries(
        thread_id="thread-a",
        entries=episode_store.get_thread_episode_entries("thread-a"),
        last_committed_stream_id="1-0",
        now=now(),
    )
    episode = __import__("kivi_memory.writeback", fromlist=["MemoryEpisodeBuilder"]).MemoryEpisodeBuilder().build(
        plan=plan,
        all_entries=episode_store.get_thread_episode_entries("thread-a"),
    ).memory_episode
    assertion = CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": "The user is applying to Jane Street SEE.",
            "memory_type": "COMMITMENT_OPEN_LOOP",
            "subject": {"text": "user", "entity_type": "PERSON"},
            "semantic_arguments": [{"role": "program", "text": "Jane Street SEE", "is_entity": True, "entity_type": "PROGRAM"}],
            "predicate_type": "APPLYING_TO",
            "modality": "FACT",
            "polarity": "POSITIVE",
            "certainty": "CERTAIN",
            "explicitness": "IMPLICIT",
            "attributed_to": "user",
            "source_spans": [
                {"message_id": "a0", "text": "Are you applying to Jane Street SEE?"},
                {"message_id": "u1", "text": "Yes, I am."},
            ],
        }
    )

    report = validate_compiler_output(CompilerOutput(assertions=[assertion]), episode)

    assert report.valid


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
        writeback_on_turn_enabled=True,
        writeback_lock_seconds=300,
    )
    return redis, ThreadEpisodeStore(redis_client=redis, config=config), WritebackStateStore(redis_client=redis, config=config)


def worker(episode_store, state_store, pipeline) -> MemoryWritebackWorker:
    return MemoryWritebackWorker(
        episode_store=episode_store,
        state_store=state_store,
        coordinator=coordinator(),
        memory_pipeline=pipeline,
    )


def coordinator(active_episodes: int = 20) -> FlushCoordinator:
    return FlushCoordinator(
        policy=FlushPolicy(max_active_episodes=active_episodes, max_active_tokens=8000),
        writeback_batch_size=3,
        idle_flush_after_seconds=900,
    )


def append_episodes(store: ThreadEpisodeStore, count: int, completed_at: datetime | None = None) -> None:
    for index in range(count):
        append_custom_episode(store, index, f"user {index}", f"assistant {index}", completed_at or now())


def append_custom_episode(
    store: ThreadEpisodeStore,
    index: int,
    user_text: str,
    assistant_text: str,
    completed_at: datetime,
) -> None:
    store.append_thread_episode(
        "thread-a",
        ThreadEpisodeBuilder().build(
            episode_id=f"episode-{index}",
            thread_id="thread-a",
            turn_index=index,
            user_text=user_text,
            assistant_text=assistant_text,
            user_message_id=f"u{index}",
            assistant_message_id=f"a{index}",
            started_at=completed_at,
            completed_at=completed_at,
        ),
    )


def assertion_with_span(text: str, message_id: str, span: str) -> CandidateSemanticAssertion:
    return CandidateSemanticAssertion.model_validate(
        {
            "canonical_text": text,
            "memory_type": "PROJECT_GOAL_TOPIC",
            "subject": {"text": "Priya", "entity_type": "PERSON"},
            "semantic_arguments": [{"role": "project", "text": "Atlas", "is_entity": True, "entity_type": "PROJECT"}],
            "predicate_type": "RESPONSIBLE_FOR",
            "modality": "FACT",
            "polarity": "POSITIVE",
            "certainty": "CERTAIN",
            "explicitness": "EXPLICIT",
            "attributed_to": "assistant",
            "source_spans": [{"message_id": message_id, "text": span}],
        }
    )


def now() -> datetime:
    return datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
