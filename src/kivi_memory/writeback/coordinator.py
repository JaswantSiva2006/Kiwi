"""Plan Redis ThreadEpisode batches for future permanent-memory writeback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from kivi_memory.common.config import KiviThreadMemoryConfig, load_thread_memory_config_from_env
from kivi_memory.working_memory import ThreadEpisodeEntry, ThreadEpisodeStore, select_active_thread_window
from kivi_memory.writeback.policy import FlushPolicy, compare_stream_ids
from kivi_memory.writeback.state import WritebackStateStore


class FlushTrigger(StrEnum):
    WINDOW_EVICTION = "WINDOW_EVICTION"
    IDLE = "IDLE"


@dataclass(frozen=True)
class WritebackBatchPlan:
    thread_id: str
    trigger: FlushTrigger
    target_entries: list[ThreadEpisodeEntry]
    first_target_stream_id: str
    last_target_stream_id: str


@dataclass(frozen=True)
class FlushCoordinator:
    """Selects already eligible episodes to package, without committing anything."""

    policy: FlushPolicy
    writeback_batch_size: int = 3
    idle_flush_after_seconds: int = 900

    @classmethod
    def from_config(cls, config: KiviThreadMemoryConfig | None = None) -> "FlushCoordinator":
        loaded = config or load_thread_memory_config_from_env()
        return cls(
            policy=FlushPolicy.from_config(loaded),
            writeback_batch_size=loaded.writeback_batch_size,
            idle_flush_after_seconds=loaded.idle_flush_after_seconds,
        )

    def plan_from_entries(
        self,
        *,
        thread_id: str,
        entries: list[ThreadEpisodeEntry],
        last_committed_stream_id: str | None = None,
        now: datetime | None = None,
    ) -> WritebackBatchPlan | None:
        if self.writeback_batch_size < 1:
            raise ValueError("writeback_batch_size must be positive")

        eligible = self.policy.get_writeback_candidates(
            entries,
            last_committed_stream_id=last_committed_stream_id,
        )
        if len(eligible) >= self.writeback_batch_size:
            return _plan(thread_id, FlushTrigger.WINDOW_EVICTION, eligible[: self.writeback_batch_size])

        idle_candidates = self._idle_candidates(entries, last_committed_stream_id, now or datetime.now(timezone.utc))
        if idle_candidates:
            return _plan(thread_id, FlushTrigger.IDLE, idle_candidates[: self.writeback_batch_size])
        return None

    def _idle_candidates(
        self,
        entries: list[ThreadEpisodeEntry],
        last_committed_stream_id: str | None,
        now: datetime,
    ) -> list[ThreadEpisodeEntry]:
        if not entries:
            return []
        latest_completed = _parse_datetime(entries[-1].episode.completed_at)
        if (now - latest_completed).total_seconds() < self.idle_flush_after_seconds:
            return []
        return [
            entry
            for entry in entries
            if last_committed_stream_id is None or compare_stream_ids(entry.stream_id, last_committed_stream_id) > 0
        ]


def plan_writeback(
    thread_id: str,
    *,
    episode_store: ThreadEpisodeStore | None = None,
    state_store: WritebackStateStore | None = None,
    coordinator: FlushCoordinator | None = None,
    now: datetime | None = None,
    stream_read_limit: int | None = None,
) -> WritebackBatchPlan | None:
    entries = (episode_store or ThreadEpisodeStore()).get_thread_episode_entries(thread_id, limit=stream_read_limit)
    state = (state_store or WritebackStateStore()).get_state(thread_id)
    return (coordinator or FlushCoordinator.from_config()).plan_from_entries(
        thread_id=thread_id,
        entries=entries,
        last_committed_stream_id=state.last_committed_stream_id,
        now=now,
    )


def _plan(thread_id: str, trigger: FlushTrigger, entries: list[ThreadEpisodeEntry]) -> WritebackBatchPlan:
    return WritebackBatchPlan(
        thread_id=thread_id,
        trigger=trigger,
        target_entries=entries,
        first_target_stream_id=entries[0].stream_id,
        last_target_stream_id=entries[-1].stream_id,
    )


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
