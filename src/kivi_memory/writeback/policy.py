"""Deterministic normal window-based writeback selection."""

from __future__ import annotations

from dataclasses import dataclass

from kivi_memory.common.config import KiviThreadMemoryConfig, load_thread_memory_config_from_env
from kivi_memory.working_memory import ThreadEpisodeEntry, ThreadEpisodeStore, select_active_thread_window
from kivi_memory.writeback.state import WritebackStateStore


@dataclass(frozen=True)
class FlushPolicy:
    max_active_episodes: int = 20
    max_active_tokens: int = 8000

    @classmethod
    def from_config(cls, config: KiviThreadMemoryConfig | None = None) -> "FlushPolicy":
        loaded = config or load_thread_memory_config_from_env()
        return cls(
            max_active_episodes=loaded.active_window_episodes,
            max_active_tokens=loaded.active_window_tokens,
        )

    def select_active_thread_window(self, entries: list[ThreadEpisodeEntry]):
        return select_active_thread_window(
            entries,
            max_active_episodes=self.max_active_episodes,
            max_active_tokens=self.max_active_tokens,
        )

    def get_writeback_candidates(
        self,
        entries: list[ThreadEpisodeEntry],
        *,
        last_committed_stream_id: str | None = None,
    ) -> list[ThreadEpisodeEntry]:
        window = self.select_active_thread_window(entries)
        return [
            entry
            for entry in window.inactive_entries
            if last_committed_stream_id is None or compare_stream_ids(entry.stream_id, last_committed_stream_id) > 0
        ]


def get_writeback_candidates(
    thread_id: str,
    *,
    episode_store: ThreadEpisodeStore | None = None,
    state_store: WritebackStateStore | None = None,
    policy: FlushPolicy | None = None,
    stream_read_limit: int | None = None,
) -> list[ThreadEpisodeEntry]:
    """Read Redis stream entries and return only uncommitted episodes outside the active window."""

    episodes = (episode_store or ThreadEpisodeStore()).get_thread_episode_entries(thread_id, limit=stream_read_limit)
    state = (state_store or WritebackStateStore()).get_state(thread_id)
    return (policy or FlushPolicy.from_config()).get_writeback_candidates(
        episodes,
        last_committed_stream_id=state.last_committed_stream_id,
    )


def compare_stream_ids(left: str, right: str) -> int:
    left_ms, left_seq = _parse_stream_id(left)
    right_ms, right_seq = _parse_stream_id(right)
    if (left_ms, left_seq) == (right_ms, right_seq):
        return 0
    return 1 if (left_ms, left_seq) > (right_ms, right_seq) else -1


def _parse_stream_id(stream_id: str) -> tuple[int, int]:
    try:
        major, minor = str(stream_id).split("-", 1)
        return int(major), int(minor)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Redis stream id: {stream_id}") from exc
