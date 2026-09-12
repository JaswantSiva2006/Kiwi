"""Helpers for carrying uncommitted Redis context across new chat threads."""

from __future__ import annotations

from dataclasses import dataclass

from kivi_memory.working_memory.active_window import ThreadEpisodeEntry
from kivi_memory.working_memory.episodes import ThreadEpisodeStore
from kivi_memory.working_memory.models import ThreadEpisode
from kivi_memory.writeback.policy import compare_stream_ids
from kivi_memory.writeback.state import WritebackStateStore

SEED_EPISODE_PREFIX = "seed-context-"


@dataclass(frozen=True)
class SeedThreadContextResult:
    target_thread_id: str
    copied_episode_count: int
    source_thread_ids: list[str]


def seed_thread_with_uncommitted_context(
    target_thread_id: str,
    *,
    episode_store: ThreadEpisodeStore | None = None,
    state_store: WritebackStateStore | None = None,
    max_source_threads: int = 5,
    max_episodes: int = 6,
    max_tokens: int = 2500,
) -> SeedThreadContextResult:
    """Copy recent uncommitted Redis episodes into a new thread as read context only.

    The copied entries are immediately marked committed for the target thread so
    they are never flushed as new durable memories from that thread.
    """

    store = episode_store or ThreadEpisodeStore()
    state = state_store or WritebackStateStore(redis_client=store.redis, config=store.config)
    source_ids = [thread_id for thread_id in store.list_thread_ids(max_source_threads + 1) if thread_id != target_thread_id]
    pending: list[tuple[str, ThreadEpisodeEntry]] = []
    used_sources: list[str] = []
    for source_thread_id in source_ids:
        entries = store.get_thread_episode_entries(source_thread_id)
        cursor = state.get_state(source_thread_id).last_committed_stream_id
        uncommitted = [
            entry
            for entry in entries
            if cursor is None or compare_stream_ids(entry.stream_id, cursor) > 0
        ]
        if uncommitted:
            used_sources.append(source_thread_id)
            pending.extend((source_thread_id, entry) for entry in uncommitted)

    selected = _select_recent_pending(pending, max_episodes=max_episodes, max_tokens=max_tokens)
    if not selected:
        return SeedThreadContextResult(target_thread_id=target_thread_id, copied_episode_count=0, source_thread_ids=[])

    next_turn_index = 0
    for source_thread_id, entry in selected:
        store.append_thread_episode(
            target_thread_id,
            _copy_episode_for_target(
                target_thread_id=target_thread_id,
                turn_index=next_turn_index,
                source_thread_id=source_thread_id,
                entry=entry,
            ),
        )
        next_turn_index += 1

    target_entries = store.get_thread_episode_entries(target_thread_id)
    if target_entries:
        state.set_last_committed_stream_id(target_thread_id, target_entries[-1].stream_id)
    return SeedThreadContextResult(
        target_thread_id=target_thread_id,
        copied_episode_count=len(selected),
        source_thread_ids=sorted(set(source for source, _entry in selected)),
    )


def is_seed_context_episode(episode: ThreadEpisode) -> bool:
    return episode.episode_id.startswith(SEED_EPISODE_PREFIX)


def _select_recent_pending(
    pending: list[tuple[str, ThreadEpisodeEntry]],
    *,
    max_episodes: int,
    max_tokens: int,
) -> list[tuple[str, ThreadEpisodeEntry]]:
    selected: list[tuple[str, ThreadEpisodeEntry]] = []
    used_tokens = 0
    for item in reversed(pending):
        entry = item[1]
        if len(selected) >= max_episodes:
            break
        if selected and used_tokens + entry.token_count > max_tokens:
            break
        if not selected and entry.token_count > max_tokens:
            selected.append(item)
            break
        selected.append(item)
        used_tokens += entry.token_count
    selected.reverse()
    return selected


def _copy_episode_for_target(
    *,
    target_thread_id: str,
    turn_index: int,
    source_thread_id: str,
    entry: ThreadEpisodeEntry,
) -> ThreadEpisode:
    source = entry.episode
    return ThreadEpisode.from_input(
        {
            "episode_id": f"{SEED_EPISODE_PREFIX}{source_thread_id}-{entry.stream_id}",
            "thread_id": target_thread_id,
            "turn_index": turn_index,
            "timezone": source.timezone,
            "locale": source.locale,
            "started_at": source.started_at,
            "completed_at": source.completed_at,
            "messages": [
                {
                    "message_id": f"{SEED_EPISODE_PREFIX}{source_thread_id}-{entry.stream_id}-{message.message_id}",
                    "role": message.role,
                    "timestamp": message.timestamp,
                    "text": message.text,
                }
                for message in source.messages
            ],
        }
    )
