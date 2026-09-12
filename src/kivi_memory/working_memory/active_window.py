"""Shared active-thread-window selection for read context and writeback."""

from __future__ import annotations

from dataclasses import dataclass

from kivi_memory.working_memory.models import ThreadEpisode


@dataclass(frozen=True)
class ThreadEpisodeEntry:
    """A completed ThreadEpisode plus its Redis stream identity."""

    stream_id: str
    episode: ThreadEpisode
    token_count: int


@dataclass(frozen=True)
class ActiveThreadWindow:
    active_entries: list[ThreadEpisodeEntry]
    inactive_entries: list[ThreadEpisodeEntry]
    boundary_stream_id: str | None


def select_active_thread_window(
    entries: list[ThreadEpisodeEntry],
    *,
    max_active_episodes: int = 20,
    max_active_tokens: int = 8000,
) -> ActiveThreadWindow:
    """Select the largest contiguous newest suffix satisfying count and token limits."""

    if max_active_episodes < 1:
        raise ValueError("max_active_episodes must be positive")
    if max_active_tokens < 1:
        raise ValueError("max_active_tokens must be positive")

    selected: list[ThreadEpisodeEntry] = []
    used_tokens = 0
    for entry in reversed(entries):
        if len(selected) >= max_active_episodes:
            break
        if selected and used_tokens + entry.token_count > max_active_tokens:
            break
        if not selected and entry.token_count > max_active_tokens:
            selected.append(entry)
            break
        selected.append(entry)
        used_tokens += entry.token_count

    active_entries = list(reversed(selected))
    inactive_count = len(entries) - len(active_entries)
    inactive_entries = entries[:inactive_count]
    boundary_stream_id = active_entries[0].stream_id if active_entries else None
    return ActiveThreadWindow(
        active_entries=active_entries,
        inactive_entries=inactive_entries,
        boundary_stream_id=boundary_stream_id,
    )


def estimate_episode_tokens(episode: ThreadEpisode) -> int:
    text = " ".join(message.text for message in episode.messages)
    return max(1, (len(text) + 3) // 4)
