"""Build backend MemoryEpisode inputs from planned ThreadEpisode writeback batches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from kivi_memory.common.schemas import MemoryEpisode
from kivi_memory.working_memory import ThreadEpisodeEntry
from kivi_memory.writeback.coordinator import WritebackBatchPlan


@dataclass(frozen=True)
class BuiltMemoryEpisode:
    memory_episode: MemoryEpisode
    ingestion_id: str
    source_thread_id: str
    first_target_stream_id: str
    last_target_stream_id: str


@dataclass(frozen=True)
class MemoryEpisodeBuilder:
    preceding_overlap_episodes: int = 1
    following_overlap_episodes: int = 1

    def build(self, *, plan: WritebackBatchPlan, all_entries: list[ThreadEpisodeEntry]) -> BuiltMemoryEpisode:
        target_ids = {entry.stream_id for entry in plan.target_entries}
        first_index = _entry_index(all_entries, plan.first_target_stream_id)
        last_index = _entry_index(all_entries, plan.last_target_stream_id)
        start = max(0, first_index - self.preceding_overlap_episodes)
        end = min(len(all_entries), last_index + self.following_overlap_episodes + 1)
        selected = all_entries[start:end]
        ingestion_id = deterministic_ingestion_id(
            plan.thread_id,
            plan.first_target_stream_id,
            plan.last_target_stream_id,
        )

        messages = []
        target_user_message_ids: list[str] = []
        context_only_message_ids: list[str] = []
        for entry in selected:
            context_only = entry.stream_id not in target_ids
            for message in entry.episode.messages:
                skip_writeback = bool(message.metadata.get("skip_semantic_writeback"))
                message_context_only = context_only or skip_writeback
                if message.role == "USER" and not message_context_only:
                    target_user_message_ids.append(message.message_id)
                elif message_context_only:
                    context_only_message_ids.append(message.message_id)
                messages.append(
                    {
                        "message_id": message.message_id,
                        "role": message.role,
                        "timestamp": message.timestamp,
                        "text": message.text,
                        "memory_eligible": (message.role == "USER" and not message_context_only),
                        "context_only": message_context_only,
                        "source_thread_episode_id": entry.episode.episode_id,
                        "source_stream_id": entry.stream_id,
                    }
                )

        tool_context = [
            {
                "tool_name": "writeback_selection",
                "source_message_id": None,
                "content": (
                    "Extract durable assertions only when grounded in these target USER message IDs: "
                    f"{', '.join(target_user_message_ids)}. "
                    "These message IDs are context-only and must not independently create durable assertions: "
                    f"{', '.join(context_only_message_ids) or 'none'}. "
                    "Assistant messages are context only."
                ),
            }
        ]

        primary_episode = plan.target_entries[0].episode
        memory_episode = MemoryEpisode.model_validate(
            {
                "episode_id": ingestion_id,
                "session_id": plan.thread_id,
                "timezone": primary_episode.timezone,
                "locale": primary_episode.locale,
                "messages": messages,
                "tool_context": tool_context,
                "grounding_context": None,
            }
        )
        return BuiltMemoryEpisode(
            memory_episode=memory_episode,
            ingestion_id=ingestion_id,
            source_thread_id=plan.thread_id,
            first_target_stream_id=plan.first_target_stream_id,
            last_target_stream_id=plan.last_target_stream_id,
        )


def deterministic_ingestion_id(thread_id: str, first_target_stream_id: str, last_target_stream_id: str) -> str:
    raw = f"{thread_id}|{first_target_stream_id}|{last_target_stream_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"thread-writeback-{digest}"


def _entry_index(entries: list[ThreadEpisodeEntry], stream_id: str) -> int:
    for index, entry in enumerate(entries):
        if entry.stream_id == stream_id:
            return index
    raise ValueError(f"stream_id {stream_id} is not present in all_entries")
