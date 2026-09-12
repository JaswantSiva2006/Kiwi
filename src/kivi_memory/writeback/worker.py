"""Background worker for Redis ThreadEpisode to permanent-memory writeback."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from kivi_memory.pipeline.memory_pipeline import MemoryPipeline
from kivi_memory.working_memory import ThreadEpisodeStore
from kivi_memory.writeback.coordinator import FlushCoordinator, WritebackBatchPlan
from kivi_memory.writeback.memory_episode_builder import MemoryEpisodeBuilder
from kivi_memory.writeback.state import WritebackStateStore


class WritebackStatus(StrEnum):
    NO_PLAN = "NO_PLAN"
    LOCKED = "LOCKED"
    SKIPPED_ALREADY_COMMITTED = "SKIPPED_ALREADY_COMMITTED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass(frozen=True)
class MemoryWritebackResult:
    thread_id: str
    status: WritebackStatus
    plan: WritebackBatchPlan | None = None
    ingestion_id: str | None = None
    pipeline_result: Any | None = None
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class MemoryWritebackWorker:
    """Runs one planned writeback batch and commits the cursor after full success."""

    def __init__(
        self,
        *,
        episode_store: ThreadEpisodeStore | None = None,
        state_store: WritebackStateStore | None = None,
        coordinator: FlushCoordinator | None = None,
        memory_episode_builder: MemoryEpisodeBuilder | None = None,
        memory_pipeline: MemoryPipeline | None = None,
    ) -> None:
        self.episode_store = episode_store or ThreadEpisodeStore()
        self.state_store = state_store or WritebackStateStore()
        self.coordinator = coordinator or FlushCoordinator.from_config(self.episode_store.config)
        self.memory_episode_builder = memory_episode_builder or MemoryEpisodeBuilder()
        self.memory_pipeline = memory_pipeline or MemoryPipeline()

    def process_thread_once(self, thread_id: str, *, now: datetime | None = None) -> MemoryWritebackResult:
        token = self.state_store.acquire_thread_lock(thread_id)
        if token is None:
            return MemoryWritebackResult(thread_id=thread_id, status=WritebackStatus.LOCKED)

        try:
            entries = self.episode_store.get_thread_episode_entries(thread_id)
            state = self.state_store.get_state(thread_id)
            plan = self.coordinator.plan_from_entries(
                thread_id=thread_id,
                entries=entries,
                last_committed_stream_id=state.last_committed_stream_id,
                now=now,
            )
            if plan is None:
                return MemoryWritebackResult(thread_id=thread_id, status=WritebackStatus.NO_PLAN)

            built = self.memory_episode_builder.build(plan=plan, all_entries=entries)
            completed_stream_id = self.state_store.completed_ingestion_stream_id(thread_id, built.ingestion_id)
            if completed_stream_id is not None:
                self.state_store.set_last_committed_stream_id(thread_id, completed_stream_id)
                return MemoryWritebackResult(
                    thread_id=thread_id,
                    status=WritebackStatus.SKIPPED_ALREADY_COMMITTED,
                    plan=plan,
                    ingestion_id=built.ingestion_id,
                    diagnostics={"cursor_advanced_to": completed_stream_id},
                )

            pipeline_result = self.memory_pipeline.process(built.memory_episode)
            self.state_store.mark_ingestion_committed(thread_id, built.ingestion_id, plan.last_target_stream_id)
            return MemoryWritebackResult(
                thread_id=thread_id,
                status=WritebackStatus.SUCCESS,
                plan=plan,
                ingestion_id=built.ingestion_id,
                pipeline_result=pipeline_result,
                diagnostics={"cursor_advanced_to": plan.last_target_stream_id},
            )
        except Exception as exc:
            return MemoryWritebackResult(
                thread_id=thread_id,
                status=WritebackStatus.FAILED,
                error=" ".join(str(exc).split()),
            )
        finally:
            self.state_store.release_thread_lock(thread_id, token)
