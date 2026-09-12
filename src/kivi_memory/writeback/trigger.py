"""Non-blocking writeback trigger helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from threading import Thread

from kivi_memory.working_memory import ThreadEpisodeStore
from kivi_memory.writeback.worker import MemoryWritebackWorker

logger = logging.getLogger(__name__)


def trigger_writeback_check(thread_id: str, *, now: datetime | None = None) -> None:
    """Schedule one best-effort writeback check outside the response latency path."""

    Thread(target=_run_once, args=(thread_id, now), daemon=True).start()


def trigger_all_idle_writeback_checks(*, now: datetime | None = None, limit: int = 100) -> None:
    """Schedule best-effort writeback checks for recently active Redis threads."""

    Thread(target=_run_all_once, args=(now, limit), daemon=True).start()


def _run_once(thread_id: str, now: datetime | None) -> None:
    result = MemoryWritebackWorker().process_thread_once(thread_id, now=now)
    if result.error:
        logger.warning("writeback_check_failed thread_id=%s status=%s error=%s", thread_id, result.status, result.error)
    else:
        logger.debug("writeback_check_complete thread_id=%s status=%s", thread_id, result.status)


def _run_all_once(now: datetime | None, limit: int) -> None:
    store = ThreadEpisodeStore()
    for thread_id in store.list_thread_ids(limit):
        _run_once(thread_id, now)
