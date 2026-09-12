"""Redis-backed per-thread writeback cursor state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from redis.exceptions import RedisError

from kivi_memory.common.config import KiviThreadMemoryConfig, load_thread_memory_config_from_env
from kivi_memory.working_memory.redis_client import get_redis_client
from kivi_memory.working_memory.store import RedisWorkingMemoryError


@dataclass(frozen=True)
class WritebackState:
    last_committed_stream_id: str | None = None
    ingestion_ids: dict[str, str] = field(default_factory=dict)


class WritebackStateStore:
    """Stores cursors that are advanced only after permanent-memory commits."""

    def __init__(
        self,
        redis_client: Any | None = None,
        config: KiviThreadMemoryConfig | None = None,
        *,
        redis_url: str | None = None,
    ) -> None:
        self.config = config or load_thread_memory_config_from_env()
        self.redis = redis_client or get_redis_client(redis_url or self.config.redis_url)

    def get_state(self, thread_id: str) -> WritebackState:
        thread_id = _validate_thread_id(thread_id)
        try:
            data = self.redis.hgetall(writeback_state_key(thread_id))
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis writeback state read failed") from exc
        return WritebackState(
            last_committed_stream_id=data.get("last_committed_stream_id") or None,
            ingestion_ids={key.removeprefix("ingestion:"): value for key, value in data.items() if key.startswith("ingestion:")},
        )

    def set_last_committed_stream_id(self, thread_id: str, stream_id: str) -> WritebackState:
        thread_id = _validate_thread_id(thread_id)
        stream_id = _validate_stream_id(stream_id)
        now = str(time.time())
        key = writeback_state_key(thread_id)
        try:
            self.redis.hset(key, mapping={"last_committed_stream_id": stream_id, "updated_at": now})
            self.redis.expire(key, self.config.ttl_seconds)
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis writeback state update failed") from exc
        return WritebackState(last_committed_stream_id=stream_id)

    def is_ingestion_completed(self, thread_id: str, ingestion_id: str) -> bool:
        return self.completed_ingestion_stream_id(thread_id, ingestion_id) is not None

    def completed_ingestion_stream_id(self, thread_id: str, ingestion_id: str) -> str | None:
        thread_id = _validate_thread_id(thread_id)
        ingestion_id = _validate_ingestion_id(ingestion_id)
        try:
            data = self.redis.hgetall(writeback_state_key(thread_id))
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis writeback state read failed") from exc
        return data.get(f"ingestion:{ingestion_id}") or None

    def mark_ingestion_committed(self, thread_id: str, ingestion_id: str, last_target_stream_id: str) -> WritebackState:
        thread_id = _validate_thread_id(thread_id)
        ingestion_id = _validate_ingestion_id(ingestion_id)
        stream_id = _validate_stream_id(last_target_stream_id)
        now = str(time.time())
        key = writeback_state_key(thread_id)
        try:
            self.redis.hset(
                key,
                mapping={
                    "last_committed_stream_id": stream_id,
                    f"ingestion:{ingestion_id}": stream_id,
                    "updated_at": now,
                },
            )
            self.redis.expire(key, self.config.ttl_seconds)
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis writeback state commit failed") from exc
        return self.get_state(thread_id)

    def acquire_thread_lock(self, thread_id: str, ttl_seconds: int | None = None) -> str | None:
        thread_id = _validate_thread_id(thread_id)
        token = str(uuid4())
        try:
            acquired = self.redis.set(
                writeback_lock_key(thread_id),
                token,
                nx=True,
                ex=ttl_seconds or self.config.writeback_lock_seconds,
            )
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis writeback lock acquire failed") from exc
        return token if acquired else None

    def release_thread_lock(self, thread_id: str, token: str) -> None:
        thread_id = _validate_thread_id(thread_id)
        try:
            key = writeback_lock_key(thread_id)
            if self.redis.get(key) == token:
                self.redis.delete(key)
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis writeback lock release failed") from exc


def writeback_state_key(thread_id: str) -> str:
    return f"kivi:thread:{_validate_thread_id(thread_id)}:writeback_state"


def writeback_lock_key(thread_id: str) -> str:
    return f"kivi:thread:{_validate_thread_id(thread_id)}:writeback_lock"


def _validate_thread_id(thread_id: str) -> str:
    value = str(thread_id or "").strip()
    if not value:
        raise ValueError("thread_id must not be empty")
    return value


def _validate_stream_id(stream_id: str) -> str:
    value = str(stream_id or "").strip()
    if not value:
        raise ValueError("stream_id must not be empty")
    return value


def _validate_ingestion_id(ingestion_id: str) -> str:
    value = str(ingestion_id or "").strip()
    if not value:
        raise ValueError("ingestion_id must not be empty")
    return value
