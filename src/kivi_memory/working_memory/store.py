"""Redis storage/read abstraction for temporary within-thread memory."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from redis.exceptions import RedisError

from kivi_memory.common.config import KiviThreadMemoryConfig, load_thread_memory_config_from_env
from kivi_memory.working_memory.models import ThreadMessage
from kivi_memory.working_memory.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class RedisWorkingMemoryError(RuntimeError):
    """Raised when Redis working memory is unavailable or rejects an operation."""


APPEND_MESSAGE_LUA = """
if redis.call('SISMEMBER', KEYS[2], ARGV[1]) == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[5])
  redis.call('EXPIRE', KEYS[2], ARGV[5])
  redis.call('EXPIRE', KEYS[3], ARGV[5])
  return 0
end
redis.call('RPUSH', KEYS[1], ARGV[2])
redis.call('LTRIM', KEYS[1], -tonumber(ARGV[3]), -1)
redis.call('SADD', KEYS[2], ARGV[1])
redis.call('HSETNX', KEYS[3], 'created_at', ARGV[4])
redis.call('HSET', KEYS[3], 'last_activity', ARGV[4])
redis.call('EXPIRE', KEYS[1], ARGV[5])
redis.call('EXPIRE', KEYS[2], ARGV[5])
redis.call('EXPIRE', KEYS[3], ARGV[5])
return 1
"""


class ThreadMemoryStore:
    """Read/write facade for Redis-backed thread-local conversational context."""

    def __init__(
        self,
        redis_client: Any | None = None,
        config: KiviThreadMemoryConfig | None = None,
        *,
        redis_url: str | None = None,
    ) -> None:
        self.config = config or load_thread_memory_config_from_env()
        self.redis = redis_client or get_redis_client(redis_url or self.config.redis_url)

    def append_message(self, thread_id: str, message: ThreadMessage | dict[str, Any]) -> ThreadMessage:
        """Append a message once by message_id and refresh the thread's sliding TTL."""

        thread_id = _validate_thread_id(thread_id)
        stored = ThreadMessage.from_input(message)
        now = _now_iso()
        started = time.perf_counter()
        try:
            self.redis.eval(
                APPEND_MESSAGE_LUA,
                3,
                self._messages_key(thread_id),
                self._message_ids_key(thread_id),
                self._meta_key(thread_id),
                stored.message_id,
                json.dumps(stored.to_dict(), ensure_ascii=False, separators=(",", ":")),
                self.config.max_messages,
                now,
                self.config.ttl_seconds,
            )
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis working memory append failed") from exc
        logger.debug("redis_append_ms=%.2f thread_id=%s", _elapsed_ms(started), thread_id)
        return stored

    def get_recent_messages(
        self,
        thread_id: str,
        limit: int | None = None,
        max_chars: int | None = None,
    ) -> list[ThreadMessage]:
        """Return bounded recent context in chronological order."""

        thread_id = _validate_thread_id(thread_id)
        limit = _positive_limit(limit or self.config.context_messages, "limit")
        max_chars = _positive_limit(max_chars or self.config.context_max_chars, "max_chars")
        started = time.perf_counter()
        try:
            raw_items = self.redis.lrange(self._messages_key(thread_id), -limit, -1)
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis working memory context read failed") from exc

        messages = _parse_messages(raw_items, thread_id)
        selected: list[ThreadMessage] = []
        char_count = 0
        for message in reversed(messages):
            text_len = len(message.text)
            if selected and char_count + text_len > max_chars:
                break
            if not selected and text_len > max_chars:
                selected.append(message)
                char_count += text_len
                break
            selected.append(message)
            char_count += text_len
        selected.reverse()
        logger.debug(
            "redis_get_context_ms=%.2f thread_id=%s message_count=%s character_count=%s",
            _elapsed_ms(started),
            thread_id,
            len(selected),
            char_count,
        )
        return selected

    def get_all_messages(self, thread_id: str) -> list[ThreadMessage]:
        thread_id = _validate_thread_id(thread_id)
        try:
            return _parse_messages(self.redis.lrange(self._messages_key(thread_id), 0, -1), thread_id)
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis working memory full read failed") from exc

    def get_thread_metadata(self, thread_id: str) -> dict[str, str]:
        thread_id = _validate_thread_id(thread_id)
        try:
            return dict(self.redis.hgetall(self._meta_key(thread_id)))
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis working memory metadata read failed") from exc

    def clear_thread(self, thread_id: str) -> int:
        thread_id = _validate_thread_id(thread_id)
        try:
            return int(
                self.redis.delete(
                    self._messages_key(thread_id),
                    self._message_ids_key(thread_id),
                    self._meta_key(thread_id),
                )
            )
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis working memory clear failed") from exc

    def thread_exists(self, thread_id: str) -> bool:
        thread_id = _validate_thread_id(thread_id)
        try:
            return bool(self.redis.exists(self._messages_key(thread_id), self._meta_key(thread_id), self._message_ids_key(thread_id)))
        except RedisError as exc:
            raise RedisWorkingMemoryError("Redis working memory existence check failed") from exc

    def _messages_key(self, thread_id: str) -> str:
        return f"kivi:thread:{thread_id}:messages"

    def _message_ids_key(self, thread_id: str) -> str:
        return f"kivi:thread:{thread_id}:message_ids"

    def _meta_key(self, thread_id: str) -> str:
        return f"kivi:thread:{thread_id}:meta"


def _validate_thread_id(thread_id: str) -> str:
    value = str(thread_id or "").strip()
    if not value:
        raise ValueError("thread_id must not be empty")
    return value


def _positive_limit(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _parse_messages(raw_items: list[str], thread_id: str) -> list[ThreadMessage]:
    messages = []
    for item in raw_items:
        try:
            messages.append(ThreadMessage.from_input(json.loads(item)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Skipping malformed Redis thread message for thread_id=%s: %s", thread_id, exc)
    return messages


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
