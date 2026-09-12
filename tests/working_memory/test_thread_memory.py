from __future__ import annotations

import json
import time

import pytest
from redis.exceptions import RedisError

from kivi_memory.common.config import KiviThreadMemoryConfig
from kivi_memory.working_memory import RedisWorkingMemoryError, ThreadMemoryStore


class FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.expiries: dict[str, float] = {}

    def eval(self, script, numkeys, messages_key, ids_key, meta_key, message_id, payload, max_messages, now, ttl_seconds):
        del script, numkeys
        self._expire_old()
        if message_id in self.sets.setdefault(ids_key, set()):
            self.expire(messages_key, int(ttl_seconds))
            self.expire(ids_key, int(ttl_seconds))
            self.expire(meta_key, int(ttl_seconds))
            return 0
        items = self.lists.setdefault(messages_key, [])
        items.append(payload)
        self.lists[messages_key] = items[-int(max_messages) :]
        self.sets[ids_key].add(message_id)
        meta = self.hashes.setdefault(meta_key, {})
        meta.setdefault("created_at", now)
        meta["last_activity"] = now
        self.expire(messages_key, int(ttl_seconds))
        self.expire(ids_key, int(ttl_seconds))
        self.expire(meta_key, int(ttl_seconds))
        return 1

    def lrange(self, key, start, end):
        self._expire_old()
        items = self.lists.get(key, [])
        if start < 0:
            start = max(len(items) + start, 0)
        if end < 0:
            end = len(items) + end
        return items[start : end + 1]

    def hgetall(self, key):
        self._expire_old()
        return dict(self.hashes.get(key, {}))

    def delete(self, *keys):
        deleted = 0
        for key in keys:
            if key in self.lists:
                del self.lists[key]
                deleted += 1
            if key in self.sets:
                del self.sets[key]
                deleted += 1
            if key in self.hashes:
                del self.hashes[key]
                deleted += 1
            self.expiries.pop(key, None)
        return deleted

    def exists(self, *keys):
        self._expire_old()
        return sum(1 for key in keys if key in self.lists or key in self.sets or key in self.hashes)

    def ttl(self, key):
        self._expire_old()
        expires_at = self.expiries.get(key)
        if expires_at is None:
            return -2
        return max(int(expires_at - time.time()), 0)

    def expire(self, key, seconds):
        if key in self.lists or key in self.sets or key in self.hashes:
            self.expiries[key] = time.time() + int(seconds)
        return True

    def _expire_old(self):
        now = time.time()
        expired = [key for key, expires_at in self.expiries.items() if expires_at <= now]
        for key in expired:
            self.lists.pop(key, None)
            self.sets.pop(key, None)
            self.hashes.pop(key, None)
            self.expiries.pop(key, None)


def store(ttl_seconds: int = 60, context_messages: int = 3, context_max_chars: int = 100) -> ThreadMemoryStore:
    return ThreadMemoryStore(
        redis_client=FakeRedis(),
        config=KiviThreadMemoryConfig(
            redis_url="redis://test",
            ttl_seconds=ttl_seconds,
            max_messages=5,
            context_messages=context_messages,
            context_max_chars=context_max_chars,
        ),
    )


def test_append_and_retrieve_recent_context_chronological_with_fields() -> None:
    memory = store()

    memory.append_message(
        "thread-a",
        {
            "message_id": "m1",
            "role": "user",
            "text": "What is Priya working on?",
            "raw_asr": "what is priya working on",
            "formatted_text": "What is Priya working on?",
            "timestamp": "2026-09-06T10:00:00+00:00",
            "metadata": {"source": "test"},
        },
    )
    memory.append_message("thread-a", {"message_id": "m2", "role": "assistant", "text": "Project Phoenix."})

    messages = memory.get_recent_messages("thread-a")

    assert [message.message_id for message in messages] == ["m1", "m2"]
    assert messages[0].role == "user"
    assert messages[0].raw_asr == "what is priya working on"
    assert messages[0].formatted_text == "What is Priya working on?"
    assert messages[0].metadata == {"source": "test"}
    assert messages[1].timestamp


def test_duplicate_message_id_is_idempotent() -> None:
    memory = store()

    memory.append_message("thread-a", {"message_id": "same", "role": "user", "text": "first"})
    memory.append_message("thread-a", {"message_id": "same", "role": "user", "text": "second"})

    assert [message.text for message in memory.get_all_messages("thread-a")] == ["first"]


def test_context_limit_returns_newest_messages_chronologically() -> None:
    memory = store(context_messages=3)

    for index in range(6):
        memory.append_message("thread-a", {"message_id": f"m{index}", "role": "user", "text": f"text {index}"})

    assert [message.message_id for message in memory.get_recent_messages("thread-a")] == ["m3", "m4", "m5"]


def test_max_chars_preserves_newest_tail() -> None:
    memory = store(context_messages=10, context_max_chars=12)

    memory.append_message("thread-a", {"message_id": "m1", "role": "user", "text": "oldest"})
    memory.append_message("thread-a", {"message_id": "m2", "role": "assistant", "text": "middle"})
    memory.append_message("thread-a", {"message_id": "m3", "role": "user", "text": "newest"})

    assert [message.message_id for message in memory.get_recent_messages("thread-a")] == ["m2", "m3"]


def test_get_all_messages_returns_retained_messages() -> None:
    memory = store()

    for index in range(7):
        memory.append_message("thread-a", {"message_id": f"m{index}", "role": "user", "text": f"text {index}"})

    assert [message.message_id for message in memory.get_all_messages("thread-a")] == ["m2", "m3", "m4", "m5", "m6"]


def test_metadata_and_ttl_are_set_and_refreshed() -> None:
    fake = FakeRedis()
    memory = ThreadMemoryStore(
        redis_client=fake,
        config=KiviThreadMemoryConfig(redis_url="redis://test", ttl_seconds=60, max_messages=5, context_messages=3, context_max_chars=100),
    )

    memory.append_message("thread-a", {"message_id": "m1", "role": "user", "text": "hello"})
    metadata = memory.get_thread_metadata("thread-a")
    ttl_before = fake.ttl("kivi:thread:thread-a:messages")
    fake.expiries["kivi:thread:thread-a:messages"] -= 10
    memory.append_message("thread-a", {"message_id": "m2", "role": "assistant", "text": "hi"})

    assert metadata["created_at"]
    assert metadata["last_activity"]
    assert fake.ttl("kivi:thread:thread-a:messages") >= ttl_before - 1
    assert fake.ttl("kivi:thread:thread-a:meta") > 0
    assert fake.ttl("kivi:thread:thread-a:message_ids") > 0


def test_thread_exists_clear_and_empty_thread() -> None:
    memory = store()

    assert memory.get_recent_messages("thread-a") == []
    assert memory.thread_exists("thread-a") is False
    memory.append_message("thread-a", {"message_id": "m1", "role": "user", "text": "hello"})
    assert memory.thread_exists("thread-a") is True

    memory.clear_thread("thread-a")

    assert memory.thread_exists("thread-a") is False
    assert memory.get_recent_messages("thread-a") == []
    assert memory.get_thread_metadata("thread-a") == {}


def test_two_threads_never_mix() -> None:
    memory = store()

    memory.append_message("thread-a", {"message_id": "a1", "role": "user", "text": "Priya"})
    memory.append_message("thread-b", {"message_id": "b1", "role": "user", "text": "Rohit"})

    assert [message.text for message in memory.get_recent_messages("thread-a")] == ["Priya"]
    assert [message.text for message in memory.get_recent_messages("thread-b")] == ["Rohit"]


def test_malformed_json_is_skipped_without_crashing() -> None:
    fake = FakeRedis()
    memory = ThreadMemoryStore(
        redis_client=fake,
        config=KiviThreadMemoryConfig(redis_url="redis://test", ttl_seconds=60, max_messages=5, context_messages=3, context_max_chars=100),
    )
    fake.lists["kivi:thread:thread-a:messages"] = [
        "not json",
        json.dumps({"message_id": "m1", "role": "user", "text": "valid", "timestamp": "now"}),
    ]

    assert [message.text for message in memory.get_recent_messages("thread-a")] == ["valid"]


def test_validation_rejects_bad_inputs() -> None:
    memory = store()

    with pytest.raises(ValueError, match="thread_id"):
        memory.append_message("", {"message_id": "m1", "role": "user", "text": "hello"})
    with pytest.raises(ValueError, match="role"):
        memory.append_message("thread-a", {"message_id": "m1", "role": "bad", "text": "hello"})
    with pytest.raises(ValueError, match="text"):
        memory.append_message("thread-a", {"message_id": "m1", "role": "user", "text": "   "})


def test_redis_connection_failures_are_clear() -> None:
    class BrokenRedis(FakeRedis):
        def lrange(self, key, start, end):
            raise RedisError("connection refused")

    memory = ThreadMemoryStore(
        redis_client=BrokenRedis(),
        config=KiviThreadMemoryConfig(redis_url="redis://test", ttl_seconds=60, max_messages=5, context_messages=3, context_max_chars=100),
    )

    with pytest.raises(RedisWorkingMemoryError, match="context read failed"):
        memory.get_recent_messages("thread-a")
