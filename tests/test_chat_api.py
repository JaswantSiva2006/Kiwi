from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from kivi_memory.api.chat import create_app
from kivi_memory.read_response import QueryResponse
from kivi_memory.working_memory import ThreadEpisodeBuilder


class FakeReadService:
    def __init__(self, store) -> None:
        self.episode_store = store
        self.calls = []

    async def handle_user_query(self, *, thread_id, user_query, current_datetime, timezone, locale, event_callback=None):
        self.calls.append((thread_id, user_query, current_datetime, timezone, locale))
        if event_callback is not None:
            event_callback({"type": "tool_activity", "label": "Searching memory..."})
            event_callback({"type": "tool_activity_done"})
            event_callback({"type": "assistant_response", "text": "Kivi answer."})
            event_callback({"type": "memory_updated"})
        episode = ThreadEpisodeBuilder().build(
            thread_id=thread_id,
            turn_index=len(self.episode_store.context),
            user_text=user_query,
            assistant_text="Kivi answer.",
            started_at=current_datetime,
            completed_at=current_datetime,
            timezone_name=timezone,
            locale=locale,
        )
        self.episode_store.context.append(episode)
        return QueryResponse(
            text="Kivi answer.",
            thread_id=thread_id,
            thread_episode_id=episode.episode_id,
            route={},
            context="",
            semantic_memory_count=0,
            calendar_event_count=0,
        )


class FakeEpisodeStore:
    def __init__(self) -> None:
        self.context = []
        self.read_calls = []

    def get_recent_thread_episodes(self, thread_id, max_episodes=None):
        self.read_calls.append((thread_id, max_episodes))
        return list(self.context)


def test_create_thread_returns_unique_id_without_episode_write() -> None:
    store = FakeEpisodeStore()
    app = create_app(read_service=FakeReadService(store), episode_store=store)
    client = TestClient(app)

    first = client.post("/api/chat/threads").json()["thread_id"]
    second = client.post("/api/chat/threads").json()["thread_id"]

    assert first
    assert second
    assert first != second
    assert store.context == []


def test_send_message_calls_read_service_and_returns_visible_answer() -> None:
    store = FakeEpisodeStore()
    service = FakeReadService(store)
    client = TestClient(create_app(read_service=service, episode_store=store))

    response = client.post(
        "/api/chat/threads/thread-1/messages",
        json={
            "message": "Hello Kivi",
            "current_datetime": "2026-09-11T09:30:00+05:30",
            "timezone": "Asia/Kolkata",
            "locale": "en-IN",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"thread_id": "thread-1", "answer": "Kivi answer."}
    assert service.calls[0][0] == "thread-1"
    assert service.calls[0][1] == "Hello Kivi"
    assert service.calls[0][2] == datetime(2026, 9, 11, 9, 30, tzinfo=ZoneInfo("Asia/Kolkata"))


def test_stream_message_emits_tool_answer_and_memory_events() -> None:
    store = FakeEpisodeStore()
    service = FakeReadService(store)
    client = TestClient(create_app(read_service=service, episode_store=store))

    with client.stream(
        "POST",
        "/api/chat/threads/thread-1/messages/stream",
        json={
            "message": "Hello Kivi",
            "current_datetime": "2026-09-11T09:30:00+05:30",
            "timezone": "Asia/Kolkata",
            "locale": "en-IN",
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: tool_activity" in body
    assert "Searching memory..." in body
    assert "event: assistant_response" in body
    assert "Kivi answer." in body
    assert "event: memory_updated" in body
    assert "event: done" in body


def test_thread_history_flattens_episodes_chronologically() -> None:
    store = FakeEpisodeStore()
    store.context.append(
        ThreadEpisodeBuilder().build(
            thread_id="thread-1",
            turn_index=0,
            user_text="Hi",
            assistant_text="Hello.",
            started_at="2026-09-11T09:30:00+05:30",
            completed_at="2026-09-11T09:30:01+05:30",
        )
    )
    client = TestClient(create_app(read_service=FakeReadService(store), episode_store=store))

    response = client.get("/api/chat/threads/thread-1/messages")

    assert response.status_code == 200
    assert response.json()["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
    ]
