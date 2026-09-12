from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from kivi_memory.llm_redis_orchestrator.controller import (
    AgentResponse,
    AssistantAppendFailure,
    MemoryPreparationFailure,
    ResponseAgentFailure,
    TurnController,
    UserAppendFailure,
    handle_user_turn,
)
from kivi_memory.llm_redis_orchestrator.models import (
    MemoryPreparationDiagnostics,
    MemoryPreparationResult,
    MergedRetrievedMemory,
)
from kivi_memory.working_memory.models import ThreadMessage


class FakeThreadStore:
    def __init__(self, fail_on_append_role: str | None = None) -> None:
        self.messages: dict[str, list[ThreadMessage]] = {}
        self.events: list[tuple[str, Any]] = []
        self.fail_on_append_role = fail_on_append_role

    def append_message(self, thread_id: str, message: ThreadMessage):
        self.events.append(("append", thread_id, message.role, message.message_id))
        if self.fail_on_append_role == message.role:
            raise RuntimeError(f"{message.role} append failed")
        thread_messages = self.messages.setdefault(thread_id, [])
        if any(existing.message_id == message.message_id for existing in thread_messages):
            return message
        thread_messages.append(message)
        return message

    def get_recent_messages(self, thread_id: str):
        self.events.append(("get_recent", thread_id))
        return list(self.messages.get(thread_id, []))


class FakePreparer:
    def __init__(self, result: MemoryPreparationResult | None = None, fail: bool = False) -> None:
        self.result = result or preparation(False)
        self.fail = fail
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("preparation failed")
        return self.result


class FakeAgent:
    def __init__(self, response: AgentResponse | None = None, fail: bool = False) -> None:
        self.response = response or AgentResponse("Binary search narrows a sorted range.")
        self.fail = fail
        self.calls = []

    def respond(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("agent failed")
        return self.response


def test_normal_no_memory_turn_appends_user_prepares_then_appends_assistant() -> None:
    store = FakeThreadStore()
    preparer = FakePreparer(preparation(False))
    agent = FakeAgent(AgentResponse("Binary search checks the middle item.", message_id="assistant-1"))

    result = TurnController(thread_store=store, memory_preparer=preparer, response_agent=agent).handle_user_turn(
        "thread-a",
        "Explain binary search.",
        message_id="user-1",
        timestamp=datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert preparer.calls == [
        {"thread_id": "thread-a", "current_query": "Explain binary search.", "current_message_id": "user-1"}
    ]
    assert result.memory_preparation.needs_long_term_memory is False
    assert result.assistant_message.message_id == "assistant-1"
    assert [message.role for message in store.messages["thread-a"]] == ["user", "assistant"]
    assert store.events[0] == ("append", "thread-a", "user", "user-1")
    assert store.events[-1] == ("append", "thread-a", "assistant", "assistant-1")


def test_long_term_memory_turn_passes_prepared_context_to_agent() -> None:
    prep = preparation(True, memories=[retrieved_memory("memory-1", "Rohit handles Project Atlas.")])
    agent = FakeAgent(AgentResponse("Rohit handles Project Atlas."))

    result = TurnController(thread_store=FakeThreadStore(), memory_preparer=FakePreparer(prep), response_agent=agent).handle_user_turn(
        "thread-a",
        "Who handles Project Atlas?",
        message_id="user-1",
    )

    assert agent.calls[0]["memory_context"] is prep
    assert result.memory_preparation.retrieved_memory_ids == ["memory-1"]
    assert result.diagnostics.needs_long_term_memory is True
    assert result.diagnostics.retrieval_query_count == 1


def test_existing_thread_order_and_current_message_exclusion_for_agent() -> None:
    store = FakeThreadStore()
    store.append_message("thread-a", ThreadMessage.from_input({"message_id": "old-1", "role": "user", "text": "What is Priya working on?"}))
    store.append_message("thread-a", ThreadMessage.from_input({"message_id": "old-2", "role": "assistant", "text": "Priya is working on Project Phoenix."}))
    agent = FakeAgent(AgentResponse("It uses Redis."))

    TurnController(thread_store=store, memory_preparer=FakePreparer(preparation(True)), response_agent=agent).handle_user_turn(
        "thread-a",
        "What technology does it use?",
        message_id="current",
    )

    assert [message.message_id for message in agent.calls[0]["prior_thread_messages"]] == ["old-1", "old-2"]
    assert [message.role for message in store.messages["thread-a"]] == ["user", "assistant", "user", "assistant"]


def test_duplicate_user_retry_does_not_duplicate_user_message() -> None:
    store = FakeThreadStore()
    controller = TurnController(
        thread_store=store,
        memory_preparer=FakePreparer(preparation(False)),
        response_agent=FakeAgent(AgentResponse("ok", message_id="assistant-1")),
    )

    controller.handle_user_turn("thread-a", "Hello", message_id="user-1")
    controller.handle_user_turn("thread-a", "Hello", message_id="user-1", assistant_message_id="assistant-1")

    user_messages = [message for message in store.messages["thread-a"] if message.role == "user"]
    assert [message.message_id for message in user_messages] == ["user-1"]


def test_response_agent_failure_keeps_user_and_adds_no_assistant() -> None:
    store = FakeThreadStore()

    with pytest.raises(ResponseAgentFailure):
        TurnController(thread_store=store, memory_preparer=FakePreparer(preparation(False)), response_agent=FakeAgent(fail=True)).handle_user_turn(
            "thread-a",
            "Hello",
            message_id="user-1",
        )

    assert [message.role for message in store.messages["thread-a"]] == ["user"]


def test_initial_redis_failure_skips_preparation() -> None:
    preparer = FakePreparer(preparation(False))

    with pytest.raises(UserAppendFailure):
        TurnController(thread_store=FakeThreadStore(fail_on_append_role="user"), memory_preparer=preparer, response_agent=FakeAgent()).handle_user_turn(
            "thread-a",
            "Hello",
            message_id="user-1",
        )

    assert preparer.calls == []


def test_memory_preparation_failure_keeps_user_and_adds_no_assistant() -> None:
    store = FakeThreadStore()

    with pytest.raises(MemoryPreparationFailure):
        TurnController(thread_store=store, memory_preparer=FakePreparer(fail=True), response_agent=FakeAgent()).handle_user_turn(
            "thread-a",
            "Hello",
            message_id="user-1",
        )

    assert [message.role for message in store.messages["thread-a"]] == ["user"]


def test_assistant_append_failure_reports_partial_failure() -> None:
    response = AgentResponse("Generated text", message_id="assistant-1")

    with pytest.raises(AssistantAppendFailure) as excinfo:
        TurnController(
            thread_store=FakeThreadStore(fail_on_append_role="assistant"),
            memory_preparer=FakePreparer(preparation(False)),
            response_agent=FakeAgent(response),
        ).handle_user_turn("thread-a", "Hello", message_id="user-1")

    assert excinfo.value.agent_response is response
    assert excinfo.value.assistant_message_id == "assistant-1"


def test_thread_isolation() -> None:
    store = FakeThreadStore()
    controller = TurnController(thread_store=store, memory_preparer=FakePreparer(preparation(False)), response_agent=FakeAgent(AgentResponse("ok")))

    controller.handle_user_turn("thread-a", "Hello A", message_id="a1", assistant_message_id="a2")
    controller.handle_user_turn("thread-b", "Hello B", message_id="b1", assistant_message_id="b2")

    assert [message.text for message in store.messages["thread-a"]] == ["Hello A", "ok"]
    assert [message.text for message in store.messages["thread-b"]] == ["Hello B", "ok"]


def test_default_response_agent_is_wired_without_redis_write(monkeypatch) -> None:
    created = []

    class DefaultAgent(FakeAgent):
        def __init__(self):
            created.append(True)
            super().__init__(AgentResponse("default response"))

    monkeypatch.setattr("kivi_memory.llm_redis_orchestrator.final_agent.HeyKiviSarvamAgent", DefaultAgent)
    store = FakeThreadStore()

    result = TurnController(thread_store=store, memory_preparer=FakePreparer(preparation(False))).handle_user_turn(
        "thread-a",
        "Hello",
        message_id="user-1",
    )

    assert created == [True]
    assert result.assistant_message.text == "default response"


def test_public_handle_user_turn_helper() -> None:
    result = handle_user_turn(
        "thread-a",
        "Hello",
        message_id="user-1",
        assistant_message_id="assistant-1",
        thread_store=FakeThreadStore(),
        memory_preparer=FakePreparer(preparation(False)),
        response_agent=FakeAgent(AgentResponse("Hi")),
    )

    assert result.user_message.message_id == "user-1"
    assert result.assistant_message.message_id == "assistant-1"


def preparation(needs_memory: bool, memories: list[MergedRetrievedMemory] | None = None) -> MemoryPreparationResult:
    return MemoryPreparationResult(
        thread_id="thread-a",
        current_query="query",
        needs_long_term_memory=needs_memory,
        retrieval_queries=["Who handles Project Atlas?"] if needs_memory else [],
        retrieved_memories=memories or [],
        diagnostics=MemoryPreparationDiagnostics(
            redis_context_ms=1.0,
            router_llm_ms=2.0,
            retrieval_ms=3.0 if needs_memory else 0.0,
            total_ms=4.0,
            router_model_call_count=1,
            retrieval_call_count=1 if needs_memory else 0,
            redis_context_message_count=0,
        ),
    )


def retrieved_memory(memory_id: str, text: str) -> MergedRetrievedMemory:
    return MergedRetrievedMemory(
        memory_id=memory_id,
        canonical_text=text,
        subject={},
        arguments=[],
        temporal={},
        evidence=[],
        matched_retrieval_queries=["Who handles Project Atlas?"],
        retrieval_metadata={},
    )
