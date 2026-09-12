from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from kivi_memory.calendar.models import CalendarOccurrence, ScheduleDiagnostics, ScheduleResult
from kivi_memory.read_orchestrator.models import ReadOrchestrationResult, ReadToolDecision, RouterToolCall
from kivi_memory.read_response import READ_RESPONSE_SYSTEM_PROMPT, ReadResponseService, handle_user_query
from kivi_memory.working_memory import ThreadEpisodeBuilder


class FakeEpisodeStore:
    def __init__(self, context=None) -> None:
        self.context = context or []
        self.load_calls = []
        self.appended = []

    def load_recent_thread_context(self, thread_id, max_episodes=None, max_tokens=None):
        self.load_calls.append((thread_id, max_episodes, max_tokens))
        return list(self.context)

    def append_thread_episode(self, thread_id, episode):
        self.appended.append((thread_id, episode))
        self.context.append(episode)
        return episode


class FakeRouter:
    def __init__(self, decision) -> None:
        self.decision = decision
        self.calls = []

    def route(self, current_query, recent_thread_context, *, current_datetime, user_timezone):
        self.calls.append((current_query, recent_thread_context, current_datetime, user_timezone))
        return ReadOrchestrationResult(
            decision=self.decision,
            router_llm_ms=2.0,
            router_model_call_count=1,
        )


class FakeSemanticRetriever:
    def __init__(self, memories=None) -> None:
        self.memories = memories or []
        self.calls = []

    def search(self, *, query):
        self.calls.append(query)
        return list(self.memories)


class FakeCalendarTool:
    def __init__(self, result=None) -> None:
        self.result = result or schedule_result()
        self.calls = []

    def get_schedule(self, start, end):
        self.calls.append((start, end))
        return self.result


class FakeRedisHistoryTool:
    def __init__(self, result=None) -> None:
        self.result = result or {"episodes": []}
        self.calls = []

    def search(self, *, thread_id, query=None, start=None, end=None, limit=8):
        self.calls.append({"thread_id": thread_id, "query": query, "start": start, "end": end, "limit": limit})
        return self.result


class FakeWebSearchTool:
    def __init__(self, result=None) -> None:
        self.result = result or web_result()
        self.calls = []

    def search(self, *, query, freshness="none"):
        self.calls.append({"query": query, "freshness": freshness})
        return self.result


class FakeFinalAnswer:
    def __init__(self, text="Final answer.") -> None:
        self.text = text
        self.calls = []

    def answer(self, *, system_prompt, context):
        self.calls.append((system_prompt, context))
        return self.text


class FailingFinalAnswer(FakeFinalAnswer):
    def answer(self, *, system_prompt, context):
        self.calls.append((system_prompt, context))
        raise RuntimeError("sarvam failed")


class FakeSarvamChatClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []
        self.chat = self

    def completions(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_semantic_only_flow_calls_memory_not_calendar_and_appends_episode() -> None:
    deps = deps_for(decision(True, False), semantic_memories=[semantic_memory()])

    response = asyncio.run(deps.service.handle_user_query(
        thread_id="thread-a",
        user_query="Who handles Atlas?",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    ))

    assert deps.store.load_calls
    assert deps.router.calls[0][1] == []
    assert deps.semantic.calls == ["Who handles Atlas?"]
    assert deps.calendar.calls == []
    assert deps.redis_history.calls == []
    assert deps.web.calls == []
    assert "<CURRENT_THREAD>" in response.context
    assert "<SEMANTIC_MEMORY>" in response.context
    assert "<CALENDAR>" not in response.context
    assert len(deps.store.appended) == 1
    assert deps.store.appended[0][1].messages[0].text == "Who handles Atlas?"
    assert deps.store.appended[0][1].messages[1].text == "Final answer."


def test_calendar_flow_also_calls_semantic_for_grounding() -> None:
    deps = deps_for(decision(False, True))

    response = asyncio.run(deps.service.handle_user_query(
        thread_id="thread-a",
        user_query="What do I have next Tuesday?",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    ))

    assert deps.semantic.calls == ["What do I have next Tuesday?"]
    assert len(deps.calendar.calls) == 1
    assert deps.redis_history.calls == []
    assert deps.web.calls == []
    assert deps.calendar.calls[0][0].isoformat() == "2026-09-15T00:00:00+05:30"
    assert "<CALENDAR>" in response.context
    assert "<SEMANTIC_MEMORY>" in response.context
    assert len(deps.store.appended) == 1


def test_both_flow_calls_memory_and_calendar() -> None:
    deps = deps_for(decision(True, True), semantic_memories=[semantic_memory()])

    response = asyncio.run(deps.service.handle_user_query(
        thread_id="thread-a",
        user_query="What do I have with Priya next Tuesday?",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    ))

    assert deps.semantic.calls == ["What do I have with Priya next Tuesday?"]
    assert len(deps.calendar.calls) == 1
    assert deps.redis_history.calls == []
    assert deps.web.calls == []
    assert "<CURRENT_THREAD>" in response.context
    assert "<SEMANTIC_MEMORY>" in response.context
    assert "<CALENDAR>" in response.context
    assert len(deps.store.appended) == 1


def test_schedule_question_with_semantic_route_also_calls_calendar() -> None:
    deps = deps_for(decision(True, False), semantic_memories=[semantic_memory()])

    response = asyncio.run(deps.service.handle_user_query(
        thread_id="thread-a",
        user_query="What is my normal Payments review schedule, and was there any exception?",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    ))

    assert deps.semantic.calls == ["What is my normal Payments review schedule, and was there any exception?"]
    assert len(deps.calendar.calls) == 1
    assert "<SEMANTIC_MEMORY>" in response.context
    assert "<CALENDAR>" in response.context


def test_redis_only_flow_uses_prior_episode_and_no_tools() -> None:
    prior = ThreadEpisodeBuilder().build(
        thread_id="thread-a",
        turn_index=0,
        user_text="Explain recursion.",
        assistant_text="A function calls itself.",
        started_at=now(),
        completed_at=now(),
    )
    deps = deps_for(decision(False, False), prior_context=[prior])

    response = asyncio.run(deps.service.handle_user_query(
        thread_id="thread-a",
        user_query="Explain your last answer in simpler words.",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    ))

    assert deps.semantic.calls == []
    assert deps.calendar.calls == []
    assert deps.redis_history.calls == []
    assert deps.web.calls == []
    assert "USER: Explain recursion." in response.context
    assert "ASSISTANT: A function calls itself." in response.context
    assert "<SEMANTIC_MEMORY>" not in response.context
    assert "<CALENDAR>" not in response.context
    assert len(deps.store.appended) == 1


def test_first_message_empty_redis_still_works() -> None:
    deps = deps_for(decision(False, False))

    response = asyncio.run(handle_user_query(
        thread_id="new-thread",
        user_query="Hello.",
        current_datetime=now(),
        timezone="Asia/Kolkata",
        service=deps.service,
    ))

    assert deps.router.calls[0][1] == []
    assert "<CURRENT_THREAD>\n(none)\n</CURRENT_THREAD>" in response.context
    assert len(deps.store.appended) == 1


def test_memory_question_with_empty_route_falls_back_to_semantic() -> None:
    deps = deps_for(decision(False, False), semantic_memories=[semantic_memory()])

    response = asyncio.run(deps.service.handle_user_query(
        thread_id="new-thread",
        user_query="Do I still have that Bengaluru trip planned?",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    ))

    assert deps.semantic.calls == ["Do I still have that Bengaluru trip planned?"]
    assert "<SEMANTIC_MEMORY>" in response.context
    assert response.route["tool_calls"] == [
        {
            "tool": "semantic_memory.search",
            "arguments": {"query": "Do I still have that Bengaluru trip planned?"},
        }
    ]


def test_failed_final_answer_does_not_append_thread_episode() -> None:
    deps = deps_for(decision(False, False), final=FailingFinalAnswer())

    try:
        asyncio.run(deps.service.handle_user_query(
            thread_id="thread-a",
            user_query="Hello.",
            current_datetime=now(),
            timezone="Asia/Kolkata",
        ))
    except RuntimeError as exc:
        assert "sarvam failed" in str(exc)
    else:
        raise AssertionError("expected final answer failure")

    assert deps.store.appended == []


def test_sarvam_read_response_retries_once_when_visible_answer_is_empty() -> None:
    from kivi_memory.common.config import KiviOrchestratorConfig
    from kivi_memory.read_response import SarvamReadResponseClient

    client = FakeSarvamChatClient([None, "Visible final answer."])
    config = KiviOrchestratorConfig(sarvam_api_key="test-key")
    final = SarvamReadResponseClient(config=config, client=client)

    answer = final.answer(system_prompt="System", context="Context")

    assert answer == "Visible final answer."
    assert len(client.calls) == 2
    assert "previous response was empty or appeared cut off" in client.calls[1]["messages"][0]["content"]


def test_sarvam_read_response_retries_once_when_answer_looks_truncated() -> None:
    from kivi_memory.common.config import KiviOrchestratorConfig
    from kivi_memory.read_response import SarvamReadResponseClient

    client = FakeSarvamChatClient(["- **Thursday", "Complete answer."])
    config = KiviOrchestratorConfig(sarvam_api_key="test-key")
    final = SarvamReadResponseClient(config=config, client=client)

    answer = final.answer(system_prompt="System", context="Context")

    assert answer == "Complete answer."
    assert len(client.calls) == 2


def test_read_response_prompt_includes_conflict_and_calendar_evidence_rules() -> None:
    assert "most recent created_at/addition time" in READ_RESPONSE_SYSTEM_PROMPT
    assert "compare the structured event time with the evidence text" in READ_RESPONSE_SYSTEM_PROMPT
    assert "older or evidence-free calendar row" in READ_RESPONSE_SYSTEM_PROMPT
    assert "changed/moved/stopped" in READ_RESPONSE_SYSTEM_PROMPT


def test_redis_history_flow_also_calls_semantic_and_adds_context_section() -> None:
    deps = deps_for(history_decision(), redis_history_result=redis_history_result())

    response = asyncio.run(deps.service.handle_user_query(
        thread_id="thread-a",
        user_query="What optimization method did I mention earlier?",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    ))

    assert deps.semantic.calls == ["What optimization method did I mention earlier?"]
    assert deps.calendar.calls == []
    assert len(deps.redis_history.calls) == 1
    assert deps.web.calls == []
    assert deps.redis_history.calls[0]["thread_id"] == "thread-a"
    assert "<REDIS_THREAD_HISTORY>" in response.context
    assert "simulated annealing" in response.context


def test_web_flow_also_calls_semantic() -> None:
    deps = deps_for(web_decision(), web_search_result=web_result())

    response = asyncio.run(deps.service.handle_user_query(
        thread_id="thread-a",
        user_query="What's the latest NVIDIA news?",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    ))

    assert deps.semantic.calls == ["What's the latest NVIDIA news?"]
    assert deps.calendar.calls == []
    assert deps.redis_history.calls == []
    assert deps.web.calls == [{"query": "latest NVIDIA news", "freshness": "day"}]
    assert "<SEMANTIC_MEMORY>" in response.context
    assert "<WEB_CONTEXT>" in response.context
    assert "[W1]" in response.context


def test_semantic_calendar_web_combination_executes_all_selected_tools() -> None:
    deps = deps_for(semantic_calendar_web_decision(), semantic_memories=[semantic_memory()], web_search_result=web_result())

    response = asyncio.run(deps.service.handle_user_query(
        thread_id="thread-a",
        user_query="Find something relevant to my interests next week that I am free to attend.",
        current_datetime=now(),
        timezone="Asia/Kolkata",
    ))

    assert deps.semantic.calls
    assert deps.calendar.calls
    assert deps.web.calls
    assert "<SEMANTIC_MEMORY>" in response.context
    assert "<CALENDAR>" in response.context
    assert "<WEB_CONTEXT>" in response.context


def decision(use_semantic: bool, use_calendar: bool) -> ReadToolDecision:
    calls = []
    if use_semantic:
        calls.append(RouterToolCall(tool="semantic_memory.search", arguments={}))
    if use_calendar:
        calls.append(
            RouterToolCall(
                tool="calendar.get_schedule",
                arguments={
                    "start": "2026-09-15T00:00:00+05:30",
                    "end": "2026-09-16T00:00:00+05:30",
                },
            )
        )
    return ReadToolDecision(tool_calls=calls)


def history_decision() -> ReadToolDecision:
    return ReadToolDecision(
        tool_calls=[
            RouterToolCall(
                tool="redis_thread_history.search",
                arguments={"query": "optimization method", "limit": 8},
            )
        ]
    )


def web_decision() -> ReadToolDecision:
    return ReadToolDecision(
        tool_calls=[
            RouterToolCall(
                tool="web.search",
                arguments={"query": "latest NVIDIA news", "freshness": "day"},
            )
        ]
    )


def semantic_calendar_web_decision() -> ReadToolDecision:
    return ReadToolDecision(
        tool_calls=[
            RouterToolCall(tool="semantic_memory.search", arguments={"query": "user interests"}),
            RouterToolCall(
                tool="calendar.get_schedule",
                arguments={
                    "start": "2026-09-14T00:00:00+05:30",
                    "end": "2026-09-21T00:00:00+05:30",
                },
            ),
            RouterToolCall(tool="web.search", arguments={"query": "events next week", "freshness": "week"}),
        ]
    )


@dataclass
class Deps:
    service: ReadResponseService
    store: FakeEpisodeStore
    router: FakeRouter
    semantic: FakeSemanticRetriever
    calendar: FakeCalendarTool
    redis_history: FakeRedisHistoryTool
    web: FakeWebSearchTool
    final: FakeFinalAnswer


def deps_for(route, semantic_memories=None, prior_context=None, final=None, redis_history_result=None, web_search_result=None) -> Deps:
    store = FakeEpisodeStore(prior_context)
    router = FakeRouter(route)
    semantic = FakeSemanticRetriever(semantic_memories)
    calendar = FakeCalendarTool()
    redis_history = FakeRedisHistoryTool(redis_history_result)
    web = FakeWebSearchTool(web_search_result)
    final = final or FakeFinalAnswer()
    service = ReadResponseService(
        episode_store=store,
        router=router,
        semantic_retriever=semantic,
        calendar_tool=calendar,
        redis_history_tool=redis_history,
        web_search_tool=web,
        final_answer_client=final,
    )
    return Deps(service, store, router, semantic, calendar, redis_history, web, final)


def semantic_memory() -> dict:
    return {
        "memory_id": "mem-1",
        "canonical_text": "Priya handles Atlas.",
        "subject": {"text": "Priya"},
        "predicate_type": "HANDLES",
        "arguments": [{"role": "project", "text": "Atlas"}],
        "temporal": {},
        "evidence": [],
    }


def schedule_result() -> ScheduleResult:
    return ScheduleResult(
        events=[
            CalendarOccurrence(
                calendar_event_id="cal-1",
                semantic_memory_id="mem-2",
                title="Atlas review",
                start="2026-09-15T15:00:00+05:30",
            )
        ],
        diagnostics=ScheduleDiagnostics(
            db_candidate_query_ms=1.0,
            recurrence_expansion_ms=0.0,
            total_ms=1.0,
            db_recurring_series_considered=0,
            occurrences_returned=1,
        ),
    )


def redis_history_result() -> dict:
    return {
        "episodes": [
            {
                "episode_id": "old-1",
                "messages": [
                    {
                        "role": "USER",
                        "text": "I said simulated annealing was the optimization method.",
                    }
                ],
            }
        ]
    }


def web_result() -> dict:
    return {
        "query": "latest NVIDIA news",
        "searched_at": "2026-09-11T10:00:00+00:00",
        "sources": [
            {
                "source_id": "W1",
                "title": "NVIDIA announces update",
                "url": "https://example.com/nvidia",
                "published_at": "2026-09-11",
                "relevant_text": "NVIDIA announced a product update today.",
            }
        ],
    }


def now() -> datetime:
    return datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
