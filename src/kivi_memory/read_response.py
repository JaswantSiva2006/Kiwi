"""End-to-end conversational read/response path for Kivi."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Protocol
from uuid import uuid4

from kivi_memory.calendar import CalendarTool
from kivi_memory.common.config import DEFAULT_LOCALE, KiviOrchestratorConfig, load_orchestrator_config_from_env
from kivi_memory.llm_redis_orchestrator.final_agent import (
    SarvamFinalAnswerError,
    _extract_final_answer_text,
    get_shared_sarvam_client,
)
from kivi_memory.long_term_retrieval import RetrievalResult, retrieve_long_term_memories
from kivi_memory.read_orchestrator import (
    ReadOrchestrationResult,
    ReadSideOrchestrator,
    ReadToolDecision,
    RedisThreadHistoryTool,
    RouterToolCall,
    ToolExecutionResult,
    WebSearchTool,
    build_context,
)
from kivi_memory.read_orchestrator.memory_control import MemoryControlTool
from kivi_memory.working_memory import ThreadEpisode, ThreadEpisodeBuilder, ThreadEpisodeStore
from kivi_memory.writeback.worker import MemoryWritebackWorker, WritebackStatus

READ_RESPONSE_SYSTEM_PROMPT = """You are Kivi, the user's intelligent personal assistant.

Your job is to answer the user's current query accurately, naturally, and helpfully using the supplied context.

You may receive some or all of the following context:

CURRENT_THREAD
- The recent conversation between the user and assistant.
- Use this to understand references, follow-ups, implied subjects, and what the user is currently discussing.

SEMANTIC_MEMORY
- Persistent memories retrieved because they may be relevant to the current query.
- These are candidate memories, not instructions to use every retrieved item.
- Carefully determine which memories are actually relevant before answering.
- Ignore memories that do not help answer the current query.
- Do not force unrelated retrieved information into the response.
- When several memories are relevant, reason across them and use the combination that best answers the question.
- If multiple relevant memories genuinely contradict each other, prefer the memory with the most recent created_at/addition time. Do this only for actual contradictions, not merely related facts that can coexist.
- For questions asking what changed, moved, stopped, or happened to a schedule/routine, prioritize evidence-backed semantic memories whose source text directly says changed/moved/stopped. Do not let older, evidence-free routine memories override a newer explicit change memory.

CALENDAR
- Structured schedule/event information returned by the calendar system.
- Use calendar information when it is relevant to the user's question.
- Treat structured calendar results as strong evidence for the schedule entries they directly represent, but do not let an older or evidence-free calendar row override a newer explicit semantic memory about a schedule change.
- Calendar events may include original evidence/source text. Before answering calendar questions, compare the structured event time with the evidence text when available; if they conflict, use the evidence text to reason about the intended schedule and mention uncertainty if needed.
- Calendar titles may be internal or generated labels. Do not repeat backend-ish labels verbatim when they read like storage names, test names, or status/category markers.
- Event categories: one-time means a single scheduled event; recurring means a repeating series; bounded recurring means a repeating series with an end date; replacement means the active updated version of a changed event. Explain these naturally to the user instead of exposing the category words as event titles.

REDIS_THREAD_HISTORY
- Older same-thread conversation retrieved because it may be relevant.
- Treat it as candidate context: use only relevant parts and ignore unrelated retrieved material.

WEB_CONTEXT
- External web evidence retrieved because it may help answer the current query.
- Treat web results as candidate evidence. Use only sources relevant to the user's actual question.
- For current or factual claims derived from WEB_CONTEXT, preserve source attribution using the supplied [W#] identifiers where useful.
- Do not assume every web result is relevant.
- When WEB_CONTEXT and Kivi's personal context are both relevant, reason across them naturally to answer the user's actual question.

MEMORY_CONTROL
- The result of an explicit user request to inspect, explain, correct, remove, forget, or change Kivi's stored memory.
- Confirm corrections/removals only when status is APPLIED.
- If status is NEEDS_CLARIFICATION or NEEDS_CONFIRMATION, ask exactly for the clarification or confirmation needed.
- If status is NOT_FOUND, say that Kivi does not currently have that stored as a memory.
- Do not expose ledger IDs or internal implementation details unless the user explicitly asks.

USER_QUERY
- The user's current request.
- This is the primary objective you must answer.

Before answering, reason internally about:
1. What exactly is the user asking?
2. Which parts of CURRENT_THREAD are needed?
3. Which retrieved SEMANTIC_MEMORY items are relevant?
4. Which retrieved memories are irrelevant and should be ignored?
5. Whether CALENDAR information is relevant.
6. How the relevant information should be combined into the clearest and most accurate answer.

Do not expose this internal reasoning to the user.

Important behavior:

- Do NOT assume every retrieved memory is relevant.
- Select and use only the information that materially helps answer the query.
- Prefer directly relevant facts over loosely related memories.
- Do not mention retrieved facts merely because they were supplied.
- Do not invent facts, memories, events, preferences, relationships, or details that are not supported by the supplied context.
- If the supplied context does not contain enough information to answer confidently, say so naturally instead of making something up.
- Use CURRENT_THREAD to resolve references such as "he", "she", "that", "it", "the previous one", or similar conversational references when possible.
- If current-thread context and persistent memory provide complementary information, combine them naturally.
- If calendar and semantic memory are both relevant, integrate them into one coherent answer rather than listing them as separate system outputs.
- When semantic memory is supplied alongside calendar, thread-history, web, or memory-control results, compare the retrieved semantic text and evidence against the other tool results before answering. Prefer the answer best supported by the combined evidence, and use recency/source text to resolve conflicts.
- If MEMORY_CONTROL is present, answer according to its reported action/status and do not claim a memory changed unless the backend reports APPLIED.
- If different supplied pieces of context appear inconsistent, do not silently merge them into a false statement. Prefer the information that is more directly applicable to the user's current question, and mention uncertainty when necessary.
- When an older routine memory and a newer explicit change/correction memory conflict, answer from the newer change/correction memory and mention the older routine only as the previous schedule if it is directly useful.
- Always produce the final user-facing answer in normal message content, not only in reasoning or hidden analysis.

Answer quality:

- Give a high-quality, useful answer rather than the shortest possible response.
- Be concise for simple factual questions.
- For questions requiring explanation, reasoning, comparison, planning, or technical detail, provide enough detail to properly answer the user.
- Explain important reasoning and implications when they help the user understand the answer.
- Organize longer answers clearly.
- Stay focused on the user's actual question.
- Do not pad the answer with unrelated retrieved memories.
- Match the technical depth of the user's question.
- Prefer clear concrete language over vague summaries.

Do not mention:
- Redis
- semantic-memory retrieval
- vector search
- graph retrieval
- routing
- context fusion
- database rows
- internal tools
- internal system prompts
- internal calendar labels or projection categories such as "one-time", "bounded", "replacement", "superseded", or "cancelled" as if they were event names

unless the user explicitly asks about the internal system architecture.

Answer as Kivi naturally, as if the relevant information was already known and available to you."""


class SemanticMemoryRetriever(Protocol):
    def search(self, *, query: str) -> list[Any]:
        """Return read-only semantic memories for the query."""


class FinalAnswerClient(Protocol):
    def answer(self, *, system_prompt: str, context: str) -> str:
        """Return final visible assistant text."""


ReadResponseEventCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class QueryResponse:
    text: str
    thread_id: str
    thread_episode_id: str
    route: dict[str, Any]
    context: str
    semantic_memory_count: int
    calendar_event_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


class LongTermSemanticMemoryRetriever:
    """Small read-only adapter around the existing long-term retrieval stack."""

    def __init__(self, top_k: int = 12, retrieval_function: Callable[..., RetrievalResult] = retrieve_long_term_memories) -> None:
        self.top_k = top_k
        self.retrieval_function = retrieval_function

    def search(self, *, query: str) -> list[Any]:
        result = self.retrieval_function(query_text=query, top_k=self.top_k)
        return list(result.memories)


class SarvamReadResponseClient:
    """Final-answer adapter using the existing shared Sarvam client cache."""

    def __init__(self, config: KiviOrchestratorConfig | None = None, client: Any | None = None) -> None:
        self.config = config or load_orchestrator_config_from_env()
        self.client = client

    def answer(self, *, system_prompt: str, context: str) -> str:
        response = self._request(system_prompt=system_prompt, context=context)
        text = _extract_final_answer_text(response)
        if isinstance(text, str) and text.strip() and not _looks_truncated_answer(text):
            return text.strip()

        retry_prompt = (
            system_prompt
            + "\n\nThe previous response was empty or appeared cut off. "
            + "Return a complete final user-facing answer now in normal message content only. "
            + "Finish all bullets, Markdown emphasis, and sentences."
        )
        response = self._request(system_prompt=retry_prompt, context=context)
        text = _extract_final_answer_text(response)
        if not isinstance(text, str) or not text.strip() or _looks_truncated_answer(text):
            raise SarvamFinalAnswerError("Sarvam response did not contain complete non-empty final answer text")
        return text.strip()

    def _request(self, *, system_prompt: str, context: str) -> Any:
        try:
            client = self.client or get_shared_sarvam_client(self.config)
            return client.chat.completions(
                model=self.config.hey_kivi_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=self.config.hey_kivi_temperature,
                reasoning_effort=self.config.hey_kivi_reasoning_effort,
                max_tokens=self.config.hey_kivi_max_tokens,
                stream=False,
            )
        except SarvamFinalAnswerError:
            raise
        except Exception as exc:
            raise SarvamFinalAnswerError(f"Sarvam final-answer request failed: {' '.join(str(exc).split())}") from exc


class ReadResponseService:
    """Coordinates read routing, retrieval, final answer, and Redis turn append."""

    def __init__(
        self,
        *,
        episode_store: ThreadEpisodeStore | None = None,
        router: ReadSideOrchestrator | None = None,
        semantic_retriever: SemanticMemoryRetriever | None = None,
        calendar_tool: CalendarTool | None = None,
        redis_history_tool: RedisThreadHistoryTool | None = None,
        web_search_tool: WebSearchTool | None = None,
        memory_control_tool: MemoryControlTool | None = None,
        final_answer_client: FinalAnswerClient | None = None,
        episode_builder: ThreadEpisodeBuilder | None = None,
        config: KiviOrchestratorConfig | None = None,
    ) -> None:
        self.config = config or load_orchestrator_config_from_env()
        self.episode_store = episode_store or ThreadEpisodeStore()
        self.router = router or ReadSideOrchestrator(config=self.config)
        self.semantic_retriever = semantic_retriever or LongTermSemanticMemoryRetriever(top_k=self.config.agent_max_memories)
        self.calendar_tool = calendar_tool or CalendarTool()
        self.redis_history_tool = redis_history_tool or RedisThreadHistoryTool(episode_store=self.episode_store)
        self.web_search_tool = web_search_tool or WebSearchTool()
        self.memory_control_tool = memory_control_tool or MemoryControlTool()
        self.final_answer_client = final_answer_client or SarvamReadResponseClient(config=self.config)
        self.episode_builder = episode_builder or ThreadEpisodeBuilder()

    async def handle_user_query(
        self,
        *,
        thread_id: str,
        user_query: str,
        current_datetime: datetime,
        timezone: str,
        locale: str = DEFAULT_LOCALE,
        event_callback: ReadResponseEventCallback | None = None,
    ) -> QueryResponse:
        if not user_query.strip():
            raise ValueError("user_query must not be empty")

        total_started = time.perf_counter()
        redis_started = time.perf_counter()
        thread_context = await asyncio.to_thread(self.episode_store.load_recent_thread_context, thread_id)
        redis_context_ms = _elapsed_ms(redis_started)

        router_started = time.perf_counter()
        route_result = await asyncio.to_thread(
            self.router.route,
            user_query,
            thread_context,
            current_datetime=current_datetime,
            user_timezone=timezone,
        )
        router_ms = _elapsed_ms(router_started)
        route = _apply_tool_fallbacks(route_result.decision, user_query, current_datetime)
        tool_results = await self._execute_tool_calls(
            route.tool_calls,
            thread_id=thread_id,
            user_query=user_query,
            current_datetime=current_datetime,
            timezone=timezone,
            locale=locale,
            event_callback=event_callback,
        )

        context_started = time.perf_counter()
        context = build_context(
            query=user_query,
            thread_context=thread_context,
            tool_results=tool_results,
            orchestrator_result=route,
        )
        context_ms = _elapsed_ms(context_started)

        final_started = time.perf_counter()
        answer_text = await asyncio.to_thread(
            self.final_answer_client.answer,
            system_prompt=READ_RESPONSE_SYSTEM_PROMPT,
            context=context,
        )
        final_ms = _elapsed_ms(final_started)
        _emit_event(event_callback, "assistant_response", text=answer_text)

        memory_control_handled = _memory_control_changed_memory(tool_results)

        append_started = time.perf_counter()
        episode = self.episode_builder.build(
            thread_id=thread_id,
            turn_index=_next_turn_index(thread_context),
            user_text=user_query,
            assistant_text=answer_text,
            user_message_id=str(uuid4()),
            assistant_message_id=str(uuid4()),
            timezone_name=timezone,
            locale=locale,
            started_at=current_datetime,
            completed_at=datetime.now(current_datetime.tzinfo),
            user_timestamp=current_datetime,
            assistant_timestamp=datetime.now(current_datetime.tzinfo),
            user_metadata=(
                {"memory_control_handled": True, "skip_semantic_writeback": True}
                if memory_control_handled
                else None
            ),
        )
        stored_episode = await asyncio.to_thread(self.episode_store.append_thread_episode, thread_id, episode)
        append_ms = _elapsed_ms(append_started)
        if getattr(getattr(self.episode_store, "config", None), "writeback_on_turn_enabled", False):
            if event_callback is None:
                from kivi_memory.writeback.trigger import trigger_writeback_check

                trigger_writeback_check(thread_id, now=current_datetime)
            else:
                writeback_result = await asyncio.to_thread(MemoryWritebackWorker().process_thread_once, thread_id, now=current_datetime)
                if writeback_result.status == WritebackStatus.SUCCESS:
                    _emit_event(
                        event_callback,
                        "memory_updated",
                        ingestion_id=writeback_result.ingestion_id,
                        cursor=writeback_result.diagnostics.get("cursor_advanced_to"),
                    )

        return QueryResponse(
            text=answer_text,
            thread_id=thread_id,
            thread_episode_id=stored_episode.episode_id,
            route=route.model_dump(mode="json"),
            context=context,
            semantic_memory_count=_semantic_memory_count(tool_results),
            calendar_event_count=_calendar_event_count(_tool_result(tool_results, "calendar.get_schedule")),
            diagnostics={
                "redis_context_ms": redis_context_ms,
                "router_ms": router_ms,
                "router_llm_ms": route_result.router_llm_ms,
                "router_model_call_count": route_result.router_model_call_count,
                "router_fallback_used": route_result.fallback_used,
                "semantic_retrieval_ms": _tool_latency(tool_results, "semantic_memory.search"),
                "calendar_retrieval_ms": _tool_latency(tool_results, "calendar.get_schedule"),
                "redis_history_retrieval_ms": _tool_latency(tool_results, "redis_thread_history.search"),
                "web_search_ms": _tool_latency(tool_results, "web.search"),
                "tool_call_count": len(tool_results),
                "context_builder_ms": context_ms,
                "final_answer_ms": final_ms,
                "redis_append_ms": append_ms,
                "total_ms": _elapsed_ms(total_started),
            },
        )

    def _search_semantic(self, query: str) -> tuple[list[Any], float]:
        started = time.perf_counter()
        return self.semantic_retriever.search(query=query), _elapsed_ms(started)

    def _get_calendar(self, start: datetime, end: datetime) -> tuple[Any, float]:
        started = time.perf_counter()
        return self.calendar_tool.get_schedule(start, end), _elapsed_ms(started)

    async def _execute_tool_calls(
        self,
        calls,
        *,
        thread_id: str,
        user_query: str,
        current_datetime: datetime,
        timezone: str,
        locale: str,
        event_callback: ReadResponseEventCallback | None = None,
    ) -> list[ToolExecutionResult]:
        calls = _ensure_semantic_grounding(calls, user_query)
        tasks = [
            asyncio.to_thread(
                self._execute_tool_call,
                call,
                thread_id=thread_id,
                user_query=user_query,
                current_datetime=current_datetime,
                timezone=timezone,
                locale=locale,
            )
            for call in calls
        ]
        if not tasks:
            return []
        _emit_event(
            event_callback,
            "tool_activity",
            label=_tool_activity_label([getattr(call, "tool", "") for call in calls]),
            tools=[getattr(call, "tool", "") for call in calls],
        )
        try:
            return list(await asyncio.gather(*tasks))
        finally:
            _emit_event(event_callback, "tool_activity_done")

    def _execute_tool_call(
        self,
        call,
        *,
        thread_id: str,
        user_query: str,
        current_datetime: datetime,
        timezone: str,
        locale: str,
    ) -> ToolExecutionResult:
        started = time.perf_counter()
        arguments = dict(call.arguments or {})
        if call.tool == "semantic_memory.search":
            result = self.semantic_retriever.search(query=arguments.get("query") or user_query)
        elif call.tool == "calendar.get_schedule":
            result = self.calendar_tool.get_schedule(_parse_datetime(arguments["start"]), _parse_datetime(arguments["end"]))
        elif call.tool == "redis_thread_history.search":
            result = self.redis_history_tool.search(
                thread_id=thread_id,
                query=arguments.get("query") or user_query,
                start=arguments.get("start"),
                end=arguments.get("end"),
                limit=int(arguments.get("limit") or 8),
            )
        elif call.tool == "web.search":
            result = self.web_search_tool.search(
                query=arguments.get("query") or user_query,
                freshness=arguments.get("freshness") or "none",
            )
        elif call.tool == "memory.control":
            result = self.memory_control_tool.execute(
                query=arguments.get("query") or user_query,
                thread_id=thread_id,
                current_datetime=current_datetime,
                timezone=timezone,
                locale=locale,
            )
        else:
            raise ValueError(f"unknown read tool {call.tool!r}")
        return ToolExecutionResult(
            tool=call.tool,
            arguments=arguments,
            result=result,
            latency_ms=_elapsed_ms(started),
        )


async def handle_user_query(
    *,
    thread_id: str,
    user_query: str,
    current_datetime: datetime,
    timezone: str,
    locale: str = DEFAULT_LOCALE,
    service: ReadResponseService | None = None,
) -> QueryResponse:
    return await (service or ReadResponseService()).handle_user_query(
        thread_id=thread_id,
        user_query=user_query,
        current_datetime=current_datetime,
        timezone=timezone,
        locale=locale,
    )


def _calendar_event_count(calendar_result: Any) -> int:
    if calendar_result is None:
        return 0
    events = getattr(calendar_result, "events", None)
    if events is None and isinstance(calendar_result, dict):
        events = calendar_result.get("events")
    return len(events or [])


def _semantic_memory_count(tool_results: list[ToolExecutionResult]) -> int:
    memories = _tool_result(tool_results, "semantic_memory.search")
    return len(memories or [])


def _tool_result(tool_results: list[ToolExecutionResult], tool: str) -> Any | None:
    for item in tool_results:
        if item.tool == tool:
            return item.result
    return None


def _tool_latency(tool_results: list[ToolExecutionResult], tool: str) -> float:
    for item in tool_results:
        if item.tool == tool:
            return item.latency_ms
    return 0.0


def _memory_control_changed_memory(tool_results: list[ToolExecutionResult]) -> bool:
    result = _tool_result(tool_results, "memory.control")
    if result is None:
        return False
    status = _get_result_field(result, "status")
    action = _get_result_field(result, "action")
    return status == "APPLIED" and action in {"FORGET", "CORRECT"}


def _get_result_field(result: Any, field: str) -> Any:
    if isinstance(result, dict):
        return result.get(field)
    return getattr(result, field, None)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("tool datetime arguments must be timezone-aware")
    return parsed


def _next_turn_index(thread_context: list[ThreadEpisode]) -> int:
    if not thread_context:
        return 0
    return max(episode.turn_index for episode in thread_context) + 1


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _emit_event(callback: ReadResponseEventCallback | None, event_type: str, **payload: Any) -> None:
    if callback is not None:
        callback({"type": event_type, **payload})


def _tool_activity_label(tool_names: list[str]) -> str:
    labels = {
        "semantic_memory.search": "Searching memory...",
        "calendar.get_schedule": "Accessing calendar...",
        "redis_thread_history.search": "Checking earlier conversation...",
        "web.search": "Searching the web...",
        "memory.control": "Updating memory...",
    }
    unique = [name for index, name in enumerate(tool_names) if name and name not in tool_names[:index]]
    if len(unique) == 1:
        return labels.get(unique[0], "Accessing tools...")
    return "Accessing tools..."


def _ensure_semantic_grounding(calls, user_query: str):
    call_list = list(calls or [])
    if not call_list:
        return call_list
    has_semantic = any(getattr(call, "tool", None) == "semantic_memory.search" for call in call_list)
    has_other_tool = any(getattr(call, "tool", None) != "semantic_memory.search" for call in call_list)
    if has_semantic or not has_other_tool:
        return call_list
    return [
        RouterToolCall(tool="semantic_memory.search", arguments={"query": user_query}),
        *call_list,
    ]


def _apply_tool_fallbacks(route: Any, user_query: str, current_datetime: datetime) -> Any:
    calls = list(getattr(route, "tool_calls", None) or [])
    if not calls and _looks_like_memory_question(user_query):
        calls.append(RouterToolCall(tool="semantic_memory.search", arguments={"query": user_query}))
    if _looks_like_calendar_question(user_query) and not any(call.tool == "calendar.get_schedule" for call in calls):
        start, end = _calendar_window_for_query(user_query, current_datetime)
        calls.append(
            RouterToolCall(
                tool="calendar.get_schedule",
                arguments={
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )
        )
    if not calls:
        return route
    return ReadToolDecision(tool_calls=calls)


def _looks_like_memory_question(query: str) -> bool:
    lowered = query.casefold()
    memory_markers = (
        "do i still",
        "did i",
        "have i",
        "what happened",
        "what did i",
        "who handles",
        "who leads",
        "remember",
        "planned",
        "plan",
        "my ",
        "that ",
    )
    return any(marker in lowered for marker in memory_markers)


def _looks_like_calendar_question(query: str) -> bool:
    lowered = query.casefold()
    return any(
        marker in lowered
        for marker in (
            "schedule",
            "calendar",
            "free",
            "available",
            "review",
            "meeting",
            "appointment",
            "when do i",
            "when is",
        )
    )


def _calendar_window_for_query(query: str, current_datetime: datetime) -> tuple[datetime, datetime]:
    lowered = query.casefold()
    history_markers = (
        "exception",
        "normal",
        "usual",
        "what happened",
        "changed",
        "moved",
        "stopped",
        "cancelled",
        "canceled",
        "was there",
    )
    if any(marker in lowered for marker in history_markers):
        return current_datetime - timedelta(days=180), current_datetime + timedelta(days=30)
    return current_datetime, current_datetime + timedelta(days=14)


def _looks_truncated_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.count("**") % 2 != 0 or stripped.count("*") % 2 != 0:
        return True
    last_line = stripped.splitlines()[-1].strip()
    if last_line in {"-", "*", "•"}:
        return True
    if last_line.startswith(("-", "*", "•")) and len(last_line) < 12:
        return True
    dangling_endings = ("**", "__", "`", ":", "—", "-", "•")
    return stripped.endswith(dangling_endings)
