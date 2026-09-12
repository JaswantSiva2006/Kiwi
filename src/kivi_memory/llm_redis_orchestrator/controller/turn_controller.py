"""Lifecycle controller for one user-facing conversation turn."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from kivi_memory.llm_redis_orchestrator.controller.agent import AgentResponse, HeyKiviAgent
from kivi_memory.llm_redis_orchestrator.controller.models import (
    AssistantMessageResult,
    MemoryPreparationSummary,
    TurnDiagnostics,
    TurnResult,
    UserMessageResult,
)
from kivi_memory.llm_redis_orchestrator.models import MemoryPreparationResult
from kivi_memory.llm_redis_orchestrator.memory_router import prepare_long_term_memory_context
from kivi_memory.working_memory import ThreadMemoryStore
from kivi_memory.working_memory.models import ThreadMessage

MemoryPreparer = Callable[..., MemoryPreparationResult]


class TurnControllerError(RuntimeError):
    """Base error for turn-controller failures."""


class UserAppendFailure(TurnControllerError):
    """Raised when the incoming user message cannot be recorded."""


class MemoryPreparationFailure(TurnControllerError):
    """Raised when memory preparation fails after the user message was recorded."""


class ResponseAgentFailure(TurnControllerError):
    """Raised when the final response agent fails; no assistant message is stored."""


class AssistantAppendFailure(TurnControllerError):
    """Raised when a generated assistant response cannot be recorded."""

    def __init__(self, message: str, *, agent_response: AgentResponse, assistant_message_id: str) -> None:
        super().__init__(message)
        self.agent_response = agent_response
        self.assistant_message_id = assistant_message_id


class MissingResponseAgentError(TurnControllerError):
    """Raised when no final response agent has been injected."""


class TurnController:
    """Owns Redis conversation writes around memory preparation and final response."""

    def __init__(
        self,
        *,
        thread_store: ThreadMemoryStore | None = None,
        memory_preparer: MemoryPreparer = prepare_long_term_memory_context,
        response_agent: HeyKiviAgent | None = None,
    ) -> None:
        self.thread_store = thread_store or ThreadMemoryStore()
        self.memory_preparer = memory_preparer
        if response_agent is None:
            from kivi_memory.llm_redis_orchestrator.final_agent import HeyKiviSarvamAgent

            response_agent = HeyKiviSarvamAgent()
        self.response_agent = response_agent

    def handle_user_turn(
        self,
        thread_id: str,
        text: str,
        message_id: str | None = None,
        timestamp: datetime | str | None = None,
        raw_asr: str | None = None,
        formatted_text: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        assistant_message_id: str | None = None,
    ) -> TurnResult:
        thread_id = _validate_non_empty(thread_id, "thread_id")
        text = _validate_non_empty(text, "text")
        turn_id = str(uuid4())
        user_message_id = str(message_id or uuid4())
        user_timestamp = _timestamp_iso(timestamp)
        started = time.perf_counter()

        user_message = ThreadMessage.from_input(
            {
                "message_id": user_message_id,
                "role": "user",
                "text": text,
                "raw_asr": raw_asr,
                "formatted_text": formatted_text,
                "timestamp": user_timestamp,
                "metadata": metadata or {},
            }
        )
        user_append_started = time.perf_counter()
        try:
            self.thread_store.append_message(thread_id, user_message)
        except Exception as exc:
            raise UserAppendFailure("failed to append user message to Redis working memory") from exc
        redis_user_append_ms = _elapsed_ms(user_append_started)

        prepare_started = time.perf_counter()
        try:
            preparation = self.memory_preparer(
                thread_id=thread_id,
                current_query=text,
                current_message_id=user_message_id,
            )
        except Exception as exc:
            raise MemoryPreparationFailure("failed to prepare long-term memory context") from exc
        memory_prepare_ms = _elapsed_ms(prepare_started)

        agent_started = time.perf_counter()
        try:
            prior_thread_messages = _exclude_current_message(self.thread_store.get_recent_messages(thread_id), user_message_id)
            agent_response = self.response_agent.respond(
                thread_id=thread_id,
                current_query=text,
                prior_thread_messages=prior_thread_messages,
                memory_context=preparation,
            )
        except Exception as exc:
            raise ResponseAgentFailure("final response agent failed") from exc
        final_agent_ms = _elapsed_ms(agent_started)

        assistant_text = _validate_non_empty(agent_response.text, "assistant response text")
        assistant_id = agent_response.message_id or assistant_message_id or str(uuid4())
        assistant_timestamp = _timestamp_iso(None)
        assistant_metadata = {
            "turn_id": turn_id,
            "used_long_term_memory": preparation.needs_long_term_memory,
            "retrieved_memory_ids": [memory.memory_id for memory in preparation.retrieved_memories],
            "supplied_memory_ids": agent_response.supplied_memory_ids,
            "supplied_evidence_message_ids": agent_response.supplied_evidence_message_ids,
            "agent_diagnostics": agent_response.diagnostics,
            **agent_response.metadata,
        }
        assistant_message = ThreadMessage.from_input(
            {
                "message_id": assistant_id,
                "role": "assistant",
                "text": assistant_text,
                "raw_asr": None,
                "formatted_text": None,
                "timestamp": assistant_timestamp,
                "metadata": assistant_metadata,
            }
        )
        assistant_append_started = time.perf_counter()
        try:
            self.thread_store.append_message(thread_id, assistant_message)
        except Exception as exc:
            raise AssistantAppendFailure(
                "failed to append assistant message to Redis working memory",
                agent_response=agent_response,
                assistant_message_id=assistant_id,
            ) from exc
        redis_assistant_append_ms = _elapsed_ms(assistant_append_started)

        retrieved_memory_ids = [memory.memory_id for memory in preparation.retrieved_memories]
        return TurnResult(
            turn_id=turn_id,
            thread_id=thread_id,
            user_message=UserMessageResult(user_message_id, text, user_timestamp),
            memory_preparation=MemoryPreparationSummary(
                needs_long_term_memory=preparation.needs_long_term_memory,
                retrieval_queries=preparation.retrieval_queries,
                retrieved_memory_ids=retrieved_memory_ids,
            ),
            assistant_message=AssistantMessageResult(assistant_id, assistant_text, assistant_timestamp),
            diagnostics=TurnDiagnostics(
                redis_user_append_ms=redis_user_append_ms,
                redis_context_ms=preparation.diagnostics.redis_context_ms,
                router_llm_ms=preparation.diagnostics.router_llm_ms,
                memory_prepare_ms=memory_prepare_ms,
                retrieval_ms=preparation.diagnostics.retrieval_ms,
                final_agent_ms=final_agent_ms,
                context_assembly_ms=float(agent_response.diagnostics.get("context_assembly_ms", 0.0)),
                final_llm_ms=float(agent_response.diagnostics.get("final_llm_ms", 0.0)),
                sarvam_api_ms=float(agent_response.diagnostics.get("sarvam_api_ms", 0.0)),
                redis_assistant_append_ms=redis_assistant_append_ms,
                total_turn_ms=_elapsed_ms(started),
                needs_long_term_memory=preparation.needs_long_term_memory,
                retrieval_query_count=len(preparation.retrieval_queries),
                retrieved_memory_count=len(retrieved_memory_ids),
            ),
            metadata={"assistant_metadata": assistant_metadata},
        )


def handle_user_turn(
    thread_id: str,
    text: str,
    message_id: str | None = None,
    timestamp: datetime | str | None = None,
    raw_asr: str | None = None,
    formatted_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    thread_store: ThreadMemoryStore | None = None,
    memory_preparer: MemoryPreparer = prepare_long_term_memory_context,
    response_agent: HeyKiviAgent | None = None,
    assistant_message_id: str | None = None,
) -> TurnResult:
    return TurnController(
        thread_store=thread_store,
        memory_preparer=memory_preparer,
        response_agent=response_agent,
    ).handle_user_turn(
        thread_id=thread_id,
        text=text,
        message_id=message_id,
        timestamp=timestamp,
        raw_asr=raw_asr,
        formatted_text=formatted_text,
        metadata=metadata,
        assistant_message_id=assistant_message_id,
    )


def _exclude_current_message(messages: list[ThreadMessage], current_message_id: str) -> list[ThreadMessage]:
    if messages and messages[-1].message_id == current_message_id:
        return messages[:-1]
    return messages


def _validate_non_empty(value: str, name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    return cleaned


def _timestamp_iso(value: datetime | str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("timestamp must not be empty")
        return value
    timestamp = value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.isoformat()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
