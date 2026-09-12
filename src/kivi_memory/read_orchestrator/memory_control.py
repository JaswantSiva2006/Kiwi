"""Explicit user controls for inspecting and changing Kivi semantic memory."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from kivi_memory.calendar import cancel_calendar_event_for_memory
from kivi_memory.common.config import DEFAULT_LOCALE
from kivi_memory.common.schemas import MemoryEpisode
from kivi_memory.ledger.repository import LedgerRepository
from kivi_memory.long_term_retrieval import HydratedRetrievedMemory, retrieve_long_term_memories
from kivi_memory.pipeline import MemoryPipeline


class MemoryControlAction(StrEnum):
    INSPECT = "INSPECT"
    EXPLAIN = "EXPLAIN"
    FORGET = "FORGET"
    CORRECT = "CORRECT"


class MemoryControlStatus(StrEnum):
    APPLIED = "APPLIED"
    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    ERROR = "ERROR"


class MemoryControlInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: MemoryControlAction
    target_query: str = ""
    corrected_text: str | None = None
    broad_destructive: bool = False


@dataclass(frozen=True)
class MemoryControlResult:
    action: str
    status: str
    affected_memory_ids: list[str] = field(default_factory=list)
    memories: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        del mode
        return {
            "action": self.action,
            "status": self.status,
            "affected_memory_ids": self.affected_memory_ids,
            "memories": self.memories,
            "provenance": self.provenance,
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


class MemoryControlTool:
    """Execute explicit user requests about Kivi's stored semantic memory."""

    def __init__(
        self,
        *,
        retriever=retrieve_long_term_memories,
        repository: LedgerRepository | None = None,
        memory_pipeline: MemoryPipeline | None = None,
    ) -> None:
        self.retriever = retriever
        self.repository = repository or LedgerRepository()
        self.memory_pipeline = memory_pipeline or MemoryPipeline()

    def execute(
        self,
        *,
        query: str,
        thread_id: str,
        current_datetime: datetime,
        timezone: str,
        locale: str = DEFAULT_LOCALE,
    ) -> MemoryControlResult:
        started = time.perf_counter()
        try:
            interpretation = interpret_memory_control_request(query)
            if interpretation.action in {MemoryControlAction.INSPECT, MemoryControlAction.EXPLAIN}:
                result = self._inspect_or_explain(interpretation)
            elif interpretation.action == MemoryControlAction.FORGET:
                result = self._forget(interpretation, query=query)
            else:
                result = self._correct(
                    interpretation,
                    query=query,
                    thread_id=thread_id,
                    current_datetime=current_datetime,
                    timezone=timezone,
                    locale=locale,
                )
            result.diagnostics["latency_ms"] = (time.perf_counter() - started) * 1000
            return result
        except Exception as exc:
            return MemoryControlResult(
                action="UNKNOWN",
                status=MemoryControlStatus.ERROR.value,
                message=f"Memory control failed: {' '.join(str(exc).split())}",
                diagnostics={"latency_ms": (time.perf_counter() - started) * 1000, "error": str(exc)},
            )

    def _inspect_or_explain(self, interpretation: MemoryControlInterpretation) -> MemoryControlResult:
        memories = _retrieve_relevant(self.retriever, interpretation.target_query, top_k=8)
        status = MemoryControlStatus.OK if memories else MemoryControlStatus.NOT_FOUND
        return MemoryControlResult(
            action=interpretation.action.value,
            status=status.value,
            memories=[_memory_payload(memory) for memory in memories],
            provenance=_provenance_payload(memories),
            message="Relevant stored memories found." if memories else "No matching stored memory was found.",
        )

    def _forget(self, interpretation: MemoryControlInterpretation, *, query: str) -> MemoryControlResult:
        memories = _retrieve_relevant(self.retriever, interpretation.target_query or query, top_k=6)
        if interpretation.broad_destructive:
            return MemoryControlResult(
                action=MemoryControlAction.FORGET.value,
                status=MemoryControlStatus.NEEDS_CONFIRMATION.value,
                memories=[_memory_payload(memory) for memory in memories],
                provenance=_provenance_payload(memories),
                message="This could affect multiple memories. Ask the user to confirm exactly what should be forgotten.",
            )
        target = _clear_single_target(memories, interpretation.target_query)
        if target is None:
            status = MemoryControlStatus.NOT_FOUND if not memories else MemoryControlStatus.NEEDS_CLARIFICATION
            return MemoryControlResult(
                action=MemoryControlAction.FORGET.value,
                status=status.value,
                memories=[_memory_payload(memory) for memory in memories],
                provenance=_provenance_payload(memories),
                message=(
                    "No matching stored memory was found."
                    if not memories
                    else "Multiple memories could match. Ask the user which one to forget."
                ),
            )
        self._retract_memory(target.memory_id, reason=query)
        return MemoryControlResult(
            action=MemoryControlAction.FORGET.value,
            status=MemoryControlStatus.APPLIED.value,
            affected_memory_ids=[target.memory_id],
            memories=[_memory_payload(target)],
            provenance=_provenance_payload([target]),
            message="The matching memory was retracted and is no longer active.",
        )

    def _correct(
        self,
        interpretation: MemoryControlInterpretation,
        *,
        query: str,
        thread_id: str,
        current_datetime: datetime,
        timezone: str,
        locale: str,
    ) -> MemoryControlResult:
        target_memories = _retrieve_relevant(self.retriever, interpretation.target_query or query, top_k=4)
        correction = interpretation.corrected_text or query
        episode = MemoryEpisode.model_validate(
            {
                "episode_id": f"memory-control-correction-{uuid4()}",
                "session_id": thread_id,
                "timezone": timezone,
                "locale": locale,
                "messages": [
                    {
                        "message_id": f"memory-control-user-{uuid4()}",
                        "role": "USER",
                        "timestamp": current_datetime.isoformat(),
                        "text": correction,
                        "memory_eligible": True,
                        "context_only": False,
                    }
                ],
                "tool_context": [
                    {
                        "tool_name": "memory_control",
                        "content": (
                            "The user is explicitly correcting Kivi's stored memory. "
                            "Treat this as authoritative user evidence and use the existing reconciliation path."
                        ),
                    }
                ],
            }
        )
        pipeline_result = self.memory_pipeline.process(episode)
        affected = _mutation_memory_ids(pipeline_result.output)
        return MemoryControlResult(
            action=MemoryControlAction.CORRECT.value,
            status=MemoryControlStatus.APPLIED.value if affected else MemoryControlStatus.OK.value,
            affected_memory_ids=affected,
            memories=[_memory_payload(memory) for memory in target_memories],
            provenance=_provenance_payload(target_memories),
            message="The correction was sent through the semantic memory pipeline.",
            diagnostics={"pipeline_result": pipeline_result.output},
        )

    def _retract_memory(self, memory_id: str, *, reason: str) -> None:
        with self.repository.connect() as conn:
            with conn.cursor() as cur:
                self.repository.lock_active_memory(cur, memory_id)
                self.repository.update_memory_status_in_transaction(cur, memory_id=memory_id, status="RETRACTED")
                cancelled = cancel_calendar_event_for_memory(cur, memory_id=memory_id)
                self.repository.insert_event_in_transaction(
                    cur,
                    memory_id=memory_id,
                    event_type="MEMORY_RETRACTED",
                    payload={
                        "source": "memory.control",
                        "reason": reason,
                        "calendar_cancelled": cancelled,
                    },
                )


def interpret_memory_control_request(query: str) -> MemoryControlInterpretation:
    text = " ".join(query.strip().split())
    lowered = text.casefold()
    if not text:
        raise ValueError("memory control query must not be empty")

    broad = any(phrase in lowered for phrase in ["forget everything", "delete everything", "remove everything", "forget all"])
    if any(word in lowered for word in ["forget", "delete", "remove", "stop remembering"]):
        return MemoryControlInterpretation(
            action=MemoryControlAction.FORGET,
            target_query=_strip_prefix(text, ["forget that", "forget", "delete", "remove", "stop remembering"]),
            broad_destructive=broad,
        )
    if any(phrase in lowered for phrase in ["why did you think", "why do you think", "where did you get", "why did you remember"]):
        return MemoryControlInterpretation(
            action=MemoryControlAction.EXPLAIN,
            target_query=_strip_prefix(text, ["why did you think", "why do you think", "where did you get", "why did you remember"]),
        )
    if _looks_like_correction(lowered):
        return MemoryControlInterpretation(
            action=MemoryControlAction.CORRECT,
            target_query=text,
            corrected_text=text,
        )
    return MemoryControlInterpretation(
        action=MemoryControlAction.INSPECT,
        target_query=_strip_prefix(text, ["what do you remember about", "what do you know about", "show me what you remember about"]),
    )


def _retrieve_relevant(retriever, query: str, *, top_k: int) -> list[HydratedRetrievedMemory]:
    if not query.strip():
        return []
    return list(retriever(query_text=query, top_k=top_k).memories)


def _clear_single_target(memories: list[HydratedRetrievedMemory], target_query: str) -> HydratedRetrievedMemory | None:
    if len(memories) == 1:
        return memories[0]
    if not memories:
        return None
    normalized_target = target_query.casefold()
    exactish = [memory for memory in memories if memory.canonical_text.casefold() in normalized_target or normalized_target in memory.canonical_text.casefold()]
    return exactish[0] if len(exactish) == 1 else None


def _memory_payload(memory: HydratedRetrievedMemory) -> dict[str, Any]:
    return {
        "memory_id": memory.memory_id,
        "canonical_text": memory.canonical_text,
        "subject": memory.subject,
        "predicate_type": memory.predicate_type,
        "memory_type": memory.memory_type,
        "temporal": memory.temporal,
        "arguments": memory.arguments,
    }


def _provenance_payload(memories: list[HydratedRetrievedMemory]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": memory.memory_id,
            "evidence": memory.evidence[:3],
        }
        for memory in memories
    ]


def _mutation_memory_ids(output: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in output.get("assertions", []):
        mutation = item.get("mutation_result") or {}
        for key in ["created_memory_id"]:
            if mutation.get(key):
                ids.append(str(mutation[key]))
        ids.extend(str(memory_id) for memory_id in mutation.get("target_memory_ids") or [])
    seen = set()
    return [memory_id for memory_id in ids if not (memory_id in seen or seen.add(memory_id))]


def _strip_prefix(text: str, prefixes: list[str]) -> str:
    lowered = text.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip(" .:;")
    return text.strip(" .:;")


def _looks_like_correction(lowered: str) -> bool:
    return (
        lowered.startswith(("no,", "no ", "actually", "correction"))
        or "doesn't" in lowered
        or "does not" in lowered
        or "anymore" in lowered
        or "instead" in lowered
    )
