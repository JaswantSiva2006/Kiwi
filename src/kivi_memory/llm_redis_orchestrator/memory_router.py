"""Prepare long-term-memory context from current Redis thread state."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from kivi_memory.common.config import KiviOrchestratorConfig, load_orchestrator_config_from_env
from kivi_memory.llm_redis_orchestrator.models import (
    MemoryPreparationDiagnostics,
    MemoryPreparationResult,
    MergedRetrievedMemory,
)
from kivi_memory.llm_redis_orchestrator.router import LongTermMemoryRouter
from kivi_memory.long_term_retrieval import retrieve_long_term_memories
from kivi_memory.long_term_retrieval.models import RetrievalResult
from kivi_memory.working_memory import ThreadMemoryStore
from kivi_memory.working_memory.models import ThreadMessage

RetrievalFunction = Callable[[str], RetrievalResult]


def prepare_long_term_memory_context(
    thread_id: str,
    current_query: str,
    current_message_id: str | None = None,
    *,
    thread_memory_store: ThreadMemoryStore | None = None,
    router: LongTermMemoryRouter | None = None,
    retrieval_function: RetrievalFunction | None = None,
    config: KiviOrchestratorConfig | None = None,
) -> MemoryPreparationResult:
    """Read current-thread context, optionally retrieve long-term memories, and stop."""

    if not current_query.strip():
        raise ValueError("current_query must not be empty")
    config = config or load_orchestrator_config_from_env()
    store = thread_memory_store or ThreadMemoryStore()
    router = router or LongTermMemoryRouter(config=config)
    retrieval_function = retrieval_function or (lambda query: retrieve_long_term_memories(query_text=query))

    total_started = time.perf_counter()
    redis_started = time.perf_counter()
    recent_messages = store.get_recent_messages(thread_id)
    prior_messages = _exclude_current_message(recent_messages, current_message_id)
    redis_context_ms = _elapsed_ms(redis_started)

    router_result = router.route(prior_messages, current_query)
    decision = router_result.output
    if not decision.needs_long_term_memory:
        return MemoryPreparationResult(
            thread_id=thread_id,
            current_query=current_query,
            needs_long_term_memory=False,
            retrieval_queries=[],
            retrieved_memories=[],
            diagnostics=MemoryPreparationDiagnostics(
                redis_context_ms=redis_context_ms,
                router_llm_ms=router_result.router_llm_ms,
                retrieval_ms=0.0,
                total_ms=_elapsed_ms(total_started),
                router_model_call_count=router_result.router_model_call_count,
                retrieval_call_count=0,
                redis_context_message_count=len(prior_messages),
                router_fallback_used=router_result.fallback_used,
                router_error=router_result.error,
            ),
        )

    retrieval_started = time.perf_counter()
    retrieval_results = _run_retrievals(decision.retrieval_queries, retrieval_function)
    retrieval_ms = _elapsed_ms(retrieval_started)
    retrieved_memories = merge_retrieval_results(retrieval_results, config.memory_limit)

    return MemoryPreparationResult(
        thread_id=thread_id,
        current_query=current_query,
        needs_long_term_memory=True,
        retrieval_queries=decision.retrieval_queries,
        retrieved_memories=retrieved_memories,
        diagnostics=MemoryPreparationDiagnostics(
            redis_context_ms=redis_context_ms,
            router_llm_ms=router_result.router_llm_ms,
            retrieval_ms=retrieval_ms,
            total_ms=_elapsed_ms(total_started),
            router_model_call_count=router_result.router_model_call_count,
            retrieval_call_count=len(decision.retrieval_queries),
            redis_context_message_count=len(prior_messages),
            router_fallback_used=router_result.fallback_used,
            router_error=router_result.error,
        ),
    )


def merge_retrieval_results(
    retrieval_results: list[RetrievalResult],
    memory_limit: int,
) -> list[MergedRetrievedMemory]:
    if memory_limit < 1:
        raise ValueError("memory_limit must be positive")

    merged: dict[str, dict] = {}
    for result in retrieval_results:
        for rank, memory in enumerate(result.memories, start=1):
            state = merged.setdefault(
                memory.memory_id,
                {
                    "memory": memory,
                    "matched_queries": [],
                    "best_final_rank": rank,
                    "best_rrf_score": memory.rrf_score,
                },
            )
            if result.query not in state["matched_queries"]:
                state["matched_queries"].append(result.query)
            if rank < state["best_final_rank"]:
                state["best_final_rank"] = rank
            if memory.rrf_score > state["best_rrf_score"]:
                state["best_rrf_score"] = memory.rrf_score

    ranked = sorted(
        merged.values(),
        key=lambda state: (
            -len(state["matched_queries"]),
            state["best_final_rank"],
            -state["best_rrf_score"],
            state["memory"].memory_id,
        ),
    )[:memory_limit]
    return [_to_merged_memory(state) for state in ranked]


def _run_retrievals(queries: list[str], retrieval_function: RetrievalFunction) -> list[RetrievalResult]:
    if len(queries) == 1:
        return [retrieval_function(queries[0])]
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        return list(executor.map(retrieval_function, queries))


def _exclude_current_message(messages: list[ThreadMessage], current_message_id: str | None) -> list[ThreadMessage]:
    if not current_message_id or not messages:
        return messages
    if messages[-1].message_id == current_message_id:
        return messages[:-1]
    return messages


def _to_merged_memory(state: dict) -> MergedRetrievedMemory:
    memory = state["memory"]
    return MergedRetrievedMemory(
        memory_id=memory.memory_id,
        canonical_text=memory.canonical_text,
        subject=memory.subject,
        arguments=memory.arguments,
        temporal=memory.temporal,
        evidence=memory.evidence,
        matched_retrieval_queries=state["matched_queries"],
        predicate_type=memory.predicate_type,
        retrieval_metadata={
            "best_final_rank": state["best_final_rank"],
            "best_rrf_score": state["best_rrf_score"],
            "retrieval_sources": memory.retrieval_sources,
            "branch_ranks": memory.branch_ranks,
            "branch_scores": memory.branch_scores,
        },
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
