from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from kivi_memory.common.config import KiviOrchestratorConfig
from kivi_memory.llm_redis_orchestrator.memory_router import (
    merge_retrieval_results,
    prepare_long_term_memory_context,
)
from kivi_memory.llm_redis_orchestrator.prompt import ROUTER_SYSTEM_PROMPT
from kivi_memory.llm_redis_orchestrator.router import (
    ROUTER_THINK,
    LongTermMemoryRouter,
    RouterOutput,
    format_router_input,
)
from kivi_memory.long_term_retrieval.models import HydratedRetrievedMemory, RetrievalDiagnostics, RetrievalResult
from kivi_memory.working_memory.models import ThreadMessage


class FakeOllamaClient:
    def __init__(self, responses: list[dict | Exception]) -> None:
        self.responses = responses
        self.calls = []

    def _post(self, path: str, payload: dict):
        self.calls.append({"path": path, "payload": payload})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {
            "message": {"content": json.dumps(response)},
            "prompt_eval_count": 11,
            "eval_count": 7,
        }


@dataclass
class FakeThreadMemoryStore:
    messages: list[ThreadMessage]

    def __post_init__(self) -> None:
        self.calls = []
        self.mutation_calls = []

    def get_recent_messages(self, thread_id: str):
        self.calls.append(("get_recent_messages", thread_id))
        return self.messages

    def append_message(self, *args, **kwargs):
        self.mutation_calls.append(("append_message", args, kwargs))

    def clear_thread(self, *args, **kwargs):
        self.mutation_calls.append(("clear_thread", args, kwargs))


def test_router_uses_qwen_4b_think_false_temperature_zero() -> None:
    client = FakeOllamaClient([{"needs_long_term_memory": False, "retrieval_queries": []}])
    router = LongTermMemoryRouter(client=client, config=config())

    result = router.route([], "Explain binary search.")

    payload = client.calls[0]["payload"]
    assert payload["model"] == "qwen3.5:4b"
    assert payload["think"] is ROUTER_THINK is False
    assert payload["options"]["temperature"] == 0
    assert payload["messages"][0]["content"] == ROUTER_SYSTEM_PROMPT
    assert result.router_model_call_count == 1


def test_router_prompt_calls_out_user_preference_queries_as_memory_dependent() -> None:
    assert "Questions about the user's own preferences" in ROUTER_SYSTEM_PROMPT


def test_format_router_input_contains_only_roles_text_and_current_query() -> None:
    text = format_router_input(
        [message("m1", "user", "What is Priya working on?"), message("m2", "assistant", "Project Phoenix.")],
        "What technology does it use?",
    )

    assert "user: What is Priya working on?" in text
    assert "assistant: Project Phoenix." in text
    assert "CURRENT USER QUERY:\nWhat technology does it use?" in text
    assert "m1" not in text
    assert "metadata" not in text
    assert "Redis" not in text


def test_router_output_contract() -> None:
    assert RouterOutput.model_validate({"needs_long_term_memory": False, "retrieval_queries": []})
    assert RouterOutput.model_validate({"needs_long_term_memory": True, "retrieval_queries": ["Who handles Atlas?"]})
    with pytest.raises(ValueError):
        RouterOutput.model_validate({"needs_long_term_memory": False, "retrieval_queries": ["Atlas"]})
    with pytest.raises(ValueError):
        RouterOutput.model_validate({"needs_long_term_memory": True, "retrieval_queries": []})
    with pytest.raises(ValueError):
        RouterOutput.model_validate({"needs_long_term_memory": True, "retrieval_queries": ["   "]})


def test_false_decision_skips_retrieval_and_does_not_mutate_redis() -> None:
    store = FakeThreadMemoryStore([message("m1", "user", "What is Priya working on?"), message("m2", "assistant", "Project Phoenix.")])
    router = LongTermMemoryRouter(
        client=FakeOllamaClient([{"needs_long_term_memory": False, "retrieval_queries": []}]),
        config=config(),
    )
    retrieval_calls = []

    result = prepare_long_term_memory_context(
        "thread-a",
        "Which project did you say she works on?",
        thread_memory_store=store,
        router=router,
        retrieval_function=lambda query: retrieval_calls.append(query),
        config=config(),
    )

    assert result.needs_long_term_memory is False
    assert result.retrieval_queries == []
    assert result.retrieved_memories == []
    assert result.diagnostics.retrieval_call_count == 0
    assert retrieval_calls == []
    assert store.mutation_calls == []


def test_obvious_user_preference_question_overrides_false_router_decision() -> None:
    store = FakeThreadMemoryStore([])
    router = LongTermMemoryRouter(
        client=FakeOllamaClient([{"needs_long_term_memory": False, "retrieval_queries": []}]),
        config=config(),
    )
    retrieval_calls = []

    result = prepare_long_term_memory_context(
        "thread-a",
        "For research discussions, how do I like explanations?",
        thread_memory_store=store,
        router=router,
        retrieval_function=lambda query: retrieval_calls.append(query) or retrieval_result(query, []),
        config=config(),
    )

    assert result.needs_long_term_memory is True
    assert retrieval_calls == ["For research discussions, how do I like explanations?"]


def test_true_decision_calls_retrieval_with_generated_query() -> None:
    store = FakeThreadMemoryStore([message("m1", "user", "What is Priya working on?"), message("m2", "assistant", "Priya is working on Project Phoenix.")])
    router = LongTermMemoryRouter(
        client=FakeOllamaClient(
            [{"needs_long_term_memory": True, "retrieval_queries": ["What technology does Project Phoenix use?"]}]
        ),
        config=config(),
    )
    retrieval_calls = []

    result = prepare_long_term_memory_context(
        "thread-a",
        "What technology does it use?",
        thread_memory_store=store,
        router=router,
        retrieval_function=lambda query: retrieval_calls.append(query) or retrieval_result(query, [memory("m1", "Project Phoenix uses Redis.")]),
        config=config(),
    )

    assert result.needs_long_term_memory is True
    assert result.retrieval_queries == ["What technology does Project Phoenix use?"]
    assert retrieval_calls == ["What technology does Project Phoenix use?"]
    assert result.retrieved_memories[0].canonical_text == "Project Phoenix uses Redis."


def test_current_message_is_excluded_when_newest() -> None:
    store = FakeThreadMemoryStore(
        [
            message("m1", "user", "What is Priya working on?"),
            message("m2", "assistant", "Project Phoenix."),
            message("current", "user", "What technology does it use?"),
        ]
    )
    client = FakeOllamaClient([{"needs_long_term_memory": False, "retrieval_queries": []}])

    prepare_long_term_memory_context(
        "thread-a",
        "What technology does it use?",
        current_message_id="current",
        thread_memory_store=store,
        router=LongTermMemoryRouter(client=client, config=config()),
        retrieval_function=lambda query: retrieval_result(query, []),
        config=config(),
    )

    user_content = client.calls[0]["payload"]["messages"][1]["content"]
    assert user_content.count("What technology does it use?") == 1
    assert "assistant: Project Phoenix." in user_content


def test_malformed_router_output_retries_once_then_falls_back_true() -> None:
    store = FakeThreadMemoryStore([])
    client = FakeOllamaClient([
        {"needs_long_term_memory": False, "retrieval_queries": ["bad"]},
        {"needs_long_term_memory": False, "retrieval_queries": ["bad again"]},
    ])
    retrieval_calls = []

    result = prepare_long_term_memory_context(
        "thread-a",
        "Who handles Project Atlas?",
        thread_memory_store=store,
        router=LongTermMemoryRouter(client=client, config=config()),
        retrieval_function=lambda query: retrieval_calls.append(query) or retrieval_result(query, []),
        config=config(),
    )

    assert result.needs_long_term_memory is True
    assert result.retrieval_queries == ["Who handles Project Atlas?"]
    assert retrieval_calls == ["Who handles Project Atlas?"]
    assert result.diagnostics.router_model_call_count == 2
    assert result.diagnostics.router_fallback_used is True


def test_max_retrieval_queries_is_three() -> None:
    client = FakeOllamaClient(
        [
            {"needs_long_term_memory": True, "retrieval_queries": ["q1", "q2", "q3", "q4"]},
            {"needs_long_term_memory": True, "retrieval_queries": ["fallback q"]},
        ]
    )
    router = LongTermMemoryRouter(client=client, config=config(max_queries=3))

    result = router.route([], "Need memory.")

    assert result.output.retrieval_queries == ["fallback q"]
    assert result.router_model_call_count == 2


def test_multiple_retrieval_results_are_merged_by_memory_id() -> None:
    merged = merge_retrieval_results(
        [
            retrieval_result("query one", [memory("m1", "First", rrf=0.5), memory("m2", "Second", rrf=0.4)]),
            retrieval_result("query two", [memory("m2", "Second", rrf=0.8), memory("m3", "Third", rrf=0.7)]),
        ],
        memory_limit=16,
    )

    assert [item.memory_id for item in merged] == ["m2", "m1", "m3"]
    assert merged[0].matched_retrieval_queries == ["query one", "query two"]
    assert merged[0].retrieval_metadata["best_final_rank"] == 1
    assert merged[0].retrieval_metadata["best_rrf_score"] == 0.8


def test_memory_limit_is_respected() -> None:
    merged = merge_retrieval_results(
        [retrieval_result("query", [memory(f"m{index}", f"Text {index}") for index in range(5)])],
        memory_limit=2,
    )

    assert len(merged) == 2


def test_thread_isolation_comes_from_store_call() -> None:
    store_a = FakeThreadMemoryStore([message("a1", "user", "Priya")])
    router = LongTermMemoryRouter(client=FakeOllamaClient([{"needs_long_term_memory": False, "retrieval_queries": []}]), config=config())

    prepare_long_term_memory_context(
        "thread-a",
        "repeat that",
        thread_memory_store=store_a,
        router=router,
        retrieval_function=lambda query: retrieval_result(query, []),
        config=config(),
    )

    assert store_a.calls == [("get_recent_messages", "thread-a")]


def test_no_final_answer_model_or_write_pipeline_is_called(monkeypatch) -> None:
    def fail_import(name, *args, **kwargs):
        forbidden = (
            "kivi_memory.pipeline.memory_pipeline",
            "kivi_memory.pipeline.write_pipeline",
            "kivi_memory.semantic_compiler",
            "kivi_memory.ledger",
        )
        if name.startswith(forbidden):
            raise AssertionError(f"unexpected import: {name}")
        return real_import(name, *args, **kwargs)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    monkeypatch.setattr("builtins.__import__", fail_import)
    store = FakeThreadMemoryStore([])
    router = LongTermMemoryRouter(client=FakeOllamaClient([{"needs_long_term_memory": False, "retrieval_queries": []}]), config=config())

    result = prepare_long_term_memory_context(
        "thread-a",
        "Explain binary search.",
        thread_memory_store=store,
        router=router,
        retrieval_function=lambda query: retrieval_result(query, []),
        config=config(),
    )

    assert result.needs_long_term_memory is False


def config(max_queries: int = 3, memory_limit: int = 16) -> KiviOrchestratorConfig:
    return KiviOrchestratorConfig(
        ollama_base_url="http://testserver",
        timeout_seconds=1,
        router_model="qwen3.5:4b",
        router_temperature=0,
        router_max_retrieval_queries=max_queries,
        memory_limit=memory_limit,
    )


def message(message_id: str, role: str, text: str) -> ThreadMessage:
    return ThreadMessage.from_input(
        {
            "message_id": message_id,
            "role": role,
            "text": text,
            "timestamp": "2026-09-06T10:00:00+00:00",
        }
    )


def memory(memory_id: str, text: str, rrf: float = 0.5) -> HydratedRetrievedMemory:
    return HydratedRetrievedMemory(
        memory_id=memory_id,
        canonical_text=text,
        status="ACTIVE",
        version=1,
        rrf_score=rrf,
        retrieval_sources=["VECTOR"],
        branch_ranks={"vector": 1, "lexical": None, "structured": None, "graph": None},
        branch_scores={"vector": 0.9, "lexical": None, "structured": None, "graph": None},
        subject={"text": "Project Phoenix", "entity_id": "entity-1", "entity_type": "PROJECT"},
        predicate_type="USES",
        memory_type="PROJECT_FACT",
        modality="FACT",
        polarity="POSITIVE",
        certainty="CERTAIN",
        explicitness="EXPLICIT",
        attributed_to="user",
        temporal={},
        arguments=[],
        evidence=[],
    )


def retrieval_result(query: str, memories: list[HydratedRetrievedMemory]) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        resolved_query_entities=[],
        memories=memories,
        diagnostics=RetrievalDiagnostics(),
    )
