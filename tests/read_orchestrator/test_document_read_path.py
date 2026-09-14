from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
import types
from typing import Any

calendar_module = types.ModuleType("kivi_memory.calendar")
calendar_module.CalendarTool = lambda *args, **kwargs: FakeCalendarTool()
sys.modules.setdefault("kivi_memory.calendar", calendar_module)

long_term_module = types.ModuleType("kivi_memory.long_term_retrieval")
long_term_module.RetrievalResult = object
long_term_module.retrieve_long_term_memories = lambda *args, **kwargs: None
sys.modules.setdefault("kivi_memory.long_term_retrieval", long_term_module)

final_agent_module = types.ModuleType("kivi_memory.llm_redis_orchestrator.final_agent")
final_agent_module.SarvamFinalAnswerError = RuntimeError
final_agent_module._extract_final_answer_text = lambda response: "answer"
final_agent_module.get_shared_sarvam_client = lambda config: None
sys.modules.setdefault("kivi_memory.llm_redis_orchestrator.final_agent", final_agent_module)

memory_control_module = types.ModuleType("kivi_memory.read_orchestrator.memory_control")
memory_control_module.MemoryControlResult = object
memory_control_module.MemoryControlTool = lambda *args, **kwargs: FakeMemoryControlTool()
sys.modules.setdefault("kivi_memory.read_orchestrator.memory_control", memory_control_module)

web_search_module = types.ModuleType("kivi_memory.read_orchestrator.web_search")
web_search_module.TavilyWebSearchProvider = object
web_search_module.WebSearchProvider = object
web_search_module.WebSearchResult = object
web_search_module.WebSearchTool = lambda *args, **kwargs: FakeWebSearchTool()
web_search_module.search_web = lambda *args, **kwargs: None
sys.modules.setdefault("kivi_memory.read_orchestrator.web_search", web_search_module)

writeback_worker_module = types.ModuleType("kivi_memory.writeback.worker")
writeback_worker_module.MemoryWritebackWorker = lambda *args, **kwargs: None
writeback_worker_module.WritebackStatus = type("WritebackStatus", (), {"SUCCESS": "SUCCESS"})
sys.modules.setdefault("kivi_memory.writeback.worker", writeback_worker_module)

from kivi_memory.read_orchestrator.models import ReadOrchestrationResult, ReadToolDecision, RouterToolCall
from kivi_memory.read_response import ReadResponseService


def test_document_search_reuses_probe_embedding() -> None:
    document_tool = FakeDocumentTool()
    service = _service(
        router=FakeRouter([RouterToolCall(tool="document.search", arguments={"query": "What does this indexed paper say about boundary conditions?", "top_k": 6, "document_ids": []})]),
        document_tool=document_tool,
    )

    response = asyncio.run(
        service.handle_user_query(
            thread_id="t1",
            user_query="What does this indexed paper say about boundary conditions?",
            current_datetime=datetime(2026, 9, 14, tzinfo=timezone.utc),
            timezone="UTC",
        )
    )

    assert "DOCUMENT_CONTEXT" in response.context
    assert document_tool.embed_count == 1
    assert document_tool.reused_embedding is True


def test_unrelated_query_does_not_call_document_tool() -> None:
    document_tool = FakeDocumentTool()
    service = _service(router=FakeRouter([]), document_tool=document_tool)

    response = asyncio.run(
        service.handle_user_query(
            thread_id="t1",
            user_query="Who handles Atlas?",
            current_datetime=datetime(2026, 9, 14, tzinfo=timezone.utc),
            timezone="UTC",
        )
    )

    assert "document.search" not in [call["tool"] for call in response.route["tool_calls"]]
    assert document_tool.search_count == 0


def test_one_shot_uploaded_document_forces_document_search() -> None:
    document_tool = FakeDocumentTool()
    service = _service(router=FakeRouter([]), document_tool=document_tool)

    response = asyncio.run(
        service.handle_user_query(
            thread_id="t1",
            user_query="give me details more",
            current_datetime=datetime(2026, 9, 14, tzinfo=timezone.utc),
            timezone="UTC",
            force_document_search=True,
            document_ids=["doc1"],
        )
    )

    tools = [call["tool"] for call in response.route["tool_calls"]]
    assert tools == ["document.search"]
    assert document_tool.search_count == 1


def test_combined_semantic_and_document_query() -> None:
    document_tool = FakeDocumentTool()
    service = _service(
        router=FakeRouter(
            [
                RouterToolCall(tool="semantic_memory.search", arguments={"query": "Atlas owner"}),
                RouterToolCall(tool="document.search", arguments={"query": "Atlas boundary conditions", "top_k": 6, "document_ids": []}),
            ]
        ),
        document_tool=document_tool,
    )

    response = asyncio.run(
        service.handle_user_query(
            thread_id="t1",
            user_query="Using what I know about Atlas and the indexed paper, summarize the boundary conditions.",
            current_datetime=datetime(2026, 9, 14, tzinfo=timezone.utc),
            timezone="UTC",
        )
    )

    tools = [call["tool"] for call in response.route["tool_calls"]]
    assert tools == ["semantic_memory.search", "document.search"]
    assert "<SEMANTIC_MEMORY>" in response.context
    assert "<DOCUMENT_CONTEXT>" in response.context


def _service(*, router: "FakeRouter", document_tool: "FakeDocumentTool") -> ReadResponseService:
    return ReadResponseService(
        episode_store=FakeEpisodeStore(),
        router=router,
        semantic_retriever=FakeSemanticRetriever(),
        calendar_tool=FakeCalendarTool(),
        redis_history_tool=FakeRedisHistoryTool(),
        web_search_tool=FakeWebSearchTool(),
        memory_control_tool=FakeMemoryControlTool(),
        document_search_tool=document_tool,
        final_answer_client=FakeFinalAnswerClient(),
        episode_builder=FakeEpisodeBuilder(),
    )


class FakeRouter:
    def __init__(self, calls: list[RouterToolCall]) -> None:
        self.calls = calls
        self.probes: list[dict[str, Any] | None] = []

    def route(self, current_query: str, recent_thread_context: list[Any], **kwargs: Any) -> ReadOrchestrationResult:
        self.probes.append(kwargs.get("document_probe"))
        return ReadOrchestrationResult(
            decision=ReadToolDecision(tool_calls=self.calls),
            router_llm_ms=1.0,
            router_model_call_count=1,
        )


class FakeDocumentTool:
    def __init__(self) -> None:
        self.embed_count = 0
        self.search_count = 0
        self.probe_embedding = [1.0, 0.0, 0.0]
        self.reused_embedding = False

    def probe(self, query: str):
        self.embed_count += 1
        return FakeProbe(
            query=query,
            query_embedding=self.probe_embedding,
            metadata={"documents_available": True, "likely_relevant": True, "matches": [{"filename": "paper.pdf", "section": "3 Method", "score": 0.82}]},
            latency_ms=2.0,
        )

    def search(self, *, query: str, top_k: int, document_ids: list[str], query_embedding: list[float] | None = None):
        self.search_count += 1
        self.reused_embedding = query_embedding is self.probe_embedding
        if query_embedding is None:
            self.embed_count += 1
        return {
            "status": "OK",
            "results": [
                {
                    "source_id": "D1",
                    "document_id": "doc1",
                    "filename": "paper.pdf",
                    "chunk_id": "doc1:chunk:00000",
                    "section_title": "3 Method",
                    "section_path": ["3 Method"],
                    "page_start": 4,
                    "page_end": 4,
                    "text": "Boundary conditions are fixed at the wall.",
                    "similarity_score": 0.86,
                }
            ],
        }


@dataclass
class FakeProbe:
    query: str
    query_embedding: list[float] | None
    metadata: dict[str, Any]
    latency_ms: float


class FakeSemanticRetriever:
    def search(self, *, query: str) -> list[dict[str, Any]]:
        return [{"memory_id": "m1", "canonical_text": "Atlas is handled by Mira."}]


class FakeFinalAnswerClient:
    def answer(self, *, system_prompt: str, context: str) -> str:
        return "answer"


class FakeEpisodeStore:
    config = type("Config", (), {"writeback_on_turn_enabled": False})()

    def load_recent_thread_context(self, thread_id: str) -> list[Any]:
        return []

    def append_thread_episode(self, thread_id: str, episode: Any) -> Any:
        return type("Stored", (), {"episode_id": "e1"})()


class FakeEpisodeBuilder:
    def build(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs


class FakeCalendarTool:
    def get_schedule(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"events": []}


class FakeRedisHistoryTool:
    def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"episodes": []}


class FakeWebSearchTool:
    def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"sources": []}


class FakeMemoryControlTool:
    def execute(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "NOOP"}
