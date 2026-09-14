from __future__ import annotations

import sys
import types

memory_control_module = types.ModuleType("kivi_memory.read_orchestrator.memory_control")
memory_control_module.MemoryControlResult = object
memory_control_module.MemoryControlTool = object
sys.modules.setdefault("kivi_memory.read_orchestrator.memory_control", memory_control_module)

web_search_module = types.ModuleType("kivi_memory.read_orchestrator.web_search")
web_search_module.TavilyWebSearchProvider = object
web_search_module.WebSearchProvider = object
web_search_module.WebSearchResult = object
web_search_module.WebSearchTool = object
web_search_module.search_web = lambda *args, **kwargs: None
sys.modules.setdefault("kivi_memory.read_orchestrator.web_search", web_search_module)

from kivi_memory.document_rag.models import DocumentChunkSearchResult
from kivi_memory.document_rag.search_tool import MAX_DOCUMENT_CONTEXT_TOKENS, select_document_evidence
from kivi_memory.read_orchestrator.context_builder import build_context


def test_mmr_dedupes_and_respects_token_cap() -> None:
    candidates = [
        _result("c1", "d1", "intro", "boundary conditions fixed wall inlet outlet", 0.91),
        _result("c2", "d1", "intro", "boundary conditions fixed wall inlet outlet", 0.90),
        _result("c3", "d1", "method", "geometry encoder mesh constraints", 0.80),
        _result("c4", "d2", "results", "boundary evaluation ablation metrics", 0.79),
        _result("c5", "d2", "long", "word " * (MAX_DOCUMENT_CONTEXT_TOKENS + 10), 0.99),
    ]

    selected = select_document_evidence(candidates, top_k=4, max_context_tokens=20)

    assert [item.chunk_id for item in selected] == ["c1", "c3", "c4"]


def test_document_context_format() -> None:
    context = build_context(
        query="What does the paper say?",
        thread_context=[],
        tool_results=[
            {
                "tool": "document.search",
                "result": {
                    "status": "OK",
                    "results": [
                        {
                            "source_id": "D1",
                            "filename": "paper.pdf",
                            "section_title": "3 Method",
                            "section_path": ["3 Method"],
                            "page_start": 12,
                            "page_end": 13,
                            "text": "Boundary conditions are described here.",
                        }
                    ],
                },
            }
        ],
    )

    assert "<DOCUMENT_CONTEXT>" in context
    assert "[D1]" in context
    assert "File: paper.pdf" in context
    assert "Pages: 12-13" in context


def _result(chunk_id: str, document_id: str, section: str, text: str, score: float) -> DocumentChunkSearchResult:
    return DocumentChunkSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.pdf",
        text=text,
        section_title=section,
        section_path=[section],
        page_start=1,
        page_end=1,
        similarity_score=score,
    )
