from __future__ import annotations

from dataclasses import dataclass

import pytest

from kivi_memory.embeddings.models import VectorMemoryCandidate, VectorRetrievalResult
from kivi_memory.long_term_retrieval.models import BranchCandidate, QueryEntityMatch
from kivi_memory.long_term_retrieval.retriever import retrieve_long_term_memories


@dataclass
class FakeRepository:
    lexical: list[BranchCandidate]
    entities: list[QueryEntityMatch]
    structured: list[BranchCandidate]
    graph: list[BranchCandidate]

    def __post_init__(self) -> None:
        self.calls: list[tuple] = []
        self.write_calls: list[tuple] = []

    def retrieve_lexical_candidates(self, query_text: str, branch_k: int) -> list[BranchCandidate]:
        self.calls.append(("lexical", query_text, branch_k))
        return self.lexical[:branch_k]

    def match_query_entities(self, query_text: str) -> list[QueryEntityMatch]:
        self.calls.append(("entities", query_text))
        return self.entities

    def retrieve_structured_candidates(self, entity_ids: list[str], branch_k: int) -> list[BranchCandidate]:
        self.calls.append(("structured", entity_ids, branch_k))
        return self.structured[:branch_k] if entity_ids else []

    def retrieve_graph_candidates(self, entity_ids: list[str], branch_k: int) -> list[BranchCandidate]:
        self.calls.append(("graph", entity_ids, branch_k))
        return self.graph[:branch_k] if entity_ids else []

    def hydrate_memories(self, fused_candidates):
        self.calls.append(("hydrate", [candidate.memory_id for candidate in fused_candidates]))
        return [
            type(
                "Hydrated",
                (),
                {
                    "memory_id": candidate.memory_id,
                    "canonical_text": f"text {candidate.memory_id}",
                    "rrf_score": candidate.rrf_score,
                    "retrieval_sources": candidate.retrieval_sources,
                    "branch_ranks": candidate.branch_ranks,
                    "branch_scores": candidate.branch_scores,
                },
            )()
            for candidate in fused_candidates
        ]

    def store_memory(self, *args, **kwargs):
        self.write_calls.append((args, kwargs))


def branch(memory_id: str, source: str, rank: int, score: float = 1.0) -> BranchCandidate:
    return BranchCandidate(memory_id=memory_id, source=source, rank=rank, raw_score=score)


def entity(entity_id: str = "entity-1") -> QueryEntityMatch:
    return QueryEntityMatch(
        entity_id=entity_id,
        canonical_name="Project Atlas",
        entity_type="PROJECT",
        matched_text="project atlas",
        match_method="exact_name",
        score=1.0,
    )


def vector_result(ids: list[str]) -> VectorRetrievalResult:
    return VectorRetrievalResult(
        candidates=[
            VectorMemoryCandidate(
                memory_id=memory_id,
                canonical_text=f"text {memory_id}",
                vector_similarity=1.0 - index / 100,
                subject_entity_id=None,
                predicate_type="TEST",
                memory_type="TEST",
                status="ACTIVE",
            )
            for index, memory_id in enumerate(ids)
        ],
        embedding_ms=1.0,
        db_search_ms=2.0,
        total_ms=3.0,
        vector_search_mode="hnsw",
        hnsw_fallback_used=False,
    )


def test_retrieval_fuses_branches_and_hydrates_only_final_top_k(monkeypatch) -> None:
    monkeypatch.setattr(
        "kivi_memory.long_term_retrieval.retriever.retrieve_vector_candidates",
        lambda query_text, top_k: vector_result(["m1", "m2", "m3"][:top_k]),
    )
    repo = FakeRepository(
        lexical=[branch("m4", "LEXICAL", 1)],
        entities=[entity()],
        structured=[branch("m2", "STRUCTURED", 1)],
        graph=[branch("m5", "GRAPH", 1)],
    )

    result = retrieve_long_term_memories("Who handles Project Atlas?", top_k=3, branch_k=2, repository=repo)

    hydrate_call = [call for call in repo.calls if call[0] == "hydrate"][0]
    assert len(hydrate_call[1]) == 3
    assert len(result.memories) == 3
    assert result.diagnostics.vector_count == 2
    assert result.diagnostics.lexical_count == 1
    assert result.diagnostics.structured_count == 1
    assert result.diagnostics.graph_count == 1
    assert result.diagnostics.hnsw_used is True


def test_vector_and_lexical_run_without_resolved_entities(monkeypatch) -> None:
    monkeypatch.setattr(
        "kivi_memory.long_term_retrieval.retriever.retrieve_vector_candidates",
        lambda query_text, top_k: vector_result(["vector-memory"]),
    )
    repo = FakeRepository(
        lexical=[branch("lexical-memory", "LEXICAL", 1)],
        entities=[],
        structured=[branch("should-not-run", "STRUCTURED", 1)],
        graph=[branch("should-not-run", "GRAPH", 1)],
    )

    result = retrieve_long_term_memories("Kafka event processing", top_k=10, branch_k=30, repository=repo)

    assert {memory.memory_id for memory in result.memories} == {"vector-memory", "lexical-memory"}
    assert ("structured", [], 30) in repo.calls
    assert ("graph", [], 30) in repo.calls


def test_branch_k_limit_is_passed_to_every_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        "kivi_memory.long_term_retrieval.retriever.retrieve_vector_candidates",
        lambda query_text, top_k: vector_result(["m1", "m2", "m3"][:top_k]),
    )
    repo = FakeRepository([branch("m4", "LEXICAL", 1)], [entity()], [], [])

    retrieve_long_term_memories("Project Atlas", top_k=10, branch_k=2, repository=repo)

    assert ("lexical", "Project Atlas", 2) in repo.calls
    assert ("structured", ["entity-1"], 2) in repo.calls
    assert ("graph", ["entity-1"], 2) in repo.calls


@pytest.mark.parametrize("top_k,branch_k", [(0, 10), (10, 0), (-1, 10), (10, -1)])
def test_invalid_limits_fail_cleanly(monkeypatch, top_k: int, branch_k: int) -> None:
    monkeypatch.setattr(
        "kivi_memory.long_term_retrieval.retriever.retrieve_vector_candidates",
        lambda query_text, top_k: vector_result([]),
    )

    with pytest.raises(ValueError):
        retrieve_long_term_memories("Project Atlas", top_k=top_k, branch_k=branch_k, repository=FakeRepository([], [], [], []))


def test_empty_ledger_returns_empty_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "kivi_memory.long_term_retrieval.retriever.retrieve_vector_candidates",
        lambda query_text, top_k: vector_result([]),
    )

    result = retrieve_long_term_memories("nothing", repository=FakeRepository([], [], [], []))

    assert result.memories == []
    assert result.diagnostics.union_unique_count == 0


def test_function_performs_no_writes_or_llm_calls(monkeypatch) -> None:
    def fail_import(name, *args, **kwargs):
        if name.startswith("kivi_memory.semantic_compiler") or name.startswith("kivi_memory.reconciliation"):
            raise AssertionError(f"unexpected LLM/reconciliation import: {name}")
        return real_import(name, *args, **kwargs)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    monkeypatch.setattr("builtins.__import__", fail_import)
    monkeypatch.setattr(
        "kivi_memory.long_term_retrieval.retriever.retrieve_vector_candidates",
        lambda query_text, top_k: vector_result(["m1"]),
    )
    repo = FakeRepository([], [], [], [])

    retrieve_long_term_memories("read only", repository=repo)

    assert repo.write_calls == []
