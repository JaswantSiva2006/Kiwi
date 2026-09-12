from __future__ import annotations

import pytest

from kivi_memory.embeddings import (
    EMBEDDING_MODEL_NAME,
    retrieve_vector_candidates,
    retrieve_vector_candidates_by_vector,
    retrieve_vector_candidates_exact,
    retrieve_vector_candidates_hnsw,
)
from kivi_memory.embeddings.models import VectorMemoryCandidate


class FakeRetrievalRepository:
    def __init__(self) -> None:
        self.rows = [
            row("memory-1", "Riya owns the billing dashboard.", 0.92, "ACTIVE", EMBEDDING_MODEL_NAME),
            row("memory-2", "The user met Kavya yesterday.", 0.85, "ACTIVE", EMBEDDING_MODEL_NAME),
            row("memory-3", "Archived memory.", 0.99, "ARCHIVED", EMBEDDING_MODEL_NAME),
            row("memory-4", "Wrong model memory.", 0.98, "ACTIVE", "other-model"),
        ]
        self.calls = []
        self.write_calls = []

    def retrieve_vector_candidates(self, query_embedding, embedding_model: str, top_k: int):
        return self.retrieve_vector_candidates_exact(query_embedding, embedding_model, top_k)

    def retrieve_vector_candidates_exact(self, query_embedding, embedding_model: str, top_k: int):
        self.calls.append(
            {
                "mode": "exact",
                "query_embedding": query_embedding,
                "embedding_model": embedding_model,
                "top_k": top_k,
            }
        )
        filtered = [
            item
            for item in self.rows
            if item["status"] == "ACTIVE" and item["embedding_model"] == embedding_model
        ]
        filtered = sorted(filtered, key=lambda item: item["vector_similarity"], reverse=True)[:top_k]
        return [
            VectorMemoryCandidate(
                memory_id=item["memory_id"],
                canonical_text=item["canonical_text"],
                vector_similarity=item["vector_similarity"],
                subject_entity_id=item["subject_entity_id"],
                predicate_type=item["predicate_type"],
                memory_type=item["memory_type"],
                status=item["status"],
            )
            for item in filtered
        ]

    def retrieve_vector_candidates_hnsw(self, query_embedding, embedding_model: str, top_k: int, ef_search: int):
        self.calls.append(
            {
                "mode": "hnsw",
                "query_embedding": query_embedding,
                "embedding_model": embedding_model,
                "top_k": top_k,
                "ef_search": ef_search,
            }
        )
        return self.retrieve_vector_candidates_exact(query_embedding, embedding_model, top_k)

    def hnsw_index_exists(self):
        return True

    def count_active_embeddings(self, embedding_model: str):
        return len([row for row in self.rows if row["status"] == "ACTIVE" and row["embedding_model"] == embedding_model])

    def store_memory_embedding(self, *args, **kwargs):
        self.write_calls.append((args, kwargs))


def fake_embedder(text: str) -> list[float]:
    return [1.0, 0.0, 0.0]


def test_returns_at_most_top_k() -> None:
    result = retrieve_vector_candidates_exact("Riya manages the dashboard.", top_k=1, repository=FakeRetrievalRepository(), embedder=fake_embedder)

    assert len(result.candidates) == 1


def test_results_sorted_descending_by_similarity() -> None:
    result = retrieve_vector_candidates_exact("Riya manages the dashboard.", repository=FakeRetrievalRepository(), embedder=fake_embedder)

    scores = [candidate.vector_similarity for candidate in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_only_active_memories_returned() -> None:
    result = retrieve_vector_candidates_exact("Riya manages the dashboard.", repository=FakeRetrievalRepository(), embedder=fake_embedder)

    assert all(candidate.status == "ACTIVE" for candidate in result.candidates)
    assert "memory-3" not in [candidate.memory_id for candidate in result.candidates]


def test_only_current_embedding_model_used() -> None:
    repo = FakeRetrievalRepository()

    result = retrieve_vector_candidates_exact("Riya manages the dashboard.", repository=repo, embedder=fake_embedder)

    assert repo.calls[0]["embedding_model"] == EMBEDDING_MODEL_NAME
    assert "memory-4" not in [candidate.memory_id for candidate in result.candidates]


def test_empty_ledger_handled_cleanly() -> None:
    repo = FakeRetrievalRepository()
    repo.rows = []

    result = retrieve_vector_candidates_exact("Nothing yet.", repository=repo, embedder=fake_embedder)

    assert result.candidates == []


@pytest.mark.parametrize("top_k", [0, -1, 101])
def test_invalid_top_k_handled_cleanly(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be between"):
        retrieve_vector_candidates_exact("Riya manages the dashboard.", top_k=top_k, repository=FakeRetrievalRepository(), embedder=fake_embedder)


def test_paraphrase_retrieves_corresponding_memory_near_top() -> None:
    result = retrieve_vector_candidates_exact("Riya is responsible for the billing dashboard.", top_k=2, repository=FakeRetrievalRepository(), embedder=fake_embedder)

    assert result.candidates[0].canonical_text == "Riya owns the billing dashboard."


def test_function_performs_no_writes() -> None:
    repo = FakeRetrievalRepository()

    retrieve_vector_candidates_exact("Riya manages the dashboard.", repository=repo, embedder=fake_embedder)

    assert repo.write_calls == []


def test_default_dispatch_uses_hnsw(monkeypatch) -> None:
    repo = FakeRetrievalRepository()

    result = retrieve_vector_candidates("Riya manages the dashboard.", repository=repo, embedder=fake_embedder)

    assert repo.calls[0]["mode"] == "hnsw"
    assert result.vector_search_mode == "hnsw"
    assert result.hnsw_ef_search >= result.vector_top_k
    assert result.hnsw_iterative_scan == "strict_order"


def test_exact_mode_still_available() -> None:
    repo = FakeRetrievalRepository()

    result = retrieve_vector_candidates_by_vector([1.0, 0.0, 0.0], top_k=2, repository=repo, search_mode="exact")

    assert repo.calls[0]["mode"] == "exact"
    assert result.vector_search_mode == "exact"


def test_hnsw_error_falls_back_to_exact() -> None:
    class FailingHnswRepository(FakeRetrievalRepository):
        def retrieve_vector_candidates_hnsw(self, query_embedding, embedding_model: str, top_k: int, ef_search: int):
            raise RuntimeError("hnsw exploded")

    result = retrieve_vector_candidates_hnsw("Riya manages the dashboard.", repository=FailingHnswRepository(), embedder=fake_embedder)

    assert result.vector_search_mode == "exact"
    assert result.hnsw_fallback_used is True
    assert "hnsw exploded" in result.hnsw_fallback_reason


def test_missing_hnsw_index_is_diagnosable_and_falls_back() -> None:
    class MissingIndexRepository(FakeRetrievalRepository):
        def hnsw_index_exists(self):
            return False

    result = retrieve_vector_candidates_hnsw("Riya manages the dashboard.", repository=MissingIndexRepository(), embedder=fake_embedder)

    assert result.hnsw_fallback_used is True
    assert "missing" in result.hnsw_fallback_reason


def row(
    memory_id: str,
    canonical_text: str,
    vector_similarity: float,
    status: str,
    embedding_model: str,
) -> dict:
    return {
        "memory_id": memory_id,
        "canonical_text": canonical_text,
        "vector_similarity": vector_similarity,
        "subject_entity_id": None,
        "predicate_type": "TEST",
        "memory_type": "ENTITY_CONTEXT",
        "status": status,
        "embedding_model": embedding_model,
    }
