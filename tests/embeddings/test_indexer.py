from __future__ import annotations

import pytest

from kivi_memory.embeddings import EMBEDDING_MODEL_NAME, canonical_text_hash, index_memory
from kivi_memory.embeddings.indexer import backfill_active_memory_embeddings


class FakeEmbeddingRepository:
    def __init__(self) -> None:
        self.memories = {"memory-1": "The user met Kavya yesterday."}
        self.embeddings: dict[tuple[str, str], dict] = {}
        self.deleted_or_modified_memory = False

    def get_memory_canonical_text(self, memory_id: str) -> str:
        return self.memories[memory_id]

    def get_embedding_text_hash(self, memory_id: str, embedding_model: str) -> str | None:
        row = self.embeddings.get((memory_id, embedding_model))
        return None if row is None else row["embedding_text_hash"]

    def store_memory_embedding(
        self,
        memory_id: str,
        embedding: list[float],
        embedding_model: str,
        embedding_text_hash: str,
    ) -> None:
        self.embeddings[(memory_id, embedding_model)] = {
            "embedding": embedding,
            "embedding_model": embedding_model,
            "embedding_text_hash": embedding_text_hash,
        }

    def list_active_memory_embedding_status(self, embedding_model: str):
        return [
            {
                "memory_id": memory_id,
                "canonical_text": canonical_text,
                "embedding_text_hash": self.get_embedding_text_hash(memory_id, embedding_model),
            }
            for memory_id, canonical_text in self.memories.items()
        ]


def fake_embedding(text: str) -> list[float]:
    return [1.0, 0.0, 0.0]


def fake_embeddings(texts: list[str]) -> list[list[float]]:
    return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]


def test_index_memory_stores_an_embedding() -> None:
    repo = FakeEmbeddingRepository()

    result = index_memory("memory-1", repository=repo, embedder=fake_embedding)

    assert result.indexed is True
    assert result.skipped is False
    assert repo.embeddings[("memory-1", EMBEDDING_MODEL_NAME)]["embedding"] == [1.0, 0.0, 0.0]


def test_unchanged_canonical_text_hash_skips_reembedding() -> None:
    repo = FakeEmbeddingRepository()
    text_hash = canonical_text_hash(repo.memories["memory-1"])
    repo.store_memory_embedding("memory-1", [0.5, 0.0, 0.0], EMBEDDING_MODEL_NAME, text_hash)

    def should_not_embed(text: str):
        raise AssertionError("embedder should not run for current hash")

    result = index_memory("memory-1", repository=repo, embedder=should_not_embed)

    assert result.indexed is False
    assert result.skipped is True
    assert repo.embeddings[("memory-1", EMBEDDING_MODEL_NAME)]["embedding"] == [0.5, 0.0, 0.0]


def test_changed_text_hash_causes_upsert() -> None:
    repo = FakeEmbeddingRepository()
    repo.store_memory_embedding("memory-1", [0.5, 0.0, 0.0], EMBEDDING_MODEL_NAME, "old-hash")

    result = index_memory("memory-1", repository=repo, embedder=fake_embedding)

    assert result.indexed is True
    row = repo.embeddings[("memory-1", EMBEDDING_MODEL_NAME)]
    assert row["embedding"] == [1.0, 0.0, 0.0]
    assert row["embedding_text_hash"] == canonical_text_hash(repo.memories["memory-1"])


def test_batch_backfill_indexes_multiple_memories_correctly() -> None:
    repo = FakeEmbeddingRepository()
    repo.memories["memory-2"] = "The design review is tomorrow."

    result = backfill_active_memory_embeddings(batch_size=2, repository=repo, embedder=fake_embeddings)

    assert result.total_active_memories == 2
    assert result.indexed == 2
    assert result.skipped_current == 0
    assert result.failed == 0
    assert ("memory-1", EMBEDDING_MODEL_NAME) in repo.embeddings
    assert ("memory-2", EMBEDDING_MODEL_NAME) in repo.embeddings


def test_embedding_failure_does_not_modify_or_delete_canonical_memory() -> None:
    repo = FakeEmbeddingRepository()

    def failing_embedder(text: str):
        raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        index_memory("memory-1", repository=repo, embedder=failing_embedder)

    assert repo.memories == {"memory-1": "The user met Kavya yesterday."}
    assert repo.embeddings == {}
