from __future__ import annotations

from typing import Any

from kivi_memory.document_rag.ingestion import ingest_parsed_document


def test_duplicate_sha_skips_embedding() -> None:
    repository = FakeRepository(existing={"document_id": "existing-doc", "status": "READY"})
    embedding_service = FakeEmbeddingService()

    result = ingest_parsed_document(_parsed(), repository=repository, embedding_service=embedding_service)

    assert result.duplicate is True
    assert result.document_id == "existing-doc"
    assert result.chunks_indexed == 0
    assert embedding_service.calls == []
    assert repository.stored is None


def test_ingestion_batches_and_stores_once() -> None:
    repository = FakeRepository(existing=None)
    embedding_service = FakeEmbeddingService()
    parsed = _parsed()

    result = ingest_parsed_document(parsed, repository=repository, embedding_service=embedding_service)

    assert result.duplicate is False
    assert result.document_id == "doc-1"
    assert result.chunks_indexed == 2
    assert embedding_service.calls == [["alpha geometry encoder", "beta architecture decoder"]]
    assert repository.stored is not None
    assert repository.stored[0] == parsed
    assert len(repository.stored[1]) == 2


class FakeRepository:
    def __init__(self, existing: dict[str, Any] | None) -> None:
        self.existing = existing
        self.stored: tuple[dict[str, Any], list[list[float]]] | None = None

    def find_document_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        return self.existing

    def store_document_with_chunks(self, parsed: dict[str, Any], embeddings: list[list[float]]) -> None:
        self.stored = (parsed, embeddings)


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


def _parsed() -> dict[str, Any]:
    return {
        "document": {
            "document_id": "doc-1",
            "filename": "sample.pdf",
            "sha256": "abc",
            "page_count": 1,
            "parser": "pymupdf",
            "parse_status": "OK",
        },
        "chunks": [
            {
                "chunk_id": "doc-1:chunk:00000",
                "document_id": "doc-1",
                "chunk_index": 0,
                "chunk_type": "TEXT",
                "text": "alpha geometry encoder",
                "section_title": "1 Overview",
                "section_path": ["1 Overview"],
                "page_start": 1,
                "page_end": 1,
                "source_blocks": ["p1:b0000"],
                "token_count": 3,
                "metadata": {"filename": "sample.pdf", "mime_type": "application/pdf"},
            },
            {
                "chunk_id": "doc-1:chunk:00001",
                "document_id": "doc-1",
                "chunk_index": 1,
                "chunk_type": "TEXT",
                "text": "beta architecture decoder",
                "section_title": "1.1 Details",
                "section_path": ["1 Overview", "1.1 Details"],
                "page_start": 1,
                "page_end": 1,
                "source_blocks": ["p1:b0001"],
                "token_count": 3,
                "metadata": {"filename": "sample.pdf", "mime_type": "application/pdf"},
            },
        ],
    }
