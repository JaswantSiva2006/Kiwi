from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    chunks_indexed: int
    duplicate: bool
    embedding_time: float
    db_write_time: float
    total_time: float


@dataclass(frozen=True)
class DocumentChunkSearchResult:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    section_title: str | None
    section_path: list[str]
    page_start: int
    page_end: int
    similarity_score: float
