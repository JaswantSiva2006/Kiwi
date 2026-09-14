from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from .embedding_service import DocumentEmbeddingService
from .models import DocumentChunkSearchResult
from .repository import DocumentRagRepository

CANDIDATE_K = 12
DEFAULT_TOP_K = 6
MAX_EVIDENCE_GROUPS = 6
MAX_DOCUMENT_CONTEXT_TOKENS = 3500
MMR_LAMBDA = 0.75
PROBE_TOP_K = 3
PROBE_RELEVANCE_THRESHOLD = 0.45


@dataclass
class DocumentProbe:
    query: str
    query_embedding: list[float] | None
    metadata: dict[str, Any]
    latency_ms: float


class DocumentSearchTool:
    def __init__(
        self,
        *,
        repository: DocumentRagRepository | None = None,
        embedding_service: DocumentEmbeddingService | None = None,
        candidate_k: int = CANDIDATE_K,
        max_evidence_groups: int = MAX_EVIDENCE_GROUPS,
        max_context_tokens: int = MAX_DOCUMENT_CONTEXT_TOKENS,
    ) -> None:
        self.repository = repository or DocumentRagRepository()
        self.embedding_service = embedding_service or DocumentEmbeddingService()
        self.candidate_k = candidate_k
        self.max_evidence_groups = max_evidence_groups
        self.max_context_tokens = max_context_tokens

    def probe(self, query: str) -> DocumentProbe:
        started = time.perf_counter()
        try:
            if self.repository.ready_document_count() <= 0:
                return DocumentProbe(query=query, query_embedding=None, metadata={"documents_available": False, "likely_relevant": False, "matches": []}, latency_ms=_elapsed_ms(started))
            embedding = self.embedding_service.embed_query(query)
            matches = self.repository.search_chunks(embedding, top_k=PROBE_TOP_K)
        except Exception:
            return DocumentProbe(query=query, query_embedding=None, metadata={"documents_available": False, "likely_relevant": False, "matches": []}, latency_ms=_elapsed_ms(started))
        metadata = {
            "documents_available": True,
            "likely_relevant": _explicit_document_intent(query) or any(item.similarity_score >= PROBE_RELEVANCE_THRESHOLD for item in matches),
            "matches": [
                {
                    "filename": item.filename,
                    "section": item.section_title,
                    "score": round(item.similarity_score, 4),
                }
                for item in matches
            ],
        }
        return DocumentProbe(query=query, query_embedding=embedding, metadata=metadata, latency_ms=_elapsed_ms(started))

    def search(
        self,
        *,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        document_ids: list[str] | None = None,
        query_embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        embedding = query_embedding or self.embedding_service.embed_query(query)
        candidates = self.repository.search_chunks(
            embedding,
            top_k=max(self.candidate_k, top_k),
            document_ids=document_ids or None,
        )
        selected = select_document_evidence(
            candidates,
            top_k=min(top_k, self.max_evidence_groups),
            max_context_tokens=self.max_context_tokens,
        )
        return {
            "status": "OK",
            "results": [
                {
                    "source_id": f"D{index}",
                    "document_id": item.document_id,
                    "filename": item.filename,
                    "chunk_id": item.chunk_id,
                    "section_title": item.section_title,
                    "section_path": item.section_path,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "text": item.text,
                    "similarity_score": round(item.similarity_score, 6),
                }
                for index, item in enumerate(selected, start=1)
            ],
        }


def select_document_evidence(
    candidates: list[DocumentChunkSearchResult],
    *,
    top_k: int = DEFAULT_TOP_K,
    max_context_tokens: int = MAX_DOCUMENT_CONTEXT_TOKENS,
    lambda_: float = MMR_LAMBDA,
) -> list[DocumentChunkSearchResult]:
    deduped = _dedupe_candidates(candidates)
    selected: list[DocumentChunkSearchResult] = []
    used_tokens = 0
    while deduped and len(selected) < top_k:
        best = max(deduped, key=lambda item: _mmr_score(item, selected, lambda_))
        deduped.remove(best)
        tokens = _token_count(best.text)
        if used_tokens + tokens > max_context_tokens:
            continue
        selected.append(best)
        used_tokens += tokens
    return selected


def _dedupe_candidates(candidates: list[DocumentChunkSearchResult]) -> list[DocumentChunkSearchResult]:
    kept: list[DocumentChunkSearchResult] = []
    seen: set[str] = set()
    for item in sorted(candidates, key=lambda c: c.similarity_score, reverse=True):
        normalized = _normalize_text(item.text)
        near_duplicate = any(_jaccard(_tokens(normalized), _tokens(_normalize_text(other.text))) >= 0.9 for other in kept)
        adjacent_overlap = any(
            item.document_id == other.document_id
            and item.section_path == other.section_path
            and abs(_chunk_number(item.chunk_id) - _chunk_number(other.chunk_id)) <= 1
            and _jaccard(_tokens(normalized), _tokens(_normalize_text(other.text))) >= 0.65
            for other in kept
        )
        if normalized in seen or near_duplicate or adjacent_overlap:
            continue
        seen.add(normalized)
        kept.append(item)
    return kept


def _mmr_score(item: DocumentChunkSearchResult, selected: list[DocumentChunkSearchResult], lambda_: float) -> float:
    if not selected:
        return item.similarity_score
    redundancy = max(_result_similarity(item, other) for other in selected)
    diversity_bonus = 0.03 if all(item.document_id != other.document_id or item.section_path != other.section_path for other in selected) else 0.0
    return lambda_ * item.similarity_score - (1 - lambda_) * redundancy + diversity_bonus


def _result_similarity(a: DocumentChunkSearchResult, b: DocumentChunkSearchResult) -> float:
    score = _jaccard(_tokens(a.text), _tokens(b.text))
    if a.document_id == b.document_id and a.section_path == b.section_path:
        score += 0.15
    return min(score, 1.0)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) > 2}


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _chunk_number(chunk_id: str) -> int:
    try:
        return int(chunk_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def _explicit_document_intent(query: str) -> bool:
    lowered = query.casefold()
    return any(marker in lowered for marker in ("document", "pdf", "paper", "report", "file", "spec", "indexed"))


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
