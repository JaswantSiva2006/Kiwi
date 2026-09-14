from __future__ import annotations

import os

DOCUMENT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCUMENT_EMBEDDING_DIMENSION = 384
DEFAULT_DOCUMENT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_DOCUMENT_HNSW_EF_SEARCH = 80
DOCUMENT_HNSW_INDEX_NAME = "document_rag_chunks_hnsw_cosine_idx"


def get_document_embedding_batch_size() -> int:
    return _positive_int_env("KIVI_DOCUMENT_EMBEDDING_BATCH_SIZE", DEFAULT_DOCUMENT_EMBEDDING_BATCH_SIZE)


def get_document_hnsw_ef_search(top_k: int) -> int:
    value = _positive_int_env("KIVI_DOCUMENT_HNSW_EF_SEARCH", DEFAULT_DOCUMENT_HNSW_EF_SEARCH)
    return max(value, top_k)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value
