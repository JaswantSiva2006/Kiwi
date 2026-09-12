"""Configuration for memory embedding projection and retrieval."""

from __future__ import annotations

import os

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
VECTOR_SEARCH_MODE_EXACT = "exact"
VECTOR_SEARCH_MODE_HNSW = "hnsw"
DEFAULT_VECTOR_SEARCH_MODE = VECTOR_SEARCH_MODE_HNSW
DEFAULT_HNSW_EF_SEARCH = 100
HNSW_ITERATIVE_SCAN = "strict_order"
HNSW_INDEX_NAME = "memory_embeddings_minilm_hnsw_cosine_idx"


def get_vector_search_mode() -> str:
    mode = os.getenv("KIVI_VECTOR_SEARCH_MODE", DEFAULT_VECTOR_SEARCH_MODE).strip().lower()
    if mode not in {VECTOR_SEARCH_MODE_EXACT, VECTOR_SEARCH_MODE_HNSW}:
        raise ValueError("KIVI_VECTOR_SEARCH_MODE must be exact or hnsw")
    return mode


def get_hnsw_ef_search(top_k: int) -> int:
    raw = os.getenv("KIVI_HNSW_EF_SEARCH", str(DEFAULT_HNSW_EF_SEARCH))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("KIVI_HNSW_EF_SEARCH must be an integer") from exc
    if value < 1:
        raise ValueError("KIVI_HNSW_EF_SEARCH must be positive")
    return max(value, top_k)
