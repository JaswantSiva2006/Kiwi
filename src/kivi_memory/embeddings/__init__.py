"""Derived memory embedding projection API."""

from kivi_memory.embeddings.config import (
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_VECTOR_SEARCH_MODE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    HNSW_INDEX_NAME,
    HNSW_ITERATIVE_SCAN,
    VECTOR_SEARCH_MODE_EXACT,
    VECTOR_SEARCH_MODE_HNSW,
)
from kivi_memory.embeddings.generator import embed_text, embed_texts
from kivi_memory.embeddings.indexer import backfill_active_memory_embeddings, canonical_text_hash, index_memory
from kivi_memory.embeddings.models import BackfillResult, IndexMemoryResult, VectorMemoryCandidate, VectorRetrievalResult
from kivi_memory.embeddings.repository import MemoryEmbeddingRepository
from kivi_memory.embeddings.retriever import (
    MAX_VECTOR_RETRIEVAL_TOP_K,
    retrieve_vector_candidates,
    retrieve_vector_candidates_by_vector,
    retrieve_vector_candidates_exact,
    retrieve_vector_candidates_exact_by_vector,
    retrieve_vector_candidates_hnsw,
    retrieve_vector_candidates_hnsw_by_vector,
)

__all__ = [
    "DEFAULT_HNSW_EF_SEARCH",
    "DEFAULT_VECTOR_SEARCH_MODE",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_MODEL_NAME",
    "HNSW_INDEX_NAME",
    "HNSW_ITERATIVE_SCAN",
    "BackfillResult",
    "IndexMemoryResult",
    "MAX_VECTOR_RETRIEVAL_TOP_K",
    "MemoryEmbeddingRepository",
    "VectorMemoryCandidate",
    "VectorRetrievalResult",
    "VECTOR_SEARCH_MODE_EXACT",
    "VECTOR_SEARCH_MODE_HNSW",
    "backfill_active_memory_embeddings",
    "canonical_text_hash",
    "embed_text",
    "embed_texts",
    "index_memory",
    "retrieve_vector_candidates",
    "retrieve_vector_candidates_by_vector",
    "retrieve_vector_candidates_exact",
    "retrieve_vector_candidates_exact_by_vector",
    "retrieve_vector_candidates_hnsw",
    "retrieve_vector_candidates_hnsw_by_vector",
]
