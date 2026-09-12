"""Retrieval candidate generation, merge, and rerank API."""

from kivi_memory.retrieval.config import (
    DEFAULT_GRAPH_MAX_BRIDGE_DEGREE,
    DEFAULT_GRAPH_RERANK_WEIGHT,
    DEFAULT_GRAPH_TOP_K,
    MAX_GRAPH_TOP_K,
    get_graph_max_bridge_degree,
    get_graph_rerank_weight,
    get_graph_top_k,
)
from kivi_memory.retrieval.graph import (
    GRAPH_DISTANCE_DIRECT,
    GRAPH_DISTANCE_ONE_EXPANSION,
    GraphMemoryRepository,
    GraphRetrievalResult,
    retrieve_graph_candidates,
)
from kivi_memory.retrieval.models import (
    GraphMemoryCandidate,
    GraphSignals,
    MergedMemoryCandidate,
    StructuredMemoryCandidate,
    StructuredSignals,
)
from kivi_memory.retrieval.reranker import (
    MAX_FINAL_TOP_K,
    MAX_SHARED_ENTITY_BOOST,
    SOURCE_GRAPH,
    SAME_MEMORY_TYPE_BOOST,
    SAME_PREDICATE_BOOST,
    SAME_SUBJECT_BOOST,
    SHARED_ENTITY_BOOST,
    SOURCE_STRUCTURED,
    SOURCE_VECTOR,
    VECTOR_SIMILARITY_WEIGHT,
    merge_and_rerank_candidates,
)
from kivi_memory.retrieval.structured import (
    MAX_STRUCTURED_RETRIEVAL_TOP_K,
    StructuredMemoryRepository,
    StructuredRetrievalResult,
    retrieve_structured_candidates,
)

__all__ = [
    "DEFAULT_GRAPH_MAX_BRIDGE_DEGREE",
    "DEFAULT_GRAPH_RERANK_WEIGHT",
    "DEFAULT_GRAPH_TOP_K",
    "GRAPH_DISTANCE_DIRECT",
    "GRAPH_DISTANCE_ONE_EXPANSION",
    "GraphMemoryCandidate",
    "GraphMemoryRepository",
    "GraphRetrievalResult",
    "GraphSignals",
    "MAX_FINAL_TOP_K",
    "MAX_GRAPH_TOP_K",
    "MAX_SHARED_ENTITY_BOOST",
    "MAX_STRUCTURED_RETRIEVAL_TOP_K",
    "MergedMemoryCandidate",
    "SOURCE_GRAPH",
    "SAME_MEMORY_TYPE_BOOST",
    "SAME_PREDICATE_BOOST",
    "SAME_SUBJECT_BOOST",
    "SHARED_ENTITY_BOOST",
    "SOURCE_STRUCTURED",
    "SOURCE_VECTOR",
    "StructuredMemoryRepository",
    "StructuredMemoryCandidate",
    "StructuredRetrievalResult",
    "StructuredSignals",
    "VECTOR_SIMILARITY_WEIGHT",
    "get_graph_max_bridge_degree",
    "get_graph_rerank_weight",
    "get_graph_top_k",
    "merge_and_rerank_candidates",
    "retrieve_graph_candidates",
    "retrieve_structured_candidates",
]
