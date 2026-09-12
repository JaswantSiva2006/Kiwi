"""Configuration for retrieval candidate generation and reranking."""

from __future__ import annotations

import os

DEFAULT_GRAPH_TOP_K = 30
MAX_GRAPH_TOP_K = 100
DEFAULT_GRAPH_MAX_BRIDGE_DEGREE = 100
DEFAULT_GRAPH_RERANK_WEIGHT = 0.20


def get_graph_top_k() -> int:
    return _positive_int("KIVI_GRAPH_TOP_K", DEFAULT_GRAPH_TOP_K, maximum=MAX_GRAPH_TOP_K)


def get_graph_max_bridge_degree() -> int:
    return _positive_int("KIVI_GRAPH_MAX_BRIDGE_DEGREE", DEFAULT_GRAPH_MAX_BRIDGE_DEGREE)


def get_graph_rerank_weight() -> float:
    raw = os.getenv("KIVI_GRAPH_RERANK_WEIGHT")
    if raw is None:
        return DEFAULT_GRAPH_RERANK_WEIGHT
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("KIVI_GRAPH_RERANK_WEIGHT must be a number") from exc
    if value < 0:
        raise ValueError("KIVI_GRAPH_RERANK_WEIGHT must be non-negative")
    return value


def _positive_int(name: str, default: int, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    if maximum is not None:
        return min(value, maximum)
    return value
