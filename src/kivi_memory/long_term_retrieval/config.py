"""Configuration for read-only long-term memory retrieval."""

from __future__ import annotations

import os

DEFAULT_READ_RETRIEVAL_TOP_K = 12
DEFAULT_READ_RETRIEVAL_BRANCH_K = 30
MAX_READ_RETRIEVAL_TOP_K = 50
MAX_READ_RETRIEVAL_BRANCH_K = 100
READ_RETRIEVAL_RRF_K = 60
QUERY_ENTITY_FUZZY_THRESHOLD = 0.85
QUERY_ENTITY_MIN_MARGIN = 0.10


def clamp_top_k(top_k: int) -> int:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    return min(top_k, MAX_READ_RETRIEVAL_TOP_K)


def clamp_branch_k(branch_k: int) -> int:
    if branch_k < 1:
        raise ValueError("branch_k must be positive")
    return min(branch_k, MAX_READ_RETRIEVAL_BRANCH_K)


def env_default_top_k() -> int:
    return clamp_top_k(_int_env("KIVI_READ_RETRIEVAL_TOP_K", DEFAULT_READ_RETRIEVAL_TOP_K))


def env_default_branch_k() -> int:
    return clamp_branch_k(_int_env("KIVI_READ_RETRIEVAL_BRANCH_K", DEFAULT_READ_RETRIEVAL_BRANCH_K))


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
