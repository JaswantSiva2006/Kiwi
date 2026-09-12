"""Reciprocal Rank Fusion for read-time memory retrieval."""

from __future__ import annotations

from kivi_memory.long_term_retrieval.config import READ_RETRIEVAL_RRF_K
from kivi_memory.long_term_retrieval.models import BranchCandidate, FusedCandidate

BRANCH_ORDER = ["VECTOR", "LEXICAL", "STRUCTURED", "GRAPH"]
BRANCH_KEYS = {"VECTOR": "vector", "LEXICAL": "lexical", "STRUCTURED": "structured", "GRAPH": "graph"}


def fuse_branch_candidates(
    branch_candidates: dict[str, list[BranchCandidate]],
    top_k: int,
) -> list[FusedCandidate]:
    """Fuse branch-local ranks with unweighted RRF and return final Top-K IDs."""

    merged: dict[str, dict] = {}
    for source in BRANCH_ORDER:
        candidates = branch_candidates.get(source, [])
        for candidate in candidates:
            if candidate.rank < 1:
                raise ValueError("branch ranks must be 1-based")
            state = merged.setdefault(
                candidate.memory_id,
                {
                    "memory_id": candidate.memory_id,
                    "sources": set(),
                    "branch_ranks": {key: None for key in BRANCH_KEYS.values()},
                    "branch_scores": {key: None for key in BRANCH_KEYS.values()},
                    "rrf_score": 0.0,
                },
            )
            if source in state["sources"]:
                key = BRANCH_KEYS[source]
                current_rank = state["branch_ranks"][key]
                if current_rank is not None and current_rank <= candidate.rank:
                    continue
                state["rrf_score"] -= 1 / (READ_RETRIEVAL_RRF_K + current_rank)
            state["sources"].add(source)
            branch_key = BRANCH_KEYS[source]
            state["branch_ranks"][branch_key] = candidate.rank
            state["branch_scores"][branch_key] = candidate.raw_score
            state["rrf_score"] += 1 / (READ_RETRIEVAL_RRF_K + candidate.rank)

    fused = [
        FusedCandidate(
            memory_id=state["memory_id"],
            rrf_score=state["rrf_score"],
            retrieval_sources=[source for source in BRANCH_ORDER if source in state["sources"]],
            branch_ranks=state["branch_ranks"],
            branch_scores=state["branch_scores"],
            best_branch_rank=min(rank for rank in state["branch_ranks"].values() if rank is not None),
        )
        for state in merged.values()
    ]
    return sorted(fused, key=_sort_key)[:top_k]


def _sort_key(candidate: FusedCandidate) -> tuple[float, int, int, str]:
    return (
        -candidate.rrf_score,
        -len(candidate.retrieval_sources),
        candidate.best_branch_rank,
        candidate.memory_id,
    )
