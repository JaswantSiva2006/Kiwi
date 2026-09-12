from __future__ import annotations

import pytest

from kivi_memory.long_term_retrieval.config import READ_RETRIEVAL_RRF_K
from kivi_memory.long_term_retrieval.fusion import fuse_branch_candidates
from kivi_memory.long_term_retrieval.models import BranchCandidate


def candidate(memory_id: str, source: str, rank: int, score: float = 1.0) -> BranchCandidate:
    return BranchCandidate(memory_id=memory_id, source=source, rank=rank, raw_score=score)


def test_duplicate_memory_from_branches_is_fused_once() -> None:
    fused = fuse_branch_candidates(
        {
            "VECTOR": [candidate("m1", "VECTOR", 1, 0.9)],
            "LEXICAL": [candidate("m1", "LEXICAL", 2, 0.4)],
            "STRUCTURED": [],
            "GRAPH": [],
        },
        top_k=10,
    )

    assert len(fused) == 1
    assert fused[0].retrieval_sources == ["VECTOR", "LEXICAL"]
    assert fused[0].branch_ranks == {"vector": 1, "lexical": 2, "structured": None, "graph": None}
    assert fused[0].branch_scores["vector"] == 0.9
    assert fused[0].branch_scores["lexical"] == 0.4


def test_rrf_uses_branch_ranks_only() -> None:
    fused = fuse_branch_candidates(
        {
            "VECTOR": [candidate("m1", "VECTOR", 1, 999.0)],
            "LEXICAL": [candidate("m2", "LEXICAL", 1, 0.1), candidate("m1", "LEXICAL", 2, 0.1)],
        },
        top_k=10,
    )

    m1 = next(item for item in fused if item.memory_id == "m1")
    assert m1.rrf_score == pytest.approx(1 / (READ_RETRIEVAL_RRF_K + 1) + 1 / (READ_RETRIEVAL_RRF_K + 2))


def test_source_count_tiebreak_promotes_multi_branch_candidate() -> None:
    fused = fuse_branch_candidates(
        {
            "VECTOR": [candidate("m1", "VECTOR", 1), candidate("m2", "VECTOR", 2)],
            "LEXICAL": [candidate("m2", "LEXICAL", 1)],
        },
        top_k=10,
    )

    assert fused[0].memory_id == "m2"


def test_sorting_and_final_top_k_are_deterministic() -> None:
    fused = fuse_branch_candidates(
        {
            "VECTOR": [candidate("b", "VECTOR", 1), candidate("a", "VECTOR", 1), candidate("c", "VECTOR", 3)],
        },
        top_k=2,
    )

    assert [item.memory_id for item in fused] == ["a", "b"]


def test_vector_only_structured_only_and_empty_inputs_work() -> None:
    fused = fuse_branch_candidates(
        {
            "VECTOR": [candidate("vector-only", "VECTOR", 1)],
            "STRUCTURED": [candidate("structured-only", "STRUCTURED", 1)],
        },
        top_k=10,
    )

    assert {item.memory_id for item in fused} == {"vector-only", "structured-only"}
    assert fuse_branch_candidates({}, top_k=10) == []


def test_duplicate_memory_ids_inside_a_branch_keep_best_rank() -> None:
    fused = fuse_branch_candidates(
        {"VECTOR": [candidate("m1", "VECTOR", 3), candidate("m1", "VECTOR", 1)]},
        top_k=10,
    )

    assert fused[0].branch_ranks["vector"] == 1
    assert fused[0].rrf_score == pytest.approx(1 / (READ_RETRIEVAL_RRF_K + 1))


def test_branch_ranks_must_be_one_based() -> None:
    with pytest.raises(ValueError, match="1-based"):
        fuse_branch_candidates({"VECTOR": [candidate("m1", "VECTOR", 0)]}, top_k=10)
