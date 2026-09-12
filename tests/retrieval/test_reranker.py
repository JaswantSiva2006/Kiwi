from __future__ import annotations

import pytest

from kivi_memory.retrieval import SOURCE_GRAPH, SOURCE_STRUCTURED, SOURCE_VECTOR, merge_and_rerank_candidates


def vector_candidate(memory_id: str, similarity: float, canonical_text: str | None = None) -> dict:
    return {
        "memory_id": memory_id,
        "canonical_text": canonical_text or f"Memory {memory_id}",
        "vector_similarity": similarity,
        "subject_entity_id": f"subject-{memory_id}",
        "predicate_type": "RELATED_TO",
        "memory_type": "ENTITY_CONTEXT",
        "status": "ACTIVE",
    }


def structured_candidate(
    memory_id: str,
    *,
    same_subject: bool = False,
    shared_entity_count: int = 0,
    same_predicate: bool = False,
    same_memory_type: bool = False,
    canonical_text: str | None = None,
) -> dict:
    return {
        "memory_id": memory_id,
        "canonical_text": canonical_text or f"Memory {memory_id}",
        "structured_signals": {
            "same_subject": same_subject,
            "shared_entity_count": shared_entity_count,
            "same_predicate": same_predicate,
            "same_memory_type": same_memory_type,
        },
        "subject_entity_id": f"subject-{memory_id}",
        "predicate_type": "RELATED_TO",
        "memory_type": "ENTITY_CONTEXT",
        "status": "ACTIVE",
    }


def graph_candidate(
    memory_id: str,
    *,
    graph_distance: str = "DIRECT",
    graph_score: float = 1.0,
    bridge_path_count: int = 0,
    canonical_text: str | None = None,
) -> dict:
    return {
        "memory_id": memory_id,
        "canonical_text": canonical_text or f"Memory {memory_id}",
        "graph_signals": {
            "graph_distance": graph_distance,
            "graph_score": graph_score,
            "direct_seed_count": 1 if graph_distance == "DIRECT" else 0,
            "bridge_path_count": bridge_path_count,
        },
        "subject_entity_id": f"subject-{memory_id}",
        "predicate_type": "RELATED_TO",
        "memory_type": "ENTITY_CONTEXT",
        "status": "ACTIVE",
    }


def test_duplicate_memory_from_vector_and_structured_becomes_one_candidate() -> None:
    result = merge_and_rerank_candidates(
        [vector_candidate("memory-1", 0.7)],
        [structured_candidate("memory-1", same_subject=True)],
    )

    assert len(result) == 1
    assert result[0].memory_id == "memory-1"


def test_signals_from_both_sources_are_preserved() -> None:
    result = merge_and_rerank_candidates(
        [vector_candidate("memory-1", 0.7)],
        [structured_candidate("memory-1", same_subject=True, shared_entity_count=2, same_predicate=True)],
    )

    candidate = result[0]
    assert candidate.vector_similarity == 0.7
    assert candidate.structured_signals.same_subject is True
    assert candidate.structured_signals.shared_entity_count == 2
    assert candidate.structured_signals.same_predicate is True
    assert candidate.retrieval_sources == [SOURCE_VECTOR, SOURCE_STRUCTURED]


def test_strong_structural_overlap_can_promote_candidate() -> None:
    result = merge_and_rerank_candidates(
        [
            vector_candidate("vector-high", 0.82),
            vector_candidate("structural", 0.55),
        ],
        [
            structured_candidate(
                "structural",
                same_subject=True,
                shared_entity_count=2,
                same_predicate=True,
                same_memory_type=True,
            )
        ],
    )

    assert result[0].memory_id == "structural"
    assert result[0].rerank_score > result[1].rerank_score


def test_candidates_remain_sorted_by_rerank_score() -> None:
    result = merge_and_rerank_candidates(
        [
            vector_candidate("a", 0.5),
            vector_candidate("b", 0.9),
            vector_candidate("c", 0.7),
        ],
        [],
    )

    scores = [candidate.rerank_score for candidate in result]
    assert scores == sorted(scores, reverse=True)


def test_final_top_k_respected() -> None:
    result = merge_and_rerank_candidates(
        [vector_candidate(str(index), 1.0 - index / 100) for index in range(5)],
        [],
        final_top_k=2,
    )

    assert len(result) == 2


def test_vector_only_and_structured_only_cases_work() -> None:
    result = merge_and_rerank_candidates(
        [vector_candidate("vector-only", 0.6)],
        [structured_candidate("structured-only", same_subject=True)],
    )

    by_id = {candidate.memory_id: candidate for candidate in result}
    assert by_id["vector-only"].retrieval_sources == [SOURCE_VECTOR]
    assert by_id["structured-only"].retrieval_sources == [SOURCE_STRUCTURED]
    assert by_id["structured-only"].vector_similarity is None


def test_graph_only_candidate_merges_and_contributes_to_score() -> None:
    result = merge_and_rerank_candidates(
        [vector_candidate("weak-vector", 0.05)],
        [],
        [graph_candidate("graph-only")],
    )

    by_id = {candidate.memory_id: candidate for candidate in result}
    assert by_id["graph-only"].retrieval_sources == [SOURCE_GRAPH]
    assert by_id["graph-only"].graph_signals.graph_score == 1.0
    assert by_id["graph-only"].rerank_score == 0.20
    assert result[0].memory_id == "graph-only"


def test_graph_signals_are_preserved_when_sources_overlap() -> None:
    result = merge_and_rerank_candidates(
        [vector_candidate("memory-1", 0.7)],
        [structured_candidate("memory-1", same_subject=True)],
        [graph_candidate("memory-1", graph_score=1.0)],
    )

    candidate = result[0]
    assert candidate.retrieval_sources == [SOURCE_VECTOR, SOURCE_STRUCTURED, SOURCE_GRAPH]
    assert candidate.graph_signals.graph_distance == "DIRECT"
    assert candidate.rerank_score == pytest.approx(1.15)


def test_direct_graph_signal_wins_over_expanded_signal() -> None:
    result = merge_and_rerank_candidates(
        [],
        [],
        [
            graph_candidate("memory-1", graph_distance="ONE_EXPANSION", graph_score=0.5, bridge_path_count=3),
            graph_candidate("memory-1", graph_distance="DIRECT", graph_score=1.0),
        ],
    )

    assert len(result) == 1
    assert result[0].graph_signals.graph_distance == "DIRECT"
    assert result[0].graph_signals.graph_score == 1.0


def test_empty_inputs_work() -> None:
    assert merge_and_rerank_candidates([], []) == []


@pytest.mark.parametrize("memory_id", ["", "   ", None])
def test_invalid_memory_ids_fail_cleanly(memory_id) -> None:
    with pytest.raises(ValueError, match="memory_id"):
        merge_and_rerank_candidates([vector_candidate(memory_id, 0.5)], [])


@pytest.mark.parametrize("final_top_k", [0, -1, 101])
def test_invalid_final_top_k_fails_cleanly(final_top_k: int) -> None:
    with pytest.raises(ValueError, match="final_top_k"):
        merge_and_rerank_candidates([], [], final_top_k=final_top_k)


def test_duplicate_memory_ids_within_one_source_are_merged() -> None:
    result = merge_and_rerank_candidates(
        [
            vector_candidate("memory-1", 0.6),
            vector_candidate("memory-1", 0.9, canonical_text="Better vector"),
        ],
        [],
    )

    assert len(result) == 1
    assert result[0].retrieval_sources == [SOURCE_VECTOR]
    assert result[0].vector_similarity == 0.9


def test_no_writes_or_external_model_calls_occur(monkeypatch) -> None:
    def fail_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers") or "psycopg" in name:
            raise AssertionError(f"unexpected external import: {name}")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)

    result = merge_and_rerank_candidates([vector_candidate("memory-1", 0.8)], [])

    assert result[0].memory_id == "memory-1"
