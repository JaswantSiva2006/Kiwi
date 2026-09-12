from __future__ import annotations

from kivi_memory.long_term_retrieval.repository import LongTermRetrievalRepository, _dedupe_entity_matches


def test_query_entity_matching_collapses_duplicate_evidence_for_same_entity() -> None:
    rows = [
        row("entity-1", "Priya", "exact_name", 1.0),
        row("entity-1", "Priya", "exact_alias", 1.0),
    ]

    matches = _dedupe_entity_matches(rows)

    assert len(matches) == 1
    assert matches[0].entity_id == "entity-1"
    assert matches[0].match_method == "exact_name"


def test_ambiguous_exact_entity_query_match_is_not_used_as_seed() -> None:
    rows = [
        row("entity-1", "Priya", "exact_alias", 1.0),
        row("entity-2", "Priya", "exact_alias", 1.0),
    ]

    assert _dedupe_entity_matches(rows) == []


def test_graph_branch_reuses_existing_graph_repository(monkeypatch) -> None:
    calls = []

    class FakeGraphRepository:
        def __init__(self, database_url=None):
            self.database_url = database_url

        def retrieve_candidates(self, *, seed_entity_ids, max_bridge_degree, top_k):
            calls.append((seed_entity_ids, max_bridge_degree, top_k, self.database_url))
            return [
                type(
                    "GraphCandidate",
                    (),
                    {
                        "memory_id": "memory-1",
                        "graph_signals": type(
                            "Signals",
                            (),
                            {"graph_score": 1.0, "graph_distance": "DIRECT", "direct_seed_count": 1, "bridge_path_count": 0},
                        )(),
                    },
                )()
            ]

    monkeypatch.setattr("kivi_memory.long_term_retrieval.repository.GraphMemoryRepository", FakeGraphRepository)

    candidates = LongTermRetrievalRepository("postgresql://example").retrieve_graph_candidates(["entity-1"], 5)

    assert calls == [(["entity-1"], 100, 5, "postgresql://example")]
    assert candidates[0].source == "GRAPH"
    assert candidates[0].rank == 1
    assert candidates[0].diagnostics["graph_distance"] == "DIRECT"


def row(entity_id: str, matched_text: str, method: str, score: float) -> dict:
    return {
        "entity_id": entity_id,
        "canonical_name": f"name {entity_id}",
        "entity_type": "PERSON",
        "matched_text": matched_text,
        "match_method": method,
        "score": score,
    }
