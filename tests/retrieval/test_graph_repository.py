from __future__ import annotations

from kivi_memory.retrieval.graph import GraphMemoryRepository


class FakeConnection:
    def cursor(self):
        return FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None


class FakeCursor:
    calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, sql, params):
        self.__class__.calls.append((sql, params))

    def fetchall(self):
        return [
            {
                "memory_id": "m1",
                "canonical_text": "Priya handles Atlas.",
                "subject_entity_id": "e1",
                "predicate_type": "RESPONSIBLE_FOR",
                "memory_type": "PROJECT_GOAL_TOPIC",
                "status": "ACTIVE",
                "graph_distance": "DIRECT",
                "graph_score": 1.0,
                "direct_seed_count": 1,
                "bridge_path_count": 0,
            }
        ]


def test_graph_repository_uses_one_bounded_active_sql_query(monkeypatch) -> None:
    FakeCursor.calls = []
    monkeypatch.setattr("kivi_memory.retrieval.graph.connect", lambda database_url=None: FakeConnection())

    result = GraphMemoryRepository().retrieve_candidates(
        seed_entity_ids=["00000000-0000-0000-0000-000000000001"],
        max_bridge_degree=100,
        top_k=30,
    )

    assert len(result) == 1
    assert len(FakeCursor.calls) == 1
    sql, params = FakeCursor.calls[0]
    normalized_sql = " ".join(sql.lower().split())
    assert "with seed_entities as" in normalized_sql
    assert "direct as" in normalized_sql
    assert "expanded as" in normalized_sql
    assert "degree_memory.status = 'active'" in normalized_sql
    assert "sm.status = 'active'" in normalized_sql
    assert "limit %s" in normalized_sql
    assert params == (["00000000-0000-0000-0000-000000000001"], ["00000000-0000-0000-0000-000000000001"], 100, 30)
