from __future__ import annotations

from kivi_memory.embeddings import EMBEDDING_MODEL_NAME
from kivi_memory.embeddings.repository import MemoryEmbeddingRepository


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def cursor(self):
        return self.cursor_obj


class FakeCursor:
    def __init__(self):
        self.sql = []
        self.params = []
        self.result = [
            {
                "memory_id": "memory-1",
                "canonical_text": "Riya owns the billing dashboard.",
                "vector_similarity": 1.0,
                "subject_entity_id": None,
                "predicate_type": "OWNS",
                "memory_type": "PROJECT_GOAL_TOPIC",
                "status": "ACTIVE",
            }
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def execute(self, sql, params=()):
        self.sql.append(sql)
        self.params.append(params)

    def fetchall(self):
        return self.result


def test_hnsw_query_uses_cosine_operator_active_filter_model_filter_and_local_settings(monkeypatch) -> None:
    connection = FakeConnection()
    monkeypatch.setattr("kivi_memory.embeddings.repository.connect", lambda database_url=None: connection)

    result = MemoryEmbeddingRepository().retrieve_vector_candidates_hnsw([1.0, 0.0], EMBEDDING_MODEL_NAME, 5, 100)

    joined_sql = "\n".join(connection.cursor_obj.sql)
    assert "<=>" in joined_sql
    assert "ORDER BY me.embedding <=>" in joined_sql
    assert "sm.status = 'ACTIVE'" in joined_sql
    assert "me.embedding_model = %s" in joined_sql
    assert "set_config('hnsw.ef_search'" in joined_sql
    assert "set_config('hnsw.iterative_scan'" in joined_sql
    assert connection.cursor_obj.params[0] == ("100",)
    assert connection.cursor_obj.params[1] == ("strict_order",)
    assert result[0].memory_id == "memory-1"


def test_exact_query_disables_indexscan_locally(monkeypatch) -> None:
    connection = FakeConnection()
    monkeypatch.setattr("kivi_memory.embeddings.repository.connect", lambda database_url=None: connection)

    MemoryEmbeddingRepository().retrieve_vector_candidates_exact([1.0, 0.0], EMBEDDING_MODEL_NAME, 5)

    joined_sql = "\n".join(connection.cursor_obj.sql)
    assert "SET LOCAL enable_indexscan = off" in joined_sql
