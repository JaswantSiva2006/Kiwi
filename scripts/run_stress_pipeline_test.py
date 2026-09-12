from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.common.schemas import MemoryEpisode
from kivi_memory.embeddings.config import EMBEDDING_MODEL_NAME
from kivi_memory.entity_resolution import create_entity
from kivi_memory.entity_resolution.repository import connect
from kivi_memory.pipeline import MemoryPipeline

OUTPUT_DIR = ROOT / "test_outputs"
LEDGER_OUTPUT = OUTPUT_DIR / "stress_test_ledger.json"
GRAPH_OUTPUT = OUTPUT_DIR / "stress_test_graph.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the clean full Kivi pipeline stress test.")
    parser.add_argument("--input", default=ROOT / "kivi_pipeline_stress_test_v1.json", type=Path)
    args = parser.parse_args()

    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Inspecting schema dependencies...")
    print(json.dumps(inspect_fk_dependencies(), indent=2))

    print("Cleaning semantic-memory test state...")
    clean_counts = clean_database()
    print("Row counts after cleanup:")
    print(json.dumps(clean_counts, indent=2))

    user_entity = create_entity("user", "person", source_episode_id=None)
    print(f"Bootstrapped current-user entity: {user_entity.entity_id}")

    pipeline = MemoryPipeline()
    run_started = time.perf_counter()
    all_episode_outputs: list[dict[str, Any]] = []
    total_messages = 0

    for raw_episode in fixture["episodes"]:
        episode = episode_from_fixture(raw_episode)
        total_messages += len(episode.messages)
        print(f"\n=== Episode {episode.episode_id} ===")
        result = pipeline.process(episode)
        all_episode_outputs.append(result.output)
        print_episode_trace(result.output)

    ledger = export_ledger()
    graph = export_graph()
    LEDGER_OUTPUT.write_text(json.dumps(ledger, indent=2, default=str) + "\n", encoding="utf-8")
    GRAPH_OUTPUT.write_text(json.dumps(graph, indent=2, default=str) + "\n", encoding="utf-8")

    summary = summarize_run(
        total_messages=total_messages,
        episode_outputs=all_episode_outputs,
        total_runtime_ms=(time.perf_counter() - run_started) * 1000,
    )
    invariants = check_invariants(all_episode_outputs)

    print("\n=== Final Validation ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Invariants ===")
    print(json.dumps(invariants, indent=2))
    print(f"\nLedger output: {LEDGER_OUTPUT}")
    print(f"Graph output: {GRAPH_OUTPUT}")
    return 0 if all(item["ok"] for item in invariants.values()) else 1


def inspect_fk_dependencies() -> list[dict[str, Any]]:
    sql = """
        SELECT
            tc.table_name,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public'
          AND (
            tc.table_name IN (
                'memory_entity_links',
                'memory_embeddings',
                'memory_arguments',
                'memory_evidence',
                'memory_events',
                'semantic_memories',
                'entity_aliases',
                'entities'
            )
            OR ccu.table_name IN ('semantic_memories', 'entities')
          )
        ORDER BY tc.table_name, kcu.column_name
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]


def clean_database() -> dict[str, int]:
    tables = [
        "memory_entity_links",
        "memory_embeddings",
        "memory_arguments",
        "memory_evidence",
        "memory_events",
        "semantic_memories",
        "entity_aliases",
        "entities",
    ]
    with connect() as conn:
        with conn.cursor() as cur:
            for table in tables:
                cur.execute(f"DELETE FROM {table}")
            counts = row_counts(cur, tables)
    return counts


def row_counts(cur, tables: list[str]) -> dict[str, int]:
    counts = {}
    for table in tables:
        cur.execute(f"SELECT count(*)::int AS count FROM {table}")
        counts[table] = int(cur.fetchone()["count"])
    return counts


def episode_from_fixture(raw_episode: dict[str, Any]) -> MemoryEpisode:
    timestamp = raw_episode["timestamp"]
    return MemoryEpisode.model_validate(
        {
            "episode_id": raw_episode["episode_id"],
            "timezone": "Asia/Kolkata",
            "messages": [
                {
                    "message_id": message["message_id"],
                    "role": message["role"].upper(),
                    "timestamp": timestamp,
                    "text": message["content"],
                }
                for message in raw_episode["messages"]
            ],
        }
    )


def print_episode_trace(output: dict[str, Any]) -> None:
    for item in output["assertions"]:
        assertion = item["semantic_assertion"]
        source_ids = [span["message_id"] for span in assertion.get("source_spans", [])]
        entity_resolution = item.get("entity_resolution") or {}
        subject = entity_resolution.get("subject") or {}
        mutation = item.get("mutation_result") or {}
        decision = item.get("reconciliation_decision") or {}
        top_candidates = [
            {
                "memory_id": candidate.get("memory_id"),
                "sources": candidate.get("retrieval_sources"),
                "score": round(float(candidate.get("rerank_score") or 0.0), 4),
                "status": candidate.get("status"),
            }
            for candidate in item.get("final_candidates", [])[:3]
        ]
        print(
            json.dumps(
                {
                    "source_message_id": ",".join(source_ids),
                    "canonical_text": assertion.get("canonical_text"),
                    "temporal_route": item.get("temporal_route"),
                    "subject_entity_id": subject.get("entity_id"),
                    "top_candidates": top_candidates,
                    "reconciliation_op": decision.get("op"),
                    "target_memory_ids": mutation.get("target_memory_ids"),
                    "created_memory_id": mutation.get("created_memory_id"),
                    "mutation_success": not item.get("errors") and bool(mutation),
                    "embedding_projection_ok": _projection_ok(mutation, "embedding"),
                    "graph_projection_ok": _projection_ok(mutation, "graph"),
                    "errors": item.get("errors"),
                },
                ensure_ascii=True,
            )
        )


def _projection_ok(mutation: dict[str, Any], projection: str) -> bool | None:
    if not mutation or not mutation.get("created_memory_id"):
        return None
    return mutation.get(f"{projection}_projection_error") is None


def export_ledger() -> list[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    memory_id::text AS memory_id,
                    canonical_text,
                    status,
                    version,
                    subject_text,
                    subject_entity_id::text AS subject_entity_id,
                    subject_entity_type,
                    predicate_type,
                    memory_type,
                    modality,
                    polarity,
                    certainty,
                    explicitness,
                    attributed_to,
                    temporal_kind,
                    valid_from,
                    valid_to,
                    event_time,
                    temporal_precision,
                    recurrence,
                    recurrence_specifics,
                    created_at,
                    updated_at
                FROM semantic_memories
                ORDER BY created_at, memory_id
                """
            )
            memories = [dict(row) for row in cur.fetchall()]
            for memory in memories:
                memory_id = memory["memory_id"]
                cur.execute(
                    """
                    SELECT
                        role,
                        text,
                        is_entity,
                        entity_id::text AS entity_id,
                        entity_type,
                        position
                    FROM memory_arguments
                    WHERE memory_id = %s
                    ORDER BY position
                    """,
                    (memory_id,),
                )
                memory["arguments"] = [dict(row) for row in cur.fetchall()]
                cur.execute(
                    """
                    SELECT episode_id, message_id, source_text, observed_at, created_at
                    FROM memory_evidence
                    WHERE memory_id = %s
                    ORDER BY created_at, evidence_id
                    """,
                    (memory_id,),
                )
                memory["evidence"] = [dict(row) for row in cur.fetchall()]
    return memories


def export_graph() -> dict[str, Any]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    mel.memory_id::text AS memory_id,
                    sm.canonical_text AS memory_canonical_text,
                    sm.status AS memory_status,
                    mel.entity_id::text AS entity_id,
                    e.canonical_name AS entity_canonical_name,
                    e.entity_type,
                    mel.link_type,
                    mel.argument_role,
                    mel.predicate_type,
                    mel.position
                FROM memory_entity_links mel
                JOIN semantic_memories sm ON sm.memory_id = mel.memory_id
                JOIN entities e ON e.entity_id = mel.entity_id
                ORDER BY e.canonical_name, mel.link_type, mel.position NULLS FIRST, sm.created_at
                """
            )
            links = [dict(row) for row in cur.fetchall()]

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        adjacency.setdefault(link["entity_canonical_name"], []).append(
            {
                "memory_id": link["memory_id"],
                "canonical_text": link["memory_canonical_text"],
                "memory_status": link["memory_status"],
                "link_type": link["link_type"],
                "argument_role": link["argument_role"],
                "predicate_type": link["predicate_type"],
                "position": link["position"],
            }
        )
    return {"links": links, "adjacency": adjacency}


def summarize_run(
    *,
    total_messages: int,
    episode_outputs: list[dict[str, Any]],
    total_runtime_ms: float,
) -> dict[str, Any]:
    op_counts = Counter()
    assertions = 0
    hnsw_calls = structured_calls = graph_calls = graph_only_useful = 0
    for output in episode_outputs:
        for item in output["assertions"]:
            assertions += 1
            decision = item.get("reconciliation_decision") or {}
            op_counts[decision.get("op") or "ERROR"] += 1
            if item.get("vector_retrieval"):
                hnsw_calls += 1
            if "structured_retrieval_ms" in item.get("latency", {}):
                structured_calls += 1
            if item.get("graph_retrieval"):
                graph_calls += 1
            vector_ids = {candidate["memory_id"] for candidate in item.get("vector_candidates", [])}
            structured_ids = {candidate["memory_id"] for candidate in item.get("structured_candidates", [])}
            for candidate in item.get("final_candidates", []):
                sources = candidate.get("retrieval_sources") or []
                if "GRAPH" in sources and candidate["memory_id"] not in vector_ids and candidate["memory_id"] not in structured_ids:
                    graph_only_useful += 1

    db_counts = final_counts()
    return {
        "input_messages": total_messages,
        "semantic_assertions_produced": assertions,
        "ADD_count": op_counts["ADD"],
        "REINFORCE_count": op_counts["REINFORCE"],
        "SUPERSEDE_count": op_counts["SUPERSEDE"],
        "RETRACT_count": op_counts["RETRACT"],
        "NO_MEMORY_count": op_counts["NO_MEMORY"],
        "ERROR_count": op_counts["ERROR"],
        "ACTIVE_memories": db_counts["active_memories"],
        "SUPERSEDED_memories": db_counts["superseded_memories"],
        "RETRACTED_memories": db_counts["retracted_memories"],
        "entity_count": db_counts["entities"],
        "graph_link_count": db_counts["graph_links"],
        "embedding_count": db_counts["embeddings"],
        "HNSW_retrieval_calls": hnsw_calls,
        "structured_retrieval_calls": structured_calls,
        "graph_retrieval_calls": graph_calls,
        "graph_only_useful_candidates_observed": graph_only_useful,
        "total_pipeline_runtime_ms": round(total_runtime_ms, 3),
    }


def final_counts() -> dict[str, int]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*)::int AS count FROM semantic_memories WHERE status = 'ACTIVE'")
            active = cur.fetchone()["count"]
            cur.execute("SELECT count(*)::int AS count FROM semantic_memories WHERE status = 'SUPERSEDED'")
            superseded = cur.fetchone()["count"]
            cur.execute("SELECT count(*)::int AS count FROM semantic_memories WHERE status = 'RETRACTED'")
            retracted = cur.fetchone()["count"]
            cur.execute("SELECT count(*)::int AS count FROM entities")
            entities = cur.fetchone()["count"]
            cur.execute("SELECT count(*)::int AS count FROM memory_entity_links")
            graph_links = cur.fetchone()["count"]
            cur.execute("SELECT count(*)::int AS count FROM memory_embeddings WHERE embedding_model = %s", (EMBEDDING_MODEL_NAME,))
            embeddings = cur.fetchone()["count"]
    return {
        "active_memories": active,
        "superseded_memories": superseded,
        "retracted_memories": retracted,
        "entities": entities,
        "graph_links": graph_links,
        "embeddings": embeddings,
    }


def check_invariants(episode_outputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)::int AS count
                FROM semantic_memories sm
                LEFT JOIN memory_embeddings me
                  ON me.memory_id = sm.memory_id
                 AND me.embedding_model = %s
                WHERE sm.status = 'ACTIVE'
                  AND me.memory_id IS NULL
                """,
                (EMBEDDING_MODEL_NAME,),
            )
            missing_embeddings = cur.fetchone()["count"]

            cur.execute(
                """
                WITH expected AS (
                    SELECT memory_id, subject_entity_id AS entity_id, 'SUBJECT'::text AS link_type, NULL::text AS argument_role, predicate_type, NULL::int AS position
                    FROM semantic_memories
                    WHERE status = 'ACTIVE' AND subject_entity_id IS NOT NULL
                    UNION ALL
                    SELECT ma.memory_id, ma.entity_id, 'ARGUMENT'::text, ma.role, sm.predicate_type, ma.position
                    FROM memory_arguments ma
                    JOIN semantic_memories sm ON sm.memory_id = ma.memory_id
                    WHERE sm.status = 'ACTIVE' AND ma.is_entity = true AND ma.entity_id IS NOT NULL
                )
                SELECT count(*)::int AS count
                FROM expected e
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM memory_entity_links mel
                    WHERE mel.memory_id = e.memory_id
                      AND mel.entity_id = e.entity_id
                      AND mel.link_type = e.link_type
                      AND COALESCE(mel.argument_role, '') = COALESCE(e.argument_role, '')
                      AND mel.predicate_type = e.predicate_type
                      AND COALESCE(mel.position, -1) = COALESCE(e.position, -1)
                )
                """
            )
            missing_graph_links = cur.fetchone()["count"]

            cur.execute(
                """
                SELECT count(*)::int AS count
                FROM memory_entity_links mel
                LEFT JOIN semantic_memories sm ON sm.memory_id = mel.memory_id
                LEFT JOIN entities e ON e.entity_id = mel.entity_id
                WHERE sm.memory_id IS NULL OR e.entity_id IS NULL
                """
            )
            dangling_graph_links = cur.fetchone()["count"]

            cur.execute(
                """
                SELECT count(*)::int AS count
                FROM (
                    SELECT memory_id, entity_id, link_type, COALESCE(argument_role, '') AS argument_role, predicate_type, COALESCE(position, -1) AS position
                    FROM memory_entity_links
                    GROUP BY memory_id, entity_id, link_type, COALESCE(argument_role, ''), predicate_type, COALESCE(position, -1)
                    HAVING count(*) > 1
                ) duplicate_links
                """
            )
            duplicate_graph_links = cur.fetchone()["count"]

            cur.execute(
                """
                SELECT count(*)::int AS count
                FROM (
                    SELECT memory_id, embedding_model
                    FROM memory_embeddings
                    GROUP BY memory_id, embedding_model
                    HAVING count(*) > 1
                ) duplicate_embeddings
                """
            )
            duplicate_embeddings = cur.fetchone()["count"]

    non_active_retrievals = []
    projection_errors = []
    for output in episode_outputs:
        for item in output["assertions"]:
            for key in ["vector_candidates", "structured_candidates", "graph_candidates", "final_candidates"]:
                for candidate in item.get(key, []):
                    if candidate.get("status") not in {None, "ACTIVE"}:
                        non_active_retrievals.append({"source": key, "memory_id": candidate.get("memory_id"), "status": candidate.get("status")})
            mutation = item.get("mutation_result") or {}
            if mutation.get("embedding_projection_error") or mutation.get("graph_projection_error"):
                projection_errors.append(
                    {
                        "created_memory_id": mutation.get("created_memory_id"),
                        "embedding_error": mutation.get("embedding_projection_error"),
                        "graph_error": mutation.get("graph_projection_error"),
                    }
                )

    return {
        "active_memories_have_embeddings": {"ok": missing_embeddings == 0, "missing_count": missing_embeddings},
        "active_resolved_entities_have_graph_links": {"ok": missing_graph_links == 0, "missing_count": missing_graph_links},
        "graph_links_reference_existing_rows": {"ok": dangling_graph_links == 0, "dangling_count": dangling_graph_links},
        "retrieval_results_are_active": {"ok": not non_active_retrievals, "violations": non_active_retrievals[:10]},
        "no_duplicate_graph_links": {"ok": duplicate_graph_links == 0, "duplicate_count": duplicate_graph_links},
        "no_duplicate_embeddings": {"ok": duplicate_embeddings == 0, "duplicate_count": duplicate_embeddings},
        "projection_failures_did_not_stop_run": {"ok": True, "projection_errors": projection_errors},
        "clean_state_used": {"ok": True, "note": "semantic-memory tables were deleted before bootstrap and run"},
    }


if __name__ == "__main__":
    raise SystemExit(main())
