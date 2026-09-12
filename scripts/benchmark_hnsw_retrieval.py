from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.embeddings import EMBEDDING_MODEL_NAME, embed_text, embed_texts
from kivi_memory.embeddings.config import HNSW_ITERATIVE_SCAN
from kivi_memory.embeddings.repository import _vector_literal
from kivi_memory.entity_resolution.repository import connect

SCHEMA = "hnsw_bench"
SEED = 1337
DEFAULT_SIZES = [1000, 10000, 25000, 50000, 100000]
DEFAULT_EF_SWEEP = [40, 80, 100, 200]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark exact vs HNSW pgvector search in isolated hnsw_bench schema.")
    parser.add_argument("--sizes", default=",".join(str(size) for size in DEFAULT_SIZES))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--ef-sweep", default=",".join(str(value) for value in DEFAULT_EF_SWEEP))
    parser.add_argument("--output", type=Path, default=Path("hnsw_benchmark_output.json"))
    args = parser.parse_args()

    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    ef_sweep = [int(value) for value in args.ef_sweep.split(",") if value.strip()]
    embed_text("warm up MiniLM")
    results = []
    for size in sizes:
        setup_benchmark_schema(size, args.batch_size)
        queries = benchmark_queries(min(args.queries, size))
        result = benchmark_size(size, queries, args.ef_search)
        if size >= 50000:
            result["ef_sweep"] = [benchmark_size(size, queries, ef, include_exact=True) for ef in ef_sweep]
        results.append(result)

    args.output.write_text(json.dumps({"embedding_model": EMBEDDING_MODEL_NAME, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark output written to {args.output}")
    return 0


def setup_benchmark_schema(size: int, batch_size: int) -> None:
    rows = generate_texts(size)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            cur.execute(f"CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.memories (
                    id integer PRIMARY KEY,
                    canonical_text text NOT NULL,
                    status text NOT NULL
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.embeddings (
                    memory_id integer PRIMARY KEY REFERENCES {SCHEMA}.memories(id),
                    embedding_model text NOT NULL,
                    embedding vector(384) NOT NULL
                )
                """
            )
            cur.execute(f"TRUNCATE {SCHEMA}.embeddings, {SCHEMA}.memories")
            for start in range(0, size, batch_size):
                batch = rows[start : start + batch_size]
                embeddings = embed_texts([row[1] for row in batch])
                cur.executemany(
                    f"INSERT INTO {SCHEMA}.memories (id, canonical_text, status) VALUES (%s, %s, %s)",
                    batch,
                )
                cur.executemany(
                    f"INSERT INTO {SCHEMA}.embeddings (memory_id, embedding_model, embedding) VALUES (%s, %s, %s::vector)",
                    [
                        (row[0], EMBEDDING_MODEL_NAME, _vector_literal(embedding))
                        for row, embedding in zip(batch, embeddings, strict=True)
                    ],
                )
    with connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS hnsw_bench_embeddings_hnsw_idx
                ON {SCHEMA}.embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                WHERE embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
                """
            )
            cur.execute(f"ANALYZE {SCHEMA}.memories")
            cur.execute(f"ANALYZE {SCHEMA}.embeddings")


def benchmark_size(size: int, queries: list[tuple[int, str]], ef_search: int, include_exact: bool = True) -> dict:
    exact_latencies = []
    hnsw_latencies = []
    recalls = {1: [], 5: [], 10: [], 20: []}
    top1_matches = 0
    short_results = 0
    for _, text in queries:
        vector = embed_text(text)
        exact_ids = []
        if include_exact:
            exact_started = time.perf_counter()
            exact_ids = query_exact(vector, 20)
            exact_latencies.append((time.perf_counter() - exact_started) * 1000)
        hnsw_started = time.perf_counter()
        hnsw_ids = query_hnsw(vector, 20, ef_search)
        hnsw_latencies.append((time.perf_counter() - hnsw_started) * 1000)
        if len(hnsw_ids) < 20:
            short_results += 1
        if exact_ids:
            if hnsw_ids[:1] == exact_ids[:1]:
                top1_matches += 1
            for k in recalls:
                recalls[k].append(len(set(hnsw_ids[:k]) & set(exact_ids[:k])) / k)
    return {
        "size": size,
        "query_count": len(queries),
        "ef_search": ef_search,
        "exact_db_search_ms": latency_stats(exact_latencies) if exact_latencies else None,
        "hnsw_db_search_ms": latency_stats(hnsw_latencies),
        "speedup_mean": (
            statistics.mean(exact_latencies) / statistics.mean(hnsw_latencies)
            if exact_latencies and hnsw_latencies
            else None
        ),
        "top1_agreement": top1_matches / len(queries) if exact_ids else None,
        "recall": {
            f"recall_at_{k}": statistics.mean(values) if values else None
            for k, values in recalls.items()
        },
        "short_hnsw_results": short_results,
    }


def query_exact(vector: list[float], top_k: int) -> list[int]:
    sql = f"""
        SELECT m.id
        FROM {SCHEMA}.embeddings e
        JOIN {SCHEMA}.memories m ON m.id = e.memory_id
        WHERE m.status = 'ACTIVE'
          AND e.embedding_model = %s
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute(sql, (EMBEDDING_MODEL_NAME, _vector_literal(vector), top_k))
            return [row["id"] for row in cur.fetchall()]


def query_hnsw(vector: list[float], top_k: int, ef_search: int) -> list[int]:
    sql = f"""
        SELECT m.id
        FROM {SCHEMA}.embeddings e
        JOIN {SCHEMA}.memories m ON m.id = e.memory_id
        WHERE m.status = 'ACTIVE'
          AND e.embedding_model = %s
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
            cur.execute("SELECT set_config('hnsw.iterative_scan', %s, true)", (HNSW_ITERATIVE_SCAN,))
            cur.execute(sql, (EMBEDDING_MODEL_NAME, _vector_literal(vector), top_k))
            return [row["id"] for row in cur.fetchall()]


def latency_stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(ordered),
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
    }


def percentile(ordered: list[float], percentile_value: int) -> float:
    if not ordered:
        return 0.0
    index = min(round((percentile_value / 100) * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


def benchmark_queries(count: int) -> list[tuple[int, str]]:
    rows = generate_texts(count * 2)
    return [(row_id, text) for row_id, text, status in rows if status == "ACTIVE"][:count]


def generate_texts(size: int) -> list[tuple[int, str, str]]:
    rng = random.Random(SEED)
    people = ["Priya", "Riya", "Rohit", "Kavya", "Aarav", "Meera", "Nikhil", "Sara"]
    projects = ["Atlas", "Orion", "billing dashboard", "migration", "audit", "vendor contract"]
    verbs = [
        "{person} handles {project}.",
        "{person} owns {project}.",
        "{person} is responsible for {project}.",
        "The user likes {project}.",
        "The user pays rent every month.",
        "{project} is due next quarter.",
        "{person} moved to {project}.",
        "The team decided to prioritize {project}.",
    ]
    rows = []
    for index in range(size):
        text = (
            rng.choice(verbs).format(person=rng.choice(people), project=rng.choice(projects))
            + f" Context marker {index + 1}."
        )
        status = "ACTIVE" if index % 5 else rng.choice(["SUPERSEDED", "RETRACTED"])
        rows.append((index + 1, text, status))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
