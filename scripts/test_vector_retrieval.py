from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kivi_memory.embeddings import retrieve_vector_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Test read-only vector memory retrieval.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        result = retrieve_vector_candidates(args.query, args.top_k)
    except Exception as exc:
        print(f"Vector retrieval failed: {exc}", file=sys.stderr)
        return 1

    print(f"embedding_ms: {result.embedding_ms:.2f}")
    print(f"db_search_ms: {result.db_search_ms:.2f}")
    print(f"total_ms: {result.total_ms:.2f}")
    print(f"search_mode: {result.vector_search_mode}")
    print(f"returned_count: {result.vector_returned_count}")
    print(f"hnsw_ef_search: {result.hnsw_ef_search}")
    print(f"hnsw_iterative_scan: {result.hnsw_iterative_scan}")
    print(f"hnsw_fallback_used: {result.hnsw_fallback_used}")
    print(f"hnsw_fallback_reason: {result.hnsw_fallback_reason}")
    for rank, candidate in enumerate(result.candidates, start=1):
        print(f"{rank}. {candidate.memory_id}")
        print(f"   similarity: {candidate.vector_similarity:.4f}")
        print(f"   canonical_text: {candidate.canonical_text}")
        print(f"   subject_entity_id: {candidate.subject_entity_id}")
        print(f"   predicate_type: {candidate.predicate_type}")
        print(f"   memory_type: {candidate.memory_type}")
        print(f"   status: {candidate.status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
