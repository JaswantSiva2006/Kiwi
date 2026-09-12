from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.long_term_retrieval import retrieve_long_term_memories

DEFAULT_QUERIES = [
    "Who handles Project Atlas?",
    "What technology does Project Phoenix use?",
    "What is Priya working on?",
    "What technology is connected to Priya's project?",
    "Where does Kavya work?",
    "What do we know about Rohit?",
    "Kafka event processing",
    "Goa December 2026",
    "What does Neha support?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug read-only long-term memory retrieval.")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--branch-k", type=int, default=30)
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    for index, query in enumerate(queries, start=1):
        result = retrieve_long_term_memories(query, top_k=args.top_k, branch_k=args.branch_k)
        print(f"\nQUERY {index}: {query}")
        print(f"resolved_entities: {[entity.canonical_name for entity in result.resolved_query_entities]}")
        for rank, memory in enumerate(result.memories, start=1):
            print(
                json.dumps(
                    {
                        "final_rank": rank,
                        "memory_id": memory.memory_id,
                        "canonical_text": memory.canonical_text,
                        "rrf_score": round(memory.rrf_score, 6),
                        "retrieval_sources": memory.retrieval_sources,
                        "vector_rank": memory.branch_ranks["vector"],
                        "lexical_rank": memory.branch_ranks["lexical"],
                        "structured_rank": memory.branch_ranks["structured"],
                        "graph_rank": memory.branch_ranks["graph"],
                    },
                    ensure_ascii=True,
                )
            )
        print(f"total_retrieval_latency_ms: {result.diagnostics.total_ms:.2f}")
        if index == 4:
            for source, candidates in result.branch_candidates.items():
                print(f"raw_{source.lower()}_top:")
                for candidate in candidates[:10]:
                    print(json.dumps(asdict(candidate), ensure_ascii=True))


def _json_default(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    return str(value)


if __name__ == "__main__":
    main()
