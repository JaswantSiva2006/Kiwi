from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.embeddings import EMBEDDING_MODEL_NAME, embed_text
from kivi_memory.embeddings.config import get_hnsw_ef_search
from kivi_memory.embeddings.repository import MemoryEmbeddingRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Kivi pgvector/HNSW retrieval health.")
    parser.add_argument("--query", default="Riya owns the billing dashboard.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    repo = MemoryEmbeddingRepository()
    health = repo.vector_health(EMBEDDING_MODEL_NAME)
    query_embedding = embed_text(args.query)
    explain = repo.explain_hnsw_query(
        query_embedding,
        EMBEDDING_MODEL_NAME,
        args.top_k,
        get_hnsw_ef_search(args.top_k),
    )
    print(json.dumps({"health": health, "explain": explain}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
