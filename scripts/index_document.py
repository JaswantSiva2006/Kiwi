from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.document_rag.ingestion import ingest_parsed_document


def main() -> int:
    parser = argparse.ArgumentParser(description="Index parsed PDF chunks into isolated Document RAG storage.")
    parser.add_argument("--input", required=True, help="Path to validated JSON from PDF_parse")
    args = parser.parse_args()

    parsed = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = ingest_parsed_document(parsed)
    print(f"document_id={result.document_id}")
    print(f"chunks_indexed={result.chunks_indexed}")
    print(f"duplicate={str(result.duplicate).lower()}")
    print(f"embedding_time={result.embedding_time:.4f}")
    print(f"db_write_time={result.db_write_time:.4f}")
    print(f"total_time={result.total_time:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
