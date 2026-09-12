from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.entity_resolution.repository import connect

MIGRATION = ROOT / "migrations" / "20260906_memory_embeddings_hnsw.sql"


def main() -> int:
    sql = MIGRATION.read_text(encoding="utf-8")
    with connect() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute("ANALYZE public.memory_embeddings")
    print(f"Applied {MIGRATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
