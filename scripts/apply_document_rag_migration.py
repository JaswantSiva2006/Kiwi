from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.document_rag.database import connect

MIGRATION = ROOT / "migrations" / "20260914_document_rag.sql"


def main() -> int:
    sql = MIGRATION.read_text(encoding="utf-8")
    with connect() as conn:
        with conn.cursor() as cur:
            for statement in _split_sql(sql):
                cur.execute(statement)
            cur.execute("ANALYZE document_rag.chunks")
    print(f"Applied {MIGRATION}")
    return 0


def _split_sql(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
