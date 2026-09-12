from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.entity_resolution.repository import connect

MIGRATION = ROOT / "migrations" / "20260906_calendar_projection.sql"


def main() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(MIGRATION.read_text(encoding="utf-8"))
    print(f"Applied {MIGRATION}")


if __name__ == "__main__":
    main()
