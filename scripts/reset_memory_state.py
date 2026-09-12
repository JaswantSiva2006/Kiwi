from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.entity_resolution.repository import connect

TABLES = [
    "calendar_event_entities",
    "calendar_events",
    "memory_embeddings",
    "memory_entity_links",
    "memory_events",
    "memory_evidence",
    "memory_arguments",
    "semantic_memories",
    "entity_aliases",
    "entities",
]

REPORTS = [
    ROOT / "evaluation" / "corpus_ingestion_report.json",
    ROOT / "evaluation" / "converted_memory_episodes.jsonl",
    ROOT / "evaluation" / "results.json",
    ROOT / "evaluation" / "results_summary.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset Kivi semantic-memory tables for a clean corpus re-run.")
    parser.add_argument("--yes", action="store_true", help="Required confirmation.")
    parser.add_argument("--keep-reports", action="store_true", help="Do not remove generated evaluation/report files.")
    args = parser.parse_args()
    if not args.yes:
        print("Refusing to reset without --yes", file=sys.stderr)
        return 2

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE")
        conn.commit()
    print("Reset PostgreSQL semantic-memory tables.")

    if not args.keep_reports:
        for path in REPORTS:
            if path.exists():
                path.unlink()
                print(f"Removed {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
