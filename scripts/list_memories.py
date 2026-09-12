from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kivi_memory.ledger import list_memories


def main() -> int:
    parser = argparse.ArgumentParser(description="List Canonical Memory Ledger summaries.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--status")
    parser.add_argument("--memory-type")
    parser.add_argument("--subject-entity-id")
    args = parser.parse_args()

    try:
        memories = list_memories(
            limit=args.limit,
            status=args.status,
            memory_type=args.memory_type,
            subject_entity_id=args.subject_entity_id,
        )
    except Exception as exc:
        print(f"List memories failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(memories, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
