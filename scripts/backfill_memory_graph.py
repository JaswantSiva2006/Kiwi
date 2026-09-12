from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.graph import backfill_active_memory_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill memory_entity_links for ACTIVE ledger memories.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = backfill_active_memory_graph(limit=args.limit)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
