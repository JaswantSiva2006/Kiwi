from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kivi_memory.embeddings import backfill_active_memory_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill derived memory embeddings for ACTIVE ledger memories.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        result = backfill_active_memory_embeddings(batch_size=args.batch_size)
    except Exception as exc:
        print(f"Embedding backfill failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(result), indent=2))
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
