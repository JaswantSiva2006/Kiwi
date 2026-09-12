from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kivi_memory.entity_resolution import create_entity


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Kivi entity and canonical alias.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--type", dest="entity_type", required=True)
    parser.add_argument("--source-episode-id")
    args = parser.parse_args()

    try:
        entity = create_entity(args.name, args.entity_type, args.source_episode_id)
    except Exception as exc:
        print(f"Create entity failed: {exc}", file=sys.stderr)
        return 1

    print(f"Created entity: {entity.canonical_name} ({entity.entity_type})")
    print(f"entity_id: {entity.entity_id}")
    print(f"normalized_name: {entity.normalized_name}")
    print("canonical_alias: created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
