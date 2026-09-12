from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kivi_memory.entity_resolution import find_entity_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Print ranked Kivi entity candidates.")
    parser.add_argument("--mention", required=True)
    parser.add_argument("--type", dest="entity_type")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    try:
        candidates = find_entity_candidates(args.mention, args.entity_type, args.limit)
    except Exception as exc:
        print(f"Entity candidate lookup failed: {exc}", file=sys.stderr)
        return 1

    if not candidates:
        print("No candidates found.")
        return 0

    for index, candidate in enumerate(candidates, start=1):
        print(f"{index}. {candidate.canonical_name} ({candidate.entity_type})")
        print(f"   entity_id: {candidate.entity_id}")
        print(f"   matched_text: {candidate.matched_text}")
        print(f"   match_type: {candidate.match_type}")
        print(f"   similarity_score: {candidate.similarity_score:.3f}")
        print(f"   type_match: {candidate.type_match}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
