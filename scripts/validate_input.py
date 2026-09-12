from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError

from kivi_memory.common.io import load_memory_episode


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_input.py input.json")
        return 2

    try:
        episode = load_memory_episode(Path(sys.argv[1]))
    except (OSError, ValidationError, ValueError) as exc:
        print(f"Input validation failed: {exc}")
        return 1

    print(f"Input valid: {episode.episode_id} ({len(episode.messages)} messages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
