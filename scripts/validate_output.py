from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError

from kivi_memory.common.io import load_compiler_output, load_memory_episode
from kivi_memory.enrichment.validator import validate_compiler_output


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/validate_output.py input.json output.json")
        return 2

    try:
        episode = load_memory_episode(Path(sys.argv[1]))
        output = load_compiler_output(Path(sys.argv[2]))
    except (OSError, ValidationError, ValueError) as exc:
        print(f"Validation failed while loading input/output: {exc}")
        return 1

    report = validate_compiler_output(output, episode)
    print(f"Output valid: {report.valid}")
    for issue in report.issues:
        location = f" assertion={issue.assertion_index}" if issue.assertion_index is not None else ""
        print(f"{issue.severity}: {issue.code}{location}: {issue.message}")

    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
