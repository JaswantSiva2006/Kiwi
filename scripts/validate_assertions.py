from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError

from kivi_memory.common.io import load_compiler_output, load_memory_episode
from kivi_memory.enrichment.validator import validate_assertions


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate candidate semantic assertions.")
    parser.add_argument("--input", default="input.json", type=Path)
    parser.add_argument("--output", default="output.json", type=Path)
    args = parser.parse_args()

    try:
        episode = load_memory_episode(args.input)
        output = load_compiler_output(args.output)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"Validation failed while loading input/output: {exc}")
        return 1

    results = validate_assertions(output.assertions, episode)
    warning_count = 0
    valid_count = 0
    for index, result in enumerate(results, start=1):
        report = result.validation_report
        warning_count += sum(1 for issue in report.issues if issue.severity == "WARNING")
        valid_count += int(report.valid)
        if report.valid and report.issues:
            status = "PASS WITH WARNINGS"
        else:
            status = "PASS" if report.valid else "FAIL"
        print(f"Assertion {index}: {status}")
        for issue in report.issues:
            print(f"  - {issue.code}")

    print()
    print("Summary:")
    print(f"{len(results)} assertions")
    print(f"{valid_count} valid")
    print(f"{len(results) - valid_count} invalid")
    print(f"{warning_count} warning")
    return 0 if valid_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
