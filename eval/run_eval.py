from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kivi_memory.semantic_compiler.compiler import SemanticCompiler, SemanticCompilerError
from kivi_memory.common.io import load_memory_episode, write_compiler_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kivi semantic compiler eval episodes.")
    parser.add_argument("--episodes-dir", type=Path, default=ROOT / "eval" / "episodes")
    parser.add_argument("--expected-dir", type=Path, default=ROOT / "eval" / "expected")
    parser.add_argument("--actual-dir", type=Path, default=ROOT / "eval" / "results")
    args = parser.parse_args()

    args.actual_dir.mkdir(parents=True, exist_ok=True)
    compiler = SemanticCompiler()
    total = 0
    passed = 0
    latencies: list[int] = []

    for episode_path in sorted(args.episodes_dir.glob("*.json")):
        total += 1
        expected_path = args.expected_dir / episode_path.name
        expected = json.loads(expected_path.read_text(encoding="utf-8")) if expected_path.exists() else {}

        try:
            episode = load_memory_episode(episode_path)
            result = compiler.compile(episode)
            actual_path = args.actual_dir / episode_path.name
            write_compiler_output(result.output, actual_path)
            ok, notes = check_expectations(result.output.model_dump(), expected)
            latencies.append(result.inference_duration_ms)
        except (OSError, SemanticCompilerError, ValueError) as exc:
            ok = False
            notes = [str(exc)]

        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {episode_path.stem}: {'; '.join(notes) if notes else 'ok'}")

    if latencies:
        print(f"Latency avg: {sum(latencies) // len(latencies)} ms")
    print(f"Eval summary: {passed}/{total} passed lightweight checks")
    return 0 if passed == total else 1


def check_expectations(actual: dict, expected: dict) -> tuple[bool, list[str]]:
    notes: list[str] = []
    assertions = actual.get("assertions", [])

    expected_count = expected.get("expected_assertion_count")
    if expected_count is not None and len(assertions) != expected_count:
        notes.append(f"expected {expected_count} assertions, got {len(assertions)}")

    memory_types = {assertion.get("memory_type") for assertion in assertions}
    for memory_type in expected.get("must_contain_memory_type", []):
        if memory_type not in memory_types:
            notes.append(f"missing memory_type {memory_type}")

    text_blob = json.dumps(actual, sort_keys=True)
    for forbidden in expected.get("must_not_contain_text", []):
        if forbidden in text_blob:
            notes.append(f"contains forbidden text {forbidden!r}")

    if expected.get("require_valid_source_spans", True):
        for index, assertion in enumerate(assertions):
            if not assertion.get("source_spans"):
                notes.append(f"assertion {index} has no source spans")

    return not notes, notes


if __name__ == "__main__":
    raise SystemExit(main())
