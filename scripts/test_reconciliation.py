from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.reconciliation import ReconciliationFailure, reconcile_memory


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug one read-only Kivi reconciliation decision.")
    parser.add_argument("--pipeline-output", required=True, help="Pipeline output JSON containing incoming assertions.")
    parser.add_argument("--assertion-index", type=int, default=0)
    parser.add_argument("--candidate-memory-id", action="append", default=[], help="Existing memory UUID candidate, in rank order.")
    args = parser.parse_args()

    data = json.loads(Path(args.pipeline_output).read_text(encoding="utf-8"))
    try:
        incoming = data["assertions"][args.assertion_index]
    except (KeyError, IndexError) as exc:
        raise SystemExit(f"assertion not found: {args.assertion_index}") from exc

    candidates = [{"memory_id": memory_id} for memory_id in args.candidate_memory_id]
    try:
        decision = reconcile_memory(incoming, candidates)
    except ReconciliationFailure as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2

    print(
        json.dumps(
            {
                "decision": decision.op.value,
                "target_memory_ids": decision.target_memory_ids,
                "candidate_hydration_ms": decision.candidate_hydration_ms,
                "formatting_ms": decision.formatting_ms,
                "llm_ms": decision.llm_ms,
                "total_ms": decision.total_ms,
                "prompt_eval_count": decision.prompt_eval_count,
                "eval_count": decision.eval_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
