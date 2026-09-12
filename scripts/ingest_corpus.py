#!/usr/bin/env python3
"""Ingest an ASR corpus through the existing Kivi MemoryPipeline.

Flow:
    ASR corpus JSONL
      -> convert_jsonl_file(...)
      -> result.episodes
      -> MemoryPipeline.process(episode)
      -> semantic-memory backend

This script intentionally does not recreate corpus conversion, does not use
Redis writeback, and does not write directly to SQL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kivi_memory.corpus_import import convert_jsonl_file
from kivi_memory.common.config import load_config_from_env
from kivi_memory.pipeline import MemoryPipeline

REPORT_PATH = ROOT / "evaluation" / "corpus_ingestion_report.json"
CONVERTED_PATH = ROOT / "evaluation" / "converted_memory_episodes.jsonl"
LEDGER_OPS = ("ADD", "REINFORCE", "SUPERSEDE", "RETRACT")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"FATAL: FileNotFoundError: {input_path}", file=sys.stderr)
        return 2

    print("Converting corpus to MemoryEpisode objects...")
    conversion = convert_jsonl_file(input_path, CONVERTED_PATH)
    if conversion.errors:
        write_report(
            {
                "complete": False,
                "input": str(input_path),
                "converted_output": str(CONVERTED_PATH),
                "conversion_errors": conversion.errors,
                "records": [],
            }
        )
        print("FATAL: corpus conversion failed")
        print(json.dumps(conversion.errors, indent=2, ensure_ascii=False))
        return 2

    episodes = sorted(conversion.episodes, key=episode_sort_key)
    start = args.start_from - 1
    stop = len(episodes) if args.limit is None else min(len(episodes), start + args.limit)
    selected = episodes[start:stop]
    if not selected:
        print("FATAL: no episodes selected for ingestion", file=sys.stderr)
        return 2

    print(f"Input:              {input_path}")
    print(f"Converted episodes: {CONVERTED_PATH}")
    print(f"Episodes available: {len(episodes)}")
    print(f"Episodes selected:  {len(selected)}")
    print("Pipeline:           kivi_memory.pipeline.MemoryPipeline")
    print()

    pipeline = MemoryPipeline()
    rows: list[dict[str, Any]] = []

    for offset, episode in enumerate(selected, start=1):
        source_index = start + offset
        record_id = episode_record_id(episode)
        episode_id = str(episode.episode_id)
        prefix = f"[{offset:03d}/{len(selected):03d}] {record_id}"
        print(f"{prefix} \u2192 ingesting...", flush=True)

        row: dict[str, Any] = {
            "source_index": source_index,
            "record_id": record_id,
            "memory_episode_id": episode_id,
            "success": False,
            "assertion_count": 0,
            "reconciliation_operations": {op: 0 for op in LEDGER_OPS},
            "memory_ids": [],
            "latency_seconds": None,
            "error": None,
        }

        episode_started = time.perf_counter()
        try:
            result = pipeline.process(episode)
            latency = time.perf_counter() - episode_started
            output = result.output
            assertions = output.get("assertions", [])
            ops, memory_ids = summarize_mutations(assertions)
            assertion_rows = summarize_assertions(assertions)

            row.update(
                {
                    "success": True,
                    "assertion_count": len(assertions),
                    "assertions": assertion_rows,
                    "reconciliation_operations": {op: ops[op] for op in LEDGER_OPS},
                    "memory_ids": memory_ids,
                    "provenance_source_ids": provenance_source_ids(assertion_rows),
                    "model_usage": model_usage(result),
                    "latency_seconds": round(latency, 4),
                    "pipeline_errors": assertion_errors(assertions),
                }
            )
            rows.append(row)

            print(f"{prefix} \u2192 SUCCESS")
            print(f"           assertions: {len(assertions)}")
            print(f"           ledger ops: {format_ops(ops)}")
            print(f"           latency: {latency:.2f}s")
        except Exception as exc:
            latency = time.perf_counter() - episode_started
            exact_error = f"{type(exc).__name__}: {exc}"
            row.update(
                {
                    "success": False,
                    "latency_seconds": round(latency, 4),
                    "error": exact_error,
                }
            )
            rows.append(row)

            print(f"{prefix} \u2192 FAILED")
            print(f"           error: {exact_error}")
            print(f"           latency: {latency:.2f}s")
            if args.debug_tracebacks:
                traceback.print_exc()

            write_current_report(input_path, rows, started, complete=False)
            if not args.continue_on_error:
                print(f"\nStopped on failure. Resume with --start-from {source_index}.")
                break
        finally:
            write_current_report(input_path, rows, started, complete=False)

    write_current_report(input_path, rows, started, complete=len(rows) == len(selected))
    print_totals(rows, started)
    return 0 if all(row.get("success") for row in rows) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a Kivi ASR corpus through MemoryPipeline.")
    parser.add_argument("--input", required=True, help="ASR corpus JSONL input path.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N episodes.")
    parser.add_argument("--start-from", type=int, default=1, help="1-based chronological episode index.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after an episode failure.")
    parser.add_argument("--debug-tracebacks", action="store_true", help="Print tracebacks in addition to exact errors.")
    args = parser.parse_args()
    if args.start_from < 1:
        parser.error("--start-from must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def episode_sort_key(episode: Any) -> tuple[str, str]:
    eligible = target_message(episode)
    timestamp = getattr(eligible, "timestamp", "") if eligible is not None else ""
    return (str(timestamp), str(episode.episode_id))


def episode_record_id(episode: Any) -> str:
    eligible = target_message(episode)
    if eligible is not None:
        record_id = getattr(eligible, "source_thread_episode_id", None)
        if record_id:
            return str(record_id)
    episode_id = str(episode.episode_id)
    return episode_id.removeprefix("corpus-")


def target_message(episode: Any) -> Any | None:
    for message in episode.messages:
        if getattr(message, "memory_eligible", False):
            return message
    return episode.messages[0] if episode.messages else None


def summarize_mutations(assertions: list[dict[str, Any]]) -> tuple[Counter, list[str]]:
    ops: Counter = Counter()
    memory_ids: list[str] = []
    seen: set[str] = set()
    for assertion in assertions:
        mutation = assertion.get("mutation_result") or {}
        op = str(mutation.get("op", "")).upper()
        if op in LEDGER_OPS and mutation.get("executed", False):
            ops[op] += 1
        for key in ("created_memory_id", "target_memory_ids"):
            value = mutation.get(key)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item is None:
                    continue
                memory_id = str(item)
                if memory_id and memory_id not in seen:
                    seen.add(memory_id)
                    memory_ids.append(memory_id)
    return ops, memory_ids


def summarize_assertions(assertions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for assertion in assertions:
        semantic = assertion.get("semantic_assertion") or {}
        mutation = assertion.get("mutation_result") or {}
        reconciliation = assertion.get("reconciliation_result") or {}
        evidence = semantic.get("source_spans") or []
        rows.append(
            {
                "index": assertion.get("index"),
                "canonical_text": semantic.get("canonical_text"),
                "memory_type": semantic.get("memory_type"),
                "predicate_type": semantic.get("predicate_type"),
                "reconciliation_op": mutation.get("op") or reconciliation.get("op"),
                "executed": mutation.get("executed"),
                "created_memory_id": mutation.get("created_memory_id"),
                "target_memory_ids": mutation.get("target_memory_ids") or reconciliation.get("target_memory_ids") or [],
                "affected_memory_ids": affected_memory_ids(mutation, reconciliation),
                "source_spans": evidence,
                "provenance_source_ids": sorted(source_id_from_span(span) for span in evidence if source_id_from_span(span)),
                "errors": assertion.get("errors", []),
            }
        )
    return rows


def affected_memory_ids(mutation: dict[str, Any], reconciliation: dict[str, Any]) -> list[str]:
    ids = []
    for value in [
        mutation.get("created_memory_id"),
        *(mutation.get("target_memory_ids") or []),
        *(reconciliation.get("target_memory_ids") or []),
    ]:
        if value and str(value) not in ids:
            ids.append(str(value))
    return ids


def source_id_from_span(span: dict[str, Any]) -> str | None:
    message_id = str(span.get("message_id") or "")
    if message_id.startswith("corpus-") and message_id.endswith("-user"):
        return message_id.removeprefix("corpus-").removesuffix("-user")
    return None


def provenance_source_ids(assertions: list[dict[str, Any]]) -> list[str]:
    ids = set()
    for assertion in assertions:
        ids.update(assertion.get("provenance_source_ids") or [])
    return sorted(ids)


def model_usage(result: Any) -> dict[str, Any]:
    config = load_config_from_env()
    output = result.output if hasattr(result, "output") else {}
    return {
        "semantic_compiler_model": getattr(config, "model", None),
        "semantic_compiler_think": getattr(config, "think", None),
        "temporal_reasoning_model": getattr(config, "temporal_reasoning_model", None),
        "temporal_normalization_model": getattr(config, "temporal_normalization_model", None),
        "pipeline_model_fields": output.get("model_usage") or output.get("models") or {},
    }


def assertion_errors(assertions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for assertion in assertions:
        for error in assertion.get("errors", []):
            errors.append({"assertion_index": assertion.get("index"), **error})
    return errors


def format_ops(ops: Counter) -> str:
    return " ".join(f"{op}={ops[op]}" for op in LEDGER_OPS)


def write_current_report(input_path: Path, rows: list[dict[str, Any]], started: float, *, complete: bool) -> None:
    op_totals = Counter()
    for row in rows:
        if row.get("success"):
            op_totals.update(row.get("reconciliation_operations", {}))
    latencies = [row["latency_seconds"] for row in rows if row.get("latency_seconds") is not None]
    payload = {
        "complete": complete,
        "input": str(input_path),
        "converted_output": str(CONVERTED_PATH),
        "report_path": str(REPORT_PATH),
        "summary": {
            "processed": len(rows),
            "successful": sum(1 for row in rows if row.get("success")),
            "failed": sum(1 for row in rows if not row.get("success")),
            "operations": {op: op_totals[op] for op in LEDGER_OPS},
            "total_runtime_seconds": round(time.perf_counter() - started, 4),
            "average_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        },
        "records": rows,
    }
    write_report(payload)


def write_report(payload: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_totals(rows: list[dict[str, Any]], started: float) -> None:
    op_totals = Counter()
    for row in rows:
        if row.get("success"):
            op_totals.update(row.get("reconciliation_operations", {}))
    elapsed = time.perf_counter() - started
    latencies = [row["latency_seconds"] for row in rows if row.get("latency_seconds") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    processed = len(rows)
    successful = sum(1 for row in rows if row.get("success"))
    failed = processed - successful

    print()
    print("Totals")
    print(f"processed: {processed}")
    print(f"successful: {successful}")
    print(f"failed: {failed}")
    print(f"ADD: {op_totals['ADD']}")
    print(f"REINFORCE: {op_totals['REINFORCE']}")
    print(f"SUPERSEDE: {op_totals['SUPERSEDE']}")
    print(f"RETRACT: {op_totals['RETRACT']}")
    print(f"total runtime: {elapsed:.2f}s")
    print(f"average latency: {avg_latency:.2f}s")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
