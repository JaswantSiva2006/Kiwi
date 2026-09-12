from __future__ import annotations

import asyncio
import argparse
import json
import re
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.common.config import load_orchestrator_config_from_env
from kivi_memory.entity_resolution.repository import connect
from kivi_memory.read_response import ReadResponseService, SarvamReadResponseClient

OUTPUT_DIR = ROOT / "evaluation"
RESULTS_PATH = OUTPUT_DIR / "results.json"
SUMMARY_PATH = OUTPUT_DIR / "results_summary.json"
CURRENT_DATETIME = datetime(2026, 9, 12, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
TIMEZONE = "Asia/Kolkata"
LOCALE = "en-IN"
EVAL_MAX_TOKENS = 8192


CaseCheck = Callable[[str], tuple[bool, str]]


def contains_all(*needles: str) -> CaseCheck:
    def check(answer: str) -> tuple[bool, str]:
        lowered = answer.casefold()
        missing = [needle for needle in needles if needle.casefold() not in lowered]
        return (not missing, f"missing: {', '.join(missing)}" if missing else "matched expected terms")

    return check


def contains_any(*needles: str) -> CaseCheck:
    def check(answer: str) -> tuple[bool, str]:
        lowered = answer.casefold()
        return (
            any(needle.casefold() in lowered for needle in needles),
            "matched one expected term" if any(needle.casefold() in lowered for needle in needles) else f"missing any of: {', '.join(needles)}",
        )

    return check


def unknown_refusal(answer: str) -> tuple[bool, str]:
    lowered = answer.casefold()
    refusal = any(term in lowered for term in ("don't have", "do not have", "no record", "not enough", "cannot find", "don't see"))
    invented = any(term in lowered for term in ("taj", "oberoi", "marriott", "hyatt", "leela", "ritz"))
    return refusal and not invented, "refused to invent" if refusal and not invented else "did not clearly refuse or invented hotel"


CASES: list[dict[str, Any]] = [
    {
        "id": 1,
        "query": "Who leads Helios now, and what is Arjun's role?",
        "expected_behavior": "Return current Helios lead and Arjun's actual/current role; avoid stale superseded owner facts.",
        "check": contains_all("Helios", "Arjun"),
    },
    {
        "id": 2,
        "query": "What happened to my usual badminton schedule?",
        "expected_behavior": "Explain that the usual badminton schedule moved from Sunday morning to Thursday evening.",
        "check": contains_all("badminton", "Thursday", "Sunday"),
    },
    {
        "id": 3,
        "query": "Do I still have that Bengaluru trip planned?",
        "expected_behavior": "Answer from current memory, preserving whether it is planned/booked/cancelled/uncertain.",
        "check": contains_any("Bengaluru", "Bangalore"),
    },
    {
        "id": 4,
        "query": "When is the Atlas launch review now?",
        "expected_behavior": "Return the current Atlas launch review time, not a stale superseded time.",
        "check": contains_all("Atlas", "review"),
    },
    {
        "id": 5,
        "query": "Where do I keep project decisions, and where do Atlas action items go?",
        "expected_behavior": "Return both storage/location preferences distinctly.",
        "check": contains_all("project decisions", "Atlas", "action"),
    },
    {
        "id": 6,
        "query": "Do I prefer all meetings in the morning?",
        "expected_behavior": "Do not overgeneralize; answer whether preference applies to all meetings or only scoped meetings.",
        "check": contains_any("not all", "don't prefer all", "do not prefer all", "only", "depends"),
    },
    {
        "id": 7,
        "query": "Am I definitely applying to the Orion research program?",
        "expected_behavior": "Preserve uncertainty; do not turn uncertain/intended application into definite fact.",
        "check": contains_any("not definitely", "might", "considering", "uncertain", "not confirmed"),
    },
    {
        "id": 8,
        "query": "Did I already buy the Pixel, or did I only decide I prefer it?",
        "expected_behavior": "Distinguish purchase from preference; do not invent a completed purchase.",
        "check": contains_all("Pixel", "prefer"),
    },
    {
        "id": 9,
        "query": "What is still missing before the Atlas review package is complete?",
        "expected_behavior": "Return the missing prerequisite for the Atlas review package.",
        "check": contains_all("Atlas", "review package"),
    },
    {
        "id": 10,
        "query": "Where are my probability, brain teaser, and market microstructure notes?",
        "expected_behavior": "Return locations for all three note categories if stored.",
        "check": contains_all("probability", "brain teaser", "market microstructure"),
    },
    {
        "id": 11,
        "query": "What is my normal Payments review schedule, and was there any exception?",
        "expected_behavior": "Return normal Wednesday 2 PM Payments review and the Thursday 11 AM exception.",
        "check": contains_all("Payments", "Wednesday", "2 PM", "Thursday", "11"),
    },
    {
        "id": 12,
        "query": "Which cities might I travel to next month? Have I booked anything?",
        "expected_behavior": "List possible cities without treating possibilities as bookings; preserve uncertainty.",
        "check": contains_any("might", "possible", "possibility", "not booked", "haven't booked", "no booking"),
    },
    {
        "id": 13,
        "query": "Who is actually responsible for Orion?",
        "expected_behavior": "Return current Orion responsibility, not stale conflicting responsibility.",
        "check": contains_all("Orion"),
    },
    {
        "id": 14,
        "query": "Is the Phoenix demo definitely on Monday?",
        "expected_behavior": "Preserve uncertainty/attribution; do not state Monday as definite if not confirmed.",
        "check": contains_any("not definite", "not confirmed", "might", "uncertain", "said"),
    },
    {
        "id": 15,
        "query": "What hotel did I book for Bengaluru?",
        "expected_behavior": "Unknown-information case; pass only if Kivi refuses to invent a hotel.",
        "check": unknown_refusal,
    },
]


async def main() -> None:
    args = parse_args()
    cases = CASES[: args.limit] if args.limit else CASES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = replace(load_orchestrator_config_from_env(), hey_kivi_max_tokens=EVAL_MAX_TOKENS)
    service = ReadResponseService(config=config, final_answer_client=SarvamReadResponseClient(config=config))
    results = []
    started_all = time.perf_counter()

    for case in cases:
        print(f"[{case['id']:02d}/{len(cases)}] {case['query']} -> running...", flush=True)
        started = time.perf_counter()
        error = None
        response = None
        try:
            response = await service.handle_user_query(
                thread_id=f"evaluation-{case['id']:02d}-{uuid4()}",
                user_query=case["query"],
                current_datetime=CURRENT_DATETIME,
                timezone=TIMEZONE,
                locale=LOCALE,
            )
            passed, reason = case["check"](response.text)
        except Exception as exc:
            passed = False
            reason = f"exception: {type(exc).__name__}: {exc}"
            error = str(exc)
        latency_ms = (time.perf_counter() - started) * 1000
        evidence = extract_evidence(response.context if response else "")
        memory_ids = sorted({item.get("memory_id") for item in evidence["retrieved_memories"] if item.get("memory_id")})
        memory_ids.extend(
            sorted({item.get("semantic_memory_id") for item in evidence["calendar_events"] if item.get("semantic_memory_id")})
        )
        ledger_state = load_ledger_state(sorted(set(memory_ids)))
        record = {
            "case_id": case["id"],
            "query": case["query"],
            "expected_behavior": case["expected_behavior"],
            "actual_answer": response.text if response else None,
            "tools_called": [call["tool"] for call in (response.route.get("tool_calls", []) if response else [])],
            "tool_calls": response.route.get("tool_calls", []) if response else [],
            "retrieved_memories": evidence["retrieved_memories"],
            "calendar_events": evidence["calendar_events"],
            "provenance_source_record_ids": sorted(evidence["record_ids"]),
            "ledger_state": ledger_state,
            "latency_ms": latency_ms,
            "diagnostics": response.diagnostics if response else {},
            "pass": passed,
            "failure_reason": None if passed else reason,
            "error": error,
        }
        results.append(record)
        print(f"[{case['id']:02d}/{len(cases)}] -> {'PASS' if passed else 'FAIL'} ({latency_ms:.0f} ms) {reason}", flush=True)
        write_outputs(results, started_all)

    write_outputs(results, started_all)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real Hey Kivi read-pipeline evaluation queries.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases, for smoke testing.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def write_outputs(results: list[dict[str, Any]], started_all: float) -> None:
    failed = [
        {"case_id": item["case_id"], "query": item["query"], "reason": item["failure_reason"]}
        for item in results
        if not item["pass"]
    ]
    summary = {
        "total": len(results),
        "passed": sum(1 for item in results if item["pass"]),
        "failed": len(failed),
        "pass_rate": (sum(1 for item in results if item["pass"]) / len(results)) if results else 0.0,
        "average_latency_ms": (sum(item["latency_ms"] for item in results) / len(results)) if results else 0.0,
        "failed_cases": failed,
        "total_runtime_ms": (time.perf_counter() - started_all) * 1000,
        "current_datetime": CURRENT_DATETIME.isoformat(),
        "timezone": TIMEZONE,
        "hey_kivi_max_tokens": EVAL_MAX_TOKENS,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def extract_evidence(context: str) -> dict[str, Any]:
    memories = []
    calendar_events = []
    record_ids: set[str] = set()
    semantic_section = section(context, "SEMANTIC_MEMORY")
    for line in semantic_section.splitlines():
        line = line.strip()
        if not line or line == "(none)":
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        memories.append(item)
        record_ids.update(record_ids_from_item(item))

    calendar_section = section(context, "CALENDAR")
    if calendar_section and calendar_section not in {"(none)", "[]"}:
        try:
            payload = json.loads(calendar_section)
            events = payload.get("events", payload if isinstance(payload, list) else [])
            if isinstance(events, list):
                calendar_events = events
                for event in events:
                    record_ids.update(record_ids_from_item(event))
        except json.JSONDecodeError:
            pass
    return {"retrieved_memories": memories, "calendar_events": calendar_events, "record_ids": record_ids}


def section(context: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", context, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def record_ids_from_item(item: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(item, dict):
        for key, value in item.items():
            if key == "episode_id" and isinstance(value, str) and value.startswith("corpus-"):
                found.add(value.removeprefix("corpus-"))
            else:
                found.update(record_ids_from_item(value))
    elif isinstance(item, list):
        for value in item:
            found.update(record_ids_from_item(value))
    return found


def load_ledger_state(memory_ids: list[str]) -> list[dict[str, Any]]:
    if not memory_ids:
        return []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    sm.memory_id::text,
                    sm.canonical_text,
                    sm.memory_type,
                    sm.predicate_type,
                    sm.status,
                    sm.version,
                    sm.created_at,
                    sm.updated_at,
                    sm.temporal_kind,
                    sm.recurrence,
                    sm.recurrence_specifics,
                    COALESCE(json_agg(DISTINCT me.episode_id) FILTER (WHERE me.episode_id IS NOT NULL), '[]') AS source_episode_ids,
                    COALESCE(json_agg(DISTINCT mev.event_type) FILTER (WHERE mev.event_type IS NOT NULL), '[]') AS ledger_events
                FROM semantic_memories sm
                LEFT JOIN memory_evidence me ON me.memory_id = sm.memory_id
                LEFT JOIN memory_events mev ON mev.memory_id = sm.memory_id
                WHERE sm.memory_id = ANY(%s)
                GROUP BY sm.memory_id
                ORDER BY sm.created_at DESC
                """,
                (memory_ids,),
            )
            return [dict(row) for row in cur.fetchall()]


if __name__ == "__main__":
    asyncio.run(main())
