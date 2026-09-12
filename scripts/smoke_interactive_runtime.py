from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.common.config import load_orchestrator_config_from_env
from kivi_memory.entity_resolution.repository import connect
from kivi_memory.llm_redis_orchestrator.controller import TurnController, TurnControllerError
from kivi_memory.llm_redis_orchestrator.final_agent import HeyKiviSarvamAgent, get_shared_sarvam_client
from kivi_memory.working_memory import ThreadMemoryStore


class RecordingSarvamClient:
    """Wrap the real Sarvam client so the smoke test can inspect the final prompt."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.payloads: list[dict[str, Any]] = []
        self.call_ms: list[float] = []
        self.chat = self

    def completions(self, **kwargs: Any) -> Any:
        self.payloads.append(kwargs)
        started = time.perf_counter()
        try:
            return self.inner.chat.completions(**kwargs)
        finally:
            self.call_ms.append(_elapsed_ms(started))


CASES = {
    "thread_only": {
        "prior": [{"role": "user", "text": 'Explain recursion simply using "tiny mirrors".'}],
        "current": "Give me an example using that phrase.",
        "expects_memory": False,
        "expected_terms": ["tiny mirrors"],
    },
    "direct_memory": {
        "prior": [],
        "current": "Who handles Project Atlas?",
        "expects_memory": True,
        "expected_terms": ["rohit"],
    },
    "redis_bridge_memory": {
        "prior": [
            {"role": "user", "text": "What is Priya working on?"},
            {"role": "assistant", "text": "Priya is working on Project Phoenix."},
        ],
        "current": "What technology does it use?",
        "expects_memory": True,
        "expected_terms": ["redis"],
    },
    "general_knowledge": {
        "prior": [],
        "current": "Explain binary search in two sentences.",
        "expects_memory": False,
        "expected_terms": ["sorted"],
    },
    "scoped_preference": {
        "prior": [],
        "current": "For research discussions, how do I like explanations?",
        "expects_memory": True,
        "expected_terms": ["detailed"],
    },
    "unsupported_history": {
        "prior": [],
        "current": "What did I decide about Project Orion?",
        "expects_memory": True,
        "expected_terms": ["not enough", "don't have", "do not have", "no memory"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real Hey Kivi interactive-runtime smoke turns.")
    parser.add_argument("--cases", nargs="*", choices=sorted(CASES), default=["thread_only", "direct_memory", "redis_bridge_memory", "general_knowledge"])
    parser.add_argument("--thread-prefix", default=f"hey-kivi-smoke-{uuid4()}")
    parser.add_argument("--output", default="test_outputs/interactive_runtime_smoke.json")
    args = parser.parse_args()

    config = load_orchestrator_config_from_env()
    store = ThreadMemoryStore()
    db_counts_before = _ledger_counts()
    results = []
    started = time.perf_counter()

    for case_name in args.cases:
        case = CASES[case_name]
        thread_id = f"{args.thread_prefix}-{case_name}"
        store.clear_thread(thread_id)
        for index, message in enumerate(case["prior"], start=1):
            store.append_message(
                thread_id,
                {
                    "message_id": f"{thread_id}-seed-{index}",
                    "role": message["role"],
                    "text": message["text"],
                },
            )
        seeded_assistant_count = sum(1 for message in store.get_all_messages(thread_id) if message.role == "assistant")

        recorder = RecordingSarvamClient(get_shared_sarvam_client(config))
        controller = TurnController(
            thread_store=store,
            response_agent=HeyKiviSarvamAgent(client=recorder, config=config),
        )

        case_started = time.perf_counter()
        try:
            result = controller.handle_user_turn(
                thread_id=thread_id,
                text=case["current"],
                message_id=f"{thread_id}-user-current",
            )
            messages = store.get_all_messages(thread_id)
            prompt_payload = recorder.payloads[-1] if recorder.payloads else {}
            prompt_user_content = _prompt_user_content(prompt_payload)
            answer = result.assistant_message.text
            checks = _checks(
                answer=answer,
                prompt_user_content=prompt_user_content,
                expects_memory=case["expects_memory"],
                expected_terms=case["expected_terms"],
                result=result,
            )
            results.append(
                {
                    "case": case_name,
                    "current_query": case["current"],
                    "expects_memory": case["expects_memory"],
                    "status": "completed",
                    "answer": answer,
                    "checks": checks,
                    "turn_result": asdict(result),
                    "redis_thread_roles": [message.role for message in messages],
                    "redis_thread_message_count": len(messages),
                    "new_assistant_appended": sum(1 for message in messages if message.role == "assistant") > seeded_assistant_count,
                    "final_model_payload": _payload_summary(prompt_payload),
                    "sarvam_call_ms": recorder.call_ms[-1] if recorder.call_ms else None,
                    "case_total_ms": _elapsed_ms(case_started),
                }
            )
        except TurnControllerError as exc:
            messages = store.get_all_messages(thread_id)
            prompt_payload = recorder.payloads[-1] if recorder.payloads else {}
            results.append(
                {
                    "case": case_name,
                    "current_query": case["current"],
                    "expects_memory": case["expects_memory"],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "error_cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
                    "error_cause": str(exc.__cause__) if exc.__cause__ else None,
                    "redis_thread_roles": [message.role for message in messages],
                    "redis_thread_message_count": len(messages),
                    "new_assistant_appended": sum(1 for message in messages if message.role == "assistant") > seeded_assistant_count,
                    "final_model_payload": _payload_summary(prompt_payload),
                    "sarvam_call_ms": recorder.call_ms[-1] if recorder.call_ms else None,
                    "case_total_ms": _elapsed_ms(case_started),
                }
            )

    output = {
        "config": {
            "router_model": config.router_model,
            "hey_kivi_provider": config.hey_kivi_provider,
            "hey_kivi_model": config.hey_kivi_model,
            "hey_kivi_reasoning_effort": config.hey_kivi_reasoning_effort,
            "hey_kivi_temperature": config.hey_kivi_temperature,
            "hey_kivi_max_tokens": config.hey_kivi_max_tokens,
            "hey_kivi_timeout_seconds": config.hey_kivi_timeout_seconds,
            "agent_max_memories": config.agent_max_memories,
        },
        "db_counts_before": db_counts_before,
        "db_counts_after": _ledger_counts(),
        "total_ms": _elapsed_ms(started),
        "results": results,
    }
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(output), indent=2), encoding="utf-8")
    print(json.dumps(_summary(output), indent=2))
    print(f"wrote {output_path}")


def _checks(
    *,
    answer: str,
    prompt_user_content: str,
    expects_memory: bool,
    expected_terms: list[str],
    result: Any,
) -> dict[str, Any]:
    lowered_answer = answer.lower()
    lowered_prompt = prompt_user_content.lower()
    return {
        "memory_section_expected": expects_memory,
        "memory_section_present": "long-term memory" in lowered_prompt,
        "current_query_once": prompt_user_content.count(result.user_message.text) == 1,
        "retrieval_expectation_met": bool(result.memory_preparation.needs_long_term_memory) is expects_memory,
        "expected_answer_term_present": any(term in lowered_answer for term in expected_terms),
        "assistant_appended": result.assistant_message.message_id is not None,
    }


def _summary(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": output["config"],
        "db_counts_before": output["db_counts_before"],
        "db_counts_after": output["db_counts_after"],
        "total_ms": output["total_ms"],
        "cases": [
            _case_summary(item)
            for item in output["results"]
        ],
    }


def _case_summary(item: dict[str, Any]) -> dict[str, Any]:
    if item["status"] == "failed":
        return {
            "case": item["case"],
            "status": item["status"],
            "error_type": item["error_type"],
            "error_cause_type": item["error_cause_type"],
            "error_cause": item["error_cause"],
            "new_assistant_appended": item["new_assistant_appended"],
            "redis_thread_roles": item["redis_thread_roles"],
            "case_total_ms": item["case_total_ms"],
            "sarvam_call_ms": item["sarvam_call_ms"],
            "final_model": item["final_model_payload"]["model"],
            "reasoning_effort": item["final_model_payload"]["reasoning_effort"],
        }
    return {
                "case": item["case"],
                "status": item["status"],
                "answer": item["answer"],
                "checks": item["checks"],
                "needs_long_term_memory": item["turn_result"]["memory_preparation"]["needs_long_term_memory"],
                "retrieved_memory_count": item["turn_result"]["diagnostics"]["retrieved_memory_count"],
                "retrieval_queries": item["turn_result"]["memory_preparation"]["retrieval_queries"],
                "diagnostics": item["turn_result"]["diagnostics"],
                "sarvam_call_ms": item["sarvam_call_ms"],
                "final_model": item["final_model_payload"]["model"],
                "reasoning_effort": item["final_model_payload"]["reasoning_effort"],
            }


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": payload.get("model"),
        "reasoning_effort": payload.get("reasoning_effort"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "stream": payload.get("stream"),
        "system_prompt": _prompt_system_content(payload),
        "user_prompt": _prompt_user_content(payload),
    }


def _prompt_system_content(payload: dict[str, Any]) -> str:
    for message in payload.get("messages") or []:
        if message.get("role") == "system":
            return str(message.get("content") or "")
    return ""


def _prompt_user_content(payload: dict[str, Any]) -> str:
    for message in payload.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _ledger_counts() -> dict[str, int]:
    tables = ["semantic_memories", "memory_arguments", "memory_evidence", "memory_events", "memory_embeddings"]
    counts = {}
    with connect() as conn:
        with conn.cursor() as cur:
            for table in tables:
                cur.execute(f"SELECT count(*) AS count FROM {table}")
                counts[table] = int(cur.fetchone()["count"])
    return counts


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


if __name__ == "__main__":
    main()
