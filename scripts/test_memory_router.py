from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.llm_redis_orchestrator import prepare_long_term_memory_context
from kivi_memory.working_memory import ThreadMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug current-thread to long-term-memory routing.")
    parser.add_argument("--thread-id", default="memory-router-demo")
    parser.add_argument("--query", default="What technology does it use?")
    parser.add_argument("--current-message-id", default="current")
    args = parser.parse_args()

    store = ThreadMemoryStore()
    store.clear_thread(args.thread_id)
    store.append_message(args.thread_id, {"message_id": "m1", "role": "user", "text": "What is Priya working on?"})
    store.append_message(args.thread_id, {"message_id": "m2", "role": "assistant", "text": "Priya is working on Project Phoenix."})
    store.append_message(args.thread_id, {"message_id": args.current_message_id, "role": "user", "text": args.query})

    result = prepare_long_term_memory_context(
        thread_id=args.thread_id,
        current_query=args.query,
        current_message_id=args.current_message_id,
        thread_memory_store=store,
    )
    print(json.dumps(asdict(result), indent=2, default=_json_default))


def _json_default(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


if __name__ == "__main__":
    main()
