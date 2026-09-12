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

from kivi_memory.llm_redis_orchestrator.controller import TurnController
from kivi_memory.working_memory import ThreadMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate the Kivi turn controller lifecycle.")
    parser.add_argument("--thread-id", default="turn-controller-demo")
    parser.add_argument("--text", default="What technology does it use?")
    parser.add_argument("--message-id", default="turn-demo-user")
    args = parser.parse_args()

    store = ThreadMemoryStore()
    store.clear_thread(args.thread_id)
    store.append_message(args.thread_id, {"message_id": "demo-1", "role": "user", "text": "What is Priya working on?"})
    store.append_message(
        args.thread_id,
        {"message_id": "demo-2", "role": "assistant", "text": "Priya is working on Project Phoenix."},
    )

    result = TurnController(thread_store=store).handle_user_turn(
        thread_id=args.thread_id,
        text=args.text,
        message_id=args.message_id,
    )
    print(json.dumps(asdict(result), indent=2, default=_json_default))
    print(
        json.dumps(
            {"redis_thread_messages": [message.to_dict() for message in store.get_all_messages(args.thread_id)]},
            indent=2,
        )
    )


def _json_default(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


if __name__ == "__main__":
    main()
