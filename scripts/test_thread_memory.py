from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.working_memory import ThreadMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate Redis within-thread working memory.")
    parser.add_argument("--thread-a", default="dev-thread-a")
    parser.add_argument("--thread-b", default="dev-thread-b")
    args = parser.parse_args()

    store = ThreadMemoryStore()
    store.clear_thread(args.thread_a)
    store.clear_thread(args.thread_b)

    store.append_message(args.thread_a, {"message_id": "a1", "role": "user", "text": "What is Priya working on?"})
    store.append_message(
        args.thread_a,
        {"message_id": "a2", "role": "assistant", "text": "Priya is currently working on Project Phoenix."},
    )
    store.append_message(args.thread_a, {"message_id": "a3", "role": "user", "text": "What technology does it use?"})

    store.append_message(args.thread_b, {"message_id": "b1", "role": "user", "text": "Remind me about badminton."})
    store.append_message(args.thread_b, {"message_id": "b2", "role": "assistant", "text": "You play on Sunday mornings."})

    output = {
        "thread_a_recent_messages": [message.to_dict() for message in store.get_recent_messages(args.thread_a)],
        "thread_b_recent_messages": [message.to_dict() for message in store.get_recent_messages(args.thread_b)],
        "thread_a_metadata": store.get_thread_metadata(args.thread_a),
        "thread_b_metadata": store.get_thread_metadata(args.thread_b),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
