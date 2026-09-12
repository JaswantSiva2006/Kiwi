from __future__ import annotations

import argparse
import json

from kivi_memory.writeback import MemoryWritebackWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Redis thread writeback check.")
    parser.add_argument("--thread-id", required=True)
    args = parser.parse_args()

    result = MemoryWritebackWorker().process_thread_once(args.thread_id)
    print(
        json.dumps(
            {
                "thread_id": result.thread_id,
                "status": result.status,
                "trigger": result.plan.trigger if result.plan else None,
                "first_target_stream_id": result.plan.first_target_stream_id if result.plan else None,
                "last_target_stream_id": result.plan.last_target_stream_id if result.plan else None,
                "ingestion_id": result.ingestion_id,
                "error": result.error,
                "diagnostics": result.diagnostics,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
