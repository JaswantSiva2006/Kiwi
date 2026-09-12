from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kivi_memory.entity_resolution.repository import connect


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Kivi semantic memory, provenance, ledger events, and calendar projection state.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--status", default=None)
    parser.add_argument("--query", default=None, help="Case-insensitive text filter over canonical text, subject, arguments, and evidence.")
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 500:
        parser.error("--limit must be between 1 and 500")

    with connect() as conn:
        with conn.cursor() as cur:
            payload = {
                "counts": counts(cur),
                "memories": memories(cur, limit=args.limit, status=args.status, query=args.query),
            }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def counts(cur) -> dict[str, int]:
    names = [
        "semantic_memories",
        "memory_arguments",
        "memory_evidence",
        "memory_events",
        "entities",
        "entity_aliases",
        "memory_entity_links",
        "memory_embeddings",
        "calendar_events",
        "calendar_event_entities",
    ]
    result = {}
    for name in names:
        cur.execute(f"SELECT count(*) AS n FROM {name}")
        result[name] = int(cur.fetchone()["n"])
    return result


def memories(cur, *, limit: int, status: str | None, query: str | None) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []
    if status:
        clauses.append("sm.status = %s")
        params.append(status)
    if query:
        clauses.append(
            """
            (
                sm.canonical_text ILIKE %s
                OR sm.subject_text ILIKE %s
                OR EXISTS (
                    SELECT 1 FROM memory_arguments ma
                    WHERE ma.memory_id = sm.memory_id AND ma.text ILIKE %s
                )
                OR EXISTS (
                    SELECT 1 FROM memory_evidence me
                    WHERE me.memory_id = sm.memory_id AND me.source_text ILIKE %s
                )
            )
            """
        )
        pattern = f"%{query}%"
        params.extend([pattern, pattern, pattern, pattern])
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    cur.execute(
        f"""
        SELECT
            sm.memory_id::text,
            sm.canonical_text,
            sm.subject_text,
            sm.subject_entity_type,
            sm.predicate_type,
            sm.memory_type,
            sm.status,
            sm.version,
            sm.created_at,
            sm.updated_at,
            sm.temporal_kind,
            sm.valid_from,
            sm.valid_to,
            sm.event_time,
            sm.recurrence,
            sm.recurrence_specifics,
            COALESCE(args.arguments, '[]') AS arguments,
            COALESCE(ev.evidence, '[]') AS evidence,
            COALESCE(events.ledger_events, '[]') AS ledger_events,
            COALESCE(cal.calendar_events, '[]') AS calendar_events
        FROM semantic_memories sm
        LEFT JOIN LATERAL (
            SELECT json_agg(json_build_object('role', role, 'text', text, 'entity_id', entity_id::text) ORDER BY position) AS arguments
            FROM memory_arguments ma
            WHERE ma.memory_id = sm.memory_id
        ) args ON true
        LEFT JOIN LATERAL (
            SELECT json_agg(json_build_object('episode_id', episode_id, 'message_id', message_id, 'source_text', source_text, 'observed_at', observed_at) ORDER BY created_at) AS evidence
            FROM memory_evidence me
            WHERE me.memory_id = sm.memory_id
        ) ev ON true
        LEFT JOIN LATERAL (
            SELECT json_agg(json_build_object('event_type', event_type, 'created_at', created_at, 'payload', payload) ORDER BY created_at) AS ledger_events
            FROM memory_events mev
            WHERE mev.memory_id = sm.memory_id
        ) events ON true
        LEFT JOIN LATERAL (
            SELECT json_agg(json_build_object(
                'calendar_event_id', calendar_event_id::text,
                'title', title,
                'start_at', start_at,
                'start_date', start_date,
                'recurrence', recurrence,
                'recurrence_specifics', recurrence_specifics,
                'status', status
            ) ORDER BY created_at) AS calendar_events
            FROM calendar_events ce
            WHERE ce.semantic_memory_id = sm.memory_id
        ) cal ON true
        {where_sql}
        ORDER BY sm.created_at DESC, sm.memory_id DESC
        LIMIT %s
        """,
        tuple(params),
    )
    return [dict(row) for row in cur.fetchall()]


if __name__ == "__main__":
    raise SystemExit(main())
