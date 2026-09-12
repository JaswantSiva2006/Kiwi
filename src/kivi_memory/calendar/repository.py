"""PostgreSQL access for structured calendar projection rows."""

from __future__ import annotations

from typing import Any

from kivi_memory.entity_resolution.repository import connect


class CalendarRepository:
    """Small repository for calendar projection writes and schedule reads."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url

    def connect(self):
        return connect(self.database_url)

    def insert_calendar_event_in_transaction(
        self,
        cur,
        *,
        event_values: dict[str, Any],
        entity_links: list[dict[str, Any]],
    ) -> str | None:
        if not hasattr(cur, "execute"):
            return None
        cur.execute(
            """
            INSERT INTO calendar_events (
                semantic_memory_id,
                title,
                event_kind,
                start_at,
                end_at,
                start_date,
                end_date,
                recurrence_until_at,
                recurrence_until_date,
                all_day,
                location_text,
                timezone_text,
                recurrence,
                recurrence_specifics,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (semantic_memory_id) DO NOTHING
            RETURNING calendar_event_id::text AS calendar_event_id
            """,
            (
                event_values["semantic_memory_id"],
                event_values["title"],
                event_values["event_kind"],
                event_values["start_at"],
                event_values["end_at"],
                event_values["start_date"],
                event_values["end_date"],
                event_values["recurrence_until_at"],
                event_values["recurrence_until_date"],
                event_values["all_day"],
                event_values["location_text"],
                event_values["timezone_text"],
                event_values["recurrence"],
                event_values["recurrence_specifics"],
                event_values["status"],
            ),
        )
        row = cur.fetchone()
        if row is None:
            return None

        calendar_event_id = row["calendar_event_id"]
        for link in entity_links:
            cur.execute(
                """
                INSERT INTO calendar_event_entities (
                    calendar_event_id,
                    entity_id,
                    role,
                    mention_text
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    calendar_event_id,
                    link["entity_id"],
                    link["role"],
                    link["mention_text"],
                ),
            )
        return calendar_event_id

    def update_calendar_status_for_memory_in_transaction(self, cur, *, memory_id: str, status: str) -> int:
        if not hasattr(cur, "execute"):
            return 0
        cur.execute(
            """
            UPDATE calendar_events
            SET status = %s,
                updated_at = now()
            WHERE semantic_memory_id = %s
            """,
            (status, memory_id),
        )
        return int(cur.rowcount or 0)

    def fetch_schedule_candidates(self, *, start, end) -> list[dict[str, Any]]:
        """Fetch SCHEDULED calendar rows that can overlap a requested interval.

        Recurring rows are prefiltered by anchor and recurrence-until bounds;
        occurrence expansion is intentionally performed in the read tool.
        """

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        ce.calendar_event_id::text AS calendar_event_id,
                        ce.semantic_memory_id::text AS semantic_memory_id,
                        sm.created_at AS memory_created_at,
                        ce.title,
                        ce.event_kind,
                        ce.start_at,
                        ce.end_at,
                        ce.start_date,
                        ce.end_date,
                        ce.recurrence_until_at,
                        ce.recurrence_until_date,
                        ce.all_day,
                        ce.location_text,
                        ce.timezone_text,
                        ce.recurrence,
                        ce.recurrence_specifics,
                        ce.status,
                        COALESCE(
                            jsonb_agg(
                                DISTINCT jsonb_build_object(
                                    'entity_id', cee.entity_id::text,
                                    'role', cee.role,
                                    'mention_text', cee.mention_text
                                )
                            ) FILTER (WHERE cee.entity_id IS NOT NULL),
                            '[]'::jsonb
                        ) AS entities,
                        COALESCE(
                            jsonb_agg(
                                DISTINCT jsonb_build_object(
                                    'episode_id', me.episode_id,
                                    'message_id', me.message_id,
                                    'source_text', me.source_text,
                                    'observed_at', me.observed_at
                                )
                            ) FILTER (WHERE me.evidence_id IS NOT NULL),
                            '[]'::jsonb
                        ) AS evidence
                    FROM calendar_events ce
                    LEFT JOIN semantic_memories sm
                        ON sm.memory_id = ce.semantic_memory_id
                    LEFT JOIN calendar_event_entities cee
                        ON cee.calendar_event_id = ce.calendar_event_id
                    LEFT JOIN memory_evidence me
                        ON me.memory_id = ce.semantic_memory_id
                    WHERE ce.status = 'SCHEDULED'
                      AND (
                          (
                              ce.recurrence IS NULL
                              AND ce.start_at IS NOT NULL
                              AND ce.start_at < %s
                              AND COALESCE(ce.end_at, ce.start_at) >= %s
                          )
                          OR (
                              ce.recurrence IS NULL
                              AND ce.start_date IS NOT NULL
                              AND ce.start_date < %s
                              AND COALESCE(ce.end_date, ce.start_date) >= %s
                          )
                          OR (
                              ce.recurrence IS NOT NULL
                              AND (
                                  ce.start_at IS NULL
                                  OR ce.start_at < %s
                              )
                              AND (
                                  ce.start_date IS NULL
                                  OR ce.start_date < %s
                              )
                              AND (
                                  ce.recurrence_until_at IS NULL
                                  OR ce.recurrence_until_at >= %s
                              )
                              AND (
                                  ce.recurrence_until_date IS NULL
                                  OR ce.recurrence_until_date >= %s
                              )
                          )
                      )
                    GROUP BY ce.calendar_event_id, sm.created_at
                    ORDER BY
                        COALESCE(ce.start_at, ce.start_date::timestamptz),
                        ce.calendar_event_id
                    """,
                    (
                        end,
                        start,
                        end.date(),
                        start.date(),
                        end,
                        end.date(),
                        start,
                        start.date(),
                    ),
                )
                return [dict(row) for row in cur.fetchall()]
