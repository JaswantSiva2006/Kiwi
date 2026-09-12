"""PostgreSQL access for entity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from kivi_memory.entity_resolution.config import get_database_url
from kivi_memory.entity_resolution.normalizer import normalize_entity_name


@dataclass(frozen=True)
class CandidateEvidence:
    entity_id: str
    canonical_name: str
    entity_type: str
    matched_text: str
    match_type: str
    similarity_score: float


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    canonical_name: str
    normalized_name: str
    entity_type: str


@dataclass(frozen=True)
class EntityAliasRecord:
    alias_id: str
    entity_id: str
    alias: str
    normalized_alias: str
    confidence: float | None
    alias_source: str | None
    source_episode_id: str | None


class EntityNotFoundError(ValueError):
    """Raised when an alias references an entity that does not exist."""


def connect(database_url: str | None = None) -> psycopg.Connection:
    """Open a row-dict psycopg connection using the shared database setting."""

    return psycopg.connect(database_url or get_database_url(), row_factory=dict_row)


class EntityRepository:
    """Entity lookup and write-side repository."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or get_database_url()

    def find_exact(self, normalized_mention: str, limit: int) -> list[CandidateEvidence]:
        """Find exact canonical-name and alias evidence for a normalized mention."""

        sql = """
            (
                SELECT
                    e.entity_id::text AS entity_id,
                    e.canonical_name,
                    e.entity_type,
                    e.normalized_name AS matched_text,
                    'EXACT_NAME' AS match_type,
                    1.0::double precision AS similarity_score
                FROM entities e
                WHERE e.normalized_name = %s
            )
            UNION ALL
            (
                SELECT
                    e.entity_id::text AS entity_id,
                    e.canonical_name,
                    e.entity_type,
                    a.normalized_alias AS matched_text,
                    'EXACT_ALIAS' AS match_type,
                    1.0::double precision AS similarity_score
                FROM entity_aliases a
                JOIN entities e ON e.entity_id = a.entity_id
                WHERE a.normalized_alias = %s
            )
            LIMIT %s
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (normalized_mention, normalized_mention, limit))
                return list(_rows_to_evidence(cur.fetchall()))

    def get_current_user_entity(self) -> EntityRecord | None:
        """Return the stable current-user entity without fuzzy lookup."""

        sql = """
            (
                SELECT
                    e.entity_id::text AS entity_id,
                    e.canonical_name,
                    e.normalized_name,
                    e.entity_type,
                    0 AS source_rank
                FROM entities e
                WHERE e.normalized_name = %s
            )
            UNION ALL
            (
                SELECT
                    e.entity_id::text AS entity_id,
                    e.canonical_name,
                    e.normalized_name,
                    e.entity_type,
                    1 AS source_rank
                FROM entity_aliases a
                JOIN entities e ON e.entity_id = a.entity_id
                WHERE a.normalized_alias = %s
            )
            ORDER BY source_rank, canonical_name
            LIMIT 1
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, ("user", "user"))
                row = cur.fetchone()
                if row is None:
                    return None
                return _row_to_entity(row)

    def find_fuzzy(
        self,
        normalized_mention: str,
        threshold: float,
        limit: int,
    ) -> list[CandidateEvidence]:
        """Find broad pg_trgm canonical-name and alias evidence."""

        sql = """
            (
                SELECT
                    e.entity_id::text AS entity_id,
                    e.canonical_name,
                    e.entity_type,
                    e.normalized_name AS matched_text,
                    'FUZZY_NAME' AS match_type,
                    similarity(e.normalized_name, %s) AS similarity_score
                FROM entities e
                WHERE e.normalized_name %% %s
                  AND similarity(e.normalized_name, %s) >= %s
                ORDER BY similarity_score DESC
                LIMIT %s
            )
            UNION ALL
            (
                SELECT
                    e.entity_id::text AS entity_id,
                    e.canonical_name,
                    e.entity_type,
                    a.normalized_alias AS matched_text,
                    'FUZZY_ALIAS' AS match_type,
                    similarity(a.normalized_alias, %s) AS similarity_score
                FROM entity_aliases a
                JOIN entities e ON e.entity_id = a.entity_id
                WHERE a.normalized_alias %% %s
                  AND similarity(a.normalized_alias, %s) >= %s
                ORDER BY similarity_score DESC
                LIMIT %s
            )
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('pg_trgm.similarity_threshold', %s, true)", (str(threshold),))
                cur.execute(
                    sql,
                    (
                        normalized_mention,
                        normalized_mention,
                        normalized_mention,
                        threshold,
                        limit,
                        normalized_mention,
                        normalized_mention,
                        normalized_mention,
                        threshold,
                        limit,
                    ),
                )
                return list(_rows_to_evidence(cur.fetchall()))

    def create_entity(
        self,
        canonical_name: str,
        entity_type: str,
        source_episode_id: str | None = None,
    ) -> EntityRecord:
        """Create an entity and its canonical alias in one transaction."""

        canonical_name = _require_non_empty(canonical_name, "canonical_name")
        entity_type = _require_non_empty(entity_type, "entity_type")
        normalized_name = normalize_entity_name(canonical_name)
        if not normalized_name:
            raise ValueError("canonical_name must contain semantic text")

        entity_sql = """
            INSERT INTO entities (canonical_name, normalized_name, entity_type)
            VALUES (%s, %s, %s)
            RETURNING
                entity_id::text AS entity_id,
                canonical_name,
                normalized_name,
                entity_type
        """
        alias_sql = """
            INSERT INTO entity_aliases (
                entity_id,
                alias,
                normalized_alias,
                confidence,
                alias_source,
                source_episode_id
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING alias_id::text AS alias_id
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(entity_sql, (canonical_name, normalized_name, entity_type))
                entity = _row_to_entity(cur.fetchone())
                cur.execute(
                    alias_sql,
                    (
                        entity.entity_id,
                        canonical_name,
                        normalized_name,
                        1.0,
                        "CANONICAL",
                        source_episode_id,
                    ),
                )
                cur.fetchone()
                return entity

    def add_entity_alias(
        self,
        entity_id: str,
        alias: str,
        confidence: float | None = None,
        alias_source: str | None = None,
        source_episode_id: str | None = None,
    ) -> EntityAliasRecord:
        """Add an alias idempotently for one entity."""

        entity_id = _require_non_empty(str(entity_id), "entity_id")
        alias = _require_non_empty(alias, "alias")
        normalized_alias = normalize_entity_name(alias)
        if not normalized_alias:
            raise ValueError("alias must contain semantic text")

        exists_sql = "SELECT 1 FROM entities WHERE entity_id = %s"
        alias_sql = """
            INSERT INTO entity_aliases (
                entity_id,
                alias,
                normalized_alias,
                confidence,
                alias_source,
                source_episode_id
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id, normalized_alias)
            DO UPDATE SET normalized_alias = entity_aliases.normalized_alias
            RETURNING
                alias_id::text AS alias_id,
                entity_id::text AS entity_id,
                alias,
                normalized_alias,
                confidence,
                alias_source,
                source_episode_id
        """
        with connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(exists_sql, (entity_id,))
                if cur.fetchone() is None:
                    raise EntityNotFoundError(f"entity does not exist: {entity_id}")
                cur.execute(
                    alias_sql,
                    (
                        entity_id,
                        alias,
                        normalized_alias,
                        confidence,
                        alias_source,
                        source_episode_id,
                    ),
                )
                return _row_to_alias(cur.fetchone())


def _rows_to_evidence(rows: list[dict]) -> Iterator[CandidateEvidence]:
    for row in rows:
        yield CandidateEvidence(
            entity_id=row["entity_id"],
            canonical_name=row["canonical_name"],
            entity_type=row["entity_type"],
            matched_text=row["matched_text"],
            match_type=row["match_type"],
            similarity_score=float(row["similarity_score"]),
        )


def _row_to_entity(row: dict | None) -> EntityRecord:
    if row is None:
        raise RuntimeError("entity insert did not return a row")
    return EntityRecord(
        entity_id=row["entity_id"],
        canonical_name=row["canonical_name"],
        normalized_name=row["normalized_name"],
        entity_type=row["entity_type"],
    )


def _row_to_alias(row: dict | None) -> EntityAliasRecord:
    if row is None:
        raise RuntimeError("alias insert did not return a row")
    return EntityAliasRecord(
        alias_id=row["alias_id"],
        entity_id=row["entity_id"],
        alias=row["alias"],
        normalized_alias=row["normalized_alias"],
        confidence=row["confidence"],
        alias_source=row["alias_source"],
        source_episode_id=row["source_episode_id"],
    )


def _require_non_empty(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped
