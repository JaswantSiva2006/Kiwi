from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv

load_dotenv()


def get_database_url() -> str:
    value = os.getenv("KIVI_DATABASE_URL")
    if not value:
        raise RuntimeError("KIVI_DATABASE_URL is required for Document RAG storage")
    return value


def connect(database_url: str | None = None):
    url = database_url or get_database_url()
    try:
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(url, row_factory=dict_row)
    except ImportError:
        return _connect_pg8000(url)


def _connect_pg8000(database_url: str) -> "_Pg8000Connection":
    import pg8000.dbapi

    parsed = urlparse(database_url)
    conn = pg8000.dbapi.connect(
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=(parsed.path or "/").lstrip("/"),
    )
    return _Pg8000Connection(conn)


class _Pg8000Connection:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __enter__(self) -> "_Pg8000Connection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self._conn.close()

    def cursor(self) -> "_Pg8000Cursor":
        return _Pg8000Cursor(self._conn.cursor())

    def rollback(self) -> None:
        self._conn.rollback()


class _Pg8000Cursor:
    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def __enter__(self) -> "_Pg8000Cursor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._cursor.close()

    def execute(self, sql: str, params: tuple[Any, ...] | list[Any] | None = None) -> None:
        self._cursor.execute(sql, params or ())

    def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> None:
        self._cursor.executemany(sql, params)

    def fetchone(self) -> dict[str, Any] | None:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def fetchall(self) -> list[dict[str, Any]]:
        return [self._row_to_dict(row) for row in self._cursor.fetchall()]

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount or 0)

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        names = [column[0] for column in self._cursor.description or []]
        return dict(zip(names, row))
