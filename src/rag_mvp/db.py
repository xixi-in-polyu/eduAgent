"""Shared synchronous PostgreSQL connection for Prisma-managed tables (e.g. materials)."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import psycopg
from psycopg_pool import ConnectionPool


_pool_lock = threading.Lock()
_pool: ConnectionPool | None = None
_pool_dsn: str | None = None


def _database_url_for_psycopg(dsn: str) -> str:
    """Prisma adds ``?schema=public``; libpq/psycopg rejects unknown URI query keys."""
    parsed = urlparse(dsn.strip())
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    schema_name: str | None = None
    out: list[tuple[str, str]] = []
    for key, val in pairs:
        if key == "schema":
            schema_name = val or "public"
            continue
        out.append((key, val))
    if schema_name and schema_name != "public":
        out.append(("options", f"-csearch_path={schema_name}"))
    query = urlencode(out) if out else ""
    return urlunparse(parsed._replace(query=query))


def connect_sync(*, autocommit: bool = False) -> psycopg.Connection:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required")
    return psycopg.connect(_database_url_for_psycopg(dsn), autocommit=autocommit)


def _connection_pool() -> ConnectionPool:
    global _pool, _pool_dsn
    raw_dsn = os.environ.get("DATABASE_URL", "").strip()
    if not raw_dsn:
        raise RuntimeError("DATABASE_URL is required")
    dsn = _database_url_for_psycopg(raw_dsn)
    if _pool is not None and _pool_dsn == dsn:
        return _pool
    with _pool_lock:
        if _pool is not None and _pool_dsn == dsn:
            return _pool
        if _pool is not None:
            _pool.close()
        min_size = max(1, int(os.environ.get("DATABASE_POOL_MIN_SIZE", "1")))
        max_size = max(min_size, int(os.environ.get("DATABASE_POOL_MAX_SIZE", "20")))
        _pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=float(os.environ.get("DATABASE_POOL_TIMEOUT", "10")),
            open=True,
        )
        _pool_dsn = dsn
        return _pool


@contextmanager
def pooled_connection() -> Iterator[psycopg.Connection]:
    """Borrow a request-scoped connection from the process-wide platform DB pool."""
    with _connection_pool().connection() as conn:
        yield conn


def close_connection_pool() -> None:
    """Close the platform DB pool during service shutdown."""
    global _pool, _pool_dsn
    with _pool_lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _pool_dsn = None
