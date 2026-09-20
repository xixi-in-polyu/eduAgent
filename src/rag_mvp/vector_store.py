"""PostgreSQL/pgvector-backed document and chunk storage.

This module owns the vector RAG persistence contract.  It intentionally has no
dependency on a graph database or a third-party RAG orchestration framework.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from loguru import logger
from psycopg_pool import ConnectionPool

from .config import settings
from .embedding_factory import embed_texts


_pool_lock = threading.Lock()
_pool: ConnectionPool | None = None
_pool_dsn: str | None = None
_schema_lock = threading.Lock()
_schema_ready_dsn: str | None = None


@dataclass(slots=True)
class VectorChunk:
    content: str
    file_path: str
    order_index: int
    page_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def stable_id(value: str, *, prefix: str) -> str:
    return prefix + hashlib.md5(value.encode("utf-8")).hexdigest()


def document_id(material_id: str) -> str:
    return stable_id(f"edu:material:{material_id}", prefix="doc-")


def chunk_id(doc_id: str, chunk: VectorChunk) -> str:
    identity = f"{doc_id}\0{chunk.file_path}\0{chunk.order_index}\0{chunk.content}"
    return stable_id(identity, prefix="chunk-")


def course_workspace(course_id: str) -> str:
    raw = str(course_id).strip().lower()
    if not raw:
        raise ValueError("course_id must be non-empty")
    return f"course_{raw}"


def personal_workspace(user_id: str) -> str:
    raw = str(user_id).strip().lower()
    if not raw:
        raise ValueError("user_id must be non-empty")
    safe = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_") or "user"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"personal_{safe[:40]}_{digest}"


def _dsn() -> str:
    dsn = os.environ.get("RAG_PG_DSN", "").strip() or os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError("RAG_PG_DSN or DATABASE_URL is required for vector RAG storage")
    parsed = urlparse(dsn)
    query = [(key, value) for key, value in parse_qsl(parsed.query) if key != "schema"]
    return urlunparse(parsed._replace(query=urlencode(query)))


def _connection_pool() -> ConnectionPool:
    global _pool, _pool_dsn
    dsn = _dsn()
    if _pool is not None and _pool_dsn == dsn:
        return _pool
    with _pool_lock:
        if _pool is not None and _pool_dsn == dsn:
            return _pool
        if _pool is not None:
            _pool.close()
        min_size = max(1, int(os.environ.get("RAG_PG_POOL_MIN_SIZE", "1")))
        max_size = max(min_size, int(os.environ.get("RAG_PG_POOL_MAX_SIZE", "20")))
        _pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=float(os.environ.get("RAG_PG_POOL_TIMEOUT", "10")),
            open=True,
        )
        _pool_dsn = dsn
        return _pool


def close_vector_pool() -> None:
    """Close the process-wide pgvector pool during service shutdown."""
    global _pool, _pool_dsn, _schema_ready_dsn
    with _pool_lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _pool_dsn = None
        _schema_ready_dsn = None


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def ensure_schema() -> None:
    global _schema_ready_dsn
    dsn = _dsn()
    if _schema_ready_dsn == dsn:
        return
    with _schema_lock:
        if _schema_ready_dsn == dsn:
            return
        _ensure_schema_once(dsn)
        _schema_ready_dsn = dsn


def _ensure_schema_once(_expected_dsn: str) -> None:
    dim = int(settings.embedding_dim)
    if dim <= 0:
        raise RuntimeError("EMBEDDING_DIM must be positive")
    with _connection_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_store_metadata (
                key text PRIMARY KEY,
                value text NOT NULL,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute("SELECT value FROM rag_store_metadata WHERE key = 'embedding_dim'")
        row = cur.fetchone()
        if row and int(row[0]) != dim:
            raise RuntimeError(
                f"Vector store uses embedding_dim={row[0]}, but EMBEDDING_DIM={dim}. "
                "Migrate or rebuild the vector index before starting the service."
            )
        cur.execute(
            """
            INSERT INTO rag_store_metadata (key, value)
            VALUES ('embedding_dim', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
            """,
            (str(dim),),
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_documents (
                workspace varchar(255) NOT NULL,
                id varchar(255) NOT NULL,
                file_path text NOT NULL DEFAULT '',
                status varchar(32) NOT NULL DEFAULT 'processed',
                chunks_count integer NOT NULL DEFAULT 0,
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (workspace, id)
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                workspace varchar(255) NOT NULL,
                id varchar(255) NOT NULL,
                document_id varchar(255) NOT NULL,
                content text NOT NULL,
                file_path text NOT NULL DEFAULT '',
                chunk_order_index integer NOT NULL,
                page_idx integer,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dim}) NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (workspace, id),
                FOREIGN KEY (workspace, document_id)
                    REFERENCES rag_documents(workspace, id) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_document "
            "ON rag_chunks (workspace, document_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_fts ON rag_chunks "
            "USING GIN (to_tsvector('simple', content))"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding ON rag_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
        conn.commit()


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), max(1, size)):
        yield items[start : start + max(1, size)]


async def replace_document(
    workspace: str,
    doc_id: str,
    chunks: Sequence[VectorChunk],
    *,
    file_path: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    clean = [chunk for chunk in chunks if chunk.content.strip()]
    if not clean:
        raise ValueError(f"No non-empty chunks for document {doc_id}")

    texts = [chunk.content for chunk in clean]
    vectors: list[list[float]] = []
    for batch in _batched(texts, settings.embedding_batch_num):
        vectors.extend(await embed_texts(list(batch)))
    if len(vectors) != len(clean):
        raise RuntimeError(f"Embedding backend returned {len(vectors)} vectors for {len(clean)} chunks")

    def _write() -> int:
        ensure_schema()
        with _connection_pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_documents (workspace, id, file_path, status, chunks_count, metadata)
                VALUES (%s, %s, %s, 'indexing', 0, %s::jsonb)
                ON CONFLICT (workspace, id) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    status = 'indexing',
                    chunks_count = 0,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                (workspace, doc_id, file_path, json.dumps(metadata or {}, ensure_ascii=False)),
            )
            cur.execute("DELETE FROM rag_chunks WHERE workspace = %s AND document_id = %s", (workspace, doc_id))
            rows = []
            for chunk, vector in zip(clean, vectors):
                chunk_meta = dict(chunk.metadata)
                if chunk.page_idx is not None:
                    chunk_meta.setdefault("page_idx", chunk.page_idx)
                rows.append(
                    (
                        workspace,
                        chunk_id(doc_id, chunk),
                        doc_id,
                        chunk.content,
                        chunk.file_path,
                        chunk.order_index,
                        chunk.page_idx,
                        json.dumps(chunk_meta, ensure_ascii=False),
                        _vector_literal(vector),
                    )
                )
            cur.executemany(
                """
                INSERT INTO rag_chunks (
                    workspace, id, document_id, content, file_path,
                    chunk_order_index, page_idx, metadata, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                """,
                rows,
            )
            cur.execute(
                """
                UPDATE rag_documents
                SET status = 'processed', chunks_count = %s, updated_at = now()
                WHERE workspace = %s AND id = %s
                """,
                (len(rows), workspace, doc_id),
            )
            conn.commit()
        return len(rows)

    count = await asyncio.to_thread(_write)
    logger.success("Indexed vector document {} in {} ({} chunks)", doc_id, workspace, count)
    return count


async def delete_document(workspace: str, doc_id: str) -> None:
    def _delete() -> None:
        ensure_schema()
        with _connection_pool().connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM rag_documents WHERE workspace = %s AND id = %s", (workspace, doc_id))
            conn.commit()

    await asyncio.to_thread(_delete)


async def clear_workspace(workspace: str) -> None:
    def _clear() -> None:
        ensure_schema()
        with _connection_pool().connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM rag_documents WHERE workspace = %s", (workspace,))
            conn.commit()

    await asyncio.to_thread(_clear)


async def vector_search(
    workspace: str,
    query: str,
    *,
    top_k: int,
    timings_ms: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    started = loop.time()
    vectors = await embed_texts([query])
    embedding_finished = loop.time()
    if timings_ms is not None:
        timings_ms["embedding"] = (embedding_finished - started) * 1000
    if not vectors:
        return []
    vector = _vector_literal(vectors[0])

    def _search() -> list[dict[str, Any]]:
        ensure_schema()
        with _connection_pool().connection() as conn, conn.cursor() as cur:
            sql = """
                SELECT id, content, file_path, document_id, page_idx, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM rag_chunks
                WHERE workspace = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
            params = (vector, workspace, vector, top_k)
            cur.execute(sql, params)
            rows = cur.fetchall()
            # A global HNSW index can find neighbours from other workspaces first;
            # PostgreSQL applies the workspace predicate afterwards and may return
            # fewer than top_k rows. Fall back to an exact scan for correctness.
            if len(rows) < top_k:
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute("SET LOCAL enable_bitmapscan = off")
                cur.execute(sql, params)
                rows = cur.fetchall()
            return [
                {
                    "chunk_id": str(row[0]),
                    "text": str(row[1]),
                    "metadata": {
                        **(row[5] or {}),
                        "file_path": row[2],
                        "document_id": row[3],
                        "page_idx": row[4],
                    },
                    "relevance_score": float(row[6] or 0.0),
                }
                for row in rows
            ]

    result = await asyncio.to_thread(_search)
    if timings_ms is not None:
        timings_ms["vector_db"] = (loop.time() - embedding_finished) * 1000
    return result


def bm25_search(workspace: str, query: str, *, top_k: int) -> list[dict[str, Any]]:
    ensure_schema()
    with _connection_pool().connection() as conn, conn.cursor() as cur:
        if re.search(r"[\u4e00-\u9fff]", query):
            terms = _cjk_bigrams(query)
            if not terms:
                return []
            cur.execute(
                """
                WITH terms AS (SELECT unnest(%s::text[]) AS term)
                SELECT c.id, c.content, c.file_path, c.document_id, c.page_idx, c.metadata,
                       avg(CASE WHEN strpos(c.content, terms.term) > 0 THEN 1.0 ELSE 0.0 END) AS score
                FROM rag_chunks c CROSS JOIN terms
                WHERE c.workspace = %s
                GROUP BY c.id, c.content, c.file_path, c.document_id, c.page_idx, c.metadata
                HAVING bool_or(strpos(c.content, terms.term) > 0)
                ORDER BY score DESC
                LIMIT %s
                """,
                (terms, workspace, top_k),
            )
        else:
            cur.execute(
                """
                SELECT id, content, file_path, document_id, page_idx, metadata,
                       ts_rank_cd(to_tsvector('simple', content), plainto_tsquery('simple', %s)) AS score
                FROM rag_chunks
                WHERE workspace = %s
                  AND to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, workspace, query, top_k),
            )
        return [
            {
                "chunk_id": str(row[0]),
                "text": str(row[1]),
                "metadata": {
                    **(row[5] or {}),
                    "file_path": row[2],
                    "document_id": row[3],
                    "page_idx": row[4],
                },
                "relevance_score": float(row[6] or 0.0),
            }
            for row in cur.fetchall()
        ]


def _cjk_bigrams(query: str) -> list[str]:
    """Return stable, de-duplicated Chinese bigrams for lexical retrieval."""
    sequences = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    return list(
        dict.fromkeys(
            sequence[index : index + 2]
            for sequence in sequences
            for index in range(len(sequence) - 1)
        )
    )


def document_page_mappings(workspace: str, doc_id: str) -> list[tuple[str, int]]:
    ensure_schema()
    with _connection_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, page_idx FROM rag_chunks
            WHERE workspace = %s AND document_id = %s AND page_idx IS NOT NULL
            """,
            (workspace, doc_id),
        )
        return [(str(row[0]), int(row[1])) for row in cur.fetchall()]


def workspace_stats(workspace: str) -> dict[str, int]:
    ensure_schema()
    with _connection_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM rag_documents WHERE workspace = %s", (workspace,))
        documents = int(cur.fetchone()[0])
        cur.execute("SELECT count(*) FROM rag_chunks WHERE workspace = %s", (workspace,))
        chunks = int(cur.fetchone()[0])
    return {"documents": documents, "chunks": chunks}
