"""Redis Stream consumer for course material RAG tasks (XREADGROUP + XACK + XAUTOCLAIM)."""

from __future__ import annotations

import os
import signal
import sys
from typing import Any

import redis
from dotenv import load_dotenv
from loguru import logger

from rag_mvp.db import connect_sync
from rag_mvp.logging_setup import configure_logging
from rag_mvp.task_handlers import get_task_handler
from rag_mvp.worker_async_loop import (
    run_worker_coroutine,
    start_worker_async_loop,
    stop_worker_async_loop,
)


def _stream_name() -> str:
    return os.environ.get("RAG_TASK_STREAM_NAME", "edu:rag:tasks:stream").strip()


def _group_name() -> str:
    return os.environ.get("RAG_TASK_STREAM_GROUP", "edu-rag-workers").strip()


def _consumer_name() -> str:
    return os.environ.get("RAG_TASK_CONSUMER_NAME", f"edu-rag-{os.getpid()}").strip()


def _claim_idle_ms() -> int:
    return int(os.environ.get("RAG_STREAM_CLAIM_IDLE_MS", "300000"))


def _material_stale_sec() -> int:
    return int(os.environ.get("RAG_MATERIAL_STALE_SEC", "1800"))


def _mark_stale_jobs_failed(conn: Any) -> None:
    """On startup: mark any PARSING/PARSED/INDEXING materials whose ``updated_at`` has
    not moved in more than ``RAG_MATERIAL_STALE_SEC`` seconds as FAILED.
    Applies to both course materials and personal materials.
    """
    stale_sec = _material_stale_sec()
    stale_sql = """
        UPDATE {table}
        SET status = 'FAILED',
            status_message = 'WORKER_ABANDONED: worker was interrupted or crashed',
            updated_at = NOW()
        WHERE is_deleted = false
          AND status IN ('PARSING', 'PARSED', 'INDEXING')
          AND updated_at < NOW() - (%s * INTERVAL '1 second')
        RETURNING id::text
    """
    for table in ("materials", "personal_materials"):
        try:
            with conn.cursor() as cur:
                cur.execute(stale_sql.format(table=table), (stale_sec,))
                rows = cur.fetchall()
            if rows:
                ids = [r[0] for r in rows]
                logger.warning(
                    "Marked {} stale {} as FAILED on startup: {}",
                    len(ids), table, ids,
                )
        except Exception:
            logger.exception("Failed to clean up stale {} on startup", table)


def _ensure_group(r: redis.Redis, stream: str, group: str) -> None:
    try:
        r.xgroup_create(name=stream, groupname=group, id="0", mkstream=True)
        logger.info("Created stream consumer group {} on {}", group, stream)
    except redis.ResponseError as exc:
        if "BUSYGROUP" in str(exc):
            return
        raise


def _coerce_field_dict(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        out: dict[str, str] = {}
        it = iter(raw)
        for k in it:
            v = next(it, None)
            if v is not None:
                out[str(k)] = str(v)
        return out
    return {}


def _parse_autoclaim_messages(resp: Any) -> list[tuple[str, dict[str, str]]]:
    """Parse XAUTOCLAIM RESP2: [cursor, [[id, [k,v,...]], ...]]."""
    if not isinstance(resp, (list, tuple)) or len(resp) < 2:
        return []
    msgs = resp[1]
    if not isinstance(msgs, list):
        return []
    out: list[tuple[str, dict[str, str]]] = []
    for item in msgs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        msg_id = str(item[0])
        out.append((msg_id, _coerce_field_dict(item[1])))
    return out


def _parse_bool_field(raw: str | None, default: bool = True) -> bool:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if not s:
        return default
    return s in ("1", "true", "yes", "on")


def _process_one(conn: Any, r: redis.Redis, fields: dict[str, str]) -> None:
    op = fields.get("operation")
    text_only = _parse_bool_field(fields.get("text_only"), default=True)
    handler = get_task_handler(op)
    if handler is None:
        raise ValueError(f"unknown operation: {op!r}")
    handler(conn, r, fields, text_only)


def _handle_entries(
    conn: Any,
    r: redis.Redis,
    stream: str,
    group: str,
    entries: list[tuple[str, dict[str, str]]],
) -> None:
    for msg_id, raw_fields in entries:
        fields = {str(k): str(v) for k, v in raw_fields.items()}
        try:
            _process_one(conn, r, fields)
            r.xack(stream, group, msg_id)
        except ValueError as exc:
            logger.error("Invalid or poison task stream_id={}: {}", msg_id, exc)
            r.xack(stream, group, msg_id)
        except Exception:
            logger.exception("Task failed stream_id={}", msg_id)
            # ACK so the message leaves the PEL immediately; material/index paths persist FAILED in DB.
            # Retries: index_only or re-enqueue (see material_processor / assignment_gen).
            try:
                r.xack(stream, group, msg_id)
            except redis.ResponseError:
                logger.exception("XACK failed stream_id={}", msg_id)


def main() -> None:
    # Load root-level .env first, then edu-platform/.env as fallback (override=False keeps
    # already-set values, so system env vars always win).
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, "..", ".."))
    load_dotenv(os.path.join(_root, ".env"))
    load_dotenv(os.path.join(_root, "edu-platform", ".env"))
    configure_logging("rag-worker")
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        logger.error("REDIS_URL is required")
        sys.exit(1)
    stream = _stream_name()
    group = _group_name()
    consumer = _consumer_name()
    idle_ms = _claim_idle_ms()
    r = redis.from_url(redis_url, decode_responses=True)
    conn = connect_sync(autocommit=True)
    from rag_mvp.vector_store import ensure_schema

    ensure_schema()
    stop = False

    def _stop(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    _ensure_group(r, stream, group)
    _mark_stale_jobs_failed(conn)
    start_worker_async_loop()
    from .config import settings as _s
    logger.info(
        "edu-rag-worker stream={} group={} consumer={} claim_idle_ms={} persistent_async_loop=on",
        stream,
        group,
        consumer,
        idle_ms,
    )
    logger.info(
        "config | llm_model={} embedding_mode={} embedding_model={} embedding_dim={}",
        _s.llm_model,
        _s.embedding_mode,
        _s.embedding_model,
        _s.embedding_dim,
    )
    try:
        while not stop:
            try:
                resp = r.execute_command(
                    "XAUTOCLAIM",
                    stream,
                    group,
                    consumer,
                    str(idle_ms),
                    "0-0",
                    "COUNT",
                    25,
                )
                claimed = _parse_autoclaim_messages(resp)
                if claimed:
                    _handle_entries(conn, r, stream, group, claimed)
                msgs: Any = r.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={stream: ">"},
                    count=5,
                    block=5000,
                )
                if msgs:
                    for _sname, entries in msgs:
                        if entries:
                            _handle_entries(conn, r, stream, group, entries)
            except redis.ConnectionError:
                logger.exception("Redis connection error")
            except Exception:
                logger.exception("Worker loop error")
    finally:
        try:
            from rag_mvp.embedding_factory import close_embedding_clients

            run_worker_coroutine(close_embedding_clients(), timeout=30)
        except Exception:
            logger.exception("Failed to close embedding clients")
        stop_worker_async_loop()
        from rag_mvp.vector_store import close_vector_pool

        close_vector_pool()
        conn.close()
        logger.info("edu-rag-worker stopped")


if __name__ == "__main__":
    main()
