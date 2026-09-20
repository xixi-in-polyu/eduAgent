"""Embedding client abstraction for the vector RAG store."""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .config import settings
from .http_env import ensure_loopback_bypass_http_proxy

ensure_loopback_bypass_http_proxy()

_MAX_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 2


@dataclass(slots=True)
class _BatchItem:
    text: str
    future: asyncio.Future[list[float]]


class _EmbeddingMicroBatcher:
    """Collect concurrent single-text requests into short, bounded batches."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: list[_BatchItem] = []
        self._timer: asyncio.TimerHandle | None = None
        self._task: asyncio.Task[None] | None = None

    async def submit(self, text: str) -> list[float]:
        future: asyncio.Future[list[float]] = self._loop.create_future()
        self._queue.append(_BatchItem(text=text, future=future))

        if len(self._queue) >= settings.embedding_query_batch_size:
            self._cancel_timer()
            self._ensure_drain_task()
        elif self._timer is None and (self._task is None or self._task.done()):
            delay = settings.embedding_query_batch_window_ms / 1000
            self._timer = self._loop.call_later(delay, self._on_timer)

        return await future

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_timer(self) -> None:
        self._timer = None
        self._ensure_drain_task()

    def _ensure_drain_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = self._loop.create_task(self._drain())

    async def _drain(self) -> None:
        try:
            while self._queue:
                size = settings.embedding_query_batch_size
                batch, self._queue = self._queue[:size], self._queue[size:]
                active = [item for item in batch if not item.future.cancelled()]
                if not active:
                    continue
                logger.debug("Dispatching embedding micro-batch with {} queries", len(active))
                try:
                    vectors = await _embed_with_retry([item.text for item in active])
                    if len(vectors) != len(active):
                        raise RuntimeError(
                            "Embedding backend returned "
                            f"{len(vectors)} vectors for {len(active)} query texts"
                        )
                    for item, vector in zip(active, vectors, strict=True):
                        if not item.future.done():
                            item.future.set_result(vector)
                except Exception as exc:
                    for item in active:
                        if not item.future.done():
                            item.future.set_exception(exc)
        finally:
            self._task = None
            if self._queue:
                if len(self._queue) >= settings.embedding_query_batch_size:
                    self._ensure_drain_task()
                elif self._timer is None:
                    delay = settings.embedding_query_batch_window_ms / 1000
                    self._timer = self._loop.call_later(delay, self._on_timer)

    async def close(self) -> None:
        self._cancel_timer()
        if self._queue:
            self._ensure_drain_task()
        if self._task is not None:
            await self._task


_batchers: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, _EmbeddingMicroBatcher
] = weakref.WeakKeyDictionary()
_ollama_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = (
    weakref.WeakKeyDictionary()
)
_openai_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = (
    weakref.WeakKeyDictionary()
)


def _base_url() -> str:
    return (settings.embedding_base_url or settings.llm_base_url or "").strip().rstrip("/")


def _api_key() -> str:
    return (settings.embedding_api_key or settings.llm_api_key or "").strip()


def _retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    retryable_words = ("connection", "timeout", "network", "reset by peer", "eof")
    return any(word in text for word in retryable_words)


async def _openai_embeddings(texts: list[str]) -> list[list[float]]:
    from openai import AsyncOpenAI

    loop = asyncio.get_running_loop()
    client = _openai_clients.get(loop)
    if client is None:
        client = AsyncOpenAI(api_key=_api_key() or "not-set", base_url=_base_url() or None)
        _openai_clients[loop] = client
    request: dict[str, Any] = {"model": settings.embedding_model, "input": texts}
    # `dimensions` is an OpenAI text-embedding-3 feature. Most compatible
    # providers (including BGE-M3 endpoints) reject the parameter outright.
    if settings.embedding_model.lower().startswith("text-embedding-3"):
        request["dimensions"] = settings.embedding_dim
    response = await client.embeddings.create(**request)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [list(item.embedding) for item in ordered]


async def _ollama_embeddings(texts: list[str]) -> list[list[float]]:
    from ollama import AsyncClient

    loop = asyncio.get_running_loop()
    client = _ollama_clients.get(loop)
    if client is None:
        kwargs: dict[str, Any] = {"host": settings.ollama_base_url.rstrip("/")}
        if settings.ollama_api_key.strip():
            kwargs["headers"] = {"Authorization": f"Bearer {settings.ollama_api_key.strip()}"}
        client = AsyncClient(**kwargs)
        _ollama_clients[loop] = client
    response = await client.embed(model=settings.embedding_model, input=texts)
    embeddings = response.get("embeddings") if isinstance(response, dict) else response.embeddings
    return [list(vector) for vector in embeddings]


def _validate_vectors(vectors: list[list[float]]) -> list[list[float]]:
    for vector in vectors:
        if len(vector) != settings.embedding_dim:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected {settings.embedding_dim}, got {len(vector)}"
            )
    return vectors


async def _embed_with_retry(texts: list[str]) -> list[list[float]]:
    call = (
        _openai_embeddings
        if settings.embedding_mode == "openai_compatible"
        else _ollama_embeddings
    )
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return _validate_vectors(await call(texts))
        except Exception as exc:
            last_error = exc
            if not _retryable(exc) or attempt == _MAX_ATTEMPTS - 1:
                raise
            delay = _RETRY_BASE_SECONDS * (2**attempt)
            logger.warning(
                "Embedding request failed (attempt {}/{}); retrying in {}s: {}",
                attempt + 1,
                _MAX_ATTEMPTS,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    raise last_error or RuntimeError("Embedding request failed")


def _batcher_for_running_loop() -> _EmbeddingMicroBatcher:
    loop = asyncio.get_running_loop()
    batcher = _batchers.get(loop)
    if batcher is None:
        batcher = _EmbeddingMicroBatcher(loop)
        _batchers[loop] = batcher
    return batcher


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts, micro-batching concurrent online single-query calls."""
    if not texts:
        return []
    if len(texts) == 1 and settings.embedding_query_batch_enabled:
        return [_validate_vectors([await _batcher_for_running_loop().submit(texts[0])])[0]]
    return await _embed_with_retry(texts)


async def close_embedding_clients() -> None:
    """Drain the current loop's batcher and close reusable HTTP clients."""
    loop = asyncio.get_running_loop()
    batcher = _batchers.pop(loop, None)
    if batcher is not None:
        await batcher.close()
    for clients in (_ollama_clients, _openai_clients):
        client = clients.pop(loop, None)
        if client is not None:
            await client.close()
