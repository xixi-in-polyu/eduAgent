from __future__ import annotations

import asyncio

import pytest

from rag_mvp import embedding_factory


def _vector(value: float) -> list[float]:
    return [value] * embedding_factory.settings.embedding_dim


@pytest.mark.asyncio
async def test_concurrent_single_queries_are_micro_batched(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [_vector(float(index)) for index, _text in enumerate(texts)]

    monkeypatch.setattr(embedding_factory.settings, "embedding_query_batch_enabled", True)
    monkeypatch.setattr(embedding_factory.settings, "embedding_query_batch_window_ms", 20)
    monkeypatch.setattr(embedding_factory.settings, "embedding_query_batch_size", 32)
    monkeypatch.setattr(embedding_factory, "_embed_with_retry", fake_embed)

    results = await asyncio.gather(
        *(embedding_factory.embed_texts([f"query-{index}"]) for index in range(10))
    )

    assert calls == [[f"query-{index}" for index in range(10)]]
    assert len(results) == 10
    assert all(len(result) == 1 for result in results)
    await embedding_factory.close_embedding_clients()


@pytest.mark.asyncio
async def test_existing_multi_text_batches_bypass_query_batcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [_vector(1.0) for _text in texts]

    monkeypatch.setattr(embedding_factory.settings, "embedding_query_batch_enabled", True)
    monkeypatch.setattr(embedding_factory, "_embed_with_retry", fake_embed)

    results = await embedding_factory.embed_texts(["chunk-a", "chunk-b"])

    assert calls == [["chunk-a", "chunk-b"]]
    assert len(results) == 2


@pytest.mark.asyncio
async def test_micro_batch_size_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        await asyncio.sleep(0)
        return [_vector(1.0) for _text in texts]

    monkeypatch.setattr(embedding_factory.settings, "embedding_query_batch_enabled", True)
    monkeypatch.setattr(embedding_factory.settings, "embedding_query_batch_window_ms", 20)
    monkeypatch.setattr(embedding_factory.settings, "embedding_query_batch_size", 2)
    monkeypatch.setattr(embedding_factory, "_embed_with_retry", fake_embed)

    await asyncio.gather(
        *(embedding_factory.embed_texts([f"query-{index}"]) for index in range(5))
    )

    assert [len(call) for call in calls] == [2, 2, 1]
    assert [text for call in calls for text in call] == [f"query-{index}" for index in range(5)]
    await embedding_factory.close_embedding_clients()
