from __future__ import annotations

from rag_mvp import vector_store


def test_ensure_schema_runs_once_per_dsn(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(vector_store, "_schema_ready_dsn", None)
    monkeypatch.setattr(vector_store, "_dsn", lambda: "postgresql://test/one")
    monkeypatch.setattr(vector_store, "_ensure_schema_once", calls.append)

    vector_store.ensure_schema()
    vector_store.ensure_schema()

    assert calls == ["postgresql://test/one"]


def test_ensure_schema_rechecks_when_dsn_changes(monkeypatch) -> None:
    calls: list[str] = []
    current = {"dsn": "postgresql://test/one"}
    monkeypatch.setattr(vector_store, "_schema_ready_dsn", None)
    monkeypatch.setattr(vector_store, "_dsn", lambda: current["dsn"])
    monkeypatch.setattr(vector_store, "_ensure_schema_once", calls.append)

    vector_store.ensure_schema()
    current["dsn"] = "postgresql://test/two"
    vector_store.ensure_schema()

    assert calls == ["postgresql://test/one", "postgresql://test/two"]
