"""High-level parsing, vector indexing, retrieval, and local CLI helpers."""

from __future__ import annotations

import asyncio
import base64
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import settings
from .document_parser import parse_document
from .llm import ensure_embedding_backend_reachable, llm_chat_model_func, vision_model_func
from .multimodal_surrogate_chunks import content_item_to_surrogate_text_async
from .parsed_document import PARSED_DOCUMENT_FILENAME, load_parsed_document
from .text_chunking import split_text
from .vector_store import (
    VectorChunk,
    bm25_search,
    clear_workspace,
    course_workspace,
    delete_document,
    document_id,
    personal_workspace,
    replace_document,
    vector_search,
)

_SUPPORTED_SUFFIXES = frozenset(
    {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt", ".md", ".jpg", ".jpeg", ".png"}
)


def material_stable_doc_id(material_id: str) -> str:
    return document_id(material_id)


def personal_user_to_workspace(user_id: str) -> str:
    return personal_workspace(user_id)


def _invalidate_personal_rag_cache(user_id: str | None = None) -> None:
    """Compatibility no-op; the vector store has no process-local storage cache."""


def _invalidate_course_rag_cache_for(course_id: str) -> None:
    """Compatibility no-op; the vector store has no process-local storage cache."""


def _sanitize_material_display_stem(raw: str) -> str:
    value = re.sub(r"\s+", "_", raw.strip())
    value = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "document")[:60]


def _make_material_file_path(material_id: str, display_stem: str) -> str:
    return f"mat_{material_id}_{_sanitize_material_display_stem(display_stem)}"


def _material_id_from_course_file_path(file_path: str | None) -> str | None:
    match = re.search(r"^mat_([0-9a-fA-F-]{36})_", str(file_path or ""))
    return match.group(1) if match else None


async def _content_list_chunks(
    content_list: list[dict[str, Any]],
    *,
    file_path: str,
    text_only: bool,
    order_base: int,
) -> list[VectorChunk]:
    chunks: list[VectorChunk] = []
    for item in content_list:
        content_type = str(item.get("type") or "unknown").strip()
        if text_only and content_type != "text":
            continue
        text = await content_item_to_surrogate_text_async(
            item,
            use_vlm_for_images=(
                not text_only and content_type == "image" and settings.ingest_surrogate_image_vlm
            ),
        )
        page_raw = item.get("page_idx")
        try:
            page_idx = int(page_raw) if page_raw is not None else None
        except (TypeError, ValueError):
            page_idx = None
        for piece in split_text(text):
            chunks.append(
                VectorChunk(
                    content=piece,
                    file_path=file_path,
                    order_index=order_base + len(chunks),
                    page_idx=page_idx,
                    metadata={"content_type": content_type},
                )
            )
    return chunks


async def _load_parsed_chunks(
    material_id: str,
    source_file: Path,
    original_filename: str | None,
    *,
    text_only: bool,
) -> tuple[list[VectorChunk], str]:
    scan_dir = settings.output_dir / source_file.stem
    document = load_parsed_document(scan_dir)
    groups = document.grouped_content_items()
    all_chunks: list[VectorChunk] = []
    first_file_path = ""
    for sub_stem, content_list in groups:
        display_stem = Path(original_filename).stem if original_filename else sub_stem
        rag_file_path = _make_material_file_path(
            material_id, sub_stem if len(groups) > 1 else display_stem
        )
        first_file_path = first_file_path or rag_file_path
        file_chunks = await _content_list_chunks(
            content_list,
            file_path=rag_file_path,
            text_only=text_only,
            order_base=len(all_chunks),
        )
        all_chunks.extend(file_chunks)
    return all_chunks, first_file_path


async def ingest_parsed_material_into_course_async(
    course_id: str,
    material_id: str,
    source_file: Path,
    original_filename: str | None = None,
    text_only: bool = True,
) -> int:
    ensure_embedding_backend_reachable()
    chunks, file_path = await _load_parsed_chunks(
        material_id, source_file, original_filename, text_only=text_only
    )
    for chunk in chunks:
        chunk.metadata["material_id"] = material_id
    return await replace_document(
        course_workspace(course_id),
        material_stable_doc_id(material_id),
        chunks,
        file_path=file_path,
        metadata={"material_id": material_id, "scope": "course"},
    )


def ingest_parsed_material_into_course_sync(
    course_id: str,
    material_id: str,
    source_file: Path,
    original_filename: str | None = None,
    text_only: bool = True,
) -> int:
    return asyncio.run(
        ingest_parsed_material_into_course_async(
            course_id, material_id, source_file, original_filename, text_only
        )
    )


async def ingest_parsed_material_into_personal_async(
    user_id: str,
    material_id: str,
    source_file: Path,
    original_filename: str | None = None,
    text_only: bool = True,
) -> int:
    ensure_embedding_backend_reachable()
    chunks, file_path = await _load_parsed_chunks(
        material_id, source_file, original_filename, text_only=text_only
    )
    for chunk in chunks:
        chunk.metadata["material_id"] = material_id
    return await replace_document(
        personal_workspace(user_id),
        material_stable_doc_id(material_id),
        chunks,
        file_path=file_path,
        metadata={"material_id": material_id, "scope": "personal"},
    )


def ingest_parsed_material_into_personal_sync(
    user_id: str,
    material_id: str,
    source_file: Path,
    original_filename: str | None = None,
    text_only: bool = True,
) -> int:
    return asyncio.run(
        ingest_parsed_material_into_personal_async(
            user_id, material_id, source_file, original_filename, text_only
        )
    )


def _plain_text_chunks(material_id: str, text: str, original_filename: str | None) -> tuple[list[VectorChunk], str]:
    display = Path(original_filename).stem if original_filename else material_id
    file_path = _make_material_file_path(material_id, display)
    chunks = [
        VectorChunk(piece, file_path, index, metadata={"material_id": material_id, "content_type": "text"})
        for index, piece in enumerate(split_text(text))
    ]
    return chunks, file_path


async def ingest_text_into_course_async(
    course_id: str,
    material_id: str,
    text: str,
    original_filename: str | None = None,
) -> int:
    chunks, file_path = _plain_text_chunks(material_id, text, original_filename)
    return await replace_document(
        course_workspace(course_id), material_stable_doc_id(material_id), chunks, file_path=file_path
    )


def ingest_text_into_course_sync(
    course_id: str,
    material_id: str,
    text: str,
    original_filename: str | None = None,
) -> int:
    return asyncio.run(ingest_text_into_course_async(course_id, material_id, text, original_filename))


async def ingest_text_into_personal_async(
    user_id: str,
    material_id: str,
    text: str,
    original_filename: str | None = None,
) -> int:
    chunks, file_path = _plain_text_chunks(material_id, text, original_filename)
    return await replace_document(
        personal_workspace(user_id), material_stable_doc_id(material_id), chunks, file_path=file_path
    )


def ingest_text_into_personal_sync(
    user_id: str,
    material_id: str,
    text: str,
    original_filename: str | None = None,
) -> int:
    return asyncio.run(ingest_text_into_personal_async(user_id, material_id, text, original_filename))


async def delete_material_course_async(course_id: str, material_id: str) -> None:
    await delete_document(course_workspace(course_id), material_stable_doc_id(material_id))


def delete_material_course_sync(course_id: str, material_id: str) -> None:
    asyncio.run(delete_material_course_async(course_id, material_id))


async def delete_material_personal_async(user_id: str, material_id: str) -> None:
    await delete_document(personal_workspace(user_id), material_stable_doc_id(material_id))


def delete_material_personal_sync(user_id: str, material_id: str) -> None:
    asyncio.run(delete_material_personal_async(user_id, material_id))


def _normalise_hits(hits: list[dict[str, Any]], *, origin: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for hit in hits:
        metadata = dict(hit.get("metadata") or {})
        metadata.setdefault("material_id", _material_id_from_course_file_path(metadata.get("file_path")))
        result.append({**hit, "metadata": metadata, "origin": origin})
    return result


def _rrf_merge(
    vector_hits: list[dict[str, Any]], lexical_hits: list[dict[str, Any]], *, top_k: int
) -> list[dict[str, Any]]:
    """Fuse dense and lexical ranks while retaining their diagnostic scores."""
    merged: dict[str, dict[str, Any]] = {}
    for source, hits in (("vector", vector_hits), ("lexical", lexical_hits)):
        for rank, hit in enumerate(hits, start=1):
            chunk_key = str(hit.get("chunk_id") or "")
            if not chunk_key:
                continue
            item = merged.setdefault(
                chunk_key,
                {**hit, "retrieval_sources": [], "fusion_score": 0.0},
            )
            item["retrieval_sources"].append(source)
            item[f"{source}_score"] = float(hit.get("relevance_score") or 0.0)
            item["fusion_score"] += 1.0 / (60 + rank)
    ranked = sorted(merged.values(), key=lambda item: item["fusion_score"], reverse=True)
    if ranked:
        best = float(ranked[0]["fusion_score"])
        for item in ranked:
            item["relevance_score"] = float(item["fusion_score"]) / best
    return ranked[:top_k]


async def _hybrid_workspace_search(
    workspace: str,
    question: str,
    *,
    top_k: int,
    timings_ms: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    fetch_k = min(max(top_k * 2, top_k), 40)
    loop = asyncio.get_running_loop()

    async def _dense() -> list[dict[str, Any]]:
        started = loop.time()
        result = await vector_search(
            workspace,
            question.strip(),
            top_k=fetch_k,
            timings_ms=timings_ms,
        )
        if timings_ms is not None:
            timings_ms["dense_total"] = (loop.time() - started) * 1000
        return result

    async def _lexical() -> list[dict[str, Any]]:
        started = loop.time()
        result = await asyncio.to_thread(
            bm25_search,
            workspace,
            question.strip(),
            top_k=fetch_k,
        )
        if timings_ms is not None:
            timings_ms["bm25"] = (loop.time() - started) * 1000
        return result

    dense, lexical = await asyncio.gather(_dense(), _lexical())
    return _rrf_merge(dense, lexical, top_k=top_k)


async def course_retrieval_hits(
    course_id: str,
    question: str,
    *,
    top_k: int,
    timings_ms: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    hits = await _hybrid_workspace_search(
        course_workspace(course_id),
        question,
        top_k=top_k,
        timings_ms=timings_ms,
    )
    return _normalise_hits(hits, origin="course")


async def personal_retrieval_hits(
    user_id: str,
    question: str,
    *,
    top_k: int,
    timings_ms: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    hits = await _hybrid_workspace_search(
        personal_workspace(user_id),
        question,
        top_k=top_k,
        timings_ms=timings_ms,
    )
    return _normalise_hits(hits, origin="personal")


def course_retrieval_hits_sync(
    course_id: str, question: str, *, top_k: int
) -> list[dict[str, Any]]:
    return asyncio.run(course_retrieval_hits(course_id, question, top_k=top_k))


def personal_retrieval_hits_sync(
    user_id: str, question: str, *, top_k: int
) -> list[dict[str, Any]]:
    return asyncio.run(personal_retrieval_hits(user_id, question, top_k=top_k))


def course_bm25_hits_sync(course_id: str, query: str, *, top_k: int) -> list[dict[str, Any]]:
    return _normalise_hits(
        bm25_search(course_workspace(course_id), query, top_k=top_k), origin="course"
    )


async def course_aquery_data(
    course_id: str, question: str, *, top_k: int
) -> dict[str, Any]:
    hits = await course_retrieval_hits(course_id, question, top_k=top_k)
    chunks = [
        {
            "id": hit["chunk_id"],
            "chunk_id": hit["chunk_id"],
            "content": hit["text"],
            "file_path": (hit.get("metadata") or {}).get("file_path"),
            "relevance_score": hit.get("relevance_score", 0.0),
        }
        for hit in hits
    ]
    return {"data": {"chunks": chunks}}


async def personal_aquery_data(
    user_id: str, question: str, *, top_k: int
) -> dict[str, Any]:
    hits = await personal_retrieval_hits(user_id, question, top_k=top_k)
    return {
        "data": {
            "chunks": [
                {
                    "id": hit["chunk_id"],
                    "chunk_id": hit["chunk_id"],
                    "content": hit["text"],
                    "file_path": (hit.get("metadata") or {}).get("file_path"),
                    "relevance_score": hit.get("relevance_score", 0.0),
                }
                for hit in hits
            ]
        }
    }


def parse_file(file_path: str | Path) -> None:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    asyncio.run(parse_document(path))


def parse_folder(folder_path: str | Path) -> None:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    files = [path for path in folder.rglob("*") if path.suffix.lower() in _SUPPORTED_SUFFIXES]

    async def _run() -> None:
        for path in files:
            await parse_document(path)

    asyncio.run(_run())


def ingest_file(file_path: str | Path) -> None:
    path = Path(file_path)
    parse_file(path)
    material_id = stable_local_material_id(path)
    ingest_parsed_material_into_personal_sync(
        "local", material_id, path, original_filename=path.name, text_only=False
    )


def ingest_folder(folder_path: str | Path) -> None:
    folder = Path(folder_path)
    for path in folder.rglob("*"):
        if path.suffix.lower() in _SUPPORTED_SUFFIXES:
            ingest_file(path)


def stable_local_material_id(path: Path) -> str:
    import hashlib

    digest = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def reindex_from_cache(output_dir: str | Path | None = None, file_path: str | Path | None = None) -> None:
    scan_dir = Path(output_dir) if output_dir else settings.output_dir
    artifact_dirs = {path.parent for path in scan_dir.rglob(PARSED_DOCUMENT_FILENAME)}
    for path in scan_dir.rglob("*_content_list.json"):
        if "_content_list_v2" in path.name:
            continue
        relative = path.relative_to(scan_dir)
        artifact_dirs.add(scan_dir / relative.parts[0] if len(relative.parts) > 1 else path.parent)
    if file_path is not None:
        stem = Path(file_path).stem
        artifact_dirs = {path for path in artifact_dirs if path.name == stem or stem in path.parts}
    seen_stems: set[str] = set()
    for artifact_dir in sorted(artifact_dirs):
        source = Path(file_path) if file_path else Path(f"{artifact_dir.name}.pdf")
        if source.stem in seen_stems:
            continue
        seen_stems.add(source.stem)
        material_id = stable_local_material_id(source)
        ingest_parsed_material_into_personal_sync(
            "local", material_id, source, original_filename=source.name, text_only=False
        )


def clear_storage() -> None:
    asyncio.run(clear_workspace(personal_workspace("local")))


def _context_from_hits(hits: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(
        f"[{index}] {hit['text']}" for index, hit in enumerate(hits, start=1)
    ) or "(No retrieved passages.)"


def query(
    question: str,
    *,
    with_refs: bool = False,
    image_paths: Sequence[Path] | None = None,
) -> str | dict[str, Any]:
    async def _run() -> str | dict[str, Any]:
        hits = await personal_retrieval_hits("local", question, top_k=8)
        context = _context_from_hits(hits)
        system = "Answer using only the retrieved passages. If they are insufficient, say so."
        prompt = f"Retrieved passages:\n{context}\n\nQuestion: {question}"
        if image_paths:
            messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for path in image_paths:
                mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            messages.append({"role": "user", "content": content})
            answer = await vision_model_func(prompt, system_prompt=system, messages=messages)
        else:
            answer = await llm_chat_model_func(prompt, system_prompt=system)
        if not with_refs:
            return answer
        return {
            "answer": answer,
            "chunks": [
                {
                    "reference_id": index,
                    "chunk_id": hit["chunk_id"],
                    "content": hit["text"],
                    "file_path": (hit.get("metadata") or {}).get("file_path", ""),
                }
                for index, hit in enumerate(hits, start=1)
            ],
            "entities": [],
            "references": [
                {"reference_id": index, "file_path": (hit.get("metadata") or {}).get("file_path", "")}
                for index, hit in enumerate(hits, start=1)
            ],
        }

    return asyncio.run(_run())
