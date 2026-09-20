"""RAG Service — FastAPI microservice exposing vector retrieval to the TS Agent.

Endpoints:
  POST /rag/query                         — course/personal/enrolled_courses vector retrieval
  POST /rag/generate-quiz                 — quiz generation from retrieved course chunks
  POST /rag/build-mindmap                 — mindmap from parsed Markdown files
  POST /rag/parse-document                — base64 → extracted text (PDF / office / image)
  POST /rag/assignment/regenerate-question — regenerate a single assignment question via RAG
  POST /rag/assignment/complete-question  — complete a teacher-written question stem with AI

Auth: X-Internal-Key header must match RAG_SERVICE_API_KEY env var.
"""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import uvicorn
from boto3 import Session as Boto3Session
from botocore.config import Config as BotocoreConfig
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel, Field

from rag_mvp.logging_setup import configure_logging

load_dotenv()
configure_logging("rag-service")

# ---------------------------------------------------------------------------
# Internal-key auth
# ---------------------------------------------------------------------------

_RAG_KEY = (os.environ.get("RAG_SERVICE_API_KEY") or "").strip()


def _require_key(request: Request) -> None:
    if not _RAG_KEY:
        return  # no key configured → allow all (dev mode)
    given = (request.headers.get("x-internal-key") or "").strip()
    if given != _RAG_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from rag_mvp.vector_store import ensure_schema

    # Schema creation and index checks belong to startup, not the query hot path.
    await asyncio.to_thread(ensure_schema)
    try:
        yield
    finally:
        from rag_mvp.db import close_connection_pool
        from rag_mvp.embedding_factory import close_embedding_clients
        from rag_mvp.vector_store import close_vector_pool

        await close_embedding_clients()
        await asyncio.to_thread(close_connection_pool)
        await asyncio.to_thread(close_vector_pool)


app = FastAPI(title="RAG Service", version="1.0.0", docs_url="/docs", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# /rag/query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    source: str  # "personal" | "course" | "all" | "enrolled_courses"
    user_id: str
    course_id: str | None = None
    accessible_course_ids: list[str] = Field(default_factory=list)
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class HitItem(BaseModel):
    chunk_id: str
    text: str
    origin: str
    course_id: str | None = None
    material_id: str | None = None
    material_title: str | None = None
    relevance_score: float = 0.0
    image_urls: list[dict[str, Any]] = Field(default_factory=list)


class QueryResponse(BaseModel):
    hits: list[HitItem]
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


def _fetch_hit_enrichment(
    material_ids: list[str], chunk_ids: list[str]
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Fetch hit metadata with one pool checkout to reduce high-concurrency contention."""
    from rag_mvp.db import pooled_connection

    titles: dict[str, str] = {}
    images: dict[str, list[dict[str, Any]]] = {}
    page_mappings: dict[str, int] = {}
    try:
        with pooled_connection() as conn, conn.cursor() as cur:
            if material_ids:
                cur.execute(
                    """
                    SELECT id::text, original_filename
                    FROM materials
                    WHERE id = ANY(%s::uuid[]) AND NOT is_deleted
                    """,
                    (material_ids,),
                )
                titles = {row[0]: row[1] for row in cur.fetchall()}

                cur.execute(
                    """
                    SELECT material_id::text, page_idx, minio_url
                    FROM material_images
                    WHERE material_id = ANY(%s::uuid[])
                    ORDER BY material_id, page_idx
                    """,
                    (material_ids,),
                )
                for mid, page_idx, minio_url in cur.fetchall():
                    images.setdefault(mid, []).append(
                        {"page_idx": page_idx, "url": minio_url}
                    )

            if chunk_ids:
                cur.execute(
                    "SELECT chunk_id, page_idx FROM chunk_page_mappings WHERE chunk_id = ANY(%s)",
                    (chunk_ids,),
                )
                page_mappings = {row[0]: row[1] for row in cur.fetchall()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("hit enrichment lookup failed: {}", exc)
    return titles, images, page_mappings


def _enrich_hits(
    raw_hits: list[dict[str, Any]],
    *,
    course_id: str | None,
    material_titles: dict[str, str],
    material_images: dict[str, list[dict[str, Any]]],
    chunk_page_mappings: dict[str, int],
) -> list[HitItem]:
    """Promote nested metadata fields and attach material_title / image_urls from DB lookup."""
    items: list[HitItem] = []
    for h in raw_hits:
        meta = h.get("metadata") or {}
        mid = meta.get("material_id") if isinstance(meta, dict) else None
        chunk_id = str(h.get("chunk_id") or "")
        page_idx = chunk_page_mappings.get(chunk_id)
        if page_idx is not None and mid:
            all_images = material_images.get(mid, [])
            image_urls = [img for img in all_images if img.get("page_idx") == page_idx]
        else:
            image_urls = []
        items.append(
            HitItem(
                chunk_id=chunk_id,
                text=str(h.get("text") or ""),
                origin=str(h.get("origin") or "unknown"),
                course_id=course_id,
                material_id=mid,
                material_title=material_titles.get(mid) if mid else None,
                relevance_score=float(h.get("relevance_score") or 0.0),
                image_urls=image_urls,
            )
        )
    return items


@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(body: QueryRequest, _auth: None = Depends(_require_key)) -> QueryResponse:
    request_started = perf_counter()
    from rag_mvp.engine import (
        course_retrieval_hits,
        personal_retrieval_hits,
    )

    source = body.source.strip().lower()
    top_k = body.top_k
    question = body.question.strip()
    warnings: list[str] = []
    retrieval_timings: dict[str, float] = {}

    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    # Hard guard: single-course sessions must not fan out to enrolled courses;
    # hub sessions must not query course-scoped sources without an active course.
    has_course_context = bool((body.course_id or "").strip())
    if has_course_context and source == "enrolled_courses":
        raise HTTPException(
            status_code=400,
            detail="source=enrolled_courses is not allowed when course_id is present",
        )
    if (not has_course_context) and source in ("course", "all"):
        raise HTTPException(
            status_code=400,
            detail="source=course/all requires course_id",
        )

    raw_hits: list[dict[str, Any]] = []

    if source == "personal":
        raw_hits = await personal_retrieval_hits(body.user_id, question, top_k=top_k)

    elif source == "course":
        if not body.course_id:
            raise HTTPException(status_code=400, detail="course_id required for source=course")
        raw_hits = await course_retrieval_hits(
            body.course_id,
            question,
            top_k=top_k,
            timings_ms=retrieval_timings,
        )

    elif source == "all":
        # personal + current course merged
        personal = await personal_retrieval_hits(body.user_id, question, top_k=top_k)
        course_hits: list[dict[str, Any]] = []
        if body.course_id:
            course_hits = await course_retrieval_hits(
                body.course_id, question, top_k=top_k
            )
        raw_hits = course_hits + personal

    elif source == "enrolled_courses":
        # Query each accessible course, merge and deduplicate by chunk_id
        seen: set[str] = set()
        for cid in (body.accessible_course_ids or []):
            try:
                hits = await course_retrieval_hits(cid, question, top_k=top_k)
                for h in hits:
                    cid_chunk = str(h.get("chunk_id") or "")
                    if cid_chunk and cid_chunk in seen:
                        continue
                    seen.add(cid_chunk)
                    # Tag with originating course_id for downstream enrichment
                    h["_course_id"] = cid
                    raw_hits.append(h)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"course {cid}: {exc}")

        # Sort merged hits by relevance_score descending, keep top_k
        raw_hits.sort(key=lambda h: float(h.get("relevance_score") or 0.0), reverse=True)
        raw_hits = raw_hits[:top_k]

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source={source!r}. Allowed: personal, course, all, enrolled_courses",
        )

    retrieval_finished = perf_counter()

    # Collect material IDs for title and image lookup
    material_ids = list(
        {
            str(h.get("metadata", {}).get("material_id") or "")
            for h in raw_hits
            if isinstance(h.get("metadata"), dict) and h["metadata"].get("material_id")
        }
    )
    chunk_ids = [str(h.get("chunk_id") or "") for h in raw_hits if h.get("chunk_id")]
    material_titles, material_images, chunk_page_mappings = await asyncio.to_thread(
        _fetch_hit_enrichment,
        material_ids,
        chunk_ids,
    )
    enrichment_finished = perf_counter()
    timings_ms = {
        **retrieval_timings,
        "retrieval": (retrieval_finished - request_started) * 1000,
        "enrichment": (enrichment_finished - retrieval_finished) * 1000,
        "total": (enrichment_finished - request_started) * 1000,
    }

    # For enrolled_courses, use per-hit _course_id; otherwise use body.course_id
    if source == "enrolled_courses":
        items: list[HitItem] = []
        for h in raw_hits:
            cid = h.pop("_course_id", None)
            meta = h.get("metadata") or {}
            mid = meta.get("material_id") if isinstance(meta, dict) else None
            chunk_id = str(h.get("chunk_id") or "")
            page_idx = chunk_page_mappings.get(chunk_id)
            if page_idx is not None and mid:
                all_images = material_images.get(mid, [])
                image_urls = [img for img in all_images if img.get("page_idx") == page_idx]
            else:
                image_urls = []
            items.append(
                HitItem(
                    chunk_id=chunk_id,
                    text=str(h.get("text") or ""),
                    origin=str(h.get("origin") or "course"),
                    course_id=cid,
                    material_id=mid,
                    material_title=material_titles.get(mid) if mid else None,
                    relevance_score=float(h.get("relevance_score") or 0.0),
                    image_urls=image_urls,
                )
            )
        return QueryResponse(hits=items, warnings=warnings, timings_ms=timings_ms)

    course_id_for_hits = body.course_id if source in ("course", "all") else None
    hits = _enrich_hits(
        raw_hits,
        course_id=course_id_for_hits,
        material_titles=material_titles,
        material_images=material_images,
        chunk_page_mappings=chunk_page_mappings,
    )
    return QueryResponse(hits=hits, warnings=warnings, timings_ms=timings_ms)


# ---------------------------------------------------------------------------
# /rag/generate-quiz
# ---------------------------------------------------------------------------

class GenerateQuizRequest(BaseModel):
    course_id: str
    count: int = Field(default=5, ge=1, le=20)
    question_type: str = "mixed"


class GenerateQuizResponse(BaseModel):
    questions: list[dict[str, Any]]
    total: int
    generated_at: str


@app.post("/rag/generate-quiz", response_model=GenerateQuizResponse)
async def rag_generate_quiz(
    body: GenerateQuizRequest, _auth: None = Depends(_require_key)
) -> GenerateQuizResponse:
    """Generate review questions from ordinary vector-retrieved course passages."""
    from rag_mvp.engine import course_retrieval_hits
    from rag_mvp.question_gen import DEFAULT_TYPE_WEIGHTS, generate_one

    requested_type = body.question_type.strip().lower()
    if requested_type != "mixed" and requested_type not in DEFAULT_TYPE_WEIGHTS:
        raise HTTPException(status_code=400, detail="Unsupported question_type")

    hits = await course_retrieval_hits(
        body.course_id,
        "课程核心概念 定义 原理 应用 重点",
        top_k=min(body.count * 3, 20),
    )
    if not hits:
        raise HTTPException(status_code=404, detail="No indexed course passages found")

    if requested_type == "mixed":
        question_types = ["single_choice", "fill_blank"]
    else:
        question_types = [requested_type]

    async def _generate(index: int, hit: dict[str, Any]) -> dict[str, Any] | None:
        metadata = hit.get("metadata") or {}
        label = Path(str(metadata.get("file_path") or "课程知识点")).stem or "课程知识点"
        return await generate_one(
            entity_name=label,
            context=str(hit.get("text") or ""),
            q_type=question_types[index % len(question_types)],
            q_id=index + 1,
            score=float(hit.get("relevance_score") or 0.0),
            chunk_ids=[str(hit.get("chunk_id") or "")],
            objective="knowledge",
            entity_names=[label],
        )

    candidates = await asyncio.gather(
        *(_generate(index, hit) for index, hit in enumerate(hits[: body.count * 2]))
    )
    questions = [question for question in candidates if question is not None][: body.count]
    for index, question in enumerate(questions, start=1):
        question["id"] = index
    return GenerateQuizResponse(
        questions=questions,
        total=len(questions),
        generated_at=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# /rag/build-mindmap
# ---------------------------------------------------------------------------

class BuildMindmapRequest(BaseModel):
    source: str
    refine: bool = False


class BuildMindmapResponse(BaseModel):
    markdown: str
    html: str


@app.post("/rag/build-mindmap", response_model=BuildMindmapResponse)
def rag_build_mindmap(
    body: BuildMindmapRequest, _auth: None = Depends(_require_key)
) -> BuildMindmapResponse:
    from rag_mvp.mindmap import build_llm_mindmap, build_structure_mindmap

    build_fn = build_llm_mindmap if body.refine else build_structure_mindmap
    try:
        html_paths = build_fn(body.source)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not html_paths:
        raise HTTPException(status_code=404, detail="No output generated for source")

    html_path = html_paths[0]
    md_path = html_path.with_suffix(".md")

    html = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
    markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    return BuildMindmapResponse(markdown=markdown, html=html)


# ---------------------------------------------------------------------------
# /rag/parse-document
# ---------------------------------------------------------------------------

class ParseDocumentRequest(BaseModel):
    filename: str = "document.pdf"
    base64_content: str


class ParseDocumentResponse(BaseModel):
    text: str
    pages: int | None = None


def _extract_text_pypdf(data: bytes) -> tuple[str, int]:
    """Fast text extraction from PDF using pypdf (no MinerU required)."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = len(reader.pages)
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts), pages


def _extract_text_plaintext(data: bytes, encoding: str = "utf-8") -> tuple[str, None]:
    try:
        return data.decode(encoding, errors="replace"), None
    except Exception:
        return data.decode("latin-1", errors="replace"), None


@app.post("/rag/parse-document", response_model=ParseDocumentResponse)
def rag_parse_document(
    body: ParseDocumentRequest, _auth: None = Depends(_require_key)
) -> ParseDocumentResponse:
    try:
        raw = base64.b64decode(body.base64_content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64_content: {exc}") from exc

    filename = Path(body.filename)
    suffix = filename.suffix.lower()

    # --- PDF ---
    if suffix == ".pdf":
        try:
            text, pages = _extract_text_pypdf(raw)
            return ParseDocumentResponse(text=text, pages=pages)
        except Exception as exc:
            logger.warning("pypdf failed for {}: {}", filename, exc)
            raise HTTPException(status_code=422, detail=f"PDF parse failed: {exc}") from exc

    # --- Plain text / Markdown ---
    if suffix in (".txt", ".md"):
        text, _ = _extract_text_plaintext(raw)
        return ParseDocumentResponse(text=text, pages=None)

    # --- Office / image: use the configured parser provider ---
    _PARSER_SUFFIXES = frozenset({".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".jpg", ".jpeg", ".png"})
    if suffix in _PARSER_SUFFIXES:
        try:
            from rag_mvp.document_parser import parse_document

            with tempfile.TemporaryDirectory(prefix="rag_parse_document_") as tmpdir:
                work_dir = Path(tmpdir)
                tmp_path = work_dir / f"input{suffix}"
                tmp_path.write_bytes(raw)
                document = asyncio.run(
                    parse_document(tmp_path, output_dir=work_dir / "parsed")
                )
                return ParseDocumentResponse(
                    text=document.extracted_text(), pages=document.page_count
                )
        except Exception as exc:
            logger.warning("Document parser failed for {}: {}", filename, exc)
            raise HTTPException(status_code=422, detail=f"Document parse failed: {exc}") from exc

    raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# /rag/assignment/regenerate-question  &  /rag/assignment/complete-question
# ---------------------------------------------------------------------------


class RegenerateQuestionRequest(BaseModel):
    course_id: str
    entity_names: list[str] = Field(default_factory=list)
    q_type: str = "single_choice"
    objective: str = "knowledge"
    q_id: int = 1
    extra_requirements: str = ""
    current_question: str = ""
    difficulty: str = "medium"


class CompleteQuestionRequest(BaseModel):
    course_id: str
    entity_names: list[str] = Field(default_factory=list)
    question_stem: str
    answer_hint: str = ""
    q_type: str = "single_choice"
    objective: str = "knowledge"
    q_id: int = 1
    difficulty: str = "medium"


@app.post("/rag/assignment/regenerate-question")
async def rag_regenerate_question(
    body: RegenerateQuestionRequest,
    _auth: None = Depends(_require_key),
) -> dict[str, Any]:
    from rag_mvp.assignment_gen import regenerate_one_question

    result = await regenerate_one_question(
        course_id=body.course_id,
        entity_names=body.entity_names,
        q_type=body.q_type,
        objective=body.objective,
        q_id=body.q_id,
        extra_requirements=body.extra_requirements,
        current_question=body.current_question,
        difficulty=body.difficulty,
    )
    if result is None:
        raise HTTPException(status_code=502, detail="Question generation failed — check RAG logs")
    return result


@app.post("/rag/assignment/complete-question")
async def rag_complete_question(
    body: CompleteQuestionRequest,
    _auth: None = Depends(_require_key),
) -> dict[str, Any]:
    from rag_mvp.assignment_gen import complete_teacher_question

    result = await complete_teacher_question(
        course_id=body.course_id,
        entity_names=body.entity_names,
        question_stem=body.question_stem,
        answer_hint=body.answer_hint,
        q_type=body.q_type,
        objective=body.objective,
        q_id=body.q_id,
        difficulty=body.difficulty,
    )
    if result is None:
        raise HTTPException(status_code=502, detail="Question completion failed — check RAG logs")
    return result



# ---------------------------------------------------------------------------
# /run-arbitrary-script  (agent run_script tool — user-approved arbitrary code)
# ---------------------------------------------------------------------------

class RunArbitraryScriptRequest(BaseModel):
    language: str           # "python" or "javascript"
    code: str               # source code, max 8000 chars
    timeout_sec: int = Field(default=30, ge=5, le=60)


class RunArbitraryScriptResponse(BaseModel):
    stdout: str
    stderr: str
    return_code: int


_LANG_CMD: dict[str, list[str]] = {
    "python":     ["python"],
    "javascript": ["node"],
}

_LANG_EXT: dict[str, str] = {
    "python":     ".py",
    "javascript": ".js",
}


@app.post("/run-arbitrary-script", response_model=RunArbitraryScriptResponse)
async def run_arbitrary_script(
    body: RunArbitraryScriptRequest,
    _auth: None = Depends(_require_key),
) -> RunArbitraryScriptResponse:
    lang = body.language.strip().lower()
    if lang not in _LANG_CMD:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang!r}")
    if not body.code.strip():
        raise HTTPException(status_code=400, detail="code must not be empty")
    if len(body.code) > 8000:
        raise HTTPException(status_code=400, detail="code exceeds 8000-character limit")

    interpreter = _LANG_CMD[lang]
    ext = _LANG_EXT[lang]

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / f"script{ext}"
        script_path.write_text(body.code, encoding="utf-8")

        cmd = interpreter + [str(script_path)]
        logger.info("run_arbitrary_script: lang={} timeout={}s", lang, body.timeout_sec)

        loop = asyncio.get_running_loop()
        try:
            proc: subprocess.CompletedProcess[str] = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=body.timeout_sec,
                    # Restrict inherited env to avoid leaking secrets
                    env={
                        **{
                            k: v
                            for k, v in os.environ.items()
                            if k in {"PATH", "PYTHONPATH", "HOME", "TEMP", "TMP", "SystemRoot",
                                     "USERPROFILE", "LANG", "LC_ALL"}
                        },
                        # Force UTF-8 I/O so emoji / CJK in print() won't raise
                        # UnicodeEncodeError on Windows (which defaults to GBK).
                        "PYTHONUTF8": "1",
                        "PYTHONIOENCODING": "utf-8",
                    },
                ),
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=408,
                detail=f"Script timed out after {body.timeout_sec}s",
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Interpreter not found: {exc}",
            ) from exc

        return RunArbitraryScriptResponse(
            stdout=proc.stdout[:50_000],
            stderr=proc.stderr[:10_000],
            return_code=proc.returncode,
        )


# ---------------------------------------------------------------------------
# /run-skill-script  (Anthropic skills mechanism — script execution proxy)
# ---------------------------------------------------------------------------

class RunSkillScriptRequest(BaseModel):
    skill: str
    script: str                           # path relative to scripts/ OR "-m module_name"
    args: list[str] = Field(default_factory=list)
    input_file_url: str | None = None     # presigned URL → downloaded to temp dir
    input_filename: str | None = None     # original filename hint (determines extension)
    output_filename: str | None = None    # filename to look for after execution → uploaded to MinIO
    user_id: str = "anonymous"
    timeout_sec: int = Field(default=60, ge=5, le=300)


class RunSkillScriptResponse(BaseModel):
    stdout: str
    stderr: str
    return_code: int
    output_key: str | None = None         # MinIO object key; TS side generates presigned URL


def _skills_root() -> Path:
    """Resolve the skills root directory.
    Defaults to <cwd>/skills; override with SKILLS_ROOT_DIR env var."""
    override = os.environ.get("SKILLS_ROOT_DIR", "").strip()
    return Path(override) if override else Path(os.getcwd()) / "skills"


def _resolve_script_cmd(skills_root: Path, skill: str, script: str) -> list[str]:
    """Return the full command list to execute the script.
    Handles both '-m module' and relative file paths."""
    # Module invocation: "-m markitdown", "-m pptxgenjs", etc.
    if script.startswith("-m "):
        module = script[3:].strip()
        # Allow: letters, digits, hyphens, underscores, dots
        if not all(c.isalnum() or c in ("-", "_", ".") for c in module):
            raise ValueError(f"Invalid module name: {module!r}")
        return ["python", "-m", module]

    # File-based script — enforce path traversal guard
    scripts_dir = (skills_root / skill / "scripts").resolve()
    resolved = (scripts_dir / script).resolve()
    if not str(resolved).startswith(str(scripts_dir) + os.sep) and resolved != scripts_dir:
        raise ValueError(f"Path traversal detected: {script!r}")
    if not resolved.exists():
        raise ValueError(f"Script not found: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".py":
        return ["python", str(resolved)]
    if suffix == ".js":
        return ["node", str(resolved)]
    raise ValueError(f"Unsupported extension {suffix!r}; only .py and .js are allowed.")


def _make_skill_s3_client():
    """Minimal boto3 S3 client using the same MinIO env vars as material_processor."""
    endpoint = os.environ["MINIO_ENDPOINT"].strip()
    if not endpoint.startswith("http"):
        use_ssl = os.environ.get("MINIO_USE_SSL", "true").lower() == "true"
        endpoint = ("https://" if use_ssl else "http://") + endpoint
    session = Boto3Session()
    return session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"].strip(),
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"].strip(),
        region_name=os.environ.get("MINIO_REGION", "us-east-1").strip(),
        config=BotocoreConfig(proxies={}),
    )


@app.post("/run-skill-script", response_model=RunSkillScriptResponse)
async def run_skill_script(
    body: RunSkillScriptRequest,
    _auth: None = Depends(_require_key),
) -> RunSkillScriptResponse:
    # --- Validate skill name (alphanumeric + hyphens/underscores) ---
    skill_safe = body.skill.strip()
    if not skill_safe or not all(c.isalnum() or c in ("-", "_") for c in skill_safe):
        raise HTTPException(status_code=400, detail=f"Invalid skill name: {skill_safe!r}")

    skills_root = _skills_root()
    skill_dir = skills_root / skill_safe
    if not skill_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{skill_safe}' not found (expected at {skill_dir})",
        )
    scripts_dir = skill_dir / "scripts"

    try:
        cmd = _resolve_script_cmd(skills_root, skill_safe, body.script)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_file_path: Path | None = None

        # --- Download input file if provided ---
        if body.input_file_url:
            in_name = (body.input_filename or "input").strip() or "input"
            input_file_path = tmp_path / in_name
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.get(body.input_file_url)
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download input_file_url: HTTP {resp.status_code}",
                    )
                input_file_path.write_bytes(resp.content)

        # --- Substitute {input_file} / {output_file} placeholders in args ---
        effective_args: list[str] = []
        for arg in body.args:
            if arg == "{input_file}":
                effective_args.append(str(input_file_path) if input_file_path else arg)
            elif arg == "{output_file}" and body.output_filename:
                effective_args.append(str(tmp_path / body.output_filename))
            else:
                effective_args.append(arg)

        full_cmd = cmd + effective_args
        cwd = str(scripts_dir) if scripts_dir.exists() else str(skill_dir)
        logger.info("run_skill_script: {} (cwd={})", " ".join(full_cmd), cwd)

        # --- Execute in thread pool to avoid blocking async loop ---
        loop = asyncio.get_running_loop()
        try:
            proc_result: subprocess.CompletedProcess[str] = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    full_cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=body.timeout_sec,
                ),
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=408,
                detail=f"Script timed out after {body.timeout_sec}s",
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Interpreter not found: {exc}",
            ) from exc

        # --- Upload output file to MinIO if requested ---
        output_key: str | None = None
        if body.output_filename:
            out_path = tmp_path / body.output_filename
            if out_path.exists():
                try:
                    s3 = _make_skill_s3_client()
                    bucket = os.environ["MINIO_BUCKET"].strip()
                    output_key = (
                        f"skill-output/{body.user_id}/{uuid.uuid4().hex}/{body.output_filename}"
                    )
                    await loop.run_in_executor(
                        None,
                        lambda: s3.upload_file(str(out_path), bucket, output_key),
                    )
                except Exception as exc:
                    logger.warning("run_skill_script: MinIO upload failed — {}", exc)
                    output_key = None

        return RunSkillScriptResponse(
            stdout=proc_result.stdout[:50_000],
            stderr=proc_result.stderr[:10_000],
            return_code=proc_result.returncode,
            output_key=output_key,
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point (pyproject.toml: rag-service = "rag_service.main:run")
# ---------------------------------------------------------------------------

def run() -> None:
    # Load .env files before reading any config — same order as worker.py.
    # System env vars already set always win (load_dotenv default: override=False).
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, "..", ".."))
    load_dotenv(os.path.join(_root, ".env"))
    load_dotenv(os.path.join(_root, "edu-platform", ".env"))

    host = os.environ.get("RAG_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("RAG_SERVICE_PORT", "8001"))
    logger.info("Starting RAG Service on {}:{}", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
