"""Shared utilities for the agentic-RAG evaluation harness.

Sets up:
  - DB connections for creating eval courses + user
  - vector RAG text-ingest helpers (naive vs full)
  - Path constants
  - Logging
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[2]
EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"
EDU_PLATFORM_DIR = REPO_ROOT / "edu-platform"
PARSED_OUTPUT_DIR = EDU_PLATFORM_DIR / "output" / "parsed"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Bootstrap: load .env and project src/ before importing engine
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Load env, add src to sys.path if needed."""
    from dotenv import load_dotenv
    # prefer project-root .env, then edu-platform/.env
    for p in [REPO_ROOT / ".env", EDU_PLATFORM_DIR / ".env"]:
        if p.exists():
            load_dotenv(p, override=False)
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

_bootstrap()


def non_thinking_extra_body(base_url: str, model: str) -> dict[str, object] | None:
    """Return the provider-specific payload for deterministic non-thinking evals."""
    if "deepseek.com" in base_url or model.lower().startswith("deepseek"):
        return {"thinking": {"type": "disabled"}}
    if "dashscope" in base_url or model.lower().startswith("qwen"):
        return {"enable_thinking": False}
    return None

# ---------------------------------------------------------------------------
# Deterministic IDs for eval infrastructure
# (these are fixed, so re-runs reuse the same DB rows and vector RAG workspace)
# RFC 4122–compliant UUIDs: third group starts with 4 (v4), fourth with 8 (variant 1)
# so they pass the strict UUID regex in edu-platform/lib/course-access.ts
# ---------------------------------------------------------------------------
EVAL_TEACHER_ID   = "e0000001-0000-4000-8000-000000000000"
EVAL_STUDENT_ID   = "e0000002-0000-4000-8000-000000000000"
EVAL_STUDENT_USERNAME = "eval_student"
EVAL_STUDENT_EMAIL    = "eval_student@eval.internal"

# Eval course IDs
COURSE_VECTOR_BASELINE = "c0000001-0000-4000-8000-000000000000"
COURSE_FRAMES         = "c0000003-0000-4000-8000-000000000000"
# For ragas_custom, use the actual production course. Pass via EVAL_RAGAS_COURSE_ID env var.

# ---------------------------------------------------------------------------
# DB helpers: create eval user + courses if they don't exist
# ---------------------------------------------------------------------------

def _get_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL must be set to create eval DB records.")
    return url


def _psycopg_dsn(url: str) -> str:
    """Convert postgresql://... to a psycopg-compatible DSN string.

    Strips Prisma-specific query parameters (e.g. ?schema=public) that
    psycopg does not understand.
    """
    url = url.replace("postgresql+asyncpg", "postgresql").replace("postgresql+psycopg", "postgresql")
    # Remove unsupported query params: psycopg only accepts standard libpq params.
    # Prisma appends ?schema=<name> which causes a ProgrammingError.
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    parsed = urlparse(url)
    if parsed.query:
        _LIBPQ_PARAMS = {
            "host", "port", "dbname", "user", "password",
            "sslmode", "sslcert", "sslkey", "sslrootcert",
            "connect_timeout", "application_name", "options",
        }
        filtered = {k: v for k, v in parse_qs(parsed.query, keep_blank_values=True).items()
                    if k in _LIBPQ_PARAMS}
        new_query = urlencode(filtered, doseq=True)
        parsed = parsed._replace(query=new_query)
        url = urlunparse(parsed)
    return url


def _hash_password(password: str) -> str:
    """Hash a password with argon2id using argon2-cffi."""
    try:
        from argon2 import PasswordHasher
        ph = PasswordHasher(time_cost=2, memory_cost=65536, parallelism=1)
        return ph.hash(password)
    except ImportError:
        # argon2-cffi not available: return a clearly-invalid placeholder.
        # The eval user is never authenticated via password (JWT is generated directly),
        # so this only affects the login endpoint which we don't use.
        placeholder = f"INVALID_HASH_ARGON2_NOT_INSTALLED__{hashlib.sha256(password.encode()).hexdigest()}"
        return placeholder


def setup_eval_db(eval_password: str = "eval_password_not_used_123!") -> None:
    """Idempotently create eval teacher, student, courses, and enrollments in the DB.

    Safe to call multiple times; uses INSERT ... ON CONFLICT DO NOTHING.
    """
    import psycopg

    db_url = _get_db_url()
    dsn = _psycopg_dsn(db_url)
    pw_hash = _hash_password(eval_password)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Teacher user
            cur.execute(
                """
                INSERT INTO users (id, username, email, password_hash, role, real_name, is_active, updated_at)
                VALUES (%s, %s, %s, %s, 'TEACHER', 'Eval Teacher', true, NOW())
                ON CONFLICT (id) DO NOTHING
                """,
                (EVAL_TEACHER_ID, "eval_teacher", "eval_teacher@eval.internal", pw_hash),
            )
            # Student user
            cur.execute(
                """
                INSERT INTO users (id, username, email, password_hash, role, real_name, is_active, updated_at)
                VALUES (%s, %s, %s, %s, 'STUDENT', 'Eval Student', true, NOW())
                ON CONFLICT (id) DO NOTHING
                """,
                (EVAL_STUDENT_ID, EVAL_STUDENT_USERNAME, EVAL_STUDENT_EMAIL, pw_hash),
            )
            # Courses
            courses = [
                (COURSE_VECTOR_BASELINE, "Eval: vector RAG baseline"),
                (COURSE_FRAMES,         "Eval: FRAMES Wikipedia"),
            ]
            for cid, cname in courses:
                cur.execute(
                    """
                    INSERT INTO courses (id, teacher_id, name, status, updated_at)
                    VALUES (%s, %s, %s, 'PUBLISHED', NOW())
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (cid, EVAL_TEACHER_ID, cname),
                )
            # Enroll student in all courses
            for cid, _ in courses:
                cur.execute(
                    """
                    INSERT INTO course_enrollments (id, course_id, student_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (course_id, student_id) DO NOTHING
                    """,
                    (str(uuid.uuid4()), cid, EVAL_STUDENT_ID),
                )
        conn.commit()
    print("[setup_eval_db] Eval DB records created/verified.")


# ---------------------------------------------------------------------------
# vector RAG ingest helpers
# ---------------------------------------------------------------------------

def _vector_dsn() -> str:
    """Return psycopg-compatible DSN for the vector database."""
    dsn = os.environ.get("RAG_PG_DSN", "").strip()
    if not dsn:
        raise RuntimeError("RAG_PG_DSN must be set (points to the vector RAG database).")
    return _psycopg_dsn(dsn)


def is_already_ingested(course_id: str, material_id: str) -> bool:
    """Return True if this material's document exists in the vector store.

    Checks (workspace, full_doc_id) in the KV text-chunk table so we can skip
    re-embedding on a resumed run, avoiding unnecessary embedding API calls.
    """
    import psycopg
    from rag_mvp.engine import material_stable_doc_id
    from rag_mvp.vector_store import course_workspace

    doc_id = material_stable_doc_id(material_id)
    workspace = course_workspace(course_id)
    try:
        with psycopg.connect(_vector_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM rag_documents"
                    " WHERE workspace = %s AND id = %s LIMIT 1",
                    (workspace, doc_id),
                )
                return cur.fetchone() is not None
    except Exception as exc:
        print(f"  [skip-check] Could not query rag_documents: {exc}")
        return False


def ingest_text_naive(
    course_id: str,
    material_id: str,
    text: str,
    original_filename: str,
) -> int:
    """Ingest plain text into the vector index."""
    from rag_mvp.engine import ingest_text_into_course_sync
    return ingest_text_into_course_sync(
        course_id,
        material_id,
        text,
        original_filename=original_filename,
    )


def ingest_text_full(
    course_id: str,
    material_id: str,
    text: str,
    original_filename: str,
) -> int:
    """Compatibility alias for the single vector-index ingest path."""
    return ingest_text_naive(course_id, material_id, text, original_filename)


def run_ingest(coro):
    """Run a single ingest coroutine in a fresh event loop (avoids cross-loop cache)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_json(path: Path | str, data) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save_json] Saved {len(data) if hasattr(data, '__len__') else '?'} records → {p}")


def load_json(path: Path | str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stable material_id helper (deterministic UUID from a string key)
# ---------------------------------------------------------------------------

def stable_material_id(key: str) -> str:
    """Generate a deterministic UUID v5 string for use as a material_id."""
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    return str(uuid.uuid5(ns, key))


# ---------------------------------------------------------------------------
# Ragas-custom eval course (fixed ID from prepare_ragas_custom.py)
# ---------------------------------------------------------------------------
COURSE_RAGAS_CUSTOM = "c8b8787f-9c7e-4f37-bab5-fb94a438d9cf"


# ---------------------------------------------------------------------------
# query_vector_direct: retrieval + controlled LLM call (no aquery_llm)
# ---------------------------------------------------------------------------

# Official vector-RAG benchmark system prompt (from Examples/run_vector.py)
# Used to ensure fair comparison with the published leaderboard results.
# In our pipeline the {context_data} placeholder is moved to the user message;
# {history} is omitted (stateless eval).
_VECTOR_BENCH_SYSTEM_PROMPT = (
    "---Role---\n"
    "You are a helpful assistant responding to user queries.\n\n"
    "---Goal---\n"
    "Generate direct and concise answers based strictly on the provided Knowledge Base.\n"
    "Respond in plain text without explanations or formatting.\n"
    "Maintain conversation continuity and use the same language as the query.\n"
    "If the answer is unknown, respond with \"I don't know\"."
)

_QUERY_REWRITE_PROMPT = (
    "You are a search query optimizer for a computer networking textbook (written in English).\n"
    "Rewrite the user question into a concise English search query optimized for semantic retrieval.\n"
    "Rules:\n"
    "- If the question is in Chinese, translate it to English.\n"
    "- Expand abbreviations and acronyms.\n"
    "- Make the subject explicit if implicit.\n"
    "- Remove filler words; keep it concise and precise.\n"
    "- Output ONLY the rewritten query, nothing else.\n\n"
    "Question: {question}"
)

_QUERY_DECOMPOSE_PROMPT = (
    "You are a query analyzer for a computer networking textbook.\n"
    "Determine if the following question contains multiple distinct sub-topics that should be "
    "retrieved separately (e.g. comparing two different concepts, asking about both causes and effects).\n"
    "Output JSON only — no extra text:\n"
    '  {{"decompose": true,  "sub_queries": ["English sub-query 1", "English sub-query 2"]}}\n'
    '  {{"decompose": false, "sub_queries": []}}\n'
    "Rules: sub_queries must be in English; max 3; only decompose if genuinely distinct "
    "(not just rephrasing the same question).\n\n"
    "Question: {question}"
)


def _rewrite_query_for_retrieval(question: str, llm) -> str:
    """Use LLM to rewrite *question* into an English semantic-search query."""
    from langchain_core.messages import HumanMessage
    try:
        resp = llm.invoke([HumanMessage(content=_QUERY_REWRITE_PROMPT.format(question=question))])
        rewritten = str(resp.content).strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def _decompose_query_for_retrieval(question: str, llm) -> list[str]:
    """Return a list of English sub-queries if the question warrants decomposition, else []."""
    from langchain_core.messages import HumanMessage
    try:
        resp = llm.invoke([HumanMessage(content=_QUERY_DECOMPOSE_PROMPT.format(question=question))])
        text = str(resp.content).strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
        if not data.get("decompose"):
            return []
        sub_queries = [q for q in data.get("sub_queries", []) if isinstance(q, str) and q.strip()]
        return sub_queries[:3] if len(sub_queries) >= 2 else []
    except Exception:
        return []


def _rrf_merge(
    vec_hits: list[dict],
    bm25_hits: list[dict],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[dict]:
    """Reciprocal Rank Fusion merge of two hit lists.

    RRF score = Σ 1 / (k + rank_i + 1)  for each ranked list i.
    Chunks appearing in both lists get scores from both ranks summed.
    """
    scores: dict[str, dict] = {}
    for rank, h in enumerate(vec_hits):
        cid = h.get("chunk_id") or str(rank)
        if cid not in scores:
            scores[cid] = {"hit": h, "score": 0.0}
        scores[cid]["score"] += 1.0 / (k + rank + 1)
    for rank, h in enumerate(bm25_hits):
        cid = h.get("chunk_id") or str(rank)
        if cid not in scores:
            scores[cid] = {"hit": h, "score": 0.0}
        scores[cid]["score"] += 1.0 / (k + rank + 1)
    merged = sorted(scores.values(), key=lambda x: -x["score"])
    result = [m["hit"] for m in merged]
    return result[:top_k] if top_k is not None else result


def _cross_encoder_rerank(
    query: str,
    hits: list[dict],
    *,
    top_k: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict]:
    """Re-rank hits with a Cross-Encoder model (sentence-transformers).

    Falls back to the original RRF order when sentence-transformers is not
    installed.  Install the package to activate:

        pip install sentence-transformers

    The chosen model (~80 MB) is downloaded automatically on first use.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
    except ImportError:
        import logging as _log
        _log.getLogger(__name__).debug(
            "cross-encoder rerank skipped (sentence-transformers not installed); "
            "returning RRF-ranked hits"
        )
        return hits[:top_k]

    encoder = CrossEncoder(model_name)
    pairs = [(query, h.get("text", "")) for h in hits]
    scores = encoder.predict(pairs)  # type: ignore[arg-type]
    ranked = sorted(zip(scores, hits), key=lambda x: -float(x[0]))
    return [h for _, h in ranked[:top_k]]


def query_vector_direct(
    course_id: str,
    question: str,
    *,
    top_k: int = 10,
    enable_rewrite: bool = True,
    enable_decompose: bool = True,
    enable_bm25: bool = True,
    rerank: bool = False,
    official: bool = False,
) -> tuple[str, list[str]]:
    """Retrieve chunks with vector RAG then synthesise an answer via a simple RAG prompt.

    Steps:
      1. (C3) Rewrite query via LLM for better semantic retrieval (translate Chinese→English,
         expand abbreviations, clarify implicit subjects).
      2. (C4) Optionally decompose into sub-queries and merge hits.
      3. Vector retrieval via vector RAG + BM25 full-text, merged with RRF.
      4. Optional cross-encoder re-ranking of merged hits (requires sentence-transformers;
         gracefully falls back to RRF order if the package is not installed).
      5. Synthesise answer using the original *question* so reply language is preserved.

    Args:
      enable_bm25: When False, skip BM25 retrieval (pure vector baseline).
      rerank: When True, apply Cross-Encoder re-ranking after RRF merge.
              Requires `sentence-transformers` (``pip install sentence-transformers``);
              falls back to RRF order silently if the package is missing.
      official: When True, aligns with the official vector-RAG benchmark evaluation protocol:
                - disables query rewrite and decompose (use raw question for retrieval)
                - disables BM25 (official benchmark uses pure vector retrieval)
                - uses the official English system prompt (VECTOR_BENCH_SYSTEM_PROMPT)
                This ensures results are comparable to the published leaderboard.

    Returns (answer_text, [chunk_text, ...]).
    """
    if official:
        enable_rewrite = False
        enable_decompose = False
        enable_bm25 = False
    from rag_mvp.engine import course_retrieval_hits_sync, course_bm25_hits_sync
    from rag_mvp.config import settings
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import SecretStr

    extra_body = non_thinking_extra_body(
        settings.effective_chat_base_url,
        settings.effective_chat_model,
    )
    llm = ChatOpenAI(
        model=settings.effective_chat_model,
        api_key=SecretStr(settings.effective_chat_api_key or "placeholder"),
        base_url=settings.effective_chat_base_url,
        temperature=0.0,
        extra_body=extra_body,
    )

    # C3: Rewrite query for better semantic retrieval
    retrieval_query = _rewrite_query_for_retrieval(question, llm) if enable_rewrite else question

    # C4: Decompose into sub-queries; collect vector hits (deduped by chunk_id)
    sub_queries: list[str] = []
    if enable_decompose:
        sub_queries = _decompose_query_for_retrieval(retrieval_query, llm)

    queries_to_run = sub_queries if sub_queries else [retrieval_query]

    # Over-request by 2x to compensate for image/code chunks filtered by _hits_from_aquery_chunks
    vec_fetch_k = top_k * 2

    all_vec_hits: list[dict] = []
    seen_vec_ids: set[str] = set()
    for sq in queries_to_run:
        for h in course_retrieval_hits_sync(course_id, sq, top_k=vec_fetch_k):
            cid = h.get("chunk_id", "")
            if cid not in seen_vec_ids:
                seen_vec_ids.add(cid)
                all_vec_hits.append(h)

    # A: BM25 full-text retrieval on the rewritten query (lexical signal)
    bm25_hits: list[dict] = []
    if enable_bm25:
        try:
            bm25_hits = course_bm25_hits_sync(course_id, retrieval_query, top_k=top_k)
        except Exception as _exc:
            import logging as _logging
            _logging.getLogger(__name__).warning("BM25 retrieval failed: %s", _exc)

    # RRF merge: combine vector and BM25 hits
    # Over-fetch for the cross-encoder: give it 2x candidates to re-score
    rrf_fetch_k = top_k * 2 if rerank else top_k
    hits = _rrf_merge(all_vec_hits, bm25_hits, k=60, top_k=rrf_fetch_k)

    # Optional Cross-Encoder re-ranking (falls back to RRF order if not installed)
    if rerank:
        hits = _cross_encoder_rerank(retrieval_query, hits, top_k=top_k)

    chunk_texts = [h["text"] for h in hits if h.get("text", "").strip()]

    if chunk_texts:
        context_block = "\n\n".join(
            f"[{i + 1}] {t.strip()}" for i, t in enumerate(chunk_texts)
        )
    else:
        context_block = "(No relevant content retrieved)" if official else "（未检索到相关内容）"

    if official:
        # Official vector-RAG benchmark format: system prompt is role+goal only;
        # context and question are both in the user message (mirrors run_vector.py behaviour).
        messages = [
            SystemMessage(content=_VECTOR_BENCH_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"---Knowledge Base---\n{context_block}\n\n"
                f"{question}"
            )),
        ]
    else:
        messages = [
            SystemMessage(content=(
                "根据提供的上下文简洁回答问题，使用与问题相同的语言作答。"
                "若上下文不足以支撑回答，直接回答 \"I don't know\"，不要编造内容。"
            )),
            HumanMessage(content=f"上下文：\n{context_block}\n\n问题：{question}"),
        ]
    response = llm.invoke(messages)
    answer = str(response.content).strip()
    return answer, chunk_texts


# ---------------------------------------------------------------------------
# make_eval_jwt: sign HS256 JWT for EVAL_STUDENT using stdlib only
# ---------------------------------------------------------------------------

def make_eval_jwt(
    user_id: str = EVAL_STUDENT_ID,
    username: str = EVAL_STUDENT_USERNAME,
    role: str = "STUDENT",
    ttl_seconds: int = 3600,
) -> str:
    """Create a signed HS256 JWT for the eval student (no external JWT library needed).

    Reads JWT_SECRET and JWT_ISS from the environment (loaded by _bootstrap).
    """
    import base64
    import hmac
    import hashlib
    import time as _time

    secret = os.environ.get("JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "JWT_SECRET not found in environment. "
            "Ensure edu-platform/.env is loaded by _bootstrap()."
        )
    iss = os.environ.get("JWT_ISS", "edu-platform")
    now = int(_time.time())

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(json.dumps(
        {"sub": user_id, "username": username, "role": role,
         "iat": now, "exp": now + ttl_seconds, "iss": iss},
        separators=(",", ":"),
    ).encode())
    sig_input = f"{header}.{payload}".encode()
    signature = _b64url(
        hmac.new(secret.encode(), sig_input, hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{signature}"


# ---------------------------------------------------------------------------
# ensure_course_enrollment: idempotent enrol eval student in a course
# ---------------------------------------------------------------------------

def ensure_course_enrollment(course_id: str) -> None:
    """Idempotently enrol EVAL_STUDENT_ID in *course_id*.

    The eval student row must already exist (created by setup_eval_db).
    This is needed so the TS chat endpoint passes getCourseIfMember().
    """
    import psycopg

    db_url = _get_db_url()
    dsn = _psycopg_dsn(db_url)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO course_enrollments (id, course_id, student_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (course_id, student_id) DO NOTHING
                """,
                (str(uuid.uuid4()), course_id, EVAL_STUDENT_ID),
            )
        conn.commit()
    print(f"[ensure_course_enrollment] eval_student enrolled in course {course_id}.")
