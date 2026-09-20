"""Text and vision model clients used by the application."""

import base64
import contextvars
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI

from .config import settings
from .http_env import ensure_loopback_bypass_http_proxy
from .tracing import end_span, generation, get_current_trace


async def openai_complete(
    model: str,
    prompt: str,
    *,
    system_prompt: str | None,
    history_messages: list | None,
    api_key: str,
    base_url: str,
    **kwargs,
) -> str:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})
    client = AsyncOpenAI(api_key=api_key or "not-set", base_url=base_url or None)
    trace_generation = generation(
        get_current_trace(),
        name=f"llm:{_llm_role.get('default')}",
        model=model,
        input=[
            {
                "role": message.get("role"),
                "content": str(message.get("content", ""))[:4000],
            }
            for message in messages
        ],
        metadata={"provider_base_url": base_url},
    )
    try:
        response = await client.chat.completions.create(model=model, messages=messages, **kwargs)
        content = response.choices[0].message.content or ""
        usage = response.usage
        end_span(
            trace_generation,
            output=content[:8000],
            usage_details={
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
            } if usage is not None else None,
        )
        return content
    except Exception as exc:
        end_span(trace_generation, level="ERROR", status_message=str(exc))
        raise

# Before any httpx / ollama calls: so Ollama and other loopback URLs ignore HTTP_PROXY.
ensure_loopback_bypass_http_proxy()


def ensure_ollama_embedding_reachable(*, timeout: float = 3.0) -> None:
    """Fail fast before vector indexing if Ollama is down.

    Only runs when ``EMBEDDING_MODE=ollama``. Uses ``OLLAMA_BASE_URL``.
    Set ``RAG_SKIP_OLLAMA_PREFLIGHT=1`` to skip (e.g. isolated unit tests).
    Set ``RAG_SKIP_EMBEDDING_PREFLIGHT=1`` to skip all embedding pref lights.
    """
    import os

    if settings.embedding_mode != "ollama":
        return
    if os.environ.get("RAG_SKIP_EMBEDDING_PREFLIGHT", "").strip() in ("1", "true", "yes"):
        return
    if os.environ.get("RAG_SKIP_OLLAMA_PREFLIGHT", "").strip() in ("1", "true", "yes"):
        return
    base = settings.ollama_base_url.rstrip("/")
    url = f"{base}/api/tags"
    headers: dict[str, str] = {}
    key = settings.ollama_api_key.strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            r = client.get(url, headers=headers or None)
            r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base} (GET {url} failed). "
            "Course material indexing requires a running Ollama with the embedding model "
            f"pulled (e.g. `ollama pull {settings.embedding_model}`). "
            "See https://ollama.com/download — "
            "or set RAG_SKIP_OLLAMA_PREFLIGHT=1 only for tests."
        ) from exc


def ensure_openai_compatible_embedding_reachable(*, timeout: float = 5.0) -> None:
    """Optional reachability hint for OpenAI-compatible embedding base URL (no hard fail).

    When ``EMBEDDING_MODE=openai_compatible``, performs a lightweight GET ``/models``.
    Set ``RAG_SKIP_EMBEDDING_PREFLIGHT=1`` to skip.
    """
    import os

    if settings.embedding_mode != "openai_compatible":
        return
    if os.environ.get("RAG_SKIP_EMBEDDING_PREFLIGHT", "").strip() in ("1", "true", "yes"):
        return

    base = (settings.embedding_base_url or settings.llm_base_url or "").strip().rstrip("/")
    if not base:
        logger.warning(
            "[embedding] EMBEDDING_MODE=openai_compatible but LLM_BASE_URL / EMBEDDING_BASE_URL is empty."
        )
        return

    url = f"{base}/models"
    headers: dict[str, str] = {}
    key = (settings.embedding_api_key or settings.llm_api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            r = client.get(url, headers=headers or None)
            if r.status_code >= 400:
                logger.warning(
                    f"[embedding] OpenAI-compatible preflight GET {url} returned {r.status_code}; "
                    "embeddings may still work if your provider omits /models."
                )
    except Exception as exc:
        logger.warning(
            f"[embedding] OpenAI-compatible preflight failed ({url}): {exc}. "
            "Continuing; set RAG_SKIP_EMBEDDING_PREFLIGHT=1 to silence."
        )


def ensure_embedding_backend_reachable(*, timeout: float = 3.0) -> None:
    """Run the appropriate embedding preflight for ``EMBEDDING_MODE``."""
    import os

    if os.environ.get("RAG_SKIP_EMBEDDING_PREFLIGHT", "").strip() in ("1", "true", "yes"):
        return
    if settings.embedding_mode == "ollama":
        ensure_ollama_embedding_reachable(timeout=timeout)
    elif settings.embedding_mode == "openai_compatible":
        ensure_openai_compatible_embedding_reachable(timeout=max(timeout, 5.0))


# ---------------------------------------------------------------------------
# Inject model-role context into rate-limit log lines.
# ---------------------------------------------------------------------------

_llm_role: contextvars.ContextVar[str] = contextvars.ContextVar("_llm_role", default="")


class _RolePrefixFilter(logging.Filter):
    """Prepend the current LLM role to rate-limit error messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        role = _llm_role.get("")
        if role and "Rate Limit" in record.getMessage():
            record.msg = f"[{role}] {record.msg}"
            record.args = ()  # args already interpolated above via getMessage()
        return True


def image_mime_type_for_suffix(suffix: str) -> str:
    """Return an IANA image MIME type for a file suffix (e.g. ``.png`` → ``image/png``)."""
    s = (suffix or "").lower()
    if s in (".jpg", ".jpeg"):
        return "image/jpeg"
    if s == ".png":
        return "image/png"
    return "image/jpeg"


def build_data_uri_from_image_path(path: Path | str) -> str:
    """Read an image file and return a ``data:{mime};base64,...`` URL for multimodal APIs."""
    p = Path(path)
    mime = image_mime_type_for_suffix(p.suffix)
    raw = p.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Debug logging helper — enabled by LLM_DEBUG=true in .env
# ---------------------------------------------------------------------------

def _dbg(
    role: str,
    prompt: str,
    kwargs: dict,
    *,
    result: str | None = None,
    exc: BaseException | None = None,
) -> None:
    """Log LLM call details when ``LLM_DEBUG=true``.

    Truncates prompt/result to ``LLM_DEBUG_MAX_CHARS`` (0 = unlimited).
    Never logs API keys.
    """
    if not settings.llm_debug:
        return
    n = settings.llm_debug_max_chars
    def _clip(s: str) -> str:
        s = (s or "").replace("\n", "↵")
        return s[:n] + "…" if n and len(s) > n else s

    # Strip sensitive keys from kwargs before logging
    safe_kwargs = {k: v for k, v in kwargs.items() if k not in ("api_key",)}

    if exc is not None:
        logger.debug(f"[LLM/{role}] ERROR  prompt={_clip(prompt)!r}  kwargs={safe_kwargs}  exc={exc!r}")
    elif result is not None:
        logger.debug(f"[LLM/{role}] OK     prompt={_clip(prompt)!r}  kwargs={safe_kwargs}  result={_clip(result)!r}")
    else:
        logger.debug(f"[LLM/{role}] CALL   prompt={_clip(prompt)!r}  kwargs={safe_kwargs}")


async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    """Default async text-generation model used by CLI and utility flows."""
    kwargs.setdefault("max_tokens", settings.llm_max_tokens)
    kwargs.setdefault("temperature", settings.llm_temperature)
    model = settings.llm_model
    if settings.llm_extra_body:
        kwargs.setdefault("extra_body", settings.llm_extra_body)
    base_url = settings.llm_base_url or ""
    if "deepseek.com" in base_url or model.lower().startswith("deepseek"):
        kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
    if "siliconflow.cn" in base_url or "qwen3" in model.lower():
        kwargs.setdefault("extra_body", {"enable_thinking": False})
    token = _llm_role.set(f"default/{model}")
    _dbg(f"default/{model}", prompt, kwargs)
    try:
        result = await openai_complete(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            **kwargs,
        )
        _dbg(f"default/{model}", prompt, kwargs, result=result)
        return result
    except Exception as exc:
        _dbg(f"default/{model}", prompt, kwargs, exc=exc)
        raise
    finally:
        _llm_role.reset(token)


async def llm_chat_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs,
) -> str:
    """Chat / assignment LLM backed by the chat provider (e.g. DeepSeek deepseek-flash).

    Falls back to ``llm_model_func`` settings when LLM_CHAT_* env vars are not set.
    Used by assignment and question generation.
    """
    kwargs.setdefault("max_tokens", settings.llm_max_tokens)
    kwargs.setdefault("temperature", settings.llm_temperature)
    model = settings.effective_chat_model
    # Disable DeepSeek thinking mode to prevent 400 errors when tool calls are present
    # (thinking mode requires reasoning_content to be echoed back in multi-turn tool calls)
    base_url = settings.effective_chat_base_url or ""
    if "deepseek.com" in base_url or model.lower().startswith("deepseek"):
        kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
    token = _llm_role.set(f"chat/{model}")
    _dbg(f"chat/{model}", prompt, kwargs)
    try:
        result = await openai_complete(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=settings.effective_chat_api_key,
            base_url=settings.effective_chat_base_url,
            **kwargs,
        )
        _dbg(f"chat/{model}", prompt, kwargs, result=result)
        return result
    except Exception as exc:
        _dbg(f"chat/{model}", prompt, kwargs, exc=exc)
        raise
    finally:
        _llm_role.reset(token)


async def vision_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    image_data: str | None = None,
    messages: list | None = None,
    image_mime: str | None = None,
    **kwargs,
) -> str:
    """Vision LLM backed by qwen-vl-max (async).

    Three calling conventions from RAG-Anything:
    1. messages already constructed (aquery_vlm_enhanced multimodal path).
    2. image_data as base64 string (ImageModalProcessor path).
    3. Text-only fallback -> llm_model_func.
    """
    kwargs.setdefault("max_tokens", settings.llm_max_tokens)
    _safe = {k: v for k, v in kwargs.items() if k in ("max_tokens", "temperature")}

    if messages is not None:
        client = AsyncOpenAI(
            api_key=settings.effective_vision_api_key,
            base_url=settings.effective_vision_base_url,
        )
        _role = f"vision/{settings.vision_model}"
        token = _llm_role.set(_role)
        _dbg(_role, prompt, _safe)
        try:
            resp = await client.chat.completions.create(
                model=settings.vision_model,
                messages=messages,
                **_safe,
            )
            content = resp.choices[0].message.content or ""
            _dbg(_role, prompt, _safe, result=content)
            return content
        except Exception as exc:
            _dbg(_role, prompt, _safe, exc=exc)
            raise
        finally:
            _llm_role.reset(token)

    if image_data is not None:
        mime = image_mime or "image/jpeg"
        content_parts: list = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
        ]
        msg_list: list = []
        if system_prompt:
            msg_list.append({"role": "system", "content": system_prompt})
        msg_list.append({"role": "user", "content": content_parts})

        client = AsyncOpenAI(
            api_key=settings.effective_vision_api_key,
            base_url=settings.effective_vision_base_url,
        )
        _role = f"vision/{settings.vision_model}"
        token = _llm_role.set(_role)
        _dbg(_role, prompt, _safe)
        try:
            resp = await client.chat.completions.create(
                model=settings.vision_model,
                messages=msg_list,
                **_safe,
            )
            content = resp.choices[0].message.content or ""
            _dbg(_role, prompt, _safe, result=content)
            return content
        except Exception as exc:
            _dbg(_role, prompt, _safe, exc=exc)
            raise
        finally:
            _llm_role.reset(token)

    return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)


# ---------------------------------------------------------------------------
# Image filter statistics (accumulated in-process for the current session)
# ---------------------------------------------------------------------------


@dataclass
class _ImageFilterStats:
    total: int = 0
    filtered: int = 0
    errors: int = 0
    _counts: dict = field(default_factory=dict)  # filename -> "USEFUL"|"USELESS"


_filter_stats = _ImageFilterStats()


def get_image_filter_stats() -> dict:
    """Return a snapshot of image filtering statistics for the current session."""
    return {
        "total_images": _filter_stats.total,
        "useful_images": _filter_stats.total - _filter_stats.filtered - _filter_stats.errors,
        "filtered_images": _filter_stats.filtered,
        "filter_errors": _filter_stats.errors,
    }


async def _call_vision_raw(
    prompt: str,
    image_data: str,
    system_prompt: str | None = None,
    max_tokens: int = 20,
    image_mime: str | None = None,
) -> str:
    """Low-level vision call with explicit token limit (used for cheap filter checks)."""
    mime = image_mime or "image/jpeg"
    content_parts: list = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
    ]
    msg_list: list = []
    if system_prompt:
        msg_list.append({"role": "system", "content": system_prompt})
    msg_list.append({"role": "user", "content": content_parts})

    client = AsyncOpenAI(api_key=settings.effective_vision_api_key, base_url=settings.effective_vision_base_url)
    token = _llm_role.set(f"vision-filter/{settings.vision_model}")
    try:
        resp = await client.chat.completions.create(
            model=settings.vision_model,
            messages=msg_list,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    finally:
        _llm_role.reset(token)
    return (resp.choices[0].message.content or "").strip().upper()


async def _filtered_vision_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    image_data: str | None = None,
    messages: list | None = None,
    image_mime: str | None = None,
    **kwargs,
) -> str:
    """vision_model_func wrapper that skips useless images via a cheap binary pre-check.

    When ``settings.enable_image_filter`` is True and ``image_data`` is provided,
    a short binary prompt is sent first (max_tokens=20).  If the response contains
    "USELESS" the image is marked as filtered and a placeholder description is
    returned without running the expensive full analysis call.
    """
    if settings.enable_image_filter and image_data is not None and messages is None:
        _filter_stats.total += 1
        try:
            verdict = await _call_vision_raw(
                prompt=settings.image_filter_prompt,
                image_data=image_data,
                max_tokens=20,
                image_mime=image_mime,
            )
        except Exception as exc:  # noqa: BLE001
            _filter_stats.errors += 1
            logger.warning(f"[image-filter] 过滤检查失败，默认保留图片: {exc}")
            verdict = "USEFUL"

        if "USELESS" in verdict:
            _filter_stats.filtered += 1
            stats = get_image_filter_stats()
            logger.debug(
                f"[image-filter] 跳过装饰性图片 (已过滤 {stats['filtered']}/{stats['total']})"
            )
            # Return a placeholder so rag-anything still gets a valid string.
            # The chunk will have minimal content and low retrieval weight.
            return "该图片为装饰性图片或无实质内容，已跳过分析。"
        else:
            logger.debug("[image-filter] 图片通过筛选，进行完整分析")

    return await vision_model_func(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        image_data=image_data,
        messages=messages,
        image_mime=image_mime,
        **kwargs,
    )
