"""Configuration loaded from environment / .env file."""

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_EMBEDDING_MODES = frozenset({"ollama", "openai_compatible"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM API (DashScope / Qwen) — default provider
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Model names
    llm_model: str = "qwen-plus-2025-04-28"
    refine_model: str = "qwen-long"   # long-context model for structure-refine phase
    vision_model: str = "qwen3.6-plus"  # vision model for image understanding

    # Chat / assignment model — separate provider (e.g. DeepSeek) with own API key.
    # Falls back to default LLM settings when unset.
    llm_chat_api_key: str = ""      # LLM_CHAT_API_KEY; empty → use llm_api_key
    llm_chat_base_url: str = ""     # LLM_CHAT_BASE_URL; empty → use llm_base_url
    llm_chat_model: str = ""        # LLM_CHAT_MODEL; empty → use llm_model

    # Vision model — separate provider/endpoint (e.g. SiliconFlow Qwen-VL).
    # Falls back to default LLM settings when unset.
    vision_api_key: str = ""        # VISION_API_KEY; empty → use llm_api_key
    vision_base_url: str = ""       # VISION_BASE_URL; empty → use llm_base_url
    # Embedding backend: ollama (local) | openai_compatible (OpenAI /v1/embeddings API, incl. DashScope compatible-mode)
    embedding_mode: str = "openai_compatible"
    # Optional overrides for openai_compatible; empty → use LLM_BASE_URL / LLM_API_KEY
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    # Model id depends on embedding_mode: Ollama tag (e.g. bge-m3) or API model (e.g. text-embedding-v1).
    embedding_model: str = "text-embedding-v1"
    # Must match embedding vectors and PostgreSQL ``vector(N)`` on rag_chunks.
    embedding_dim: int = 1024
    embedding_max_tokens: int = 8192
    # Vector indexing and chunking tuning.
    embedding_batch_num: int = Field(
        default=3,
        validation_alias="EMBEDDING_BATCH_NUM",
        description="Texts per embedding request; lower reduces peak memory and latency.",
    )
    chunk_token_size: int = Field(default=1000, validation_alias="CHUNK_SIZE")
    """Text chunk size in tokens (env CHUNK_SIZE)."""
    chunk_overlap_token_size: int = Field(default=100, validation_alias="CHUNK_OVERLAP_SIZE")
    """Overlap between consecutive chunks (env CHUNK_OVERLAP_SIZE)."""
    embedding_timeout_seconds: int = Field(default=120, validation_alias="EMBEDDING_TIMEOUT")
    """Embedding request timeout in seconds."""
    embedding_query_batch_enabled: bool = Field(
        default=True,
        validation_alias="EMBEDDING_QUERY_BATCH_ENABLED",
    )
    """Dynamically batch concurrent single-query embedding calls in one process."""
    embedding_query_batch_window_ms: int = Field(
        default=10,
        validation_alias="EMBEDDING_QUERY_BATCH_WINDOW_MS",
        ge=1,
        le=100,
    )
    """Maximum collection window for online-query embedding micro-batches."""
    embedding_query_batch_size: int = Field(
        default=16,
        validation_alias="EMBEDDING_QUERY_BATCH_SIZE",
        ge=2,
        le=256,
    )
    """Maximum number of online queries sent in one embedding request."""

    @field_validator("embedding_mode", mode="before")
    @classmethod
    def _normalize_embedding_mode(cls, v: object) -> str:
        if v is None:
            return "ollama"
        s = str(v).strip().lower()
        if not s:
            return "ollama"
        if s in ("dashscope", "openai", "compatible", "openai-compatible"):
            return "openai_compatible"
        return s

    @model_validator(mode="after")
    def _validate_embedding_mode_and_warn_ollama_cloud_model(self) -> "Settings":
        from loguru import logger

        em = self.embedding_mode.strip().lower()
        if em not in _EMBEDDING_MODES:
            logger.error(
                f"Invalid EMBEDDING_MODE={self.embedding_mode!r}; "
                f"allowed: {sorted(_EMBEDDING_MODES)}. Falling back to 'ollama'."
            )
            self.embedding_mode = "ollama"
            em = "ollama"
        if em == "ollama" and "text-embedding" in self.embedding_model.lower():
            logger.error(
                f"EMBEDDING_MODEL={self.embedding_model!r} looks like a cloud/OpenAI-style id, "
                "but EMBEDDING_MODE=ollama uses Ollama (local model tags). "
                "Set EMBEDDING_MODE=openai_compatible for DashScope text-embedding-v1, "
                "or use an Ollama tag such as bge-m3."
            )
        return self

    @property
    def effective_chat_api_key(self) -> str:
        """API key for chat/assignment LLM (falls back to default llm_api_key)."""
        return self.llm_chat_api_key.strip() or self.llm_api_key

    @property
    def effective_chat_base_url(self) -> str:
        """Base URL for chat/assignment LLM (falls back to default llm_base_url)."""
        return self.llm_chat_base_url.strip() or self.llm_base_url

    @property
    def effective_chat_model(self) -> str:
        """Model name for chat/assignment LLM (falls back to default llm_model)."""
        return self.llm_chat_model.strip() or self.llm_model

    @property
    def effective_vision_api_key(self) -> str:
        """API key for vision LLM (falls back to default llm_api_key)."""
        return self.vision_api_key.strip() or self.llm_api_key

    @property
    def effective_vision_base_url(self) -> str:
        """Base URL for vision LLM (falls back to default llm_base_url)."""
        return self.vision_base_url.strip() or self.llm_base_url

    # LLM generation
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1
    llm_debug: bool = False  # LLM_DEBUG=true → log each call's prompt / response / error
    llm_debug_max_chars: int = 400  # LLM_DEBUG_MAX_CHARS — truncate prompt/result in debug log (0 = no limit)
    llm_extra_body: dict = Field(
        default_factory=dict,
        validation_alias="LLM_EXTRA_BODY",
        description="Optional JSON dict forwarded to OpenAI-compatible chat extra_body (e.g. {\"enable_thinking\": false}).",
    )

    # Paths
    output_dir: Path = Path("output/parsed")
    transcript_output_dir: Path = Path("output/transcripts")

    # Video → Whisper (CLI `rag video-ingest`; requires `uv sync --extra video` + ffmpeg)
    ffmpeg_path: str = "ffmpeg"
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    # Empty string = auto language detection in faster-whisper
    whisper_language: str = ""
    # Absolute path to a pre-downloaded faster-whisper model directory.
    # When non-empty, passed directly to WhisperModel() instead of whisper_model_size,
    # bypassing all HuggingFace Hub network calls.
    # Download once with: HF_ENDPOINT=https://hf-mirror.com python -c
    #   "from huggingface_hub import snapshot_download; snapshot_download(
    #    'Systran/faster-whisper-base', local_dir='models/faster-whisper-base',
    #    local_dir_use_symlinks=False)"
    whisper_model_path: str = ""

    # Video structured summary (before RAG ingest; LLM segments → .summary.md)
    video_summary_target_segment_seconds: int = 300
    video_summary_max_segment_seconds: int = 600
    # Empty = use refine_model for per-chunk JSON summaries
    video_summary_llm_model: str = ""
    # Empty = use built-in prompt in video_transcript_summary.py
    video_summary_system_prompt: str = ""
    # Comma or newline-separated list of correct domain terms (e.g. "harness engineering,Kubernetes").
    # When non-empty, appended to the system prompt so the LLM can fix Whisper mis-recognitions.
    video_summary_domain_terms: str = ""

    # Document structured summary (non-video; generated during parse_and_index pipeline)
    document_summary_enabled: bool = Field(default=True, validation_alias="DOCUMENT_SUMMARY_ENABLED")
    """Generate a Map-Reduce summary for non-video materials during ingestion (env DOCUMENT_SUMMARY_ENABLED)."""
    document_summary_chunk_tokens: int = Field(default=4000, validation_alias="DOCUMENT_SUMMARY_CHUNK_TOKENS")
    """Tokens per summarisation chunk (independent of CHUNK_SIZE; env DOCUMENT_SUMMARY_CHUNK_TOKENS)."""
    document_summary_max_chunks: int = Field(default=100, validation_alias="DOCUMENT_SUMMARY_MAX_CHUNKS")
    """Cap on map-phase chunks (~600 pages at 4000 tok/chunk; env DOCUMENT_SUMMARY_MAX_CHUNKS)."""
    document_summary_two_level_threshold: int = Field(default=40, validation_alias="DOCUMENT_SUMMARY_TWO_LEVEL_THRESHOLD")
    """Use two-level Map-Reduce when chunk count exceeds this value (env DOCUMENT_SUMMARY_TWO_LEVEL_THRESHOLD)."""

    # MinerU
    mineru_backend: str = "pipeline"
    mineru_device: str = "cpu"
    mineru_source: str = "modelscope"
    mineru_lang: str = "ch"

    # Document parsing
    parser: str = "mineru"
    parse_method: str = "auto"

    # Proxy (used by wikipedia tool; leave empty to disable)
    http_proxy: str = ""

    # Concurrency limits (lower to reduce 429 rate-limit errors and asyncpg pool races)
    llm_max_async: int = 4          # parallel LLM (chat) requests
    embedding_max_async: int = 2    # parallel embedding calls (lower for Ollama CPU / Windows + PG pool)
    max_parallel_insert: int = 1    # parallel document inserts (was 2; safer with asyncio.run)

    # Multimodal ingest: optional VLM caption for images
    ingest_surrogate_image_vlm: bool = Field(
        default=False,
        validation_alias="INGEST_SURROGATE_IMAGE_VLM",
        description="If true, course surrogate-ingest calls vision on each image file to append a short summary before text embedding (extra latency/cost).",
    )
    ingest_surrogate_image_vlm_max_tokens: int = Field(
        default=400,
        validation_alias="INGEST_SURROGATE_IMAGE_VLM_MAX_TOKENS",
        ge=32,
        le=4096,
    )
    ingest_surrogate_image_vlm_max_concurrency: int = Field(
        default=2,
        validation_alias="INGEST_SURROGATE_IMAGE_VLM_MAX_CONCURRENCY",
        ge=1,
        le=16,
    )
    ingest_surrogate_image_vlm_max_bytes: int = Field(
        default=4_000_000,
        validation_alias="INGEST_SURROGATE_IMAGE_VLM_MAX_BYTES",
        ge=50_000,
        description="Skip VLM if image file is larger than this (bytes); avoids huge reads.",
    )
    ingest_surrogate_image_vlm_system_prompt: str = Field(
        default="你是文档检索助手。用户会上传一页中的插图。请用中文写2到4句客观描述，便于后续文本向量检索。不要标题、不要markdown、不要编造图中没有的内容。",
        validation_alias="INGEST_SURROGATE_IMAGE_VLM_SYSTEM_PROMPT",
    )
    ingest_surrogate_image_vlm_user_prompt: str = Field(
        default="请描述这张图片的关键信息（主体、图表类型、文字要点），用于知识库检索。",
        validation_alias="INGEST_SURROGATE_IMAGE_VLM_USER_PROMPT",
    )

    # Image filtering (skip decorative / useless images before full vision analysis)
    enable_image_filter: bool = False
    # Prompt sent to vision model for the quick filter check (binary USEFUL/USELESS)
    image_filter_prompt: str = (
        "判断这张图片是否包含有实质信息（如图表、流程图、表格、电路图、技术示意图等）。"
        "如果图片是装饰性图片、空白页面、纯色背景、水印、页眉页脚图标，或者截取部分无法传达任何实质信息的图，回答 USELESS。"
        "只有包含完整或可理解的图表、流程图、表格、电路图或技术示意图才回答 USEFUL。"
        "只回答 USEFUL 或 USELESS，不要有其他任何内容。"
    )

    ollama_api_key: str = ""
    tavily_api_key: str = ""

    # MinerU Cloud API (mineru.net/apiManage/docs — 精准解析 v4)
    # Set MINERU_CLOUD_API_KEY to enable; leave empty to use local MinerU only.
    mineru_cloud_enabled: bool = True
    mineru_cloud_api_key: str = ""
    # model_version: pipeline | vlm (recommended) | MinerU-HTML
    mineru_cloud_model_version: str = "vlm"
    # Polling timeout (seconds) waiting for cloud task to complete
    mineru_cloud_timeout: int = 600
    # Interval (seconds) between status-poll requests
    mineru_cloud_poll_interval: int = 5
    # Fall back to local MinerU when cloud fails; set False to hard-fail instead
    mineru_cloud_fallback_local: bool = True


settings = Settings()
