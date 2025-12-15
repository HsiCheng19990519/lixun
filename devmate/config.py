from __future__ import annotations

"""
Configuration loader for DevMate.

Priority:
1) explicit kwargs when instantiating Settings
2) environment variables
3) .env file
4) config.toml (local defaults)
5) class defaults
"""

from pathlib import Path
from typing import Any, Dict, Optional
import tomllib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_toml_config(_: BaseSettings | None = None) -> Dict[str, Any]:
    """Load values from config.toml if present; ignore errors silently."""
    path = Path("config.toml")
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except Exception:
        return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unexpected keys from env/config
        alias_generator=lambda s: s.upper(),  # Accept upper-case env/config keys
        populate_by_name=True,
    )

    # ---------- LLM ----------
    llm_mode: str = "closed_source"  # open_source | closed_source
    llm_provider: str = "zhipu"   # ollama | openai | deepseek | qwen_api | zhipu
    model_name: str = "glm-4.6"
    ai_base_url: Optional[str] = None
    api_key: Optional[str] = None

    # ---------- Embeddings ----------
    embedding_mode: str = "open_source"  # open_source | closed_source
    embedding_provider: str = "huggingface"
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_cache_dir: str = "data/hf_cache"

    # ---------- RAG / splitting ----------
    vector_store_dir: str = "data/vector_store"
    chunk_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chroma_host: str = "127.0.0.1"
    chroma_http_port: int = 8000
    chroma_ssl: bool = False
    rag_distance_keep_threshold: float = Field(default=0.6, alias="DEV_RAG_DISTANCE_KEEP_THRESHOLD")
    rag_distance_requery_threshold: float = Field(default=0.8, alias="DEV_RAG_DISTANCE_REQUERY_THRESHOLD")
    rag_max_k: int = Field(default=8, alias="DEV_RAG_MAX_K")
    rag_multi_hop: bool = Field(default=False, alias="DEV_RAG_MULTI_HOP")
    rag_max_subqueries: int = Field(default=3, alias="DEV_RAG_MAX_SUBQUERIES")
    rag_rewrite: bool = Field(default=True, alias="DEV_RAG_REWRITE")
    rag_rewrite_max_chars: int = Field(default=200, alias="DEV_RAG_REWRITE_MAX_CHARS")

    # ---------- Web search ----------
    tavily_api_key: Optional[str] = None

    # ---------- Observability ----------
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = "devmate"
    langsmith_endpoint: Optional[str] = None

    # ---------- MCP client ----------
    mcp_transport: str = "http"  # http | stdio | sse
    mcp_http_url: Optional[str] = None

    # ---------- Logging ----------
    log_level: str = "INFO"
    log_file: str = "logs/devmate.log"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Order keeps env vars higher priority than config.toml defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _load_toml_config,
            file_secret_settings,
        )

    @property
    def base_url(self) -> Optional[str]:
        """Alias kept for backward compatibility."""
        return self.ai_base_url
