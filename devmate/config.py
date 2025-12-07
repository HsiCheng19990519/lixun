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
        extra="ignore",
        alias_generator=lambda s: s.upper(),
        populate_by_name=True,
    )

    # ---------- LLM ----------
    llm_mode: str = "open_source"  # open_source | closed_source
    llm_provider: str = "ollama"   # ollama | openai | deepseek | qwen_api
    model_name: str = "qwen2.5-coder:7b"
    ai_base_url: Optional[str] = None
    api_key: Optional[str] = None

    # ---------- Embeddings ----------
    embedding_mode: str = "open_source"  # open_source | closed_source
    embedding_provider: str = "huggingface"
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None

    # ---------- RAG / splitting ----------
    vector_store_dir: str = "data/vector_store"
    chunk_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---------- Web search ----------
    tavily_api_key: Optional[str] = None

    # ---------- Observability ----------
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = "devmate"
    langsmith_endpoint: Optional[str] = None

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
