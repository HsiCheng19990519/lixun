from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    ai_base_url: str = "https://api.openai.com/v1"
    api_key: str
    model_name: str = "gpt-4o-mini"
    embedding_model_name: str = "text-embedding-3-large"

    tavily_api_key: str
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None

    chroma_persist_path: str = ".chroma"
    docs_path: str = "docs"

    class Config:
        env_prefix = ""

    model_config = SettingsConfigDict(env_file=('.env',), extra='ignore')


def load_settings() -> Settings:
    """Load settings from environment with `.env` support."""

    return Settings()
