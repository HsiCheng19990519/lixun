from __future__ import annotations

import os
from typing import Any


def configure_langsmith(settings: Any) -> None:
    """Configure LangChain tracing / LangSmith based on settings."""

    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    # Allow the caller to set additional endpoints through env variables.
    if getattr(settings, "langchain_endpoint", None):
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
