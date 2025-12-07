from __future__ import annotations

"""
Observability helpers for LangSmith / LangChain tracing.

Stage 4 requirement: tracing must be enabled so tool calls and LLM runs show up in a single trace.
"""

import logging
import os
from typing import Optional

from langchain_core.runnables import RunnableConfig

from devmate.config import Settings

logger = logging.getLogger(__name__)


def setup_observability(settings: Settings) -> None:
    """
    Configure LangChain/LangSmith tracing based on settings/env.

    - LANGCHAIN_TRACING_V2 toggles tracing.
    - LANGCHAIN_API_KEY or LANGSMITH_API_KEY/PROJECT/ENDPOINT are propagated.
    """
    if settings.langchain_tracing_v2 or settings.langchain_api_key or settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        logger.info(
            "LangSmith tracing enabled project=%s endpoint=%s",
            os.environ.get("LANGSMITH_PROJECT", ""),
            os.environ.get("LANGSMITH_ENDPOINT", ""),
        )
    else:
        logger.info("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable).")


def build_tracing_config(run_name: Optional[str] = None, session_name: Optional[str] = None) -> RunnableConfig:
    """
    Helper to create a RunnableConfig with run/session metadata for tracing.
    """
    cfg: RunnableConfig = {}
    if run_name:
        cfg["run_name"] = run_name
    if session_name:
        cfg["configurable"] = {"session_id": session_name}
    return cfg
