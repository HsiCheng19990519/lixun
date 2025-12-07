from __future__ import annotations

"""
LLM and embedding factory functions driven by Settings.

Requirement highlights:
- ChatOpenAI and ChatDeepSeek are supported and wired to config/env variables.
- All model names and endpoints are configurable; nothing is hard-coded.
"""

import logging
from typing import Any, Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings

logger = logging.getLogger(__name__)


def _build_common_params(settings: Settings) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": settings.model_name,
        "temperature": 0,
    }
    if settings.api_key:
        params["api_key"] = settings.api_key
    if settings.ai_base_url:
        params["base_url"] = settings.ai_base_url
    return params


def _import_chat_deepseek():
    """
    Import ChatDeepSeek from available providers.

    LangChain versions may expose it under langchain_community or langchain_deepseek.
    """
    try:
        from langchain_community.chat_models import ChatDeepSeek  # type: ignore
        return ChatDeepSeek
    except Exception:
        try:
            from langchain_deepseek import ChatDeepSeek  # type: ignore
            return ChatDeepSeek
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "ChatDeepSeek is not available. Install langchain-deepseek "
                "or upgrade langchain_community."
            ) from exc


def build_chat_model(settings: Settings):
    """
    Create a chat model according to the configured provider.

    - deepseek -> ChatDeepSeek
    - otherwise -> ChatOpenAI (works for OpenAI and compatible endpoints)
    """
    provider = settings.llm_provider.lower()
    params = _build_common_params(settings)

    if provider == "deepseek":
        logger.info("Initializing ChatDeepSeek with model=%s", settings.model_name)
        ChatDeepSeek = _import_chat_deepseek()
        return ChatDeepSeek(**params)

    logger.info(
        "Initializing ChatOpenAI with model=%s base_url=%s",
        settings.model_name,
        settings.ai_base_url,
    )
    return ChatOpenAI(**params)


def build_embedding_model(settings: Settings):
    """
    Create an embedding model from settings.

    - open_source -> HuggingFaceEmbeddings
    - closed_source -> OpenAIEmbeddings (uses embedding_* overrides when provided)
    """
    if settings.embedding_mode.lower() == "closed_source":
        api_key = settings.embedding_api_key or settings.api_key
        base_url = settings.embedding_base_url or settings.ai_base_url
        logger.info(
            "Initializing OpenAIEmbeddings model=%s base_url=%s",
            settings.embedding_model_name,
            base_url,
        )
        return OpenAIEmbeddings(
            model=settings.embedding_model_name,
            api_key=api_key,
            base_url=base_url,
        )

    logger.info(
        "Initializing HuggingFaceEmbeddings model=%s device=%s",
        settings.embedding_model_name,
        settings.embedding_device,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.embedding_device},
    )
