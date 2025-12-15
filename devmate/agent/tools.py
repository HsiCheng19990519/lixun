from __future__ import annotations

"""
Tool definitions for the DevMate agent (Stage 4).

Exposes:
- search_knowledge_base: local RAG over persisted Chroma store
- search_web: MCP-based Tavily search
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.tools import tool

from devmate.config import Settings
from devmate.rag.retriever import search_knowledge_base
from devmate.mcp_client.client import call_search_web_sync
from devmate.agent.run_state import AgentRunFlags

logger = logging.getLogger(__name__)

ALLOWED_DEPTHS = {"basic", "advanced"}


def _resolve_transport(explicit: Optional[str], settings: Settings) -> str:
    """
    Decide MCP transport based on explicit arg > env > settings default.
    """
    if explicit:
        return explicit
    env_transport = os.environ.get("MCP_TRANSPORT")
    if env_transport:
        return env_transport
    return settings.mcp_transport


def _resolve_http_url(explicit: Optional[str], settings: Settings) -> Optional[str]:
    if explicit:
        return explicit
    return settings.mcp_http_url


def build_tools(
    settings: Optional[Settings] = None,
    *,
    transport: Optional[str] = None,
    http_url: Optional[str] = None,
    default_k: int = 4,
    run_flags: Optional[AgentRunFlags] = None,
):
    """
    Return a list of LangChain tools for the agent.
    """
    cfg = settings or Settings()
    resolved_transport = _resolve_transport(transport, cfg)
    resolved_http_url = _resolve_http_url(http_url, cfg)

    @tool("search_knowledge_base", response_format="content_and_artifact", return_direct=False)
    def search_knowledge_base_tool(query: str, k: int = default_k) -> tuple[str, List[Any]]:
        """
        Query the local knowledge base (Chroma). Returns matched chunks with metadata.
        Content is sent to the model; artifacts carry raw Documents with metadata+scores.
        """
        try:
            logger.info("Tool search_knowledge_base called query=%s k=%s", query, k)
            if run_flags:
                run_flags.used_rag = True
            result, documents = search_knowledge_base(
                query=query,
                settings=cfg,
                persist_dir=None,
                k=k,
                return_documents=True,
            )
            if not result.get("results"):
                return "No local knowledge base hits.", []
            serialized = "\n\n".join(
                f"Source: {item.get('source')} | File: {item.get('filename')} | Score: {item.get('score')}\n"
                f"{item.get('content')}"
                for item in result["results"]
            )
            return serialized, documents
        except Exception as exc:
            logger.exception("search_knowledge_base failed: %s", exc)
            return f"search_knowledge_base_failed: {exc}", []

    @tool("search_web", return_direct=False)
    def search_web_tool(query: str, max_results: int = 5, search_depth: str = "basic") -> Dict[str, Any]:
        """
        MCP Tavily web search. Uses the configured transport (http/stdio/sse).
        """
        try:
            depth = search_depth.lower() if isinstance(search_depth, str) else "basic"
            if depth not in ALLOWED_DEPTHS:
                logger.warning("search_web depth=%s is invalid; falling back to 'basic'", search_depth)
                depth = "basic"
            logger.info(
                "Tool search_web called query=%s max_results=%s depth=%s transport=%s http_url=%s",
                query,
                max_results,
                depth,
                resolved_transport,
                resolved_http_url,
            )
            if run_flags:
                run_flags.used_web = True
            result = call_search_web_sync(
                query=query,
                max_results=max_results,
                search_depth=depth,
                transport=resolved_transport,
                http_url=resolved_http_url,
            )
            # normalize to dict (call_search_web may return dict already)
            return result if isinstance(result, dict) else dict(result)
        except Exception as exc:
            logger.exception("search_web failed: %s", exc)
            return {"error": "search_web_failed", "message": str(exc), "transport": resolved_transport}

    return [search_knowledge_base_tool, search_web_tool]
