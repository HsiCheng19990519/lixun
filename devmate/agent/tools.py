from __future__ import annotations

"""
Tool definitions for the DevMate agent (Stage 4).

Exposes:
- search_knowledge_base: local RAG over persisted Chroma store
- search_web: MCP-based Tavily search
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

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

    Adaptive RAG knobs (env):
    - DEV_RAG_DISTANCE_KEEP_THRESHOLD: drop results with distance higher than this (default 0.6).
    - DEV_RAG_DISTANCE_REQUERY_THRESHOLD: if best distance is higher, re-query with larger k (default 0.8).
    - DEV_RAG_MAX_K: upper bound for adaptive k (default 8).
    """
    cfg = settings or Settings()
    resolved_transport = _resolve_transport(transport, cfg)
    resolved_http_url = _resolve_http_url(http_url, cfg)
    # Adaptive knobs (env-tunable to avoid config churn).
    distance_keep_threshold = float(os.getenv("DEV_RAG_DISTANCE_KEEP_THRESHOLD", "0.6"))  # lower is better
    distance_requery_threshold = float(os.getenv("DEV_RAG_DISTANCE_REQUERY_THRESHOLD", "0.8"))
    max_adaptive_k = int(os.getenv("DEV_RAG_MAX_K", "8"))

    def _filter_by_score(
        results: List[Dict[str, Any]],
        documents: List[Any],
        keep_threshold: float,
    ) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Keep results with distance <= keep_threshold (Chroma returns distance: lower is better).
        If everything is filtered out, fall back to the best 1-2 hits to avoid empty context.
        """
        filtered_pairs: List[Tuple[Dict[str, Any], Any]] = []
        for item, doc in zip(results, documents):
            score = item.get("score")
            if score is None or score <= keep_threshold:
                filtered_pairs.append((item, doc))
        if not filtered_pairs and results:
            # Fallback: keep the top 2 to avoid losing all context.
            filtered_pairs = list(zip(results[:2], documents[:2]))
        filtered_results = [r for r, _ in filtered_pairs]
        filtered_docs = [d for _, d in filtered_pairs]
        return filtered_results, filtered_docs

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
            notes: List[str] = [f"[RAG] initial k={k}"]
            # First pass
            result, documents = search_knowledge_base(
                query=query,
                settings=cfg,
                persist_dir=None,
                k=k,
                return_documents=True,
            )
            best_score = result["results"][0]["score"] if result.get("results") else None
            # If top score looks weak (distance too high) and we can fetch more, retry with larger k.
            needs_more = (best_score is None) or (best_score > distance_requery_threshold)
            if needs_more and k < max_adaptive_k:
                boosted_k = min(max_adaptive_k, max(k * 2, k + 2))
                logger.info(
                    "Top score %s exceeds requery threshold %s; boosting k to %s",
                    best_score,
                    distance_requery_threshold,
                    boosted_k,
                )
                notes.append(
                    f"[RAG] boosted k from {k} to {boosted_k} (best_score={best_score})"
                )
                boosted_result, boosted_docs = search_knowledge_base(
                    query=query,
                    settings=cfg,
                    persist_dir=None,
                    k=boosted_k,
                    return_documents=True,
                )
                # Merge, keeping order from boosted_result (already top-sorted).
                result, documents = boosted_result, boosted_docs
                k = boosted_k  # reflect the effective k used

            if not result.get("results"):
                return "No local knowledge base hits.", []
            filtered_results, filtered_docs = _filter_by_score(
                result["results"],
                documents,
                distance_keep_threshold,
            )
            notes.append(
                f"[RAG] filtered by distance <= {distance_keep_threshold}; kept {len(filtered_results)} of {len(result['results'])} (effective k={k})"
            )
            serialized = "\n".join(notes) + "\n\n" + "\n\n".join(
                f"Source: {item.get('source')} | File: {item.get('filename')} | "
                f"Score: {item.get('score')}\n{item.get('content')}"
                for item in filtered_results
            )
            return serialized, filtered_docs
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
