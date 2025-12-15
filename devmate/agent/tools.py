from __future__ import annotations

"""
Tool definitions for the DevMate agent (Stage 4).

Exposes:
- search_knowledge_base: local RAG over persisted Chroma store
- search_web: MCP-based Tavily search
- write_todos: manage the agent's todo list during a run
"""

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool

from devmate.config import Settings
from devmate.rag.retriever import search_knowledge_base
from devmate.mcp_client.client import call_search_web_sync
from devmate.agent.run_state import AgentRunFlags, TodoItem

logger = logging.getLogger(__name__)

ALLOWED_DEPTHS = {"basic", "advanced"}
ALLOWED_TODO_STATUSES = {"todo", "doing", "done"}


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
    llm: Any = None,
):
    """
    Return a list of LangChain tools for the agent.

    Adaptive RAG knobs (env):
    - DEV_RAG_DISTANCE_KEEP_THRESHOLD: drop results with distance higher than this (default 0.6).
    - DEV_RAG_DISTANCE_REQUERY_THRESHOLD: if best distance is higher, re-query with larger k (default 0.8).
    - DEV_RAG_MAX_K: upper bound for adaptive k (default 8).
    - DEV_RAG_MULTI_HOP: if true, generate subqueries and run multi-hop retrieval (default false).
    - DEV_RAG_MAX_SUBQUERIES: cap number of subqueries when multi-hop (default 3).
    """
    cfg = settings or Settings()
    resolved_transport = _resolve_transport(transport, cfg)
    resolved_http_url = _resolve_http_url(http_url, cfg)
    distance_keep_threshold = cfg.rag_distance_keep_threshold  # lower is better
    distance_requery_threshold = cfg.rag_distance_requery_threshold
    max_adaptive_k = cfg.rag_max_k
    multi_hop_enabled = cfg.rag_multi_hop
    max_subqueries = cfg.rag_max_subqueries

    def _propose_subqueries(query: str) -> List[str]:
        """
        Use the LLM to generate subqueries for multi-hop retrieval.
        Fallback to the original query on errors or empty outputs.
        """
        if not (multi_hop_enabled and llm):
            return [query]
        try:
            prompt = (
                "Given a user goal, propose up to "
                f"{max_subqueries} concise retrieval queries (one per line). "
                "Prefer multiple complementary subqueries when the task has several parts "
                "(e.g., APIs to use, frontend framework, deployment). "
                "Keep key entities/APIs/locations. If everything fits one query, return just one line."
            )
            resp = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": query}])
            content = getattr(resp, "content", resp)
            text = "\n".join(content) if isinstance(content, list) else str(content)
            candidates = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
            deduped: List[str] = []
            for cand in candidates:
                if cand and cand not in deduped:
                    deduped.append(cand)
                if len(deduped) >= max_subqueries:
                    break
            return deduped or [query]
        except Exception as exc:
            logger.warning("subquery proposal failed; using original query. err=%s", exc)
            return [query]

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
            subqueries = _propose_subqueries(query)
            logger.info(
                "Tool search_knowledge_base called query=%s k=%s subqueries=%s",
                query,
                k,
                subqueries,
            )
            if run_flags:
                run_flags.used_rag = True
            hop_serialized: List[str] = []
            all_docs: List[Any] = []

            for hop_idx, subquery in enumerate(subqueries, start=1):
                notes: List[str] = [f"[RAG] hop {hop_idx}/{len(subqueries)} | query={subquery} | initial k={k}"]
                result, documents = search_knowledge_base(
                    query=subquery,
                    settings=cfg,
                    persist_dir=None,
                    k=k,
                    return_documents=True,
                )
                best_score = result["results"][0]["score"] if result.get("results") else None
                needs_more = (best_score is None) or (best_score > distance_requery_threshold)
                effective_k = k
                if needs_more and k < max_adaptive_k:
                    boosted_k = min(max_adaptive_k, max(k * 2, k + 2))
                    logger.info(
                        "Hop %s: top score %s exceeds requery threshold %s; boosting k to %s",
                        hop_idx,
                        best_score,
                        distance_requery_threshold,
                        boosted_k,
                    )
                    notes.append(
                        f"[RAG] boosted k from {k} to {boosted_k} (best_score={best_score})"
                    )
                    boosted_result, boosted_docs = search_knowledge_base(
                        query=subquery,
                        settings=cfg,
                        persist_dir=None,
                        k=boosted_k,
                        return_documents=True,
                    )
                    result, documents = boosted_result, boosted_docs
                    effective_k = boosted_k

                if not result.get("results"):
                    notes.append("[RAG] no local hits")
                    hop_serialized.append("\n".join(notes))
                    continue
                filtered_results, filtered_docs = _filter_by_score(
                    result["results"],
                    documents,
                    distance_keep_threshold,
                )
                all_docs.extend(filtered_docs)
                notes.append(
                    f"[RAG] filtered by distance <= {distance_keep_threshold}; kept {len(filtered_results)} of {len(result['results'])} (effective k={effective_k})"
                )
                hop_serialized.append(
                    "\n".join(notes)
                    + "\n\n"
                    + "\n\n".join(
                        f"Source: {item.get('source')} | File: {item.get('filename')} | "
                        f"Score: {item.get('score')}\n{item.get('content')}"
                        for item in filtered_results
                    )
                )

            if not hop_serialized:
                return "No local knowledge base hits.", []

            return "\n\n---\n\n".join(hop_serialized), all_docs
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

    def _normalize_status(status: Optional[str]) -> str:
        if not status:
            return "todo"
        normalized = str(status).strip().lower()
        return normalized if normalized in ALLOWED_TODO_STATUSES else "todo"

    def _normalize_todos(items: Any) -> List[TodoItem]:
        """
        Convert LLM-provided items to a clean list of TodoItem.
        Accepts list/dict/str for resilience; ignores malformed entries.
        """
        if items is None:
            return []
        if isinstance(items, dict):
            items = [items]
        if isinstance(items, str):
            # If the model wrapped a JSON array/object in a string, try to parse it first.
            try:
                parsed = json.loads(items)
                if isinstance(parsed, (list, dict)):
                    items = parsed if isinstance(parsed, list) else [parsed]
                else:
                    items = [items]
            except Exception:
                items = [items]
        if not isinstance(items, list):
            return []

        todos: List[TodoItem] = []
        for raw in items:
            if isinstance(raw, str):
                title = raw.strip()
                if title:
                    todos.append(TodoItem(title=title))
                continue
            if not isinstance(raw, dict):
                continue
            title = str(
                raw.get("title")
                or raw.get("task")
                or raw.get("name")
                or ""
            ).strip()
            if not title:
                continue
            status = _normalize_status(raw.get("status"))
            note = str(raw.get("note") or raw.get("details") or "").strip()
            todos.append(TodoItem(title=title, status=status, note=note))
        return todos

    @tool("write_todos", return_direct=False)
    def write_todos(items: Any) -> Dict[str, Any]:
        """
        Replace the current todo list with the provided items.
        Each item can include: title (required), status (todo/doing/done), note (optional).
        """
        todos = _normalize_todos(items)
        if not todos and run_flags and run_flags.todos:
            return {
                "error": "invalid_todos",
                "message": "No valid todos provided.",
                "todos": [asdict(t) for t in run_flags.todos],
            }

        if run_flags is not None:
            run_flags.todos = todos
        result_todos = run_flags.todos if run_flags is not None else todos
        try:
            logger.info(
                "write_todos updated plan: %s",
                "; ".join(f"{t.status.upper()} - {t.title}" for t in result_todos) or "empty",
            )
        except Exception:
            logger.debug("write_todos logging skipped due to serialization error")
        return {"todos": [asdict(t) for t in result_todos]}

    return [search_knowledge_base_tool, search_web_tool, write_todos]
