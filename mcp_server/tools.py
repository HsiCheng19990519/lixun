from __future__ import annotations

import logging
from typing import Any, Dict, List

from tavily import TavilyClient

from devmate.config import Settings

logger = logging.getLogger(__name__)


class SearchService:
    """Wrapper around Tavily search to expose a simple MCP-friendly interface."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not self.settings.tavily_api_key:
            logger.warning("TAVILY_API_KEY is not set; search_web will fail until provided.")
        self.client = TavilyClient(api_key=self.settings.tavily_api_key or "")

    def search_web(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> Dict[str, Any]:
        """Run a Tavily search and normalize results for the MCP response."""
        logger.info("Running Tavily search query=%s max_results=%s depth=%s", query, max_results, search_depth)
        resp = self.client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )
        results: List[Dict[str, Any]] = []
        for item in resp.get("results", []):
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content"),
                }
            )
        return {
            "query": query,
            "results": results,
            "raw": resp,
        }
