from __future__ import annotations

import os
from typing import Any, Dict

import httpx
from mcp.server.fastmcp import FastMCP

from devmate.config import Settings
from devmate.logging_utils import setup_logging

settings = Settings()
mcp_transport = os.environ.get("MCP_TRANSPORT", "stdio")  # stdio | sse | streamable-http
host = os.environ.get("MCP_HOST", "127.0.0.1")
port = int(os.environ.get("MCP_PORT", "8010"))
mcp = FastMCP("devmate-mcp-search", json_response=True, host=host, port=port)
logger = setup_logging(settings)


async def call_tavily(query: str, max_results: int = 5, search_depth: str = "basic") -> Dict[str, Any]:
    """Call Tavily search API; raises if key missing."""
    api_key = settings.tavily_api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def search_web(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> Dict[str, Any]:
    """MCP tool: Tavily web search."""
    logger.info("search_web query=%s max_results=%s depth=%s", query, max_results, search_depth)
    return await call_tavily(query, max_results, search_depth)


def main() -> None:
    """
    Entry point to start the MCP server (stdio/SSE/streamable-http decided by env).
    """
    logger.info("Starting MCP server transport=%s host=%s port=%s ...", mcp_transport, host, port)
    mcp.run(transport=mcp_transport)


if __name__ == "__main__":
    main()
