from __future__ import annotations

from tavily import TavilyClient
from mcp.server.fastmcp import FastMCP

from .config import load_settings


mcp = FastMCP("devmate-mcp")


@mcp.tool()
async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using Tavily and return structured results."""

    settings = load_settings()
    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query, max_results=max_results)
    return response.get("results", response)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
