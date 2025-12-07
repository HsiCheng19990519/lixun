from __future__ import annotations

import argparse
import asyncio
import json

from devmate.mcp_client.client import call_search_web
from devmate.logging_utils import setup_logging
from devmate.config import Settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test MCP search_web tool via Tavily.")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--search-depth", default="basic", choices=["basic", "advanced"])
    parser.add_argument("--transport", default="stdio", choices=["http", "sse", "stdio"])
    parser.add_argument("--http-url", default=None, help="Streamable HTTP endpoint (default http://127.0.0.1:8010/mcp)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout seconds for MCP calls")
    args = parser.parse_args()

    settings = Settings()
    setup_logging(settings)

    result = await call_search_web(
        query=args.query,
        max_results=args.max_results,
        search_depth=args.search_depth,
        transport=args.transport,
        http_url=args.http_url,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
