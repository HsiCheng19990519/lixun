from __future__ import annotations

import argparse
import asyncio
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test MCP search_web tool over Streamable HTTP")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument(
        "--url",
        default=os.environ.get("MCP_HTTP_URL", "http://127.0.0.1:8000/mcp"),
        help="MCP streamable HTTP endpoint (default from MCP_HTTP_URL or http://127.0.0.1:8000/mcp)",
    )
    parser.add_argument("--max-results", type=int, default=5, help="Max results for search_web")
    parser.add_argument(
        "--search-depth",
        default="basic",
        choices=["basic", "advanced"],
        help="Tavily search depth",
    )
    args = parser.parse_args()

    async with streamablehttp_client(args.url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])
            if "search_web" not in [t.name for t in tools.tools]:
                raise RuntimeError("search_web tool not found on server")

            result = await session.call_tool(
                "search_web",
                arguments={
                    "query": args.query,
                    "max_results": args.max_results,
                    "search_depth": args.search_depth,
                },
            )
            # 控制台编码在 Windows 可能不是 UTF-8，这里使用 ensure_ascii=True 避免编码异常
            print(result.model_dump_json(ensure_ascii=True, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
