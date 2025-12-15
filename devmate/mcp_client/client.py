from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client

from devmate.config import Settings
from devmate.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _default_server_params() -> StdioServerParameters:
    """
    Default to launching the MCP search server using the current Python executable.
    Pass through env/cwd to ensure .env and dependencies are visible.
    """
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["MCP_TRANSPORT"] = env.get("MCP_TRANSPORT", "stdio")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.main"],
        env=env,
        cwd=str(root),
    )


def _resolve_timeout(timeout_seconds: Optional[float]) -> float:
    """
    Choose timeout for MCP calls. Prefers explicit arg, then env override, then fallback.
    """
    if timeout_seconds is not None:
        return timeout_seconds
    env_val = os.environ.get("MCP_TIMEOUT_SECONDS")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            logger.warning("Invalid MCP_TIMEOUT_SECONDS=%s; falling back to default", env_val)
    return 240.0


async def call_search_web(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    server_params: Optional[StdioServerParameters] = None,
    timeout_seconds: Optional[float] = None,
    transport: str = "stdio",
    http_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Connect to the MCP server over stdio and call search_web tool.
    Returns the raw CallToolResult model_dump() for flexibility.
    """
    timeout = _resolve_timeout(timeout_seconds)

    if transport == "stdio":
        params = server_params or _default_server_params()
        log_path = Path("logs/mcp_server_stderr.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        err_file: TextIO | None = log_path.open("a", encoding="utf-8")

        try:
            async with stdio_client(params, errlog=err_file) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(session.initialize(), timeout=timeout)
                    result = await asyncio.wait_for(
                        session.call_tool(
                            "search_web",
                            arguments={
                                "query": query,
                                "max_results": max_results,
                                "search_depth": search_depth,
                            },
                        ),
                        timeout=timeout,
                    )
                    return result.model_dump()
        finally:
            if err_file:
                err_file.close()

    elif transport == "sse":
        # SSE transport: server must be running and listening (FastMCP.run_sse_async)
        sse_url = os.environ.get("MCP_SSE_URL", "http://127.0.0.1:8000/sse")
        url = sse_url
        async with sse_client(url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=timeout)
                result = await asyncio.wait_for(
                    session.call_tool(
                        "search_web",
                        arguments={
                            "query": query,
                            "max_results": max_results,
                            "search_depth": search_depth,
                        },
                    ),
                    timeout=timeout,
                )
                return result.model_dump()

    elif transport == "http":
        url = http_url or os.environ.get("MCP_HTTP_URL", "http://127.0.0.1:8010/mcp")
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(url) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=timeout)
                result = await asyncio.wait_for(
                    session.call_tool(
                        "search_web",
                        arguments={
                            "query": query,
                            "max_results": max_results,
                            "search_depth": search_depth,
                        },
                    ),
                    timeout=timeout,
                )
                return result.model_dump()

    else:
        raise ValueError(f"Unsupported transport: {transport}")


def run_cli(query: str, max_results: int = 5, search_depth: str = "basic") -> None:
    """
    Convenience runner for manual testing.
    """
    settings = Settings()
    setup_logging(settings)
    asyncio.run(call_search_web(query, max_results=max_results, search_depth=search_depth))


def call_search_web_sync(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    transport: str = "stdio",
    http_url: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper around call_search_web for agent/tool usage.
    """
    return asyncio.run(
        call_search_web(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            transport=transport,
            http_url=http_url,
            timeout_seconds=timeout_seconds,
        )
    )
