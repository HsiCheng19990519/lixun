from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Iterable

from mcp.client.stdio import StdioTransport
from mcp.client.session import ClientSession


@asynccontextmanager
async def mcp_session(server_command: list[str]):
    """Create an MCP client session against the provided server command."""

    transport = StdioTransport(server_command[0], server_command[1:])
    async with ClientSession(transport=transport) as session:
        await session.initialize()
        await session.enumerate_tools()
        yield session


async def invoke_search(query: str, server_command: list[str]) -> Any:
    """Invoke the MCP search tool and return results."""

    async with mcp_session(server_command) as session:
        result = await session.call_tool("search_web", {"query": query})
        if hasattr(result, "content"):
            return result.content
        return result
