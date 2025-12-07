# DevMate (Stage 1 & 2 Snapshot)

This branch is focused on environment setup (Stage 1) and MCP search (Stage 2). The Agent/RAG/observability parts are not covered here.

## What’s implemented
- **Stage 1 (env & deps)**: Python 3.13, uv-managed project, LangChain 1.x and related deps defined in `pyproject.toml`. Configuration via `.env`/`config.toml` and `Settings` (no hardcoded keys/models). Secrets ignored by `.gitignore`.
- **Stage 2 (MCP web search)**:
  - MCP server (`mcp_server/main.py`) using FastMCP with `streamable-http` transport.
  - Tool `search_web` calls Tavily (via `TAVILY_API_KEY`); returns structured JSON plus text content.
  - Client test (`scripts/test_streamable_http_client.py`) initializes MCP session, lists tools, and calls `search_web`.
- **Direct HTTP fallback** (for debugging): `devmate/mcp_client/client.py` supports a `http-direct` path to `/tools/search_web` if MCP transport ever misbehaves.

## Run MCP server (Stage 2)
```powershell
# In repo root
set TAVILY_API_KEY=your_key_here
# optional: set MCP_PORT=8010
uv run python -m mcp_server.main
```
Logs should show the server listening (default `http://127.0.0.1:8000/mcp` or the port you set).

## Test MCP client (Stage 2)
```powershell
# Default URL http://127.0.0.1:8000/mcp
uv run python scripts/test_streamable_http_client.py --query "model context protocol"

# If you changed the port
uv run python scripts/test_streamable_http_client.py --query "model context protocol" --url http://127.0.0.1:8010/mcp
```
Expected: prints available tools (`search_web`) and Tavily search results (JSON).

## Env/config keys
- `TAVILY_API_KEY` (required for search_web)
- `MCP_PORT` / `MCP_HOST` (optional override for server)
- `MCP_HTTP_URL` (optional override for client)
- Other LLM/embedding/LangSmith keys are defined in `.env` / `config.toml` for later stages.

## Notes / limits
- Only Stage 1 and Stage 2 are addressed here. Agent loop, RAG, Docker, observability, etc. are out of scope in this snapshot.
- Use `streamable-http` transport (default). SSE/stdio remain unsupported on this branch due to Windows transport quirks.***
