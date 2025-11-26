# DevMate - AI Coding Assistant

DevMate is an AI-driven coding assistant that combines LangChain, MCP-based web search, and RAG over local docs. It is configured for Python 3.13 and managed with [`uv`](https://github.com/astral-sh/uv).

## Quickstart

1. Install dependencies with `uv`:
   ```bash
   uv venv
   uv pip install -e .
   ```
2. Copy `.env.example` to `.env` and fill required keys.
3. Ingest sample docs and run the agent:
   ```bash
   ingest-docs
   devmate "I want to build a site showing nearby hiking trails"
   ```

## Configuration

Env vars (see `.env.example`):
- `AI_BASE_URL`, `API_KEY` for LLM endpoint.
- `MODEL_NAME`, `EMBEDDING_MODEL_NAME` to configure chat and embedding models.
- `TAVILY_API_KEY` for MCP search server.
- `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` for LangSmith.

## MCP and RAG

- `src/devmate/mcp_server.py` exposes Tavily search via MCP.
- `src/devmate/mcp_client.py` connects the agent to the MCP server over stdio.
- `src/devmate/rag.py` ingests markdown docs into Chroma for retrieval.

## Docker

Use Docker to run DevMate and dependencies:
```bash
docker compose up --build
```
The compose file starts the app and ensures volumes for Chroma persistence.
