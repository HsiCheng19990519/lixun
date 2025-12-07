from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from devmate.agent.core import run_agent
from devmate.config import Settings
from devmate.logging_utils import setup_logging
from devmate.observability import setup_observability


def _str2bool(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    lowered = val.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {val}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DevMate Agent CLI (Stage 4)")
    parser.add_argument("-m", "--message", help="User request (if omitted, will prompt)")
    parser.add_argument("--transport", choices=["http", "stdio", "sse"], default=None, help="MCP transport")
    parser.add_argument("--k", type=int, default=4, help="Top-k for RAG search_knowledge_base")
    parser.add_argument("--max-iterations", type=int, default=6, help="Max tool/LLM iterations")
    parser.add_argument("--session-name", default=None, help="Tracing session name")

    # LLM
    parser.add_argument("--llm-mode", dest="llm_mode", default=None, help="LLM mode (open_source/closed_source)")
    parser.add_argument("--provider", dest="llm_provider", default=None, help="LLM provider (ollama/openai/deepseek/qwen_api/...)")
    parser.add_argument("--model", dest="model_name", default=None, help="LLM model name")
    parser.add_argument("--ai-base-url", dest="ai_base_url", default=None, help="AI_BASE_URL for OpenAI-compatible endpoints")
    parser.add_argument("--api-key", dest="api_key", default=None, help="API key for LLM provider")

    # Embeddings
    parser.add_argument("--embedding-mode", dest="embedding_mode", default=None, help="Embedding mode (open_source/closed_source)")
    parser.add_argument("--embedding-provider", dest="embedding_provider", default=None, help="Embedding provider")
    parser.add_argument("--embedding-model-name", dest="embedding_model_name", default=None, help="Embedding model name")
    parser.add_argument("--embedding-device", dest="embedding_device", default=None, help="Embedding device (cpu/cuda:0)")
    parser.add_argument("--embedding-base-url", dest="embedding_base_url", default=None, help="Embedding base URL")
    parser.add_argument("--embedding-api-key", dest="embedding_api_key", default=None, help="Embedding API key")

    # RAG / splitting
    parser.add_argument("--vector-store-dir", dest="vector_store_dir", default=None, help="Vector store directory")
    parser.add_argument("--chunk-strategy", dest="chunk_strategy", default=None, help="Chunk strategy")
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=None, help="Chunk size")
    parser.add_argument("--chunk-overlap", dest="chunk_overlap", type=int, default=None, help="Chunk overlap")

    # Tavily / search
    parser.add_argument("--tavily-api-key", dest="tavily_api_key", default=None, help="Tavily API key")

    # Observability
    parser.add_argument("--langchain-tracing-v2", dest="langchain_tracing_v2", type=_str2bool, default=None, help="Enable LangChain tracing v2 (true/false)")
    parser.add_argument("--langchain-api-key", dest="langchain_api_key", default=None, help="LangChain API key")
    parser.add_argument("--langsmith-api-key", dest="langsmith_api_key", default=None, help="LangSmith API key")
    parser.add_argument("--langsmith-project", dest="langsmith_project", default=None, help="LangSmith project name")
    parser.add_argument("--langsmith-endpoint", dest="langsmith_endpoint", default=None, help="LangSmith endpoint")

    # MCP
    parser.add_argument("--mcp-transport", dest="mcp_transport", choices=["http", "stdio", "sse"], default=None, help="MCP client transport")
    parser.add_argument("--mcp-http-url", dest="mcp_http_url", default=None, help="MCP streamable HTTP URL")

    # Logging
    parser.add_argument("--log-level", dest="log_level", default=None, help="Log level")
    parser.add_argument("--log-file", dest="log_file", default=None, help="Log file path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    msg = args.message or input("Enter your request: ").strip()

    overrides: Dict[str, Any] = {
        k: v
        for k, v in {
            # LLM
            "llm_mode": args.llm_mode,
            "llm_provider": args.llm_provider,
            "model_name": args.model_name,
            "ai_base_url": args.ai_base_url,
            "api_key": args.api_key,
            # Embeddings
            "embedding_mode": args.embedding_mode,
            "embedding_provider": args.embedding_provider,
            "embedding_model_name": args.embedding_model_name,
            "embedding_device": args.embedding_device,
            "embedding_base_url": args.embedding_base_url,
            "embedding_api_key": args.embedding_api_key,
            # RAG
            "vector_store_dir": args.vector_store_dir,
            "chunk_strategy": args.chunk_strategy,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            # Tavily
            "tavily_api_key": args.tavily_api_key,
            # Observability
            "langchain_tracing_v2": args.langchain_tracing_v2,
            "langchain_api_key": args.langchain_api_key,
            "langsmith_api_key": args.langsmith_api_key,
            "langsmith_project": args.langsmith_project,
            "langsmith_endpoint": args.langsmith_endpoint,
            # MCP
            "mcp_transport": args.mcp_transport,
            "mcp_http_url": args.mcp_http_url,
            # Logging
            "log_level": args.log_level,
            "log_file": args.log_file,
        }.items()
        if v is not None
    }

    settings = Settings(**overrides)
    setup_logging(settings)
    setup_observability(settings)
    output = run_agent(
        msg,
        settings=settings,
        transport=args.transport,
        rag_k=args.k,
        max_iterations=args.max_iterations,
        session_name=args.session_name,
    )
    print(output)


if __name__ == "__main__":
    main()
