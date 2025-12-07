# DevMate（Stage 1-3）

本分支覆盖 Stage 1（环境/依赖/配置基线）、Stage 2（MCP 搜索工具）与 Stage 3（本地 RAG 检索）。Stage 4 及之后的 Agent/容器/观测性未实现。

## 已完成
- Stage 1：使用 `uv` 管理 Python 3.13；`pyproject.toml` 声明 LangChain 1.x、langchain-chroma、ChatDeepSeek/HF 等依赖；`devmate/config.py::Settings` 统一读取配置（env > .env > config.toml），日志和敏感文件忽略已处理。
- Stage 2：MCP 搜索（FastMCP `streamable-http`）  
  - 服务器：`mcp_server/main.py`，工具 `search_web` 调 Tavily（需 `TAVILY_API_KEY`），返回 JSON 与文本。  
  - 客户端测试：`scripts/test_streamable_http_client.py` 初始化会话、列工具并调用 `search_web`。  
  - HTTP 直连兜底：`devmate/mcp_client/client.py` 支持 `http-direct` 调 `/tools/search_web`。
- Stage 3：RAG（本地知识库）  
  - 文档：`docs/internal_guidelines.md`、`docs/templates.md`、`docs/internal_fastapi_guidelines.md`（可继续扩充）。  
  - 向量库：`devmate/rag/ingest.py` + `langchain-chroma` + `BAAI/bge-m3`，持久化在 `data/vector_store`（遥测默认关闭）。  
  - 检索：`devmate/rag/retriever.py::search_knowledge_base`，脚本 `scripts/test_rag.py` 可手测。

## 快速开始
1) 安装依赖  
```
uv sync
```

2) 配置变量  
- Tavily：`TAVILY_API_KEY`  
- LLM/Embedding：`MODEL_NAME`、`EMBEDDING_MODEL_NAME`，闭源时 `AI_BASE_URL`、`API_KEY`  
- RAG：`CHUNK_SIZE`、`CHUNK_OVERLAP`、`VECTOR_STORE_DIR`（默认 `data/vector_store`）  
- 观测性：`LANGCHAIN_TRACING_V2`、`LANGCHAIN_API_KEY`、`LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`

3) 环境快速验证（Stage 1）  
```
uv run python scripts/check_env.py
```
预期输出示例（类名可能随配置变化）：  
```
LLM: <class 'langchain_openai.chat_models.base.ChatOpenAI'>
Embedding: <class 'langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings'>
```

4) 启动 MCP 服务器（Stage 2，默认端口 8010）  
```
set TAVILY_API_KEY=your_key
uv run python -m mcp_server.main
```
预期日志：  
```
Starting MCP server (streamable-http) on 127.0.0.1:8010 ...
```
默认监听 `http://127.0.0.1:8010/mcp`（可用 `MCP_PORT` 覆盖）。

5) 测试 MCP 客户端（Stage 2 验证）  
```
uv run python scripts/test_streamable_http_client.py --query "model context protocol" --url http://127.0.0.1:8010/mcp
```
预期输出（精简）：  
```
Available tools: ['search_web']
{
  "meta": null,
  "content": [{"type": "text", "text": "{...Tavily结果JSON...}"}],
  "structuredContent": {"result": {"query": "...", "results": [...], "raw": {...}}},
  "isError": false
}
```

6) 文档摄入（Stage 3）  
```
uv run python scripts/ingest_docs.py --rebuild
```
预期日志结尾类似：  
```
Ingestion completed: <N> documents -> <M> chunks
```
说明：读取 `docs/`，切分并写入 `data/vector_store`。

7) 知识库查询（Stage 3 验证）  
```
uv run python scripts/test_rag.py --query "project guidelines"
```
预期输出包含本地文档片段（文件名视 docs/ 内容而定，例如）：  
```
"results": [
  {"filename": "internal_guidelines.md", ...},
  {"filename": "templates.md", ...},
  {"filename": "internal_fastapi_guidelines.md", ...}
]
```

## 文件速览
- `pyproject.toml`：LangChain 1.x、langchain-chroma、langchain-deepseek、langchain-huggingface、sentence-transformers 等依赖；脚本入口。
- `.env` / `config.toml`：LLM/Embedding/Tavily/LangSmith 等配置示例。
- `devmate/config.py`：配置加载逻辑（env 优先，忽略未知字段，支持大写别名）。
- `devmate/llm.py`：ChatOpenAI/ChatDeepSeek、OpenAI/HuggingFace Embeddings 工厂。
- `devmate/logging_utils.py`：stderr + 滚动日志。
- `mcp_server/main.py`、`scripts/test_streamable_http_client.py`：MCP 搜索 server/client。
- `devmate/rag/ingest.py`、`devmate/rag/retriever.py`、`scripts/ingest_docs.py`、`scripts/test_rag.py`：RAG 摄入与检索。
- `docs/internal_guidelines.md`、`docs/templates.md`、`docs/internal_fastapi_guidelines.md`：本地知识库示例文档。

## 限制
- 示例文档为简单示例，需按需扩充真实内容。
