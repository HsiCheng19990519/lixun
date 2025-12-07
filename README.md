# DevMate（Stage 1 & 2）

本分支仅覆盖 Stage 1（环境/依赖/配置基线）与 Stage 2（MCP 搜索工具）。后续 Agent、RAG、容器、观测性等未在此分支实现。

## 已实现
- Stage 1：使用 uv 管理 Python 3.13 项目；`pyproject.toml` 声明 LangChain 1.x 及 ChatDeepSeek、langchain-huggingface、sentence-transformers 等依赖；配置由 `devmate/config.py::Settings` 统一读取（env > .env > config.toml），日志与敏感文件忽略已处理。
- Stage 2：MCP 搜索（FastMCP `streamable-http`）
  - 服务器：`mcp_server/main.py`，工具 `search_web` 调 Tavily（需 `TAVILY_API_KEY`），返回结构化 JSON 和文本。
  - 客户端测试：`scripts/test_streamable_http_client.py` 初始化会话、列工具并调用 `search_web`。
  - HTTP 直连兜底：`devmate/mcp_client/client.py` 支持 `http-direct` 调用 `/tools/search_web`（备用）。

## 快速开始
1) 安装依赖  
```
uv sync
```

2) 配置变量  
- Tavily：`TAVILY_API_KEY`  
- LLM/Embedding：`MODEL_NAME`、`EMBEDDING_MODEL_NAME`，闭源时 `AI_BASE_URL`、`API_KEY`  
- 观测性：`LANGCHAIN_TRACING_V2`、`LANGCHAIN_API_KEY`、`LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`

3) 最小环境验证（仅实例化，不会发真实请求）  
```
uv run python scripts/check_env.py
```
预期输出示例（类名视配置可能不同）：  
```
LLM: <class 'langchain_openai.chat_models.base.ChatOpenAI'>
Embedding: <class 'langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings'>
```

4) 启动 MCP 服务器（Stage 2，默认端口 8010）  
```
set TAVILY_API_KEY=your_key
uv run python -m mcp_server.main
```
默认监听 `http://127.0.0.1:8010/mcp`（可用 `MCP_PORT` 覆盖）。

5) 测试 MCP 客户端  
```
uv run python scripts/test_streamable_http_client.py --query "model context protocol" --url http://127.0.0.1:8010/mcp
```
预期输出示例（精简）：
```
Available tools: ['search_web']
{
  "meta": null,
  "content": [
    {"type": "text", "text": "{...Tavily结果JSON...}", ...}
  ],
  "structuredContent": {
    "result": {
      "query": "model context protocol",
      "results": [{"title": "...", "url": "...", "snippet": "..."}],
      "raw": {...}
    }
  },
  "isError": false
}
```
其中 `structuredContent.result` 可直接用于后续逻辑。

## 文件速览
- `pyproject.toml`：LangChain 1.x、langchain-deepseek、langchain-huggingface、sentence-transformers 等依赖；脚本入口。
- `.env` / `config.toml`：LLM/Embedding/Tavily/LangSmith 等配置示例。
- `devmate/config.py`：配置加载逻辑（env 优先，忽略未知字段，支持大写别名）。
- `devmate/llm.py`：ChatOpenAI/ChatDeepSeek、OpenAI/HuggingFace Embeddings 工厂。
- `devmate/logging_utils.py`：stderr + 滚动日志。
- `scripts/check_env.py`：环境快速验证。
- `scripts/test_streamable_http_client.py`：MCP 搜索手工测试。

## 限制
- 仅涵盖 Stage 1/2；后续 Agent、RAG、Docker/compose、观测链路等未提供。
- 默认使用 `streamable-http`；SSE/stdio 在 Windows 未适配，建议沿用当前传输。
