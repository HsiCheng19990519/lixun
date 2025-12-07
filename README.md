# DevMate（Stage 1 & 2）

本分支覆盖 Stage 1（环境/依赖/配置基线）与 Stage 2（MCP 搜索工具）。后续 Agent、RAG、容器、观测性等未实现。

## 已完成
- Stage 1：使用 `uv` 管理 Python 3.13；`pyproject.toml` 声明 LangChain 1.x、ChatDeepSeek/HF 等依赖；`devmate/config.py::Settings` 统一读取（env > .env > config.toml），日志与敏感文件忽略已处理。
- Stage 2：MCP 搜索（FastMCP，集成 Tavily）  
  - 服务器：`mcp_server/main.py` 暴露 `search_web`（需 `TAVILY_API_KEY`），支持 `stdio` / `sse` / `streamable-http`（默认 stdio，可通过 `MCP_TRANSPORT` 切换）。  
  - 客户端：`devmate/mcp_client/client.py`，`scripts/test_mcp_client.py` 默认使用 stdio，可切换 SSE/HTTP。  
  - Tavily 调用：`mcp_server/tools.py::SearchService` / `mcp_server/main.py::call_tavily`。

## 快速开始
1) 安装依赖  
```
uv sync
```

2) 配置变量  
- Tavily：`TAVILY_API_KEY`  
- LLM/Embedding：`MODEL_NAME`、`EMBEDDING_MODEL_NAME`，闭源时 `AI_BASE_URL`、`API_KEY`  
- 观测性（未用到，但可预设）：`LANGCHAIN_TRACING_V2`、`LANGCHAIN_API_KEY` 等

3) 最小环境验证（仅实例化，不发请求）  
```
uv run python scripts/check_env.py
```

4) 启动 MCP 服务器（默认 stdio，可切换 transport）  
```
set TAVILY_API_KEY=your_key
# 默认 stdio：子进程启动时读取 MCP_TRANSPORT=stdio
uv run python -m mcp_server.main
# 如需 HTTP：set MCP_TRANSPORT=streamable-http
# 如需 SSE：set MCP_TRANSPORT=sse
```

5) 测试 MCP 客户端（默认 stdio）  
```
uv run python scripts/test_mcp_client.py --query "model context protocol"
```
- HTTP：`--transport http --http-url http://127.0.0.1:8010/mcp`  
- SSE：`--transport sse`（需 server 以 sse 启动）  

预期：打印 Tavily 搜索结果（缺 key 时会提示错误），`logs/mcp_server_stderr.log` 可见 `search_web query=...`。

## 文件速览
- `pyproject.toml`：LangChain 1.x、langchain-deepseek、langchain-huggingface、sentence-transformers 等依赖；脚本入口。
- `.env` / `config.toml`：LLM/Embedding/Tavily 等配置示例。
- `devmate/config.py`：配置加载逻辑（env 优先，忽略未知字段，支持大写别名）。
- `devmate/llm.py`：ChatOpenAI/ChatDeepSeek、OpenAI/HuggingFace Embeddings 工厂。
- `devmate/logging_utils.py`：stderr + 滚动日志。
- `mcp_server/main.py`、`mcp_server/tools.py`：MCP server + Tavily 搜索。
- `devmate/mcp_client/client.py`、`scripts/test_mcp_client.py`：MCP 客户端与测试脚本。

## 限制
- 仅涵盖 Stage 1/2；Agent、RAG、Docker/compose、观测链路未提供。  
- Windows 环境下 stdio 可用；SSE/HTTP 需对应 transport/端口。  
- 必须设置 `TAVILY_API_KEY` 才能获得真实搜索结果。  

## 修改记录（MCP 客户端超时修复）
- 曾遇到 stdio 模式 `ClientSession` 初始化超时，已通过在客户端侧使用 `async with ClientSession(...)` 包裹会话修复。  
- 参考讨论：https://stackoverflow.com/questions/79692462/fastmcp-client-timing-out-while-initializing-the-session  
