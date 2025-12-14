# DevMate  
本分支已完成阶段性目标：Stage 1（环境/依赖/配置基线）、Stage 2（MCP 搜索工具）、Stage 3（本地 RAG）、Stage 4（Agent 编排）、Stage 5（徒步路线网站场景，自动写文件与报告）、Stage 6（容器化与 Compose 运行 DevMate），以及 Stage 7 & 8（代码审查、文档与交付）。  

## 已完成
- Stage 1：`uv` 管理 Python 3.13；`pyproject.toml` 声明 LangChain 1.x、langchain-chroma、ChatDeepSeek/HF 等依赖；`devmate/config.py::Settings` 统一读取（env > .env > config.toml），敏感文件忽略已处理。  
- Stage 2：MCP 搜索（FastMCP，集成 Tavily）  
  - 服务器：`mcp_server/main.py` 暴露 `search_web`，支持 `stdio` / `sse` / `streamable-http`（默认 stdio，可用 `MCP_TRANSPORT` 切换），需 `TAVILY_API_KEY`。  
  - 客户端：`devmate/mcp_client/client.py`、`scripts/test_mcp_client.py` 默认 stdio，可切换 SSE/HTTP。  
  - Tavily 调用：`mcp_server/tools.py::SearchService` / `mcp_server/main.py::call_tavily`。  
- Stage 3：RAG  
  - 文档：`docs/internal_guidelines.md`、`docs/templates.md`、`docs/internal_fastapi_guidelines.md`。  
  - 向量库：`devmate/rag/ingest.py`（langchain-chroma + BAAI/bge-m3，持久化 `data/vector_store`，遥测关闭）。  
  - 检索：`devmate/rag/retriever.py::search_knowledge_base`，脚本 `scripts/test_rag.py` 可验证。  
- Stage 4：Agent（工具编排）  
  - 核心：`devmate/agent/core.py`，系统提示强制先查本地（RAG）再查网络（MCP），三段式输出（Plan/Findings/Files）。  
  - 工具：`devmate/agent/tools.py` 包装 `search_knowledge_base` + `search_web`，校验 `search_depth` 非法值回退到 `basic`。  
  - 入口：`main.py` / `devmate/cli.py` 支持命令行参数覆盖模型、transport、超时等；`observability.py` 可开启 LangSmith（需配置环境变量）。  
- Stage 5：徒步路线网站场景，`--stage5` 默认徒步提示，自动写文件到 `data/stage5_output/`，生成 `agent_output.md` 与 `stage5_report.json`，并检查是否调用本地/网络搜索及是否生成 `main.py`/`pyproject.toml`。系统提示要求包含文件代码块、入口与示例运行命令。  
- Stage 6：容器化（Docker + docker-compose），单镜像复用 app/ingest，外部 Chroma 容器（host/port）+ app 内部 stdio 启动 MCP，支持卷挂载写入输出与日志。  
- Stage 7 & 8：代码审查、文档与交付在开发过程中已自然覆盖。  

## 快速开始
1) 安装依赖  
```
uv sync
```

2) 配置变量  
- Tavily：`TAVILY_API_KEY`  
- LLM/Embedding：`MODEL_NAME`、`EMBEDDING_MODEL_NAME`，闭源时 `AI_BASE_URL`、`API_KEY`  
- RAG：`VECTOR_STORE_DIR`、`CHUNK_SIZE`、`CHUNK_OVERLAP`（可选）  

3) 最小环境验证（仅实例化，不发请求）  
```
uv run python scripts/check_env.py
```
预期输出（示例，类名可能随配置变化）：  
```
LLM: <class 'langchain_openai.chat_models.base.ChatOpenAI'>
Embedding: <class 'langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings'>
```

4) 启动 MCP 服务器（默认 stdio，可切换 transport）  
```
set TAVILY_API_KEY=your_key
# 默认 stdio：MCP_TRANSPORT=stdio
uv run python -m mcp_server.main
# 如需 HTTP：set MCP_TRANSPORT=streamable-http
# 如需 SSE：set MCP_TRANSPORT=sse
```

5) 测试 MCP 客户端（默认 stdio）  
```
uv run python -m scripts.test_mcp_client --query "model context protocol"
```
- HTTP：`--transport http --http-url http://127.0.0.1:8010/mcp`  
- SSE：`--transport sse`（需 server 以 sse 启动）  
预期：输出 Tavily 结果（缺 key 时会提示错误），`logs/mcp_server_stderr.log` 可见 `search_web query=...`。

6) 文档摄入（Stage 3）  
```
uv run python -m scripts.ingest_docs --rebuild
```
说明：读取 `docs/`，切分并写入 `data/vector_store`（遥测已禁用）；首次会将 `BAAI/bge-m3` 下载并缓存到 `data/hf_cache`（可用 `EMBEDDING_CACHE_DIR` 覆盖；同时会自动设置 `HF_HOME`/`TRANSFORMERS_CACHE`/`HUGGINGFACE_HUB_CACHE` 指向该目录），后续复用；如果速度过慢，可以先将 `HF_ENDPOINT` 设为 `https://hf-mirror.com`。

7) 知识库查询（Stage 3 验证）  
```
uv run python scripts/test_rag.py --query "project guidelines"
```
预期命中本地文档片段（文件名视 docs/ 内容而定）。

8) 运行 Agent（Stage 4 演示）  
```
uv run python legacy_main.py --message "我想构建一个展示附近徒步路线的网站项目" --transport stdio --k 4 \
  --provider ollama --model qwen2.5:14b-instruct --ai-base-url http://127.0.0.1:11434/v1
```
- 可用 `--transport http --mcp-http-url http://127.0.0.1:8010/mcp` 切换 HTTP；需要有效 `TAVILY_API_KEY`。  
- 观测性：设置 `LANGCHAIN_TRACING_V2=true` 及 LangSmith 相关 key 后，CLI 会初始化 tracing。
预期输出（示例，具体内容随模型/搜索结果变化）：  
```
### 计划
...（规划要点）

### 找到资料
- 本地文档：internal_guidelines.md / templates.md 等摘录
- 网络搜索：若 Tavily 200，则列出标题+URL 摘要；若失败，会注明

### 文件
- 列出建议生成/修改的文件与示例代码块
```

9) 运行 Agent（Stage 5 检测示例，自动写文件+报告）  
- 闭源模型（DashScope OpenAI 兼容，如 qwen3-max）：  
```
uv run python main.py --stage5 --transport stdio --k 6 --max-iterations 8 \
  --llm-mode closed_source --provider openai \
  --model qwen3-max \
```
- 开源/本地模型（Ollama 示例）：  
```
uv run python main.py --stage5 --transport stdio --k 6 --max-iterations 8 \
  --llm-mode open_source --provider ollama \
  --model qwen2.5:14b-instruct \
  --ai-base-url http://127.0.0.1:11434/v1
```
预期输出：写入 `data/stage5_output/`（至少包含 `main.py`、`pyproject.toml`、`.env.example` 等），并生成 `agent_output.md` / `stage5_report.json`。`stage5_report.json` 中应包含 `used_local_docs=true`、`used_web_search=true`，以及 `has_main_py=true`、`has_pyproject_toml=true` 的检查结果。  

10) Docker 与 Compose（运行 DevMate 本身）
- 构建镜像（基于项目根目录）：  
```
docker compose -f docker/docker-compose.yml build
```
- 启动依赖（仅 Chroma，MCP 由 app 进程以 stdio 自行拉起）：  
```
docker compose -f docker/docker-compose.yml up -d vector-db
```
- 构建/刷新向量库（HTTP 模式写入远程 Chroma 服务）：  
```
docker compose -f docker/docker-compose.yml run --rm ingest
```
- 运行 Agent（Stage5 默认徒步提示，使用容器内 venv 的 Python）：  
```
docker compose -f docker/docker-compose.yml run --rm \
  -e MCP_TRANSPORT=stdio \
  app \
  /app/.venv/bin/python -m devmate.cli \
  --stage5 --transport stdio \
  --k 6 --max-iterations 8 \
  --llm-mode closed_source --provider openai --model qwen3-max \
  --write-files --output-dir /app/data/stage5_output
```

## 文件速览
- `pyproject.toml`：LangChain 1.x、langchain-chroma、langchain-deepseek、langchain-huggingface、sentence-transformers 等依赖；脚本入口。  
- `.env` / `config.toml`：LLM/Embedding/Tavily 配置示例。  
- `devmate/config.py`：配置加载逻辑（env 优先，忽略未知字段，支持大写别名）。  
- `devmate/llm.py`：ChatOpenAI/ChatDeepSeek、OpenAI/HuggingFace Embeddings 工厂。  
- `devmate/logging_utils.py`：stderr + 滚动日志。  
- `mcp_server/main.py`、`mcp_server/tools.py`：MCP server + Tavily 搜索。  
- `devmate/mcp_client/client.py`、`scripts/test_mcp_client.py`：MCP 客户端与测试脚本。  
- `devmate/rag/ingest.py`、`devmate/rag/retriever.py`、`scripts/ingest_docs.py`、`scripts/test_rag.py`：RAG 摄入与检索（Chroma 支持本地持久化或 host/port 远程模式）。  
- `docker/Dockerfile`、`docker/docker-compose.yml`：Stage 6 容器化与编排（app + Chroma，MCP 以 stdio 方式由 app 内部拉起），默认使用 Chroma host/port 连接。  
- `docs/*`：本地知识库示例文档。  

## 部分问题解决记录
1) MCP 客户端 stdio 初始化超时：客户端侧使用 `async with ClientSession(...)` 包裹会话，避免 `session.initialize()` 卡死。参考讨论 https://stackoverflow.com/questions/79692462/fastmcp-client-timing-out-while-initializing-the-session。  
2) 网络搜索工具 400（search_depth 非法）：Agent 曾传 `search_depth=medium` 触发 Tavily 400。现工具侧校验并回退为 `basic`（Tavily 仅接受 basic/advanced，参见 https://docs.tavily.com/documentation/api-reference/endpoint/search）。
