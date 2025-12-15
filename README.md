# DevMate  
本分支已完成所有阶段性目标：Stage 1（环境/依赖/配置基线）、Stage 2（MCP 搜索工具）、Stage 3（本地 RAG）、Stage 4（Agent 编排）、Stage 5（徒步路线网站场景，自动写文件与报告）、Stage 6（容器化与 Compose 运行 DevMate），以及 Stage 7 & 8（代码审查、文档与交付），并进行了额外的开发工作，包括自适应 k、查询重写、计划节点等功能。

## 分支说明
- `main` 分支：当前可运行的 DevMate（Agent + RAG + MCP + Docker/Compose）版本。
- 如需复现阶段性检查（ `checklist.md` 里的 Stages 1 - 5），请切换到对应 stage 分支查看当时可运行代码与输出。因后续开发修改了之前的文件，在本分支运行中间阶段的代码可能会出现错误。

## 环境要求
- Python：建议 3.13+（以项目依赖为准）。
- 包管理：uv（用于同步依赖）。
- 容器：Docker + Docker Compose。
- 外部服务：按需提供 Tavily Key 与 LLM/Embedding 相关配置。

## 快速开始
1) 安装依赖  
```
uv sync
```

2) 配置变量  
- Tavily：`TAVILY_API_KEY`  
- LLM/Embedding：`MODEL_NAME`、`EMBEDDING_MODEL_NAME`，闭源时 `AI_BASE_URL`、`API_KEY`  
- RAG：`VECTOR_STORE_DIR`、`CHUNK_SIZE`、`CHUNK_OVERLAP`（可选）  

3) Docker 与 Compose（运行 DevMate 本身）
- 构建镜像（基于项目根目录）：  
```
docker compose -f docker/docker-compose.yml build --progress plain
```
- 启动依赖（仅 Chroma，MCP 由 app 进程以 stdio 自行拉起）：  
```
docker compose -f docker/docker-compose.yml up -d vector-db
```
- 构建/刷新向量库（HTTP 模式写入远程 Chroma 服务）：  
```
docker compose -f docker/docker-compose.yml run --rm ingest
```
- 运行 Agent（Stage5，使用容器内 venv 的 Python）：  
```
docker compose -f docker/docker-compose.yml run --rm \
  -e MCP_TRANSPORT=stdio \
  app \
  /app/.venv/bin/python -m devmate.cli \
  --stage5 --transport stdio \
  --k 6 --max-iterations 12 \
  --llm-mode closed_source --provider openai --model glm-4.6 \
  --write-files --output-dir /app/data/stage5_output
```

4) 输出与产物
- 模型会落盘生成的代码文件、原始回答与报告，输出目录与文件名可通过 CLI 参数调整（见下文参数一览）。

### 命令行参数一览（可改项 + 默认值）
覆盖优先级：CLI > 环境变量 > `.env` > `config.toml` > `Settings` 默认。
- **基础**
  - `--message/-m`：用户请求内容；不传时启动后会提示输入。  
  - `--session-name`：LangSmith/LangChain Tracing 会话名，默认 `default`。  
  - `--transport`：MCP 客户端传输层，`http`/`stdio`/`sse`，默认跟 env `MCP_TRANSPORT`，都无则 `http`。  
  - `--mcp-http-url`：MCP streamable HTTP 服务地址，默认空（仅 HTTP 模式需要）。  
  - `MCP_TRANSPORT`（env）：直接指定 MCP 传输层，默认 `http`。  
- **Stage5 与输出**
  - `--stage5`：启用徒步场景，自动写文件并生成报告，默认 `false`。  
  - `--write-files`：是否把 agent 生成的文件块落盘，默认 `false`；Stage5 时强制为 `true`。  
  - `--output-dir`：写入基目录，Stage5 默认 `data/stage5_output`（容器内 `/app/data/stage5_output`），否则按 agent 返回路径。  
  - `--stage5-raw-output`：保存原始 markdown 输出的路径，默认 `<output_dir>/agent_output.md`。  
  - `--stage5-report`：保存 Stage5 JSON 报告的路径，默认 `<output_dir>/stage5_report.json`。  
  - `--fail-on-missing-search`：Stage5 若未调用本地或网络搜索则退出非零，默认 `false`。  
- **迭代控制**
  - `--k`：初始 RAG top-k，必要时自动升高，默认 `4`。  
  - `--max-iterations`：Agent 工具/LLM 最大轮数，默认 `6`。  
  - `--recursion-limit`：LangGraph 递归上限，默认 `max_iterations * 2`（防止深层计划溢出）。  
- **LLM**
  - `--llm-mode`：选择开源/闭源模型，默认 `closed_source`。  
  - `--provider`：LLM 提供方（如 openai/ollama/deepseek/zhipu），默认 `zhipu`。  
  - `--model`：模型名称，默认 `glm-4.6`。  
  - `--ai-base-url`：OpenAI 兼容接口基址（私有网关时设置），默认空。  
  - `--api-key`：LLM API Key，默认空（需根据提供方填写）。  
- **Embedding**
  - `--embedding-mode`：开源/闭源向量模型，默认 `open_source`。  
  - `--embedding-provider`：向量提供方，默认 `huggingface`。  
  - `--embedding-model-name`：向量模型名，默认 `BAAI/bge-m3`。  
  - `--embedding-device`：运行设备，默认 `cpu`（可填 `cuda:0` 等）。  
  - `--embedding-base-url` / `--embedding-api-key`：私有或云端向量服务的地址与 Key，默认空。  
- **RAG / 切分**
  - `--vector-store-dir`：本地 Chroma 向量库目录，默认 `data/vector_store`。  
  - `--chunk-strategy`：文本切分策略，默认 `recursive`。  
  - `--chunk-size`：切分块大小，默认 `1000`。  
  - `--chunk-overlap`：切分重叠字符数，默认 `200`。  
  - `--rag-distance-keep-threshold`：保留向量距离上限（越低越严格），默认 `0.6`。  
  - `--rag-distance-requery-threshold`：若最优距离高于该值则放大 k 重查，默认 `0.8`。  
  - `--rag-max-k`：自适应重查时的最大 k，默认 `8`。  
  - `--rag-multi-hop`：是否开启多跳子查询，默认 `False`。  
  - `--rag-max-subqueries`：多跳时生成的最大子查询数，默认 `3`。  
  - `--rag-rewrite`：是否启用查询改写中间件，默认 `True`。  
  - `--rag-rewrite-max-chars`：改写结果的最大长度，默认 `200`。  
- **Tavily**
  - `--tavily-api-key`：Tavily 搜索 Key，默认空（必填否则无法搜索）。  
- **可观测性**
  - `--langchain-tracing-v2`：开启 LangChain Tracing v2，默认 `True`。  
  - `--langchain-api-key`：LangChain API Key，默认空。  
  - `--langsmith-api-key`：LangSmith API Key，默认空。  
  - `--langsmith-project`：LangSmith 项目名，默认 `devmate`。  
  - `--langsmith-endpoint`：LangSmith 自定义 Endpoint，默认空。  
- **日志**
  - `--log-level`：日志级别，默认 `INFO`。  
  - `--log-file`：日志文件路径，默认 `logs/devmate.log`。  

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
2) 网络搜索工具 400（search_depth 非法）：历史上曾出现传入 medium 导致 Tavily 400。当前工具侧会做校验并在必要时回退为 basic（Tavily 接受 basic/advanced，参见 https://docs.tavily.com/documentation/api-reference/endpoint/search）。

## 未来可改进方向
- 临时文件系统做上下文卸载：为每次运行分配隔离目录，暴露 `ls/read_file/write_file/edit_file` 工具，长上下文写文件、引用路径，减小 token 压力并便于后续复用。  
- 子任务/子 Agent 隔离：按职责拆分网络搜索、本地检索、代码生成等子 agent，主 agent 只负责协调与汇总，降低上下文污染。  
- 持久化记忆：接入 LangGraph Store 等长期存储，把关键决策/中间产物跨会话复用，并设计清理/迁移策略。  
- 可插拔中间件：在工具调用外层加拦截/审计/截断等中间件，统一处理日志、敏感信息、重试策略。  
- 性能与耗时：一系列增加的功能让运行时间大幅度上升；可后续加入本地嵌入缓存、并行多路搜索、热点文档预加载、LLM 温启动/复用、超时/截断保护等手段提升速度。  
- 中间阶段修复：后续开发中修改了之前的文件，导致本分支在运行中间阶段的命令时会出错，不利于读者理解或检查，之后可修复相关冲突，使每个阶段的命令都能得到正确结果。
