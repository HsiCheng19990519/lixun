# DevMate —— Stage 1（环境与配置基线）

本分支仅覆盖考核第一阶段：环境、依赖、配置基线。后续 MCP、RAG、Agent、容器化、观测性等未在此分支实现。

## 已完成
- Python 3.13，使用 uv 管理依赖与虚拟环境（`pyproject.toml` + `uv.lock`，无 `requirements.txt`）。
- LangChain 1.x 相关依赖已声明，包含 OpenAI/Community/Text splitters、Chroma、tavily、HuggingFace，以及 ChatDeepSeek（`langchain-deepseek`）。
- 配置管理：`.env` / `config.toml`，由 `devmate/config.py::Settings` 统一加载，优先级 env > `.env` > `config.toml`，支持大写别名。
- LLM/Embedding 工厂：`devmate/llm.py`（ChatOpenAI / ChatDeepSeek；OpenAI / HuggingFace Embeddings），无硬编码模型/密钥，全部走配置。
- 日志：`devmate/logging_utils.py` 配置 stderr 控制台 + 滚动文件 `logs/devmate.log`。
- `.gitignore` 忽略 `.env`、`config.toml` 等敏感文件。

## 快速开始
前置：安装 uv，确保 Python 3.13 可用。

1) 安装依赖  
```
uv sync
```

2) 配置变量（`.env` / `config.toml` 或直接环境变量）  
- 必填：`MODEL_NAME`、`EMBEDDING_MODEL_NAME`  
- 闭源模型：`AI_BASE_URL`、`API_KEY`  
- Tavily：`TAVILY_API_KEY`  
- 观测性：`LANGCHAIN_TRACING_V2`、`LANGCHAIN_API_KEY`、`LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`

3) 最小环境验证（仅实例化，不会发真实请求）  
```
uv run python scripts/check_env.py
```
预期输出示例（类名可能因配置不同略有差异）：  
```
LLM: <class 'langchain_openai.chat_models.base.ChatOpenAI'>
Embedding: <class 'langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings'>
```

## 文件速览
- `pyproject.toml`：LangChain 1.x、langchain-deepseek 等依赖；脚本入口。
- `.env` / `config.toml`：配置示例（LLM/Embedding/Tavily/LangSmith 等）。
- `devmate/config.py`：配置加载（env 优先，忽略多余字段，支持大写别名）。
- `devmate/llm.py`：ChatOpenAI/ChatDeepSeek、OpenAI/HuggingFace Embeddings 工厂。
- `devmate/logging_utils.py`：stderr + 滚动日志配置。
- `scripts/check_env.py`：环境与依赖快速验证脚本。

## 注意
- 使用闭源模型或 Tavily 前，请设置相关 key，否则仅能验证实例化。***
