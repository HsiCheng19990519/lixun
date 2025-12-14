from __future__ import annotations

"""
Core agent loop for Stage 4.

Responsibilities:
- Wire LLM with RAG + MCP tools.
- Apply system guidance (use local docs + web search, output plan + citations + file content).
- Ensure observability via LangSmith tracing.
"""

import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from devmate.agent.tools import build_tools
from devmate.agent.run_state import AgentRunFlags
from devmate.config import Settings
from devmate.llm import build_chat_model
from devmate.observability import build_tracing_config

logger = logging.getLogger(__name__)


@dataclass
class AgentFileResult:
    path: str
    language: str
    content: str
    saved_path: Optional[str] = None


@dataclass
class AgentRunResult:
    query: str
    raw_text: str
    files: List[AgentFileResult]
    used_rag: bool
    used_web: bool
    written_paths: List[str]


SYSTEM_PROMPT = """You are DevMate, a coding copilot.
Follow this workflow strictly:
1) Understand the request and outline a short plan.
2) You MUST call BOTH tools before answering:
   - Call `search_knowledge_base` first for local guidelines/templates. Do NOT claim you searched without calling it. Cite filename+chunk when used.
   - Call `search_web` (MCP/Tavily) next for external best practices/API docs. Do NOT fabricate web findings.
   Do not deliver the final answer before both calls complete.
   The final answer MUST include concrete and completed code/file blocks (not just plans) to satisfy the user's requirements. For example, if the user plans to create a website, you should give the code to generate the website.
3) Combine local + web findings, then propose files to create/modify.
4) When generating code, always show: file path + code block. Use this exact format per file:
```path: relative/original/path.ext
```<language>
<content>
```
5) Do not dump raw tool JSON; summarize key points. If a tool returns an error, continue with available info and note the failure.
Final answer MUST have three sections:
- Plan
- Findings (local docs / web, with sources)
- Files (path + content or actions)
"""
# Deliverables must include at least two files: `main.py` as the entrypoint (with `main()`) and `pyproject.toml` (Python 3.13, uv/LangChain deps). Also provide a sample CLI run command such as `uv run python main.py`.


FILE_BLOCK_RE = re.compile(
    r"```path:\s*(?P<path>[^\n]+)\n```(?P<lang>\w+)?\n(?P<content>.*?)```",
    re.DOTALL,
)


def extract_files_from_markdown(text: str) -> List[AgentFileResult]:
    files: List[AgentFileResult] = []
    for match in FILE_BLOCK_RE.finditer(text):
        path = match.group("path").strip()
        lang = (match.group("lang") or "").strip() or "text"
        content = match.group("content").strip()
        files.append(AgentFileResult(path=path, language=lang, content=content))
    return files


def _persist_files(files: List[AgentFileResult], output_dir: Optional[str | Path]) -> List[str]:
    """
    Write extracted files to disk (optional). Returns list of absolute paths written.
    If output_dir is provided, paths are resolved relative to it; otherwise the file path is used as-is.
    """
    written: List[str] = []
    base = Path(output_dir).resolve() if output_dir else None

    for file in files:
        raw_path = file.path.strip()
        if not raw_path:
            logger.warning("Skipping file with empty path")
            continue

        target = Path(raw_path)
        target = (base / target).resolve() if base else target.resolve()

        if base and not target.is_relative_to(base):
            logger.warning("Skipping file outside output_dir: %s", target)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file.content, encoding="utf-8")
        abs_path = str(target)
        file.saved_path = abs_path
        written.append(abs_path)
        logger.info("Wrote file: %s", abs_path)

    return written


def run_agent(
    message: str,
    *,
    settings: Optional[Settings] = None,
    transport: Optional[str] = None,
    rag_k: int = 4,
    max_iterations: int = 6,
    session_name: Optional[str] = None,
    write_files: bool = False,
    output_dir: Optional[str | Path] = None,
) -> AgentRunResult:
    """
    Execute the agent for a single user message and return structured result.
    """
    cfg = settings or Settings()
    logger.info("Agent starting query=%s", message)
    run_flags = AgentRunFlags()

    llm = build_chat_model(cfg)
    tools = build_tools(cfg, transport=transport, default_k=rag_k, run_flags=run_flags)

    agent = create_agent(
        llm,
        tools,
        system_prompt=SYSTEM_PROMPT,
    ).with_config({"recursion_limit": max_iterations})

    run_cfg = build_tracing_config(run_name="devmate-agent", session_name=session_name or "default")
    result = agent.invoke({"messages": [HumanMessage(content=message)]}, config=run_cfg)
    final_text = ""
    if isinstance(result, dict) and "messages" in result:
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content"):
                final_text = str(msg.content)
                break
    else:
        final_text = str(result)

    file_results = extract_files_from_markdown(final_text)
    written_paths = _persist_files(file_results, output_dir) if write_files and file_results else []

    return AgentRunResult(
        query=message,
        raw_text=final_text,
        files=file_results,
        used_rag=run_flags.used_rag,
        used_web=run_flags.used_web,
        written_paths=written_paths,
    )
