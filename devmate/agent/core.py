from __future__ import annotations

"""
Core agent loop for Stage 4.

Responsibilities:
- Wire LLM with RAG + MCP tools.
- Apply system guidance (use local docs + web search, output plan + citations + file content).
- Ensure observability via LangSmith tracing.
"""

import logging
from typing import Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from devmate.agent.tools import build_tools
from devmate.config import Settings
from devmate.llm import build_chat_model
from devmate.observability import build_tracing_config

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are DevMate, a coding copilot.
Follow this workflow strictly:
1) Understand the request and outline a short plan.
2) When the user asks to build/create a project/service/website/app, you MUST call BOTH tools before answering:
   - Call `search_knowledge_base` first for local guidelines/templates. Do NOT claim you searched without calling it. Cite filename+chunk when used.
   - Call `search_web` (MCP/Tavily) next for external best practices/API docs. Do NOT fabricate web findings.
   Do not deliver the final answer before both calls complete.
3) For other requests, prefer local first; use web search if local is empty or outdated.
4) Combine local + web findings, then propose files to create/modify.
5) When generating code, always show: file path + code block.
6) Do not dump raw tool JSON; summarize key points. If a tool returns an error, continue with available info and note the failure.
Final answer MUST have three sections:
- Plan
- Findings (local docs / web, with sources)
- Files (path + content or actions)
"""


def run_agent(
    message: str,
    *,
    settings: Optional[Settings] = None,
    transport: Optional[str] = None,
    rag_k: int = 4,
    max_iterations: int = 6,
    session_name: Optional[str] = None,
) -> str:
    """
    Execute the agent for a single user message.
    """
    cfg = settings or Settings()
    logger.info("Agent starting query=%s", message)

    llm = build_chat_model(cfg)
    tools = build_tools(cfg, transport=transport, default_k=rag_k)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    ).with_config({"recursion_limit": max_iterations})

    run_cfg = build_tracing_config(run_name="devmate-agent", session_name=session_name or "default")
    result = agent.invoke({"messages": [HumanMessage(content=message)]}, config=run_cfg)
    # create_agent returns a state dict with "messages"; last AI message is the response
    if isinstance(result, dict) and "messages" in result:
        # find last AI message content
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content"):
                return str(msg.content)
    return str(result)
