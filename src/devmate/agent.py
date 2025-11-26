from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

from .config import Settings
from .mcp_client import invoke_search
from .rag import KnowledgeBase, search_knowledge_base

SYSTEM_PROMPT = """You are DevMate, an AI agent that builds and improves software projects.
Always consider using both the web search tool and the knowledge base retriever when
researching a task. When asked to generate project files, provide concise
recommendations and cite which tools you used."""


def build_agent(settings: Settings, kb: KnowledgeBase) -> AgentExecutor:
    llm = ChatOpenAI(
        model=settings.model_name,
        base_url=settings.ai_base_url,
        api_key=settings.api_key,
        temperature=0.2,
    )

    async def web_search(query: str) -> Any:
        return await invoke_search(query, ["python", "-m", "devmate.mcp_server"])

    async def kb_search(query: str) -> str:
        return search_knowledge_base(query, kb)

    tools = [
        Tool(
            name="web_search",
            func=web_search,
            description="Search the web via the MCP Tavily server",
        ),
        Tool(
            name="knowledge_base",
            func=kb_search,
            description="Retrieve internal guidelines and docs",
        ),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="structured-chat-zero-shot-react-description",
        verbose=True,
        prompt=prompt,
    )
    return agent


def run_interaction(settings: Settings, query: str) -> str:
    kb = KnowledgeBase(settings)
    kb.ingest()
    agent = build_agent(settings, kb)
    result = asyncio.run(agent.ainvoke({"input": query}))
    return result["output"] if isinstance(result, dict) else str(result)
