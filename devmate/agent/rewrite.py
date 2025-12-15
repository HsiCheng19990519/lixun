from __future__ import annotations

"""
Query rewrite middleware for RAG.

Rewrites the latest human message into a retrieval-friendly query before the model runs,
adds a system note for transparency, and stores rewrite_info in state for downstream use.
"""

import logging
from typing import Any, Dict, Optional

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, SystemMessage

from devmate.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "You are a query rewrite helper for retrieval. "
    "Rewrite the user message into a concise search query (Chinese or English is fine). "
    "Keep key entities/locations/tasks, drop pleasantries. Output only the rewritten query."
)


class QueryRewriteMiddleware(AgentMiddleware[AgentState]):
    def __init__(
        self,
        model,
        *,
        settings: Settings,
        prompt: str = DEFAULT_PROMPT,
        max_chars: Optional[int] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.settings = settings
        self.prompt = prompt
        self.max_chars = max_chars if max_chars is not None else settings.rag_rewrite_max_chars
        self.enabled = enabled

    def _rewrite(self, text: str) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            response = self.model.invoke(
                [
                    SystemMessage(content=self.prompt),
                    HumanMessage(content=text),
                ]
            )
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            rewritten = str(content).strip()
            if not rewritten:
                return None
            if len(rewritten) > self.max_chars:
                rewritten = rewritten[: self.max_chars]
            return rewritten
        except Exception as exc:
            logger.warning("Query rewrite failed; using original query. err=%s", exc)
            return None

    def before_model(self, state: AgentState) -> Dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, HumanMessage):
            return None
        if state.get("rewrite_info"):
            # Already rewritten
            return None

        rewritten = self._rewrite(last.content)
        if not rewritten or rewritten == last.content:
            return {"rewrite_info": {"original": last.content, "rewritten": last.content, "applied": False}}

        note = SystemMessage(
            content=f"[RAG] rewritten query: {rewritten}\noriginal: {last.content}"
        )
        updated_messages = [*messages[:-1], note, last.model_copy(update={"content": rewritten})]
        return {
            "messages": updated_messages,
            "rewrite_info": {"original": last.content, "rewritten": rewritten, "applied": True},
        }


def build_rewrite_middleware(model, settings: Settings) -> Optional[QueryRewriteMiddleware]:
    """
    Build the rewrite middleware if enabled via settings (env/CLI).

    Controls are read from Settings (CLI/env):
    - rag_rewrite: toggle (default on).
    - rag_rewrite_max_chars: cap length of rewritten query (default 200).
    """
    enabled = settings.rag_rewrite
    max_chars = settings.rag_rewrite_max_chars
    if not enabled:
        logger.info("Query rewrite middleware disabled via rag_rewrite/DEV_RAG_REWRITE")
        return None
    return QueryRewriteMiddleware(
        model,
        settings=settings,
        max_chars=max_chars,
        enabled=enabled,
    )
