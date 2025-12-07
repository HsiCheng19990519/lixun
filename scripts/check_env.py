from __future__ import annotations

"""
Stage 1/2 环境快速验证：构造 LLM 和 Embedding，不触发真实请求。
"""

from devmate.config import Settings
from devmate.llm import build_chat_model, build_embedding_model


def main() -> None:
    s = Settings()
    chat = build_chat_model(s)
    emb = build_embedding_model(s)
    print("LLM:", type(chat))
    print("Embedding:", type(emb))


if __name__ == "__main__":
    main()
