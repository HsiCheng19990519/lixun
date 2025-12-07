from __future__ import annotations

"""
快速验证 Stage 1 环境：构造 LLM 和 Embedding，不发真实请求。
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
