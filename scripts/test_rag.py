from __future__ import annotations

import argparse
import json

from devmate.config import Settings
from devmate.rag.retriever import search_knowledge_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local knowledge base.")
    parser.add_argument("--query", default="project guidelines", help="Search query text")
    parser.add_argument("--k", type=int, default=4, help="Number of results to return")
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Directory of persisted Chroma store (default: Settings.vector_store_dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    result = search_knowledge_base(
        query=args.query,
        settings=settings,
        persist_dir=args.persist_dir,
        k=args.k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
