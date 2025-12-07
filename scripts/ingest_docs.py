from __future__ import annotations

import argparse
from pathlib import Path

from devmate.config import Settings
from devmate.rag.ingest import ingest_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest docs into local Chroma vector store.")
    parser.add_argument("--docs-dir", default="docs", help="Directory containing markdown/text docs")
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Directory to persist Chroma store (default: Settings.vector_store_dir)",
    )
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the vector store")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    docs_dir = Path(args.docs_dir)
    persist_dir = Path(args.persist_dir) if args.persist_dir else None
    num_docs, num_chunks = ingest_documents(
        settings=settings,
        docs_dir=docs_dir,
        persist_dir=persist_dir,
        rebuild=args.rebuild,
    )
    print(f"Ingestion completed: {num_docs} documents -> {num_chunks} chunks")


if __name__ == "__main__":
    main()
