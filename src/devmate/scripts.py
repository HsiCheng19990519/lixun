from __future__ import annotations

import argparse

from .config import load_settings
from .rag import KnowledgeBase


def ingest_cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into Chroma")
    parser.parse_args()
    settings = load_settings()
    kb = KnowledgeBase(settings)
    kb.ingest()
    print(f"Ingested documents from {settings.docs_path} into {settings.chroma_persist_path}")
