from __future__ import annotations

"""
Knowledge base retrieval for Stage 3.

Loads the persisted Chroma store and exposes search_knowledge_base().
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from chromadb import HttpClient

from devmate.config import Settings
from devmate.llm import build_embedding_model
from devmate.logging_utils import setup_logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "devmate-docs"


def _load_store(settings: Settings, persist_dir: Optional[Path] = None) -> Chroma:
    store_path = persist_dir or Path(settings.vector_store_dir)
    use_http = bool(settings.chroma_host)
    if not use_http and not store_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {store_path}. Run scripts/ingest_docs.py first."
        )
    embedding = build_embedding_model(settings)
    client = HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_http_port,
        ssl=settings.chroma_ssl,
    ) if use_http else None
    persist_directory = None if use_http else str(store_path)
    return Chroma(
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        client=client,
        host=settings.chroma_host if use_http else None,
        port=settings.chroma_http_port if use_http else None,
        ssl=settings.chroma_ssl if use_http else None,
    )


def search_knowledge_base(
    query: str,
    *,
    settings: Optional[Settings] = None,
    persist_dir: Optional[Path] = None,
    k: int = 4,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Similarity search over the local vector store.

    Returns a dict with query and normalized results.
    """
    cfg = settings or Settings()
    setup_logging(cfg)
    logger.info("Searching knowledge base query=%s k=%s", query, k)

    store = _load_store(cfg, persist_dir)
    docs_with_scores = store.similarity_search_with_score(query, k=k)

    results: List[Dict[str, object]] = []
    for doc, score in docs_with_scores:
        results.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "filename": doc.metadata.get("filename"),
                "score": score,
                "chunk_id": doc.metadata.get("chunk_id"),
            }
        )

    return {"query": query, "results": results}
