from __future__ import annotations

"""
Document ingestion for the local knowledge base (Stage 3).

Reads markdown/text files, chunks them, and writes a persisted Chroma store.
"""

import logging
import shutil
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Disable Chroma telemetry before importing Chroma to prevent external calls.
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "0")
os.environ.setdefault("POSTHOG_DISABLE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings

from devmate.config import Settings
from devmate.llm import build_embedding_model
from devmate.logging_utils import setup_logging

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
COLLECTION_NAME = "devmate-docs"


def _load_documents(docs_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in docs_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "filename": path.name},
                )
            )
    return docs


def _split_documents(
    docs: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(list(docs))


def ingest_documents(
    settings: Settings,
    docs_dir: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    rebuild: bool = False,
) -> Tuple[int, int]:
    """
    Ingest docs into a persisted Chroma vector store.

    Returns (num_documents, num_chunks).
    """
    docs_path = docs_dir or Path("docs")
    persist_path = persist_dir or Path(settings.vector_store_dir)

    setup_logging(settings)
    logger.info("Starting ingestion docs_dir=%s persist_dir=%s rebuild=%s", docs_path, persist_path, rebuild)

    if not docs_path.exists():
        raise FileNotFoundError(f"docs directory not found: {docs_path}")

    raw_docs = _load_documents(docs_path)
    if not raw_docs:
        raise RuntimeError(f"No documents found in {docs_path}. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")

    chunks = _split_documents(raw_docs, settings.chunk_size, settings.chunk_overlap)
    for idx, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = idx

    if rebuild and persist_path.exists():
        shutil.rmtree(persist_path)
        logger.info("Removed existing vector store at %s", persist_path)

    persist_path.mkdir(parents=True, exist_ok=True)

    embedding = build_embedding_model(settings)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_path),
        client_settings=ChromaSettings(
            anonymized_telemetry=False,
            persist_directory=str(persist_path),
        ),
    )

    logger.info("Ingestion completed: %s documents -> %s chunks", len(raw_docs), len(chunks))
    return len(raw_docs), len(chunks)
