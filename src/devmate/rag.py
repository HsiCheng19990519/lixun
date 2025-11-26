from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from .config import Settings


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.ai_base_url,
            api_key=settings.api_key,
        )
        self.persist_directory = settings.chroma_persist_path

    def ingest(self) -> None:
        """Load docs from the configured path and persist to Chroma."""

        docs_path = Path(self.settings.docs_path)
        docs_path.mkdir(parents=True, exist_ok=True)
        documents = []
        for file_path in docs_path.rglob("*.md"):
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        splits = splitter.split_documents(documents)
        Chroma.from_documents(
            documents=splits,
            embedding=self._embeddings,
            persist_directory=self.persist_directory,
        )

    def retriever(self):
        return Chroma(
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        ).as_retriever(search_kwargs={"k": 4})


def search_knowledge_base(query: str, kb: KnowledgeBase):
    retriever = kb.retriever()
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)
