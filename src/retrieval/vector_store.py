# src/retrieval/vector_store.py
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.models.embeddings import get_embeddings
from src.config.config import settings


def get_vector_store(persist_directory: Path | None = None) -> Chroma:
    """
    Returns a Chroma vector store instance (persistent).
    """
    if persist_directory is None:
        persist_directory = settings.vector_store_dir

    embeddings = get_embeddings()
    vs = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    return vs


def add_documents(
    docs: List[Document],
    metadata_list: List[Dict[str, Any]] | None = None,
    persist_directory: Path | None = None,
) -> None:
    """
    Adds documents to the vector store.
    """
    vs = get_vector_store(persist_directory)
    if metadata_list is not None:
        assert len(docs) == len(metadata_list), "Docs and metadata length mismatch."
        vs.add_documents(
            [Document(page_content=d.page_content, metadata=m) for d, m in zip(docs, metadata_list)]
        )
    else:
        vs.add_documents(docs)
    vs.persist()


def retrieve(query: str, top_k: int | None = None) -> List[Document]:
    """
    Semantic retrieval from the vector store.
    """
    if top_k is None:
        top_k = settings.top_k

    vs = get_vector_store()
    return vs.similarity_search(query, k=top_k)