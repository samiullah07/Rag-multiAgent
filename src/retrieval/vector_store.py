# src/retrieval/vector_store.py
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.models.embeddings import get_embeddings
from src.config.config import settings

_default_vs = None


def get_vector_store(
    persist_directory: Path | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """
    Returns a Chroma vector store instance (persistent).

    collection_name defaults to settings.collection_name (the permanent KB)
    when not given, so every existing call site is unaffected.
    Caches the default instance to avoid re-initializing embeddings per call.
    """
    global _default_vs
    if persist_directory is None:
        persist_directory = settings.vector_store_dir
    if collection_name is None:
        collection_name = settings.collection_name

    is_default = (str(persist_directory) == str(settings.vector_store_dir)
                  and collection_name == settings.collection_name)
    if is_default and _default_vs is not None:
        return _default_vs

    embeddings = get_embeddings()
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    if is_default:
        _default_vs = vs
    return vs


def reset_default_cache():
    """Invalidate the cached default vector store (call after rebuilding the KB)."""
    global _default_vs
    _default_vs = None


def add_documents(
    docs: List[Document],
    metadata_list: List[Dict[str, Any]] | None = None,
    persist_directory: Path | None = None,
) -> None:
    """
    Adds documents to the vector store.
    """
    reset_default_cache()
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


def retrieve_from_upload_session(query: str, session_id: str, top_k: int = 3) -> List[Document]:
    """
    Retrieves from a session-scoped upload collection created by the
    document-upload page (pages/1_📚_Knowledge_Base.py). This is completely
    separate from the permanent KB used by build_kb.py / run_eval.py —
    collection name and persist_directory both differ, so this can never
    touch or be confused with the permanent collection.

    Returns an empty list (never raises) if the session's collection
    doesn't exist or retrieval fails for any reason — a missing/empty
    upload collection must never break the main chat pipeline.
    """
    try:
        vs = get_vector_store(
            persist_directory=Path(f"data/index/session_uploads/{session_id}"),
            collection_name=f"session_{session_id}",
        )
        return vs.similarity_search(query, k=top_k)
    except Exception:
        return []