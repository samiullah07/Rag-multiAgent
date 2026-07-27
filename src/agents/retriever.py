# src/agents/retriever.py
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.retrieval.vector_store import retrieve as vs_retrieve
from src.agents.schemas import RetrievedDoc


def retriever_agent(query: str, top_k: int | None = None, upload_session_id: str | None = None) -> List[RetrievedDoc]:
    """
    Retriever Agent: returns top-k relevant documents as RetrievedDoc objects.
    If upload_session_id is provided, retrieves from session uploads first (max 3 docs),
    then merges with permanent KB results, biasing toward uploaded content.
    """
    retrieved = []

    # Step 1: Retrieve from session uploads (if any) - max 3 chunks
    if upload_session_id:
        session_dir = Path(f"data/index/session_uploads/{upload_session_id}")
        if session_dir.exists():
            embedding = get_embeddings()
            vs = Chroma(
                collection_name=f"session_{upload_session_id}",
                embedding_function=embedding,
                persist_directory=str(session_dir),
            )
            upload_docs: List[Document] = vs.similarity_search(query, k=3)
            for idx, d in enumerate(upload_docs):
                retrieved.append(
                    RetrievedDoc(
                        id=d.metadata.get("id", f"doc_{idx}"),
                        text=d.page_content,
                        metadata=d.metadata,
                        score=d.metadata.get("score"),
                    )
                )

    # Step 2: If no upload docs or still need more, retrieve from permanent KB
    if top_k is None:
        from src.config.config import settings
        top_k = settings.top_k

    # Only query KB if we have fewer than top_k docs
    if len(retrieved) < top_k:
        needed = top_k - len(retrieved)
        kb_docs: List[Document] = vs_retrieve(query, top_k=needed)

        # Deduplicate by ID (upload docs take precedence, keep first occurrence)
        existing_ids = {d.id for d in retrieved}
        for idx, d in enumerate(kb_docs):
            doc_id = d.metadata.get("id", f"doc_{idx}")
            if doc_id not in existing_ids:
                retrieved.append(
                    RetrievedDoc(
                        id=doc_id,
                        text=d.page_content,
                        metadata=d.metadata,
                        score=d.metadata.get("score"),
                    )
                )

    return retrieved[:top_k]