# src/agents/retriever.py
from typing import List
from langchain_core.documents import Document

from src.retrieval.vector_store import retrieve as vs_retrieve
from src.agents.schemas import RetrievedDoc


def retriever_agent(query: str, top_k: int | None = None) -> List[RetrievedDoc]:
    """
    Retriever Agent: returns top-k relevant documents as RetrievedDoc objects.
    """
    docs: List[Document] = vs_retrieve(query, top_k=top_k)
    retrieved = []
    for idx, d in enumerate(docs):
        retrieved.append(
            RetrievedDoc(
                id=d.metadata.get("id", f"doc_{idx}"),
                text=d.page_content,
                metadata=d.metadata,
                score=d.metadata.get("score"),
            )
        )
    return retrieved