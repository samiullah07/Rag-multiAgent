# src/models/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from src.config.config import settings

_embeddings_instance = None


def get_embeddings():
    """Returns a cached embeddings model instance for the vector store."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings_instance