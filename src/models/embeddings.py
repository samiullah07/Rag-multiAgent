# src/models/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from src.config.config import settings

def get_embeddings():
    """
    Returns an embeddings model for the vector store.
    Currently uses OpenAIEmbeddings, configured via generic embedding_* settings.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)