# src/models/llm.py
from langchain_groq import ChatGroq
from src.config.config import settings
def get_chat_llm():
    """
    Returns a LangChain ChatModel instance configured to use Groq.
    """
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
    )