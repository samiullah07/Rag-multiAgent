# src/config/config.py
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Optional Tavily API key for web search fallback
    tavily_api_key: str | None = None
    # Groq LLM configuration
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"  # active Groq model

    # Embeddings configuration (local, Hugging Face)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Vector store
    vector_store_dir: Path = Path("data/index/chroma_db")
    collection_name: str = "multi_agent_rag"

    # Retrieval
    top_k: int = 5

    # Contradiction detection
    use_nli_detection: bool = True
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_contradiction_threshold: float = 0.8

    # Evaluation
    eval_questions_path: Path = Path("data/eval/questions.jsonl")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()