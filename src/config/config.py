# src/config/config.py
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq LLM configuration
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"  # updated, supported model

    # Embeddings configuration (local, Hugging Face)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Vector store
    vector_store_dir: Path = Path("data/index/chroma_db")
    collection_name: str = "multi_agent_rag"

    # Retrieval
    top_k: int = 5

    # Evaluation
    eval_questions_path: Path = Path("data/eval/questions.jsonl")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()