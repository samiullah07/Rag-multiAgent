# scripts/test_groq.py
from src.models.llm import get_chat_llm
from src.models.embeddings import get_embeddings
from src.retrieval.vector_store import get_vector_store

def main():
    llm = get_chat_llm()
    resp = llm.invoke("Say 'Groq is wired correctly.'")
    print("LLM response:", resp.content)

    emb = get_embeddings()
    vecs = emb.embed_documents(["hello world", "another test"])
    print("Embeddings shape:", len(vecs), "x", len(vecs[0]))

    vs = get_vector_store()
    print("Vector store ready, collection name:", vs._collection.name)

if __name__ == "__main__":
    main()