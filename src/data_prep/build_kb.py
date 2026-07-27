# src/data_prep/build_kb.py
from pathlib import Path
from typing import List, Dict, Any
import json

from langchain_core.documents import Document

from src.retrieval.vector_store import add_documents


def load_plaintext_dir(root: Path, source_label: str) -> List[Document]:
    docs: List[Document] = []
    for f in root.glob("*.txt"):
        text = f.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"id": f.stem, "source": source_label, "filename": str(f)},
            )
        )
    return docs


def load_contradiction_jsonl(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj["text"],
                    metadata={
                        "id": obj["id"],
                        "source": obj.get("source", "synthetic_contradiction"),
                        "label": obj.get("label"),
                        "publication_date": obj.get("publication_date"),
                    },
                )
            )
    return docs


def main():
    raw_dir = Path("data/raw")
    contr_dir = Path("data/contradictions")

    all_docs: List[Document] = []
    metadata_list: List[Dict[str, Any]] = []

    # Real-world documents
    raw_docs = load_plaintext_dir(raw_dir, source_label="real_world")
    all_docs.extend(raw_docs)
    metadata_list.extend([d.metadata for d in raw_docs])

    # Synthetic contradictions: assume .jsonl files
    for jsonl_file in contr_dir.glob("*.jsonl"):
        cdocs = load_contradiction_jsonl(jsonl_file)
        all_docs.extend(cdocs)
        metadata_list.extend([d.metadata for d in cdocs])

    if not all_docs:
        raise RuntimeError("No documents found in data/raw or data/contradictions.")

    add_documents(all_docs, metadata_list)
    print(f"Indexed {len(all_docs)} documents into vector store.")


if __name__ == "__main__":
    main()