# Generated according to Phase 0 execution plan (Claude Code)
"""Utility functions to compute retrieval recall and precision for a query.

If the query does not exist in ``data/eval/questions.jsonl`` the functions
return ``None`` so the API can emit ``null`` per the specification.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Dict

QUESTIONS_FILE = Path("data/eval/questions.jsonl")

_cache: Optional[Dict[str, List[str]]] = None

def _load_questions() -> Dict[str, List[str]]:
    """Load the evaluation questions mapping ``query -> list of relevant doc ids``.

    Returns an empty dict if the file does not exist.
    """
    if not QUESTIONS_FILE.is_file():
        return {}
    mapping: Dict[str, List[str]] = {}
    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            mapping[rec["query"]] = rec.get("relevant_doc_ids", [])
    return mapping

def _get_questions() -> Dict[str, List[str]]:
    global _cache
    if _cache is None:
        _cache = _load_questions()
    return _cache

def compute_recall(query: str, retrieved_ids: List[str]) -> Optional[float]:
    """Return ``1.0`` if any retrieved doc matches a relevant doc, else ``0.0``.
    Returns ``None`` when the query is not part of the evaluation set.
    """
    relevant = _get_questions().get(query)
    if relevant is None:
        return None
    return 1.0 if any(doc_id in relevant for doc_id in retrieved_ids) else 0.0

def compute_precision(query: str, retrieved_ids: List[str]) -> Optional[float]:
    """Return the fraction of retrieved docs that are relevant.
    Returns ``None`` when the query is not part of the evaluation set or
    ``retrieved_ids`` is empty.
    """
    relevant = _get_questions().get(query)
    if relevant is None or not retrieved_ids:
        return None
    intersect = sum(1 for doc_id in retrieved_ids if doc_id in relevant)
    return intersect / len(retrieved_ids)
