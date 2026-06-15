# src/agents/contradiction_detector.py
import re
import uuid
from typing import List, Tuple

from src.agents.schemas import RetrievedDoc, ConflictCluster
from src.models.llm import get_chat_llm


def _extract_numeric_claims(text: str) -> List[str]:
    """
    Very simple heuristic to extract numeric claims (numbers and dates).
    """
    # This is intentionally simple to match DPP requirement of rule-based checks
    pattern = r"\b\d{4}\b|\b\d+(\.\d+)?\b"
    return re.findall(pattern, text)


def _find_obvious_conflicts(docs: List[RetrievedDoc]) -> List[ConflictCluster]:
    """
    Compares numeric tokens across docs; if the same context has different numbers,
    mark as a potential conflict cluster.
    """
    clusters: List[ConflictCluster] = []
    # naive: if any docs contain different numeric tokens for a given query,
    # flag a single cluster.
    all_nums_per_doc = [set(_extract_numeric_claims(d.text)) for d in docs]
    all_unique = set.union(*all_nums_per_doc) if all_nums_per_doc else set()

    if len(all_unique) <= 1:
        return clusters

    # If there are multiple distinct numeric values, treat as one cluster
    cluster = ConflictCluster(
        cluster_id=str(uuid.uuid4()),
        description="Potential numeric/date conflict between documents.",
        doc_ids=[d.id for d in docs],
    )
    clusters.append(cluster)
    return clusters


def _llm_refine_conflicts(query: str, docs: List[RetrievedDoc]) -> Tuple[bool, List[ConflictCluster]]:
    """
    Uses an LLM to decide whether the retrieved docs contain conflicting statements
    about the query, and optionally split into multiple clusters.
    """
    llm = get_chat_llm()
    passages = "\n\n".join(
        [f"Doc {i} (id={d.id}):\n{d.text}" for i, d in enumerate(docs, start=1)]
    )
    prompt = (
        "You are analysing retrieved documents for contradictions.\n"
        f"User query: {query}\n\n"
        f"Documents:\n{passages}\n\n"
        "Question: Do these documents contain conflicting statements about the query?\n"
        "Answer in strict JSON with fields:\n"
        "{\n"
        '  "has_conflict": true/false,\n'
        '  "clusters": [\n'
        '    {"description": "...", "doc_ids": ["doc_id1", "doc_id2", ...]},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
    )

    resp = llm.invoke(prompt)
    import json

    try:
        data = json.loads(resp.content)
        has_conflict = bool(data.get("has_conflict", False))
        clusters_raw = data.get("clusters", [])
        clusters: List[ConflictCluster] = []
        for c in clusters_raw:
            clusters.append(
                ConflictCluster(
                    cluster_id=str(uuid.uuid4()),
                    description=c.get("description", ""),
                    doc_ids=c.get("doc_ids", []),
                )
            )
        return has_conflict, clusters
    except Exception:
        # Fallback: treat as no conflicts if parsing fails
        return False, []


def contradiction_detection_agent(query: str, docs: List[RetrievedDoc]):
    """
    Combines simple rule-based checks (numeric conflicts) with LLM-based refinement.
    Returns: has_conflict, conflict_clusters
    """
    # Step 1: rule-based numeric/date conflicts
    rule_clusters = _find_obvious_conflicts(docs)

    # Step 2: LLM-based refinement
    has_conflict_llm, llm_clusters = _llm_refine_conflicts(query, docs)

    # Merge clusters
    clusters = rule_clusters + llm_clusters
    has_conflict = bool(clusters) or has_conflict_llm

    return has_conflict, clusters