# src/agents/resolution.py
from typing import List, Dict
from src.agents.schemas import (
    RetrievedDoc,
    ConflictCluster,
    ResolvedEvidence,
    ResolvedStatus,
)
from src.models.llm import get_chat_llm
from src.retrieval.vector_store import retrieve as vs_retrieve


def _format_docs_for_llm(doc_ids: List[str], docs: List[RetrievedDoc]) -> str:
    id_to_doc = {d.id: d for d in docs}
    out = []
    for did in doc_ids:
        d = id_to_doc.get(did)
        if not d:
            continue
        out.append(f"Document id={d.id}\nSource: {d.metadata.get('source', 'unknown')}\nText:\n{d.text}")
    return "\n\n".join(out)


def _parse_date(value):
    """
    Parse a date-like string for most_recent sorting.
    If parsing fails, return an empty string so it sorts last.
    """
    from datetime import datetime

    if not value:
        return ""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return ""


def _choose_most_recent(doc_ids: List[str], docs: List[RetrievedDoc]) -> List[str]:
    id_to_doc = {d.id: d for d in docs}
    ranked = sorted(
        doc_ids,
        key=lambda doc_id: _parse_date(id_to_doc[doc_id].metadata.get("publication_date")),
        reverse=True,
    )
    return [ranked[0]] if ranked else []


def _choose_most_authoritative(doc_ids: List[str], docs: List[RetrievedDoc]) -> List[str]:
    priority = {"real_world": 0, "wikipedia": 1, "synthetic": 2}
    id_to_doc = {d.id: d for d in docs}

    def rank(doc_id: str) -> int:
        source = id_to_doc[doc_id].metadata.get("source", "")
        return priority.get(source, len(priority))

    ranked = sorted(doc_ids, key=rank)
    return [ranked[0]] if ranked else []


def resolution_agent(
    query: str,
    docs: List[RetrievedDoc],
    clusters: List[ConflictCluster],
    strategy: str = "most_recent",
) -> ResolvedEvidence:
    """
    For each conflict cluster, attempts to decide which doc(s) are more reliable,
    optionally performing an extra retrieval step for more evidence.
    """
    llm = get_chat_llm()
    resolutions: Dict[str, ResolvedStatus] = {}

    for cluster in clusters:
        docs_text = _format_docs_for_llm(cluster.doc_ids, docs)

        if strategy == "most_recent":
            chosen_doc_ids = _choose_most_recent(cluster.doc_ids, docs)
            status = "resolved" if chosen_doc_ids else "unresolved"
            rationale = "Selected the most recently published source."
        elif strategy == "most_authoritative":
            chosen_doc_ids = _choose_most_authoritative(cluster.doc_ids, docs)
            status = "resolved" if chosen_doc_ids else "unresolved"
            rationale = "Selected the most authoritative source."
        elif strategy == "explain_both":
            chosen_doc_ids = cluster.doc_ids
            status = "unresolved"
            rationale = "Both sides are presented because strategy='explain_both'."
        else:
            # Optional self-correction loop: request extra evidence
            extra_docs = vs_retrieve(query=f"{query} (clarify conflicting information)", top_k=3)
            extra_str = "\n\n".join([f"Extra doc:\n{d.page_content}" for d in extra_docs])

            prompt = (
                "You are resolving conflicting information in a knowledge base.\n"
                f"User query: {query}\n\n"
                "Conflicting documents:\n"
                f"{docs_text}\n\n"
                "Additional evidence (may or may not help):\n"
                f"{extra_str}\n\n"
                "Based on all evidence, decide if you can select the most reliable statement.\n"
                "Respond in strict JSON:\n"
                "{\n"
                '  "status": "resolved" or "unresolved",\n'
                '  "chosen_doc_ids": ["id1", "id2", ...],\n'
                '  "rationale": "short explanation"\n'
                "}\n"
            )

            resp = llm.invoke(prompt)
            import json

            try:
                data = json.loads(resp.content)
                status = data.get("status", "unresolved")
                chosen_doc_ids = data.get("chosen_doc_ids", [])
                rationale = data.get("rationale", "")
            except Exception:
                status = "unresolved"
                chosen_doc_ids = []
                rationale = "Parsing error; treated as unresolved."

        resolutions[cluster.cluster_id] = ResolvedStatus(
            status=status, chosen_doc_ids=chosen_doc_ids, rationale=rationale
        )

    return ResolvedEvidence(
        query=query,
        has_conflict=bool(clusters),
        conflict_clusters=clusters,
        resolutions=resolutions,
        docs=docs,
    )