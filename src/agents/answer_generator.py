# src/agents/answer_generator.py
from typing import List, Optional

from src.agents.schemas import RetrievedDoc, ResolvedEvidence
from src.models.llm import get_chat_llm


# Safety cap on how much text from a single document goes into the LLM
# prompt. Added after a real, reproducible groq.APIStatusError (413,
# "Request too large") caused by the document-upload feature: a large
# uploaded file can produce several ~1000-character chunks, which on top
# of the existing top_k permanent-KB chunks pushed a single prompt over
# the model's TPM limit. This caps each doc's contribution regardless of
# source (upload or permanent KB), since any source could in principle be
# large.
MAX_CHARS_PER_DOC = 2000


def _build_context_from_docs(docs: List[RetrievedDoc], doc_ids: Optional[List[str]] = None) -> str:
    if doc_ids is None:
        selected = docs
    else:
        idx = {d.id: d for d in docs}
        selected = [idx[did] for did in doc_ids if did in idx]

    chunks = []
    for d in selected:
        text = d.text
        if len(text) > MAX_CHARS_PER_DOC:
            text = text[:MAX_CHARS_PER_DOC] + "..."
        chunks.append(f"Doc id={d.id}\nSource={d.metadata.get('source', 'unknown')}\nText:\n{text}")
    return "\n\n".join(chunks)


def answer_generation_agent(
    query: str,
    docs: List[RetrievedDoc],
    resolved: ResolvedEvidence | None = None,
) -> str:
    """
    Uses resolved evidence (if provided) or raw docs to produce final answer.
    Explicitly highlights unresolved conflicts when necessary.
    """
    llm = get_chat_llm()

    if resolved is None or not resolved.has_conflict:
        # Standard RAG answer
        context = _build_context_from_docs(docs)
        prompt = (
            "You are an assistant answering a question using the provided context.\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Provide a concise, factual answer. If information is missing, say you are uncertain."
        )
        resp = llm.invoke(prompt)
        return resp.content

    # There are conflicts; construct a more explicit prompt
    pieces = []
    for cluster in resolved.conflict_clusters:
        res = resolved.resolutions.get(cluster.cluster_id)
        if res and res.status == "resolved":
            ctx = _build_context_from_docs(docs, doc_ids=res.chosen_doc_ids)
            pieces.append(
                f"Resolved cluster:\nDescription: {cluster.description}\n"
                f"Chosen docs: {res.chosen_doc_ids}\nRationale: {res.rationale}\n\n"
                f"Supporting text:\n{ctx}"
            )
        else:
            # unresolved
            ctx = _build_context_from_docs(docs, doc_ids=cluster.doc_ids)
            pieces.append(
                f"Unresolved cluster:\nDescription: {cluster.description}\n"
                f"Rationale: {res.rationale if res else 'No clear resolution.'}\n\n"
                f"Conflicting text:\n{ctx}"
            )

    context = "\n\n---\n\n".join(pieces)
    prompt = (
        "You are answering a question given evidence that may contain conflicts.\n"
        "Use resolved evidence when available; if some conflicts remain unresolved, "
        "explicitly mention the uncertainty.\n\n"
        f"Question: {query}\n\n"
        f"Evidence summary:\n{context}\n\n"
        "Produce a final answer. If conflicts remain unresolved, state that clearly."
    )

    resp = llm.invoke(prompt)
    return resp.content