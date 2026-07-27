"""
Citation-Grounding Verifier.

Motivation (not hypothetical — observed directly in this project's own
evaluation data): when the answer-generation agent cannot find a fact in the
retrieved context, it sometimes fabricates a plausible-looking supporting
document — e.g. inventing "Doc id=astronomy_01" complete with realistic
markdown formatting — rather than admitting uncertainty. See
eval_nonconflict_03 in data/eval/multi_agent_records_explain_both.json for a
verbatim, captured example of this exact failure.

This module adds a verification step that runs after answer generation:

1. Deterministic citation check (no LLM call, instant, free): every
   "Doc id=X" the final answer text cites is compared against the IDs of
   documents that were ACTUALLY retrieved for this query. Any cited ID that
   doesn't exist in the real retrieved set is a fabricated citation — this
   alone would have caught the astronomy_01/02/03 case immediately.

2. Lightweight semantic grounding check (one LLM call): for answers that
   don't cite any fake ID but may still be relying on outside/parametric
   knowledge not actually present in the retrieved text, ask the LLM to
   verify whether the claims are supported by the provided documents.

The result is written into state["faithful"] (bool) and
state["faithfulness_notes"] (str), which app.py's insights panel already
reads and renders — no UI changes are required for the basic faithful/
not-faithful badge to start working. state["fabricated_citations"] is also
populated for the more specific "which IDs were faked" detail.
"""

import re
from typing import Dict, List, Tuple

from src.agents.schemas import RetrievedDoc
from src.models.llm import get_chat_llm

# Matches "Doc id=foo_bar", "doc id = 'foo-bar'", "Document Id: foo_bar", etc.
_DOC_ID_PATTERN = re.compile(r"[Dd]oc(?:ument)?\s*[Ii]d\s*[=:]\s*['\"]?([A-Za-z0-9_\-]+)")


def find_cited_doc_ids(answer_text: str) -> List[str]:
    """Extract every doc id the answer text explicitly cites (e.g. 'Doc id=science_07')."""
    return _DOC_ID_PATTERN.findall(answer_text or "")


def check_fabricated_citations(answer_text: str, retrieved_docs: List[RetrievedDoc]) -> List[str]:
    """
    Deterministic check: any cited doc id that was never actually retrieved
    for this query is a fabricated citation. Order-preserving, de-duplicated.
    """
    real_ids = {d.id for d in retrieved_docs}
    cited_ids = find_cited_doc_ids(answer_text)

    seen = set()
    fabricated: List[str] = []
    for cid in cited_ids:
        if cid not in real_ids and cid not in seen:
            seen.add(cid)
            fabricated.append(cid)
    return fabricated


def check_semantic_grounding(
    query: str, answer_text: str, retrieved_docs: List[RetrievedDoc]
) -> Tuple[bool, str]:
    """
    Lightweight LLM-based faithfulness check, used only for answers that
    passed the (free) fabricated-citation check. Deliberately a single,
    cheap call with a one-line structured response rather than a heavier
    multi-step self-consistency scheme, so it's affordable to run on every
    answer rather than being a separate opt-in step.
    """
    context = "\n\n".join(f"Doc id={d.id}\n{d.text}" for d in retrieved_docs)
    prompt = (
        "You are a strict fact-checker. Given a question, the documents that "
        "were actually retrieved, and a generated answer, decide whether the "
        "answer's factual claims are genuinely supported by the documents, "
        "or whether the answer relies on outside knowledge not present in "
        "them.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved documents:\n{context}\n\n"
        f"Answer to check:\n{answer_text}\n\n"
        "Respond with exactly one line, no other text, in this exact format:\n"
        "GROUNDED: yes|no | REASON: <one short sentence>"
    )
    resp = get_chat_llm().invoke(prompt)
    content = (resp.content or "").strip()
    is_grounded = "GROUNDED: YES" in content.upper()

    # Store just the human-readable reason, not the raw "GROUNDED: yes|no |"
    # prefix — that machine-formatted prefix was leaking into the UI caption
    # verbatim and looking like broken/overlapping text.
    reason_match = re.search(r"REASON\s*:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
    clean_reason = reason_match.group(1).strip() if reason_match else (content or "No verification response received.")

    # Self-consistency safety check: observed directly in live testing — the
    # LLM can say "GROUNDED: YES" while its own reason text admits the
    # opposite (a real case showed a "✓ Faithful" badge next to a reason
    # stating the answer "relies on outside knowledge not present in the
    # provided documents"). When the verdict and the reasoning disagree,
    # trust the reasoning, not the literal yes/no line.
    # IMPORTANT: check for negation before matching — "does NOT rely on
    # outside knowledge" must NOT trigger this override. Negation words
    # within 3 tokens before the phrase indicate the model is affirming
    # groundedness, not admitting a failure.
    _contradiction_phrases = (
        "outside knowledge",
        "external knowledge",
        "not present in the",
        "not supported by the documents",
        "not grounded in the",
    )
    _negation_words = ("not", "no", "never", "doesn't", "does not", "isn't", "is not")

    def _phrase_is_negated(text: str, phrase: str) -> bool:
        """Return True if the phrase appears but is preceded by a negation word."""
        idx = text.find(phrase)
        if idx == -1:
            return False
        # look at the 40 characters immediately before the phrase
        window = text[max(0, idx - 40):idx].lower()
        return any(neg in window for neg in _negation_words)

    reason_lower = clean_reason.lower()
    if is_grounded:
        for phrase in _contradiction_phrases:
            if phrase in reason_lower and not _phrase_is_negated(reason_lower, phrase):
                is_grounded = False
                clean_reason = (
                    "Self-consistency override (model said grounded, but its own "
                    f"reasoning contradicted that): {clean_reason}"
                )
                break

    return is_grounded, clean_reason


def verify_grounding(
    query: str,
    answer_text: str,
    retrieved_docs: List[RetrievedDoc],
    run_semantic_check: bool = True,
) -> Dict:
    """
    Main entry point for the grounding verifier.

    Returns a dict with:
      - faithful (bool)
      - faithfulness_notes (str)   -- consumed directly by app.py
      - fabricated_citations (List[str])
    """
    fabricated = check_fabricated_citations(answer_text, retrieved_docs)

    if fabricated:
        plural = "s" if len(fabricated) != 1 else ""
        return {
            "faithful": False,
            "faithfulness_notes": (
                f"Answer cites document{plural} that were never actually "
                f"retrieved for this query: {', '.join(fabricated)}. This is "
                f"a fabricated citation, not a grounded answer."
            ),
            "fabricated_citations": fabricated,
        }

    if not run_semantic_check:
        return {
            "faithful": True,
            "faithfulness_notes": "No fabricated citations detected (semantic check skipped).",
            "fabricated_citations": [],
        }

    is_grounded, explanation = check_semantic_grounding(query, answer_text, retrieved_docs)
    return {
        "faithful": is_grounded,
        "faithfulness_notes": explanation,
        "fabricated_citations": [],
    }
