# src/agents/contradiction_detector.py
import re
import uuid
from itertools import combinations
from typing import List, Tuple

from src.agents.schemas import RetrievedDoc, ConflictCluster
from src.config.config import settings
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


_nli_model = None


def _get_nli_model():
    """Lazy-load the NLI cross-encoder (CPU, no API cost)."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(settings.nli_model)
    return _nli_model


def _softmax(logits):
    """Convert raw logits to probabilities."""
    import numpy as np
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


_spacy_nlp = None


def _get_spacy():
    """Lazy-load spaCy English model for POS tagging."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def _extract_query_content_words(query: str) -> List[str]:
    """
    POS-based content-word extraction (spaCy).
    Keeps nouns, proper nouns, and named-entity tokens from the query.
    Returns lowercased lemmas for matching.
    """
    nlp = _get_spacy()
    doc = nlp(query)
    content_pos = {"NOUN", "PROPN"}
    keywords = set()
    for token in doc:
        if token.pos_ in content_pos:
            keywords.add(token.lemma_.lower())
    for ent in doc.ents:
        for token in ent:
            keywords.add(token.lemma_.lower())
    return list(keywords)


def _extract_query_relevant_sentences(query: str, text: str) -> List[str]:
    """
    Extract the single most query-relevant sentence from a document.
    Uses spaCy POS tagging to identify content words (nouns, proper nouns,
    named entities) in the query, then scores document sentences by overlap.
    """
    sentences = re.split(r'[\n\r]+|(?<=[.!?])\s+', text)
    sentences = [s.strip().lstrip('- ').strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s) > 30 and not s.startswith('#')
                 and 'document covers' not in s.lower()]

    query_keywords = _extract_query_content_words(query)

    if not query_keywords:
        return sentences[:1] if sentences else []

    scored = []
    for s in sentences:
        s_lower = s.lower()
        overlap = sum(1 for kw in query_keywords if kw in s_lower)
        if overlap > 0:
            scored.append((overlap, s))

    if not scored:
        return sentences[:1] if sentences else []

    scored.sort(reverse=True)
    return [scored[0][1]]


def _nli_detect_conflicts(query: str, docs: List[RetrievedDoc]) -> Tuple[bool, List[ConflictCluster]]:
    """
    Claim-level NLI contradiction detection (DeBERTa-v3-small cross-encoder).
    Extracts query-relevant sentences from each doc, then runs NLI on
    sentence pairs across docs. Flags a conflict when the model predicts
    CONTRADICTION above threshold.
    """
    if len(docs) < 2:
        return False, []

    model = _get_nli_model()
    threshold = settings.nli_contradiction_threshold

    # Extract query-relevant sentences from each doc
    doc_sentences = []
    for d in docs:
        sents = _extract_query_relevant_sentences(query, d.text)
        doc_sentences.append(sents)

    # Build sentence pairs across doc combinations (both directions)
    pairs = []
    pair_doc_indices = []
    for i, j in combinations(range(len(docs)), 2):
        for sent_i in doc_sentences[i]:
            for sent_j in doc_sentences[j]:
                pairs.append((sent_i, sent_j))
                pair_doc_indices.append((i, j))
                pairs.append((sent_j, sent_i))
                pair_doc_indices.append((j, i))

    if not pairs:
        return False, []

    # Model outputs raw logits [contradiction, entailment, neutral]
    raw_scores = model.predict(pairs)
    probs = _softmax(raw_scores)

    conflicting_pairs = set()
    for idx, prob in enumerate(probs):
        contradiction_prob = prob[0]
        if contradiction_prob > threshold:
            i, j = pair_doc_indices[idx]
            conflicting_pairs.add((min(i, j), max(i, j)))

    if not conflicting_pairs:
        return False, []

    clusters = []
    for i, j in conflicting_pairs:
        cluster = ConflictCluster(
            cluster_id=str(uuid.uuid4()),
            description=f"NLI-detected contradiction between '{docs[i].id}' and '{docs[j].id}'",
            doc_ids=[docs[i].id, docs[j].id],
        )
        clusters.append(cluster)

    return True, clusters


def contradiction_detection_agent(query: str, docs: List[RetrievedDoc]):
    """
    Hybrid NLI + rule-based + LLM contradiction detection.
    - NLI (cross-encoder/nli-deberta-v3-small) is the primary detector: deterministic,
      high precision at threshold 0.8, catches claim-level entailment contradictions.
    - Rule-based numeric check acts as a pre-filter for cases NLI misses (e.g.
      "approximately 300M" vs "exactly 299,792,458" where NLI sees them as compatible).
    - LLM is invoked ONLY when the rule-based check raises suspicion but NLI did not
      fire — this limits LLM non-determinism to the narrow gap where numeric
      differences exist but NLI's textual entailment model can't resolve them.
    Returns: has_conflict, conflict_clusters
    """
    if settings.use_nli_detection:
        # Step 1: NLI-based detection (deterministic, high precision at 0.8)
        nli_has_conflict, nli_clusters = _nli_detect_conflicts(query, docs)

        if nli_has_conflict:
            # NLI is confident — use LLM only for richer cluster descriptions
            _has_conflict_llm, llm_clusters = _llm_refine_conflicts(query, docs)
            clusters = llm_clusters if llm_clusters else nli_clusters
            return True, clusters

        # Step 2: NLI did not fire — check for numeric conflicts in
        # query-relevant sentences specifically (not document-wide)
        doc_sentences = [_extract_query_relevant_sentences(query, d.text) for d in docs]
        relevant_nums = []
        for sents in doc_sentences:
            nums = set()
            for s in sents:
                # Extract numbers with at least 3 digits (filter noise)
                found = re.findall(r'\b\d[\d,]*(?:\.\d+)?\b', s)
                for n in found:
                    clean = n.replace(',', '')
                    try:
                        val = float(clean)
                        if val >= 100:
                            nums.add(val)
                    except ValueError:
                        pass
            relevant_nums.append(nums)

        # Check if different docs have different numbers for this query
        all_nums = set()
        for nums in relevant_nums:
            all_nums.update(nums)

        # Any two distinct numbers in query-relevant sentences = discrepancy.
        # The set already deduplicates identical values; sentence extraction
        # ensures only query-focused content contributes.
        has_numeric_discrepancy = len(all_nums) > 1

        if not has_numeric_discrepancy:
            return False, []

        # Step 3: Numeric discrepancy in query-relevant sentences —
        # these docs give different specific numbers for the same query,
        # which constitutes a factual conflict even if the values are close
        conflict_doc_ids = [docs[i].id for i, nums in enumerate(relevant_nums) if nums]
        if len(conflict_doc_ids) >= 2:
            cluster = ConflictCluster(
                cluster_id=str(uuid.uuid4()),
                description=f"Numeric discrepancy: documents provide different specific values ({', '.join(str(int(n)) for n in sorted(all_nums))})",
                doc_ids=conflict_doc_ids[:2],
            )
            return True, [cluster]
        return False, []
    else:
        # Fallback: original rule-based + LLM-only pipeline (for ablation)
        rule_clusters = _find_obvious_conflicts(docs)
        has_conflict_llm, llm_clusters = _llm_refine_conflicts(query, docs)
        has_conflict = has_conflict_llm
        clusters = llm_clusters if has_conflict_llm else []
        return has_conflict, clusters