# src/evaluation/metrics.py
from dataclasses import dataclass
from typing import List, Dict, Any

import time
import json


@dataclass
class EvalItem:
    id: str
    query: str
    ground_truth: str
    has_conflict: bool
    resolution_label: str | None = None  # optional id of correct doc/claim
    correct_doc_ids: List[str] = None
    should_flag_uncertain: bool = False
    out_of_kb: bool = False

def load_eval_questions(path) -> List[EvalItem]:
    items: List[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            items.append(
                EvalItem(
                    id=obj["id"],
                    query=obj["query"],
                    ground_truth=obj["ground_truth"],
                    has_conflict=bool(obj.get("has_conflict", False)),
                    resolution_label=obj.get("resolution_label"),
                    correct_doc_ids=obj.get("correct_doc_ids", []),
                    should_flag_uncertain=bool(obj.get("should_flag_uncertain", False)),
                    out_of_kb=bool(obj.get("out_of_kb", False)),
                )
            )
    return items


def simple_exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def simple_fuzzy_match(a: str, b: str) -> bool:
    """Return True if strings match closely.
    Uses substring containment (case‑insensitive) as primary check, falling back to
    a 90 % similarity ratio via difflib.SequenceMatcher.
    """
    a_low = a.strip().lower()
    b_low = b.strip().lower()
    if b_low in a_low or a_low in b_low:
        return True
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a_low, b_low).ratio() > 0.9


def compute_answer_accuracy(records: List[Dict[str, Any]]) -> float:
    """Compute answer accuracy using fuzzy matching.
    Returns proportion of records where the answer is considered correct.
    """
    correct = 0
    total = 0
    for r in records:
        if r["ground_truth"] is None:
            continue
        total += 1
        if simple_fuzzy_match(r["answer"], r["ground_truth"]):
            correct += 1
    return correct / total if total else 0.0

def compute_in_kb_accuracy(records: List[Dict[str, Any]], eval_items: List[EvalItem]) -> float:
    """Accuracy over only in-KB questions (where the answer exists in the corpus)."""
    lookup = {item.id: item for item in eval_items}
    correct = 0
    total = 0
    for r in records:
        item = lookup.get(r.get("id"))
        if not item or r["ground_truth"] is None:
            continue
        if item.out_of_kb:
            continue
        total += 1
        if simple_fuzzy_match(r["answer"], r["ground_truth"]):
            correct += 1
    return correct / total if total else 0.0


def compute_appropriate_refusal_rate(records: List[Dict[str, Any]], eval_items: List[EvalItem]) -> float:
    """Fraction of out-of-KB questions where the system correctly refused to answer.
    A correct refusal = the answer does NOT contain the ground truth (i.e. the system
    did not hallucinate it) AND faithful==False or the answer contains refusal language.
    """
    lookup = {item.id: item for item in eval_items}
    total = 0
    correct_refusals = 0
    refusal_phrases = ["uncertain", "not mentioned", "not contain", "does not",
                       "no information", "cannot answer", "don't have"]
    for r in records:
        item = lookup.get(r.get("id"))
        if not item or not item.out_of_kb:
            continue
        total += 1
        answer = r.get("answer", "")
        gt = r.get("ground_truth", "")
        contains_gt = simple_fuzzy_match(answer, gt)
        if contains_gt:
            continue
        faithful = r.get("faithful", True)
        has_refusal_language = any(p in answer.lower() for p in refusal_phrases)
        if not faithful or has_refusal_language:
            correct_refusals += 1
    return correct_refusals / total if total else 0.0


# Helper to map evaluation IDs to EvalItem objects
def _build_eval_lookup(eval_items: List[EvalItem]) -> Dict[str, EvalItem]:
    return {item.id: item for item in eval_items}

def compute_conflict_detection_rate(records: List[Dict[str, Any]], eval_items: List[EvalItem]) -> float:
    """Legacy helper returning recall (TP / (TP + FN))."""
    metrics = compute_conflict_detection_metrics(records, eval_items)
    return metrics['recall']

def compute_conflict_detection_metrics(records: List[Dict[str, Any]], eval_items: List[EvalItem]) -> Dict[str, Any]:
    """Return TP, FP, FN, TN and derived precision, recall, F1 for conflict detection.
    Gold label is item.has_conflict (bool). System flag is rec["has_conflict_system"].
    """
    gold_lookup = _build_eval_lookup(eval_items)
    tp = fp = fn = tn = 0
    for rec in records:
        item = gold_lookup.get(rec.get("id"))
        if not item:
            continue
        gold = bool(getattr(item, "has_conflict", False))
        system = bool(rec.get("has_conflict_system", False))
        if gold:
            if system:
                tp += 1
            else:
                fn += 1
        else:
            if system:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_resolution_quality(records: List[Dict[str, Any]], eval_items: List[EvalItem]) -> float:
    """
    Resolution Quality:
    Among questions where:
      - gold has_conflict == True
      - should_flag_uncertain == False (i.e., there is a resolvable ground truth)
    measure how often the system's chosen_doc_ids intersects with correct_doc_ids.
    """
    gold_lookup = _build_eval_lookup(eval_items)
    total_resolvable = 0
    correct_resolutions = 0
    for rec in records:
        item = gold_lookup.get(rec.get("id"))
        if not item:
            continue
        if not getattr(item, "has_conflict", False):
            continue
        if getattr(item, "should_flag_uncertain", False):
            continue
        gold_ids = set(getattr(item, "correct_doc_ids", []))
        if not gold_ids:
            continue
        chosen_ids = set(rec.get("chosen_doc_ids", []))
        # DEBUG output for each conflict record
        intersect = bool(gold_ids & chosen_ids)
        print(rec["id"], gold_ids, chosen_ids, intersect)
        total_resolvable += 1
        if intersect:
            correct_resolutions += 1
    result = correct_resolutions / total_resolvable if total_resolvable else 0.0
    print("Resolution quality debug: total_resolvable=", total_resolvable, "correct_resolutions=", correct_resolutions, "result=", result)
    return result

def compute_uncertain_flag_rate(records: List[Dict[str, Any]], eval_items: List[EvalItem]) -> float:
    """
    Uncertain Flag Rate (correct):
    Among questions where should_flag_uncertain == True (gold says conflict is genuinely unresolvable),
    measure how often the system correctly indicates uncertainty.
    """
    gold_lookup = _build_eval_lookup(eval_items)
    total_uncertain_cases = 0
    correct_uncertain_flags = 0
    # heuristic phrases indicating uncertainty
    uncertainty_phrases = [
        "cannot be resolved",
        "uncertain",
        "insufficient evidence",
        "conflicting sources",
    ]
    for rec in records:
        item = gold_lookup.get(rec.get("id"))
        if not item:
            continue
        if not getattr(item, "should_flag_uncertain", False):
            continue
        total_uncertain_cases += 1
        flagged = rec.get("flagged_uncertain")
        if flagged is None:
            # fallback heuristic on answer text
            answer_text = str(rec.get("answer", "")).lower()
            flagged = any(phrase in answer_text for phrase in uncertainty_phrases)
        if flagged:
            correct_uncertain_flags += 1
    return correct_uncertain_flags / total_uncertain_cases if total_uncertain_cases else 0.0