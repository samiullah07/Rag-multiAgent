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
                )
            )
    return items


def simple_exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def compute_answer_accuracy(records: List[Dict[str, Any]]) -> float:
    # existing implementation remains unchanged
    correct = 0
    total = 0
    for r in records:
        if r["ground_truth"] is None:
            continue
        total += 1
        if simple_exact_match(r["answer"], r["ground_truth"]):
            correct += 1
    return correct / total if total else 0.0

# Helper to map evaluation IDs to EvalItem objects
def _build_eval_lookup(eval_items: List[EvalItem]) -> Dict[str, EvalItem]:
    return {item.id: item for item in eval_items}

def compute_conflict_detection_rate(records: List[Dict[str, Any]], eval_items: List[EvalItem]) -> float:
    """
    Conflict Detection Rate:
    Among questions where the gold label has_conflict == True,
    how often does the system's has_conflict_system == True.
    """
    gold_lookup = _build_eval_lookup(eval_items)
    total_conflict_cases = 0
    correct_conflict_flags = 0
    for rec in records:
        item = gold_lookup.get(rec.get("id"))
        if not item or not getattr(item, "has_conflict", False):
            continue
        total_conflict_cases += 1
        if rec.get("has_conflict_system", False):
            correct_conflict_flags += 1
    return correct_conflict_flags / total_conflict_cases if total_conflict_cases else 0.0

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
        total_resolvable += 1
        if gold_ids & chosen_ids:
            correct_resolutions += 1
    return correct_resolutions / total_resolvable if total_resolvable else 0.0

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
    correct = 0
    total = 0
    for r in records:
        if r["ground_truth"] is None:
            continue
        total += 1
        if simple_exact_match(r["answer"], r["ground_truth"]):
            correct += 1
    return correct / total if total else 0.0