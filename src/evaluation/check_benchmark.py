"""
Benchmark regression guard.

Reads the saved evaluation records and recomputes the four headline metrics
directly from disk (never from a script's printed summary), then compares them
against a locked-in benchmark baseline. Exits non-zero if any metric has
regressed beyond a small tolerance.

This exists because, repeatedly during development, automated progress reports
described results that did not match what was actually written to disk. This
guard makes that class of error impossible to miss: it reads the real file,
does the real arithmetic, and fails loudly if the numbers moved the wrong way.

Usage:
    uv run python -m src.evaluation.check_benchmark

    # Update the locked baseline after a *deliberate, verified* improvement:
    uv run python -m src.evaluation.check_benchmark --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RECORDS_PATH = Path("data/eval/multi_agent_records.json")
QUESTIONS_PATH = Path("data/eval/questions.jsonl")
BASELINE_PATH = Path("data/eval/benchmark_baseline.json")

# How much a metric may drop before it counts as a regression.
# Small tolerance absorbs LLM run-to-run non-determinism without hiding real drops.
TOLERANCE = 0.05


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum() or ch.isspace()).strip()


def fuzzy_match(answer: str, truth: str) -> bool:
    return _norm(truth) in _norm(answer)


def _load_questions_lookup() -> dict[str, dict]:
    """Load questions.jsonl into a dict keyed by id."""
    if not QUESTIONS_PATH.exists():
        return {}
    lookup = {}
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            lookup[obj["id"]] = obj
    return lookup


def compute_metrics_from_records(records: list[dict]) -> dict:
    """Recompute all headline metrics from raw saved records."""
    n = len(records)
    if n == 0:
        return {"answer_accuracy": 0.0, "in_kb_accuracy": 0.0,
                "appropriate_refusal_rate": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "resolution_quality": 0.0, "n": 0}

    questions = _load_questions_lookup()

    # Answer accuracy (fuzzy substring) — overall
    correct = sum(1 for r in records if fuzzy_match(r.get("answer", ""), r.get("ground_truth", "")))
    accuracy = correct / n

    # In-KB accuracy: only questions where out_of_kb is not True
    in_kb_records = [r for r in records if not questions.get(r.get("id"), {}).get("out_of_kb", False)]
    in_kb_correct = sum(1 for r in in_kb_records if fuzzy_match(r.get("answer", ""), r.get("ground_truth", "")))
    in_kb_accuracy = in_kb_correct / len(in_kb_records) if in_kb_records else 0.0

    # Appropriate refusal rate: out-of-KB questions where system correctly declined
    out_of_kb_records = [r for r in records if questions.get(r.get("id"), {}).get("out_of_kb", False)]
    refusal_phrases = ["uncertain", "not mentioned", "not contain", "does not",
                       "no information", "cannot answer", "don't have"]
    correct_refusals = 0
    for r in out_of_kb_records:
        answer = r.get("answer", "")
        gt = r.get("ground_truth", "")
        if fuzzy_match(answer, gt):
            continue
        faithful = r.get("faithful", True)
        has_refusal = any(p in answer.lower() for p in refusal_phrases)
        if not faithful or has_refusal:
            correct_refusals += 1
    appropriate_refusal_rate = correct_refusals / len(out_of_kb_records) if out_of_kb_records else 0.0

    # Conflict detection confusion matrix
    tp = fp = fn = tn = 0
    for r in records:
        gold = bool(r.get("has_conflict_gold", False))
        pred = bool(r.get("has_conflict_system", False))
        if gold and pred: tp += 1
        elif not gold and pred: fp += 1
        elif gold and not pred: fn += 1
        else: tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Resolution quality: of gold-conflict records, fraction whose chosen docs
    # include at least one gold-correct doc.
    resolvable = [r for r in records if r.get("has_conflict_gold") and r.get("correct_doc_ids")]
    if resolvable:
        good = 0
        for r in resolvable:
            chosen = set(r.get("chosen_doc_ids", []))
            gold = set(r.get("correct_doc_ids", []))
            if chosen & gold:
                good += 1
        resolution_quality = good / len(resolvable)
    else:
        resolution_quality = None

    return {
        "answer_accuracy": round(accuracy, 4),
        "in_kb_accuracy": round(in_kb_accuracy, 4),
        "appropriate_refusal_rate": round(appropriate_refusal_rate, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "resolution_quality": round(resolution_quality, 4) if resolution_quality is not None else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n": n,
    }


def load_records() -> list[dict]:
    if not RECORDS_PATH.exists():
        print(f"ERROR: {RECORDS_PATH} does not exist. Run the evaluation first.", file=sys.stderr)
        sys.exit(2)
    return json.loads(RECORDS_PATH.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true",
                    help="Lock the current metrics as the new baseline (use only after a verified improvement).")
    args = ap.parse_args()

    records = load_records()
    current = compute_metrics_from_records(records)

    print("=" * 60)
    print("BENCHMARK CHECK — metrics recomputed directly from disk")
    print("=" * 60)
    print(f"  Records evaluated:      {current['n']}")
    print(f"  Answer accuracy:        {current['answer_accuracy']}")
    print(f"  In-KB accuracy:         {current['in_kb_accuracy']}")
    print(f"  Appropriate refusal:    {current['appropriate_refusal_rate']}")
    print(f"  Conflict precision:     {current['precision']}")
    print(f"  Conflict recall:        {current['recall']}")
    print(f"  Conflict F1:            {current['f1']}")
    print(f"  Resolution quality:     {current['resolution_quality']}")
    print(f"  Confusion (TP/FP/FN/TN):{current['confusion']}")
    print("=" * 60)

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Baseline UPDATED and locked at {BASELINE_PATH}")
        return

    if not BASELINE_PATH.exists():
        print("No baseline locked yet. Run with --update-baseline to establish one.")
        print("(Not treating this as a failure.)")
        return

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    regressed = []
    for metric in ("answer_accuracy", "precision", "recall", "f1", "resolution_quality"):
        base_val = baseline.get(metric)
        cur_val = current.get(metric)
        if base_val is None or cur_val is None:
            continue
        if cur_val < base_val - TOLERANCE:
            regressed.append((metric, base_val, cur_val))

    if regressed:
        print("\n*** BENCHMARK REGRESSION DETECTED ***")
        for metric, base_val, cur_val in regressed:
            print(f"  {metric}: baseline {base_val} -> current {cur_val}  (drop {round(base_val - cur_val, 4)})")
        print("\nA metric dropped beyond tolerance. Investigate before committing.")
        print("If this drop is intended and verified, re-lock with --update-baseline.")
        sys.exit(1)

    print("\nAll metrics within tolerance of the locked baseline. No regression.")


if __name__ == "__main__":
    main()
