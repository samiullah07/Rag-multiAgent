# src/evaluation/run_eval.py
"""
Multi-agent RAG evaluation runner.

Supports repeated runs for statistical significance:
    uv run python -m src.evaluation.run_eval --runs 3
"""
import argparse
import json
import math
import pathlib
import time
from typing import Dict, Any, List

from src.config.config import settings
from src.evaluation.metrics import (
    load_eval_questions,
    compute_answer_accuracy,
    compute_in_kb_accuracy,
    compute_appropriate_refusal_rate,
    compute_conflict_detection_metrics,
    compute_resolution_quality,
)
from src.graphs.multi_agent_graph import build_multi_agent_app


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def run_single(app, questions) -> List[Dict[str, Any]]:
    """Execute one full pass over the eval set, return per-question records."""
    records = []
    for item in questions:
        state = {"query": item.query, "strategy": "most_recent"}
        start = time.time()
        final_state = app.invoke(state)
        elapsed = time.time() - start

        answer = final_state.get("answer", "")
        resolved = final_state.get("resolved")
        chosen_ids: list[str] = []
        flagged_uncertain = False
        if resolved is not None:
            for rs in resolved.resolutions.values():
                chosen_ids.extend(rs.chosen_doc_ids)
                if rs.status == "unresolved":
                    flagged_uncertain = True
        chosen_ids = list(dict.fromkeys(chosen_ids))

        records.append({
            "id": item.id,
            "query": item.query,
            "answer": answer,
            "ground_truth": item.ground_truth,
            "has_conflict_gold": item.has_conflict,
            "has_conflict_system": final_state.get("has_conflict", False),
            "correct_doc_ids": item.correct_doc_ids or [],
            "chosen_doc_ids": chosen_ids,
            "flagged_uncertain": flagged_uncertain,
            "faithful": final_state.get("faithful"),
            "faithfulness_notes": final_state.get("faithfulness_notes", ""),
            "fabricated_citations": final_state.get("fabricated_citations", []),
            "runtime": elapsed,
        })
        time.sleep(12)

    return records


def compute_all_metrics(records: List[Dict[str, Any]], questions) -> Dict[str, float]:
    """Compute all headline metrics from a single run's records."""
    accuracy = compute_answer_accuracy(records)
    in_kb_acc = compute_in_kb_accuracy(records, questions)
    refusal = compute_appropriate_refusal_rate(records, questions)
    conflict = compute_conflict_detection_metrics(records, questions)
    resolution = compute_resolution_quality(records, questions)

    return {
        "answer_accuracy": accuracy,
        "in_kb_accuracy": in_kb_acc,
        "appropriate_refusal_rate": refusal,
        "precision": conflict["precision"],
        "recall": conflict["recall"],
        "f1": conflict["f1"],
        "resolution_quality": resolution,
        "tp": conflict["tp"],
        "fp": conflict["fp"],
        "fn": conflict["fn"],
        "tn": conflict["tn"],
    }


def main():
    ap = argparse.ArgumentParser(description="Run the multi-agent RAG evaluation.")
    ap.add_argument("--runs", type=int, default=1,
                    help="Number of full evaluation passes (default: 1). "
                         "Multiple runs report mean ± std for all metrics.")
    args = ap.parse_args()
    n_runs = max(1, args.runs)

    questions = load_eval_questions(settings.eval_questions_path)
    app = build_multi_agent_app()

    out_dir = pathlib.Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, float]] = []
    all_records: List[List[Dict[str, Any]]] = []

    for run_idx in range(n_runs):
        if n_runs > 1:
            print(f"\n{'='*60}")
            print(f"  RUN {run_idx + 1} / {n_runs}")
            print(f"{'='*60}")

        records = run_single(app, questions)
        metrics = compute_all_metrics(records, questions)
        all_metrics.append(metrics)
        all_records.append(records)

        # Save each run's records
        suffix = f"_run{run_idx + 1}" if n_runs > 1 else ""
        (out_dir / f"multi_agent_records{suffix}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"\n  Run {run_idx + 1} results:")
        print(f"    Answer accuracy:     {metrics['answer_accuracy']:.4f}")
        print(f"    In-KB accuracy:      {metrics['in_kb_accuracy']:.4f}")
        print(f"    Appropriate refusal: {metrics['appropriate_refusal_rate']:.4f}")
        print(f"    Precision:           {metrics['precision']:.4f}")
        print(f"    Recall:              {metrics['recall']:.4f}")
        print(f"    F1:                  {metrics['f1']:.4f}")
        print(f"    Resolution quality:  {metrics['resolution_quality']:.4f}")
        print(f"    Confusion: TP={metrics['tp']} FP={metrics['fp']} "
              f"FN={metrics['fn']} TN={metrics['tn']}")

    # Summary
    if n_runs > 1:
        print(f"\n{'='*60}")
        print(f"  AGGREGATE ({n_runs} runs): mean ± std")
        print(f"{'='*60}")

        metric_keys = ["answer_accuracy", "in_kb_accuracy", "appropriate_refusal_rate",
                       "precision", "recall", "f1", "resolution_quality"]
        summary = {}
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            mean = _mean(values)
            std = _std(values)
            summary[key] = {"mean": round(mean, 4), "std": round(std, 4),
                            "values": [round(v, 4) for v in values]}
            print(f"    {key:28s} {mean:.4f} ± {std:.4f}  {[round(v,4) for v in values]}")

        # Confusion matrix aggregates
        for cm_key in ["tp", "fp", "fn", "tn"]:
            values = [m[cm_key] for m in all_metrics]
            mean = _mean(values)
            std = _std(values)
            summary[cm_key] = {"mean": round(mean, 2), "std": round(std, 2),
                               "values": values}
            print(f"    {cm_key:28s} {mean:.2f} ± {std:.2f}  {values}")

        (out_dir / "multi_run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  Per-run records saved as multi_agent_records_run{{1..{n_runs}}}.json")
        print(f"  Summary saved to data/eval/multi_run_summary.json")

    # Also always save a "latest" copy (for check_benchmark compatibility)
    (out_dir / "multi_agent_records.json").write_text(
        json.dumps(all_records[-1], ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
