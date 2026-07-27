# src/evaluation/run_ablation.py
"""
Ablation study runner.

Evaluates five configurations on the full eval set (one run each) and
produces a comparison table for the dissertation results chapter.

Usage:
    uv run python -m src.evaluation.run_ablation
"""
import json
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
from src.graphs.baseline_rag import build_baseline_app

OUT_DIR = pathlib.Path("data/eval")

CONFIGS = [
    {
        "name": "full",
        "description": "All nodes enabled (default system)",
        "build_kwargs": {},
    },
    {
        "name": "no_detection",
        "description": "Contradiction detection + resolution disabled",
        "build_kwargs": {
            "enable_contradiction_detection": False,
        },
    },
    {
        "name": "no_resolution",
        "description": "Detection on, resolution disabled",
        "build_kwargs": {
            "enable_resolution": False,
        },
    },
    {
        "name": "no_verifier",
        "description": "Grounding verification disabled",
        "build_kwargs": {
            "enable_grounding_verification": False,
        },
    },
    {
        "name": "baseline",
        "description": "2-stage baseline (retrieve → generate only)",
        "build_kwargs": None,  # sentinel: use baseline graph
    },
]


def run_config(app, questions, is_baseline: bool = False) -> List[Dict[str, Any]]:
    """Execute one full pass over the eval set for a given config."""
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


def compute_all_metrics(records: List[Dict[str, Any]], questions) -> Dict[str, Any]:
    accuracy = compute_answer_accuracy(records)
    in_kb_acc = compute_in_kb_accuracy(records, questions)
    refusal = compute_appropriate_refusal_rate(records, questions)
    conflict = compute_conflict_detection_metrics(records, questions)
    resolution = compute_resolution_quality(records, questions)

    total_fabricated = sum(
        len(r.get("fabricated_citations", [])) for r in records
    )
    mean_latency = (
        sum(r["runtime"] for r in records) / len(records) if records else 0.0
    )

    return {
        "answer_accuracy": round(accuracy, 4),
        "in_kb_accuracy": round(in_kb_acc, 4),
        "appropriate_refusal_rate": round(refusal, 4),
        "precision": round(conflict["precision"], 4),
        "recall": round(conflict["recall"], 4),
        "f1": round(conflict["f1"], 4),
        "resolution_quality": round(resolution, 4),
        "fabricated_citation_count": total_fabricated,
        "mean_latency_s": round(mean_latency, 2),
        "tp": conflict["tp"],
        "fp": conflict["fp"],
        "fn": conflict["fn"],
        "tn": conflict["tn"],
    }


def print_table(results: List[Dict[str, Any]]):
    """Print a plain-text comparison table."""
    cols = [
        ("config", 15),
        ("accuracy", 8),
        ("in_kb_acc", 9),
        ("refusal", 7),
        ("prec", 6),
        ("recall", 6),
        ("F1", 6),
        ("res_qual", 8),
        ("fab_cit", 7),
        ("lat(s)", 6),
    ]

    header = "| " + " | ".join(f"{name:>{width}}" for name, width in cols) + " |"
    sep = "|-" + "-|-".join("-" * width for _, width in cols) + "-|"
    print(sep)
    print(header)
    print(sep)

    for r in results:
        m = r["metrics"]
        row_vals = [
            f"{r['name']:>15}",
            f"{m['answer_accuracy']:>8.4f}",
            f"{m['in_kb_accuracy']:>9.4f}",
            f"{m['appropriate_refusal_rate']:>7.4f}",
            f"{m['precision']:>6.4f}",
            f"{m['recall']:>6.4f}",
            f"{m['f1']:>6.4f}",
            f"{m['resolution_quality']:>8.4f}",
            f"{m['fabricated_citation_count']:>7d}",
            f"{m['mean_latency_s']:>6.1f}",
        ]
        print("| " + " | ".join(row_vals) + " |")

    print(sep)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_eval_questions(settings.eval_questions_path)

    all_results = []

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"  CONFIG: {name} — {cfg['description']}")
        print(f"{'='*60}")

        if cfg["build_kwargs"] is None:
            app = build_baseline_app()
        else:
            app = build_multi_agent_app(**cfg["build_kwargs"])

        records = run_config(app, questions, is_baseline=(name == "baseline"))
        metrics = compute_all_metrics(records, questions)

        # Save per-config records
        (OUT_DIR / f"ablation_{name}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        all_results.append({
            "name": name,
            "description": cfg["description"],
            "metrics": metrics,
        })

        print(f"  Answer accuracy:           {metrics['answer_accuracy']}")
        print(f"  In-KB accuracy:            {metrics['in_kb_accuracy']}")
        print(f"  Appropriate refusal:       {metrics['appropriate_refusal_rate']}")
        print(f"  Precision:                 {metrics['precision']}")
        print(f"  Recall:                    {metrics['recall']}")
        print(f"  F1:                        {metrics['f1']}")
        print(f"  Resolution quality:        {metrics['resolution_quality']}")
        print(f"  Fabricated citations:      {metrics['fabricated_citation_count']}")
        print(f"  Mean latency:              {metrics['mean_latency_s']}s")

    # Save combined summary
    (OUT_DIR / "ablation_summary.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print comparison table
    print(f"\n\n{'='*60}")
    print("  ABLATION STUDY — COMPARISON TABLE")
    print(f"{'='*60}\n")
    print_table(all_results)
    print(f"\n  Results saved to data/eval/ablation_summary.json")
    print(f"  Per-config records: data/eval/ablation_<config>.json")


if __name__ == "__main__":
    main()
