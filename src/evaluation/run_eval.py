# src/evaluation/run_eval.py
import time
from typing import Dict, Any

from src.config.config import settings
from src.evaluation.metrics import load_eval_questions, compute_answer_accuracy
from src.graphs.multi_agent_graph import build_multi_agent_app
from src.graphs.baseline_rag import build_baseline_app


def run_system(app, label: str) -> Dict[str, Any]:
    questions = load_eval_questions(settings.eval_questions_path)
    records = []
    total_time = 0.0

    for item in questions:
        state = {"query": item.query}
        start = time.time()
        final_state = app.invoke(state)
        elapsed = time.time() - start
        total_time += elapsed

        answer = final_state.get("answer", "")
        records.append(
            {
                "id": item.id,
                "query": item.query,
                "answer": answer,
                "ground_truth": item.ground_truth,
                "has_conflict_gold": item.has_conflict,
                    "has_conflict_system": final_state.get("has_conflict", False),
                "runtime": elapsed,
            }
        )

    accuracy = compute_answer_accuracy(records)
    avg_time = total_time / len(records) if records else 0.0

    return {
        "label": label,
        "accuracy": accuracy,
        "avg_response_time": avg_time,
        "records": records,
    }


def main():
    multi_agent_app = build_multi_agent_app()
    baseline_app = build_baseline_app()

    multi_results = run_system(multi_agent_app, label="multi_agent")
    base_results = run_system(baseline_app, label="baseline")

    print("Multi-agent accuracy:", multi_results["accuracy"])
    print("Baseline accuracy:", base_results["accuracy"])
    print("Multi-agent avg time:", multi_results["avg_response_time"])
    print("Baseline avg time:", base_results["avg_response_time"])


if __name__ == "__main__":
    main()

# Example (to be implemented later once records include chosen doc IDs and uncertainty flags):
# from src.evaluation.metrics import (
#     compute_conflict_detection_rate,
#     compute_resolution_quality,
#     compute_uncertain_flag_rate,
# )
#
# multi_conflict_detection = compute_conflict_detection_rate(
#     multi_results["records"], load_eval_questions(settings.eval_questions_path)
# )
# print("Multi-agent conflict detection rate:", multi_conflict_detection)
