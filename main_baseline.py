# main_baseline.py
import argparse

from src.graphs.baseline_rag import build_baseline_app


def main():
    parser = argparse.ArgumentParser(
        description="Baseline single-pass RAG (no explicit conflict detection)."
    )
    parser.add_argument("--query", type=str, required=True, help="User question.")
    args = parser.parse_args()

    app = build_baseline_app()
    state = {"query": args.query}
    final_state = app.invoke(state)

    print("QUESTION:", args.query)
    print("ANSWER:\n", final_state.get("answer", ""))


if __name__ == "__main__":
    main()