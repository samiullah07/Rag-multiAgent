# main_multi_agent.py
import argparse

from src.graphs.multi_agent_graph import build_multi_agent_app


def main():
    parser = argparse.ArgumentParser(
        description="Self-correcting multi-agent RAG for contradictory knowledge bases."
    )
    parser.add_argument("--query", type=str, required=True, help="User question.")
    args = parser.parse_args()

    app = build_multi_agent_app()
    state = {"query": args.query}
    final_state = app.invoke(state)

    print("QUESTION:", args.query)
    print("ANSWER:\n", final_state.get("answer", ""))


if __name__ == "__main__":
    main()