# src/graphs/baseline_rag.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from src.agents.retriever import retriever_agent
from src.agents.answer_generator import answer_generation_agent


class BaselineState(Dict[str, Any]):
    """
    Keys:
      - query: str
      - retrieved_docs: List[RetrievedDoc]
      - answer: str
    """


def retriever_node(state: BaselineState) -> BaselineState:
    docs = retriever_agent(state["query"])
    state["retrieved_docs"] = docs
    return state


def answer_node(state: BaselineState) -> BaselineState:
    answer = answer_generation_agent(state["query"], state["retrieved_docs"], resolved=None)
    state["answer"] = answer
    return state


def build_baseline_app():
    graph = StateGraph(BaselineState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", END)
    return graph.compile()