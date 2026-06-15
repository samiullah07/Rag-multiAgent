# src/graphs/multi_agent_graph.py
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from src.agents.retriever import retriever_agent
from src.agents.contradiction_detector import contradiction_detection_agent
from src.agents.resolution import resolution_agent
from src.agents.answer_generator import answer_generation_agent
from src.agents.schemas import RetrievedDoc, ResolvedEvidence, ConflictCluster


# Graph state type
class GraphState(Dict[str, Any]):
    """
    Keys:
      - query: str
      - retrieved_docs: List[RetrievedDoc]
      - has_conflict: bool
      - conflict_clusters: List[ConflictCluster]
      - resolved: ResolvedEvidence | None
      - answer: str | None
    """


def retriever_node(state: GraphState) -> GraphState:
    docs = retriever_agent(state["query"])
    state["retrieved_docs"] = docs
    return state


def contradiction_detection_node(state: GraphState) -> GraphState:
    docs = state["retrieved_docs"]
    has_conflict, clusters = contradiction_detection_agent(state["query"], docs)
    state["has_conflict"] = has_conflict
    state["conflict_clusters"] = clusters
    return state


def resolution_node(state: GraphState) -> GraphState:
    docs: list[RetrievedDoc] = state["retrieved_docs"]
    clusters: list[ConflictCluster] = state.get("conflict_clusters", [])
    resolved: ResolvedEvidence = resolution_agent(state["query"], docs, clusters)
    state["resolved"] = resolved
    return state


def answer_generation_node(state: GraphState) -> GraphState:
    docs: list[RetrievedDoc] = state["retrieved_docs"]
    resolved: ResolvedEvidence | None = state.get("resolved")
    answer = answer_generation_agent(state["query"], docs, resolved=resolved)
    state["answer"] = answer
    return state


def should_resolve(state: GraphState) -> str:
    """
    Conditional edge: if has_conflict -> go to resolution, else directly to answer.
    """
    if state.get("has_conflict"):
        return "resolution"
    return "answer"


def build_multi_agent_app():
    graph = StateGraph(GraphState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("contradiction_detection", contradiction_detection_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("answer_generation", answer_generation_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "contradiction_detection")
    graph.add_conditional_edges(
      "contradiction_detection", should_resolve,
      {
        "resolution": "resolution",
        "answer": "answer_generation"
      }
    )
    graph.add_edge("resolution", "answer_generation")
    graph.add_edge("answer_generation", END)

    return graph.compile()