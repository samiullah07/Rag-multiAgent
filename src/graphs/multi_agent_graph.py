# src/graphs/multi_agent_graph.py
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END

from src.agents.retriever import retriever_agent
from src.agents.contradiction_detector import contradiction_detection_agent
from src.agents.resolution import resolution_agent
from src.agents.answer_generator import answer_generation_agent
from src.agents.grounding_verifier import verify_grounding
from src.agents.schemas import RetrievedDoc, ResolvedEvidence, ConflictCluster


# Treat state as a plain dict; do NOT enforce a strict state type at graph level.
GraphState = Dict[str, Any]


def retriever_node(state: GraphState) -> GraphState:
    # Trace
    state.setdefault('trace', []).append({'agent': 'retriever', 'summary': 'Started retrieval'})
    # We expect 'query' to be present in the input dict from app.py
    query: str = state["query"]
    top_k: int = int(state.get("top_k", 5))
    docs = retriever_agent(query, top_k=top_k)

    # If this query comes from a session with an active document upload,
    # bias retrieval toward that upload — merged FIRST, ahead of the
    # permanent knowledge base. upload_session_id is None/absent in every
    # normal case, including every run_eval.py invocation (it never sets
    # this key), so behavior there is byte-for-byte unchanged from before.
    # NOTE: this is done here rather than by changing retriever_agent()'s
    # signature, since that function is also called directly by the CLI
    # scripts (main_multi_agent.py / main_baseline.py) and must keep its
    # existing simple contract.
    upload_session_id = state.get("upload_session_id")
    if upload_session_id:
        from src.retrieval.vector_store import retrieve_from_upload_session

        upload_lc_docs = retrieve_from_upload_session(query, upload_session_id, top_k=3)
        upload_docs = [
            RetrievedDoc(
                id=d.metadata.get("id") or f"upload_{upload_session_id[:8]}_{i}",
                text=d.page_content,
                metadata=d.metadata,
                score=d.metadata.get("score"),
            )
            for i, d in enumerate(upload_lc_docs)
        ]

        seen_ids = set()
        merged: List[RetrievedDoc] = []
        for d in upload_docs + docs:
            if d.id not in seen_ids:
                seen_ids.add(d.id)
                merged.append(d)
        docs = merged[:top_k]

    state["retrieved_docs"] = docs
    return state


def contradiction_detection_node(state: GraphState) -> GraphState:
    # Trace
    state.setdefault('trace', []).append({'agent': 'contradiction_detection', 'summary': 'Detected conflicts'})
    docs: List[RetrievedDoc] = state["retrieved_docs"]
    has_conflict, clusters = contradiction_detection_agent(state["query"], docs)
    state["has_conflict"] = has_conflict
    state["conflict_clusters"] = clusters
    return state


def resolution_node(state: GraphState) -> GraphState:
    # Trace
    state.setdefault('trace', []).append({'agent': 'resolution', 'summary': 'Resolved conflicts'})
    docs: List[RetrievedDoc] = state["retrieved_docs"]
    clusters: List[ConflictCluster] = state.get("conflict_clusters", [])
    strategy: str = state.get("strategy", "explain_both")
    resolved: ResolvedEvidence = resolution_agent(
        state["query"],
        docs,
        clusters,
        strategy=strategy,
    )
    state["resolved"] = resolved
    return state


def answer_generation_node(state: GraphState) -> GraphState:
    # Trace
    state.setdefault('trace', []).append({'agent': 'answer_generation', 'summary': 'Generated answer from retrieved evidence'})
    docs: list[RetrievedDoc] = state["retrieved_docs"]
    resolved: ResolvedEvidence | None = state.get("resolved")
    model: str = state.get("model", "llama3-70b-8192")
    answer = answer_generation_agent(
        state["query"],
        docs,
        resolved=resolved,
        # model=model,  # <- remove this
    )
    state["answer"] = answer
    return state


def grounding_verification_node(state: GraphState) -> GraphState:
    """
    Verifies the generated answer against the documents that were actually
    retrieved — catches fabricated citations (e.g. a fake "Doc id=astronomy_01"
    that was never retrieved) and, more generally, ungrounded claims that rely
    on outside knowledge rather than the retrieved context. See
    src/agents/grounding_verifier.py for the motivation and implementation.
    """
    # Trace
    state.setdefault('trace', []).append({'agent': 'grounding_verification', 'summary': 'Verified answer is grounded in retrieved documents'})
    docs: List[RetrievedDoc] = state.get("retrieved_docs", [])
    answer: str = state.get("answer", "")
    result = verify_grounding(state["query"], answer, docs)
    state["faithful"] = result["faithful"]
    state["faithfulness_notes"] = result["faithfulness_notes"]
    state["fabricated_citations"] = result["fabricated_citations"]
    return state


def should_resolve(state: GraphState) -> str:
    if state.get("has_conflict"):
        return "resolution"
    return "answer"


def build_multi_agent_app(
    enable_contradiction_detection: bool = True,
    enable_resolution: bool = True,
    enable_grounding_verification: bool = True,
):
    """
    Build the multi-agent RAG graph with optional ablation flags.

    When a flag is False the corresponding node is bypassed — the graph routes
    around it. Disabling contradiction detection also forces resolution off
    (you can't resolve conflicts you never detected).
    Default (all True) produces byte-for-byte identical behavior to before.
    """
    if not enable_contradiction_detection:
        enable_resolution = False

    graph = StateGraph(dict)

    graph.add_node("retriever", retriever_node)
    graph.add_node("answer_generation", answer_generation_node)

    graph.set_entry_point("retriever")

    if enable_contradiction_detection:
        graph.add_node("contradiction_detection", contradiction_detection_node)
        graph.add_edge("retriever", "contradiction_detection")

        if enable_resolution:
            graph.add_node("resolution", resolution_node)
            graph.add_conditional_edges(
                "contradiction_detection",
                should_resolve,
                {
                    "resolution": "resolution",
                    "answer": "answer_generation",
                },
            )
            graph.add_edge("resolution", "answer_generation")
        else:
            graph.add_edge("contradiction_detection", "answer_generation")
    else:
        graph.add_edge("retriever", "answer_generation")

    if enable_grounding_verification:
        graph.add_node("grounding_verification", grounding_verification_node)
        graph.add_edge("answer_generation", "grounding_verification")
        graph.add_edge("grounding_verification", END)
    else:
        graph.add_edge("answer_generation", END)

    return graph.compile()
