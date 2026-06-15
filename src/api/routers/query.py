# Generated according to Phase 0 execution plan (Claude Code)
"""Router that fulfills the API contract defined in CLAUDE.md."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.schemas import ChatRequest, ChatResponse
from src.graphs.multi_agent_graph import build_multi_agent_app
from src.graphs.baseline_rag import build_baseline_app
from src.api.utils import experiments
from src.api.utils import retrieval_metrics
import json
import time
import asyncio

router = APIRouter(tags=["query"])

_multi_agent_app = build_multi_agent_app()
_baseline_app = build_baseline_app()

def _format_response(state: dict, request: ChatRequest) -> dict:
    # Extract fields required by the response schema
    conflict_type = None
    if state.get("has_conflict") and state.get("conflict_clusters"):
        # Simple heuristic: use first cluster's type or "other"
        for cluster in state.get("conflict_clusters", []):
            if hasattr(cluster, "description") and "numeric" in cluster.description.lower():
                conflict_type = "numeric"
                break
            elif hasattr(cluster, "type") and cluster.type:
                conflict_type = cluster.type.lower()
                break

    # Gather chosen doc IDs from the resolution node (if any)
    chosen_doc_ids: list[str] = []
    if state.get("resolved"):
        for cluster in state.get("conflict_clusters", []):
            if cluster.cluster_id in state["resolved"].resolutions:
                chosen_doc_ids.extend(
                    state["resolved"].resolutions[cluster.cluster_id].chosen_doc_ids
                )

    flagged_uncertain = bool(
        state.get("resolved")
        and state.get("conflict_clusters")
        and any(r.status == "unresolved" for r in state["resolved"].resolutions.values())
    )

    # Build the response data according to CLAUDE.md
    response_data = {
        "answer": state.get("answer", ""),
        "has_conflict": bool(state.get("has_conflict")),
        "conflict_type": conflict_type,
        "chosen_doc_ids": chosen_doc_ids,
        "flagged_uncertain": flagged_uncertain,
        "retrieved_docs": [
            {"id": d.id, "text": d.text, "metadata": d.metadata} for d in state.get("retrieved_docs", [])
        ],
        "retriever_recall": retrieval_metrics.compute_recall(
            request.query, [d["id"] for d in state.get("retrieved_docs", [])]
        ),
        "retriever_precision": retrieval_metrics.compute_precision(
            request.query, [d["id"] for d in state.get("retrieved_docs", [])]
        ),
        "faithful": state.get("faithful"),
        "faithfulness_notes": state.get("faithfulness_notes"),
        "runtime_ms": state.get("processing_time_ms", 0),
        "mode_used": request.strategy or "most_recent",
    }

    # Log the experiment
    experiments.append_run(
        {
            "timestamp": time.time(),
            "query": request.query,
            "system_type": "multi_agent",
            "strategy": request.strategy,
            "model": request.model,
            "answer": response_data["answer"],
            "has_conflict": response_data["has_conflict"],
            "conflict_type": response_data["conflict_type"],
            "chosen_doc_ids": response_data["chosen_doc_ids"],
            "flagged_uncertain": response_data["flagged_uncertain"],
            "faithful": response_data["faithful"],
            "retriever_recall": response_data["retriever_recall"],
            "retriever_precision": response_data["retriever_precision"],
            "runtime_ms": response_data["runtime_ms"],
        }
    )

    return response_data


@router.post("/multi-agent", response_model=ChatResponse)
async def multi_agent_query(request: ChatRequest):
    try:
        state = {
            "query": request.query,
            "strategy": request.strategy or "most_recent",
            "top_k": request.top_k,
        }
        result = _multi_agent_app.invoke(state)
        return ChatResponse(**_format_response(result, request))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/baseline", response_model=ChatResponse)
async def baseline_query(request: ChatRequest):
    try:
        state = {"query": request.query, "top_k": request.top_k}
        result = _baseline_app.invoke(state)
        # Baseline does not produce conflict/faithfulness fields; fill with defaults
        result["has_conflict"] = False
        result["conflict_type"] = None
        result["chosen_doc_ids"] = []
        result["flagged_uncertain"] = False
        result["faithful"] = None
        result["faithfulness_notes"] = None
        return ChatResponse(**_format_response(result, request))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_query(request: ChatRequest):
    """SSE streaming endpoint for multi‑agent queries."""
    try:
        state = {
            "query": request.query,
            "strategy": request.strategy or "most_recent",
            "top_k": request.top_k,
        }

        async def event_generator():
            # Initial status event
            yield f"data: {json.dumps({'type': 'status', 'value': 'retrieving'})}\\n\\n"
            # Run the graph synchronously (it’s CPU‑bound, not async)
            result = _multi_agent_app.invoke(state)

            # Emit intermediate status events
            yield f"data: {json.dumps({'type': 'status', 'value': 'detecting_conflicts'})}\\n\\n"
            yield f"data: {json.dumps({'type': 'status', 'value': 'resolving'})}\\n\\n"

            # Stream answer tokens (simple implementation: split on spaces)
            answer = result.get("answer", "")
            for token in answer.split():
                yield f"data: {json.dumps({'type': 'token', 'value': token})}\\n\\n"
                await asyncio.sleep(0.02)  # modest pacing for the UI

            # Final done event with full payload
            final_payload = _format_response(result, request)
            final_payload["type"] = "done"
            yield f"data: {json.dumps(final_payload)}\\n\\n"

            # Log the experiment (same payload as the non‑streaming endpoint)
            experiments.append_run(
                {
                    "timestamp": time.time(),
                    "query": request.query,
                    "system_type": "multi_agent",
                    "strategy": request.strategy,
                    "model": request.model,
                    "answer": final_payload["answer"],
                    "has_conflict": final_payload["has_conflict"],
                    "conflict_type": final_payload["conflict_type"],
                    "chosen_doc_ids": final_payload["chosen_doc_ids"],
                    "flagged_uncertain": final_payload["flagged_uncertain"],
                    "faithful": final_payload["faithful"],
                    "retriever_recall": final_payload["retriever_recall"],
                    "retriever_precision": final_payload["retriever_precision"],
                    "runtime_ms": final_payload["runtime_ms"],
                }
            )

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))