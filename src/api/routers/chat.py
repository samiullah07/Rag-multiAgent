"""Chat API router – thin shim forwarding to the new query router.

Provides backward-compatible ``/chat`` and ``/chat/stream`` endpoints that
delegate to the implementations in ``src.api.routers.query``.
"""
# Generated according to Phase 0 execution plan (Claude Code)

from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatRequest, ChatResponse
from src.api.routers import query as query_router

router = APIRouter(tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Compatibility endpoint that forwards to the multi-agent query."""
    try:
        return await query_router.multi_agent_query(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """Compatibility endpoint that forwards to the streaming multi-agent query."""
    try:
        return await query_router.stream_query(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))