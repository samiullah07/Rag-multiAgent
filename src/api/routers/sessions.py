"""Session management router."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime

from src.api.schemas import SessionSummary, SessionDetail
from src.session.session_manager import SessionManager

router = APIRouter(tags=["sessions"])
session_manager = SessionManager()


@router.post("", response_model=SessionSummary)
async def create_session():
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    session_manager.create_session()
    return SessionSummary(session_id=session_id, created_at=now, last_active=now, document_count=0)


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    # Placeholder
    raise HTTPException(status_code=404, detail="Session retrieval not yet implemented")


@router.delete("/{session_id}", response_model=SessionSummary)
async def delete_session(session_id: str):
    # Placeholder
    raise HTTPException(status_code=404, detail="Session deletion not yet implemented")