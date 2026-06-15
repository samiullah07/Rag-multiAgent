# Generated according to Phase 0 execution plan (Claude Code)
"""Pydantic models for the API layer (extended per CLAUDE.md)."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import uuid4

# ---------- Request / Response Models ----------
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = 5
    mode: str = "multi_agent"  # "multi_agent" | "baseline"
    # New fields per spec
    strategy: Optional[str] = None  # most_recent | most_authoritative | explain_both
    model: Optional[str] = None  # e.g., "llama3-70b-8192"

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    message_id: str
    sources: List[dict] = []
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0
    flagged: bool = False
    # Extended fields per spec
    has_conflict: bool = False
    conflict_type: Optional[str] = None
    chosen_doc_ids: List[str] = []
    flagged_uncertain: bool = False
    retrieved_docs: List[dict] = []
    retriever_recall: Optional[float] = None
    retriever_precision: Optional[float] = None
    faithful: Optional[bool] = None
    faithfulness_notes: Optional[str] = None
    runtime_ms: int = 0
    mode_used: str = "multi_agent"

class Document(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    mime_type: str
    size_bytes: int
    content_hash: str
    metadata: dict = Field(default={})

class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    indexed_chunks: int

class DocumentListEntry(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    indexed_at: datetime

class DocumentDeleteResponse(BaseModel):
    status: str

# ---------- Evaluation Metrics ----------
class MetricResult(BaseModel):
    metric: str
    value: float
    unit: str
    reference: Optional[str] = None

class EvalResult(BaseModel):
    metrics: List[MetricResult]
    overall_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ---------- Session Models ----------
class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime
    last_active: datetime
    document_count: int = 0

class SessionDetail(SessionSummary):
    messages: List[dict]  # simplified representation

# ---------- Comparison / Verification ----------
class ComparisonResult(BaseModel):
    baseline: dict
    multi_agent: dict
    differences: List[dict]
    recommendation: Optional[str] = None
