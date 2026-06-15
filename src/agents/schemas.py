# src/agents/schemas.py
from typing import List, Dict, Any, Literal
from pydantic import BaseModel


class RetrievedDoc(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float | None = None


class ConflictCluster(BaseModel):
    cluster_id: str
    description: str
    doc_ids: List[str]


class ResolvedStatus(BaseModel):
    status: Literal["resolved", "unresolved"]
    chosen_doc_ids: List[str]
    rationale: str


class ResolvedEvidence(BaseModel):
    query: str
    has_conflict: bool
    conflict_clusters: List[ConflictCluster]
    resolutions: Dict[str, ResolvedStatus]  # keyed by cluster_id
    docs: List[RetrievedDoc]