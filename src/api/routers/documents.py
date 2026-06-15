"""Document management router."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
import hashlib
from pathlib import Path

from src.api.schemas import Document, DocumentUploadResponse, DocumentListEntry

router = APIRouter(tags=["documents"])

DOCS_DIR = Path("data/raw")
DOCS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    content = await file.read()
    suffix = Path(file.filename).suffix or ".txt"
    dest = DOCS_DIR / f"{doc_id}{suffix}"
    dest.write_bytes(content)

    doc = Document(
        document_id=doc_id,
        filename=file.filename or "unknown",
        mime_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        content_hash=hashlib.sha256(content).hexdigest(),
    )
    return DocumentUploadResponse(
        document_id=doc_id,
        status="indexed",
        indexed_chunks=0,
    )


@router.get("", response_model=list[DocumentListEntry])
async def list_documents():
    entries = []
    for p in DOCS_DIR.iterdir():
        if p.is_file():
            entries.append(
                DocumentListEntry(
                    document_id=p.stem,
                    filename=p.name,
                    size_bytes=p.stat().st_size,
                    indexed_at=p.stat().st_mtime,
                )
            )
    return entries


@router.get("/{doc_id}", response_model=Document)
async def get_document(doc_id: str):
    raise HTTPException(status_code=404, detail="Not implemented")


@router.delete("/{doc_id}", response_model=DocumentUploadResponse)
async def delete_document(doc_id: str):
    raise HTTPException(status_code=404, detail="Not implemented")