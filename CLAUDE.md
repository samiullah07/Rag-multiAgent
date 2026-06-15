# CLAUDE.md — MSc-Level RAG System: Complete Implementation Guide

> **Project**: Self-Correcting Multi-Agent RAG with Contradiction Detection
> **Stack**: Python 3.13, LangGraph, ChromaDB, Groq (Llama-3.3-70B), FastAPI, React + Vite + TypeScript
> **Current State**: Backend-only CLI system — no UI, no API, no advanced retrieval
> **Goal**: MSc-grade research system with pixel-perfect professional web UI

---

## 1. CURRENT PROJECT ARCHITECTURE (What Already Exists)

```
RAG-gurkirat/
├── src/
│   ├── agents/
│   │   ├── answer_generator.py      # LLM-based answer synthesis
│   │   ├── contradiction_detector.py # Rule-based + LLM conflict detection
│   │   ├── resolution.py            # Conflict resolution with extra retrieval
│   │   ├── retriever.py             # Semantic retrieval wrapper
│   │   └── schemas.py               # Pydantic models (RetrievedDoc, ConflictCluster, etc.)
│   ├── graphs/
│   │   ├── baseline_rag.py          # Simple retrieve→answer graph
│   │   └── multi_agent_graph.py     # retrieve→detect→resolve→answer graph
│   ├── retrieval/
│   │   └── vector_store.py          # ChromaDB vector store
│   ├── models/
│   │   ├── embeddings.py            # HuggingFace BAAI/bge-small-en-v1.5
│   │   └── llm.py                   # Groq ChatGroq client
│   ├── data_prep/
│   │   └── build_kb.py              # .txt + JSONL knowledge base loader
│   ├── evaluation/
│   │   ├── metrics.py               # Accuracy, conflict detection rate, resolution quality
│   │   └── run_eval.py              # Evaluation runner (multi-agent vs baseline)
│   └── config/
│       └── config.py                # Pydantic Settings (env-based config)
├── main_multi_agent.py              # CLI entry point (--query flag)
├── main_baseline.py                 # CLI entry point for baseline
└── pyproject.toml                   # uv-managed dependencies
```

**What works**: CLI queries, vector store, contradiction detection, conflict resolution, basic eval.
**Critical gaps**: No REST API, no web UI, no streaming, no document upload, no advanced retrieval, no session management, no caching, no logging infrastructure.

---

## 2. MSc-LEVEL FEATURES TO ADD (Priority Order)

### 2.1 FastAPI REST API Layer (HIGHEST PRIORITY)

Create `src/api/` with full async FastAPI application.

**File: `src/api/main.py`**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.api.routers import chat, documents, eval, health, sessions

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up models, verify vector store
    yield
    # Shutdown: cleanup

app = FastAPI(
    title="Multi-Agent RAG API",
    description="Self-correcting RAG with contradiction detection",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(eval.router, prefix="/api/v1")
```

**Endpoints to implement:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check + model status |
| POST | `/api/v1/chat` | Single-turn RAG query |
| POST | `/api/v1/chat/stream` | Streaming RAG query (SSE) |
| POST | `/api/v1/sessions` | Create new chat session |
| GET | `/api/v1/sessions/{id}` | Get session history |
| DELETE | `/api/v1/sessions/{id}` | Delete session |
| POST | `/api/v1/documents/upload` | Upload documents to KB |
| GET | `/api/v1/documents` | List all documents in KB |
| DELETE | `/api/v1/documents/{id}` | Remove document from KB |
| GET | `/api/v1/documents/{id}` | Get document details |
| POST | `/api/v1/eval/run` | Run evaluation benchmark |
| GET | `/api/v1/eval/results` | Get latest eval results |
| GET | `/api/v1/kb/stats` | Vector store statistics |

**File: `src/api/schemas.py`** (API request/response models)
```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: str = "multi_agent"  # "multi_agent" | "baseline"
    top_k: int = 5
    include_sources: bool = True
    include_conflict_info: bool = True

class SourceDoc(BaseModel):
    id: str
    text: str
    source: str
    score: float
    metadata: dict

class ConflictInfo(BaseModel):
    has_conflict: bool
    clusters: List[dict]
    resolutions: dict

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    message_id: str
    sources: List[SourceDoc]
    conflict_info: Optional[ConflictInfo]
    confidence: float              # 0.0–1.0
    processing_time_ms: float
    mode_used: str
    timestamp: datetime

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    status: str

class SessionMessage(BaseModel):
    role: str                      # "user" | "assistant"
    content: str
    timestamp: datetime
    metadata: Optional[dict]

class Session(BaseModel):
    id: str
    created_at: datetime
    messages: List[SessionMessage]
    document_count: int
```

---

### 2.2 Streaming Responses with Server-Sent Events

**File: `src/api/routers/chat.py`** (streaming endpoint)
```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio, json

router = APIRouter(tags=["chat"])

@router.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    async def event_generator():
        # Yield SSE events:
        # 1. status: "retrieving"
        # 2. status: "detecting_conflicts"  
        # 3. status: "resolving"
        # 4. tokens: streamed answer tokens
        # 5. sources: final source list
        # 6. metadata: conflict info, confidence
        
        yield f"data: {json.dumps({'type': 'status', 'value': 'retrieving'})}\n\n"
        docs = await asyncio.to_thread(retriever_agent, request.query)
        
        yield f"data: {json.dumps({'type': 'status', 'value': 'detecting_conflicts'})}\n\n"
        has_conflict, clusters = await asyncio.to_thread(...)
        
        # Stream LLM tokens
        async for token in llm.astream(prompt):
            yield f"data: {json.dumps({'type': 'token', 'value': token.content})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done', 'sources': [...]})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

### 2.3 Hybrid Retrieval (BM25 + Dense Semantic + Reranking)

**File: `src/retrieval/hybrid_retriever.py`**
```python
"""
Hybrid retrieval: BM25 sparse + dense semantic + cross-encoder reranking.
MSc requirement: demonstrates retrieval research depth.
"""
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.retrieval.vector_store import get_vector_store

class HybridRetriever:
    def __init__(self, alpha: float = 0.6):
        """alpha = weight of dense retrieval (1-alpha = BM25 weight)"""
        self.alpha = alpha
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def retrieve(self, query: str, top_k: int = 10, rerank_top: int = 5):
        # 1. Dense retrieval from ChromaDB
        dense_docs = self._dense_retrieve(query, top_k)
        
        # 2. BM25 sparse retrieval from in-memory corpus
        bm25_docs = self._bm25_retrieve(query, top_k)
        
        # 3. Reciprocal Rank Fusion (RRF) to merge results
        merged = self._rrf_merge(dense_docs, bm25_docs)
        
        # 4. Cross-encoder reranking of top candidates
        reranked = self._cross_encoder_rerank(query, merged[:20], rerank_top)
        
        return reranked
    
    def _rrf_merge(self, dense, sparse, k=60):
        scores = {}
        for rank, doc in enumerate(dense):
            scores[doc.id] = scores.get(doc.id, 0) + self.alpha / (k + rank + 1)
        for rank, doc in enumerate(sparse):
            scores[doc.id] = scores.get(doc.id, 0) + (1 - self.alpha) / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def _cross_encoder_rerank(self, query, candidates, top_k):
        pairs = [(query, doc.text) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
```

**Add to `pyproject.toml`:**
```toml
"rank-bm25>=0.2.2",
"sentence-transformers>=5.5.1",   # already present
```

---

### 2.4 Advanced Query Intelligence

**File: `src/agents/query_processor.py`**
```python
"""
Three advanced query techniques:
1. HyDE (Hypothetical Document Embeddings) - generate hypothetical answer, embed it
2. Query Decomposition - break complex queries into sub-questions
3. Query Expansion - add synonyms and related terms
"""
from src.models.llm import get_chat_llm
from src.models.embeddings import get_embeddings

class QueryProcessor:
    
    def hyde_query(self, original_query: str) -> str:
        """
        Generate a hypothetical ideal document that would answer this query,
        then embed THAT to retrieve real documents.
        Source: Gao et al. 2022 "Precise Zero-Shot Dense Retrieval without Relevance Labels"
        """
        llm = get_chat_llm()
        prompt = f"""Write a short, factual paragraph that would perfectly answer this question.
        Question: {original_query}
        Hypothetical answer paragraph:"""
        hypothetical_doc = llm.invoke(prompt).content
        return hypothetical_doc
    
    def decompose_query(self, query: str) -> list[str]:
        """
        Break a complex query into atomic sub-questions.
        Run sub-questions in parallel, merge results.
        """
        llm = get_chat_llm()
        prompt = f"""Break this complex question into 2-4 simpler sub-questions.
        Return ONLY a JSON array of strings.
        Question: {query}
        Sub-questions:"""
        resp = llm.invoke(prompt)
        import json
        return json.loads(resp.content)
    
    def expand_query(self, query: str) -> list[str]:
        """Generate semantic variations of the query for multi-query retrieval."""
        llm = get_chat_llm()
        prompt = f"""Generate 3 alternative phrasings of this question for search.
        Return ONLY a JSON array of strings.
        Original: {query}"""
        resp = llm.invoke(prompt)
        import json
        variants = json.loads(resp.content)
        return [query] + variants  # include original
```

---

### 2.5 Multi-Format Document Processing

**File: `src/data_prep/document_loader.py`**
```python
"""
Supports: PDF, DOCX, CSV, JSON, JSONL, TXT, URLs, YouTube transcripts.
Add dependencies: pypdf, python-docx, unstructured
"""
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    WebBaseLoader,
    UnstructuredURLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, SemanticChunker

class DocumentProcessor:
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
    
    def load_file(self, file_path: Path) -> list:
        ext = file_path.suffix.lower()
        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
            ".txt": lambda p: [type("Doc", (), {"page_content": p.read_text()})()],
        }
        loader_cls = loaders.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported format: {ext}")
        docs = loader_cls(str(file_path)).load()
        return self.text_splitter.split_documents(docs)
    
    def load_url(self, url: str) -> list:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return self.text_splitter.split_documents(docs)
    
    def load_bytes(self, content: bytes, filename: str, mime_type: str) -> list:
        """For API file uploads - write temp file and load."""
        import tempfile
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            return self.load_file(Path(f.name))
```

**Add to `pyproject.toml`:**
```toml
"pypdf>=5.0.0",
"python-docx>=1.1.0",
"unstructured>=0.16.0",
"beautifulsoup4>=4.12.0",
```

---

### 2.6 Hallucination Detection Agent

**File: `src/agents/hallucination_detector.py`**
```python
"""
Faithfulness checking: verify every claim in the answer is grounded in the retrieved docs.
MSc contribution: adds a self-correction feedback loop.
"""
from src.models.llm import get_chat_llm
from src.agents.schemas import RetrievedDoc
from typing import List
import json

class HallucinationDetector:
    
    def check_faithfulness(
        self, 
        query: str, 
        answer: str, 
        docs: List[RetrievedDoc]
    ) -> dict:
        """
        Returns: {
            "is_faithful": bool,
            "faithfulness_score": float,   # 0.0 to 1.0
            "unfaithful_claims": list[str],
            "grounded_claims": list[str]
        }
        """
        context = "\n\n".join([f"[{d.id}]: {d.text}" for d in docs])
        llm = get_chat_llm()
        
        prompt = f"""You are verifying whether an answer is faithful to the retrieved context.
        
Question: {query}
Context: {context}
Answer to verify: {answer}

For each claim in the answer, check if it is supported by the context.
Return ONLY valid JSON:
{{
  "is_faithful": true/false,
  "faithfulness_score": 0.0-1.0,
  "unfaithful_claims": ["claim1", ...],
  "grounded_claims": ["claim1", ...]
}}"""
        
        resp = llm.invoke(prompt)
        try:
            return json.loads(resp.content)
        except:
            return {"is_faithful": True, "faithfulness_score": 1.0, 
                   "unfaithful_claims": [], "grounded_claims": []}
    
    def regenerate_if_unfaithful(
        self, query, answer, docs, max_attempts=2
    ) -> tuple[str, float]:
        """Self-correction loop: regenerate answer if unfaithful."""
        for attempt in range(max_attempts):
            result = self.check_faithfulness(query, answer, docs)
            if result["faithfulness_score"] >= 0.85:
                return answer, result["faithfulness_score"]
            
            # Regenerate with explicit grounding constraint
            context = "\n\n".join([d.text for d in docs])
            llm = get_chat_llm()
            prompt = f"""The previous answer contained ungrounded claims: {result['unfaithful_claims']}
            
Rewrite the answer using ONLY information from the context below.
Question: {query}
Context: {context}
Faithful answer:"""
            answer = llm.invoke(prompt).content
        
        return answer, result["faithfulness_score"]
```

---

### 2.7 Confidence Scoring System

**File: `src/agents/confidence_scorer.py`**
```python
"""
Multi-dimensional confidence scoring:
- Retrieval confidence: avg similarity scores from vector store
- Consistency confidence: agreement across retrieved docs
- Resolution confidence: was conflict resolved or remained unresolved?
- Faithfulness confidence: from hallucination detector
"""
from typing import List
from src.agents.schemas import RetrievedDoc, ResolvedEvidence

class ConfidenceScorer:
    
    def compute_confidence(
        self,
        docs: List[RetrievedDoc],
        resolved: ResolvedEvidence | None,
        faithfulness_score: float = 1.0,
    ) -> dict:
        retrieval_conf = self._retrieval_confidence(docs)
        conflict_penalty = self._conflict_penalty(resolved)
        
        overall = (
            0.35 * retrieval_conf +
            0.35 * faithfulness_score +
            0.30 * (1.0 - conflict_penalty)
        )
        
        return {
            "overall": round(overall, 3),
            "retrieval": round(retrieval_conf, 3),
            "faithfulness": round(faithfulness_score, 3),
            "conflict_penalty": round(conflict_penalty, 3),
            "label": self._label(overall),
        }
    
    def _retrieval_confidence(self, docs: List[RetrievedDoc]) -> float:
        scores = [d.score for d in docs if d.score is not None]
        return sum(scores) / len(scores) if scores else 0.5
    
    def _conflict_penalty(self, resolved: ResolvedEvidence | None) -> float:
        if not resolved or not resolved.has_conflict:
            return 0.0
        unresolved = sum(
            1 for r in resolved.resolutions.values()
            if r.status == "unresolved"
        )
        return min(1.0, unresolved * 0.3)
    
    def _label(self, score: float) -> str:
        if score >= 0.85: return "HIGH"
        if score >= 0.65: return "MEDIUM"
        if score >= 0.45: return "LOW"
        return "VERY_LOW"
```

---

### 2.8 Session Management with Conversation Memory

**File: `src/session/session_manager.py`**
```python
"""
SQLite-backed session management.
Stores: session ID, conversation history, document context per session.
"""
import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DB_PATH = Path("data/sessions.db")

class SessionManager:
    
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    metadata TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?)",
                (session_id, now, now)
            )
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, metadata: dict = None) -> str:
        msg_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (msg_id, session_id, role, content, json.dumps(metadata or {}), now)
            )
        return msg_id
    
    def get_history(self, session_id: str, last_n: int = 10) -> List[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT role, content, metadata, timestamp FROM messages "
                "WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                (session_id, last_n)
            ).fetchall()
        return [{"role": r[0], "content": r[1], 
                 "metadata": json.loads(r[2]), "timestamp": r[3]} 
                for r in reversed(rows)]
    
    def build_context_prompt(self, session_id: str, new_query: str) -> str:
        """Build a conversation-aware prompt with history."""
        history = self.get_history(session_id, last_n=6)
        if not history:
            return new_query
        
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}" for m in history
        ])
        return f"""Previous conversation:
{history_text}

Current question: {new_query}
Answer the current question, using conversation context where relevant."""
```

---

### 2.9 Response Caching

**File: `src/cache/query_cache.py`**
```python
"""
SQLite-based query cache with TTL.
Avoids redundant LLM calls for repeated/similar queries.
"""
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DB = Path("data/query_cache.db")
DEFAULT_TTL_HOURS = 24

class QueryCache:
    
    def __init__(self, ttl_hours: int = DEFAULT_TTL_HOURS):
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    created_at TEXT,
                    hits INTEGER DEFAULT 0
                )
            """)
    
    def _make_key(self, query: str, mode: str, top_k: int) -> str:
        payload = f"{query.lower().strip()}|{mode}|{top_k}"
        return hashlib.sha256(payload.encode()).hexdigest()
    
    def get(self, query: str, mode: str, top_k: int) -> dict | None:
        key = self._make_key(query, mode, top_k)
        with sqlite3.connect(CACHE_DB) as conn:
            row = conn.execute(
                "SELECT response, created_at, hits FROM cache WHERE key=?", (key,)
            ).fetchone()
            if not row:
                return None
            created = datetime.fromisoformat(row[1])
            if datetime.utcnow() - created > self.ttl:
                conn.execute("DELETE FROM cache WHERE key=?", (key,))
                return None
            conn.execute("UPDATE cache SET hits=hits+1 WHERE key=?", (key,))
            return json.loads(row[0])
    
    def set(self, query: str, mode: str, top_k: int, response: dict):
        key = self._make_key(query, mode, top_k)
        with sqlite3.connect(CACHE_DB) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, 0)",
                (key, json.dumps(response), datetime.utcnow().isoformat())
            )
```

---

### 2.10 Structured Logging & Observability

**File: `src/utils/logging.py`**
```python
"""
Structured logging with loguru. 
Each request gets a trace_id for end-to-end tracking.
"""
import sys
import uuid
from loguru import logger
from contextvars import ContextVar
from functools import wraps
import time

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

def setup_logging(level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stdout,
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | '
               '<cyan>trace_id={extra[trace_id]}</cyan> | {message}',
        level=level,
    )
    logger.add(
        "logs/app.jsonl",
        format="{time} | {level} | {extra[trace_id]} | {message}",
        rotation="10 MB",
        retention="30 days",
        serialize=True,
    )

def get_logger():
    return logger.bind(trace_id=trace_id_var.get())

def with_trace(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tid = str(uuid.uuid4())[:8]
        trace_id_var.set(tid)
        log = get_logger()
        t0 = time.time()
        log.info(f"START {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            log.info(f"OK {func.__name__} | {(time.time()-t0)*1000:.1f}ms")
            return result
        except Exception as e:
            log.error(f"ERROR {func.__name__} | {e}")
            raise
    return wrapper
```

**Add to `pyproject.toml`:**
```toml
"loguru>=0.7.2",
```

---

### 2.11 Advanced Evaluation Framework

**Extend `src/evaluation/metrics.py`** with:

```python
# Additional MSc-grade evaluation metrics

def compute_ragas_metrics(records: List[Dict]) -> Dict:
    """
    RAGAS framework metrics:
    - Faithfulness: are claims grounded in context?
    - Answer Relevancy: is the answer relevant to the question?
    - Context Recall: what fraction of ground truth is in context?
    - Context Precision: how many retrieved docs are actually relevant?
    """
    # Requires: pip install ragas
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    ...

def compute_bleu_rouge(predictions: List[str], references: List[str]) -> Dict:
    """BLEU-4 and ROUGE-L for answer quality."""
    from rouge_score import rouge_scorer
    import sacrebleu
    ...

def compute_mean_reciprocal_rank(retrieved_ids_list, relevant_ids_list) -> float:
    """MRR for retrieval quality evaluation."""
    mrr = 0.0
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                mrr += 1.0 / rank
                break
    return mrr / len(retrieved_ids_list)

def compute_ndcg(retrieved_ids_list, relevant_ids_list, k=10) -> float:
    """NDCG@k for ranked retrieval evaluation."""
    import math
    ndcg_scores = []
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(retrieved[:k], start=1)
            if doc_id in relevant
        )
        ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        ndcg_scores.append(dcg / ideal_dcg if ideal_dcg else 0.0)
    return sum(ndcg_scores) / len(ndcg_scores)

def compute_latency_percentiles(latencies: List[float]) -> Dict:
    """P50, P95, P99 latency statistics."""
    import numpy as np
    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
    }
```

**Add to `pyproject.toml`:**
```toml
"ragas>=0.1.21",
"rouge-score>=0.1.2",
"sacrebleu>=2.4.0",
```

---

### 2.12 Knowledge Graph Integration

**File: `src/knowledge_graph/extractor.py`**
```python
"""
Entity and relationship extraction using spaCy + LLM.
Builds a NetworkX graph for structured knowledge queries.
MSc novelty: graph-augmented RAG (GraphRAG pattern).
"""
import spacy
import networkx as nx
from src.models.llm import get_chat_llm

class KnowledgeGraphBuilder:
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.graph = nx.DiGraph()
    
    def extract_entities(self, text: str) -> list[dict]:
        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    
    def extract_relations_llm(self, text: str) -> list[tuple]:
        llm = get_chat_llm()
        prompt = f"""Extract (subject, relation, object) triples from:
        {text}
        Return ONLY a JSON array of [subject, relation, object] arrays."""
        resp = llm.invoke(prompt)
        import json
        return json.loads(resp.content)
    
    def build_graph(self, docs):
        for doc in docs:
            triples = self.extract_relations_llm(doc.text)
            for subj, rel, obj in triples:
                self.graph.add_edge(subj, obj, relation=rel, source=doc.id)
    
    def query_graph(self, entity: str, depth: int = 2) -> dict:
        if entity not in self.graph:
            return {"nodes": [], "edges": []}
        subgraph = nx.ego_graph(self.graph, entity, radius=depth)
        return {
            "nodes": list(subgraph.nodes()),
            "edges": [(u, v, d) for u, v, d in subgraph.edges(data=True)],
        }
```

**Add to `pyproject.toml`:**
```toml
"spacy>=3.7.0",
"networkx>=3.3",
```

---

### 2.13 Enhanced Multi-Agent Graph (Updated)

**Update `src/graphs/multi_agent_graph.py`** to include the new agents:

```python
# New node order (MSc-level pipeline):
# 1. query_processing  (HyDE / decomposition / expansion)
# 2. hybrid_retrieval  (BM25 + dense + reranking)
# 3. contradiction_detection
# 4. resolution  (if conflict)
# 5. answer_generation
# 6. hallucination_detection  (faithfulness check)
# 7. confidence_scoring
# → return enriched state

def hallucination_check_node(state: GraphState) -> GraphState:
    detector = HallucinationDetector()
    answer, faith_score = detector.regenerate_if_unfaithful(
        state["query"], state["answer"], state["retrieved_docs"]
    )
    state["answer"] = answer
    state["faithfulness_score"] = faith_score
    return state

def confidence_node(state: GraphState) -> GraphState:
    scorer = ConfidenceScorer()
    confidence = scorer.compute_confidence(
        docs=state["retrieved_docs"],
        resolved=state.get("resolved"),
        faithfulness_score=state.get("faithfulness_score", 1.0),
    )
    state["confidence"] = confidence
    return state

# Add conditional edge: if faithfulness_score < 0.7, loop back to answer_generation (max 2 retries)
```

---

## 3. COMPLETE UI IMPLEMENTATION

### 3.1 Technology Stack

```
frontend/
├── src/
│   ├── components/
│   │   ├── chat/           # Chat interface
│   │   ├── upload/         # Document upload
│   │   ├── dashboard/      # Evaluation metrics
│   │   ├── knowledge/      # KB document browser
│   │   └── settings/       # Configuration panel
│   ├── hooks/              # Custom React hooks
│   ├── stores/             # Zustand state management
│   ├── api/                # API client
│   └── types/              # TypeScript types
├── index.html
├── vite.config.ts
└── package.json
```

### 3.2 Initialize Frontend Project

```bash
# Run from project root: C:\Users\BEST LAPTOP\Desktop\RAG-gurkirat\
cd C:\Users\BEST LAPTOP\Desktop\RAG-gurkirat
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install

# UI dependencies
npm install zustand @tanstack/react-query axios
npm install lucide-react
npm install react-markdown remark-gfm
npm install react-dropzone
npm install recharts
npm install framer-motion
npm install react-router-dom
npm install clsx tailwind-merge

# Tailwind CSS v4
npm install -D tailwindcss @tailwindcss/vite

# Dev tools
npm install -D @types/node
```

**`frontend/vite.config.ts`:**
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

---

### 3.3 Design System & Global Styles

**`frontend/src/index.css`** — Dark, professional research tool aesthetic:
```css
@import "tailwindcss";

:root {
  --bg-base: #0a0b0f;
  --bg-surface: #111318;
  --bg-elevated: #181b23;
  --bg-overlay: #1e2230;
  --border: #2a2f3e;
  --border-subtle: #1d2030;
  --text-primary: #e8eaf0;
  --text-secondary: #8b93aa;
  --text-muted: #535a70;
  --accent: #4f7bff;
  --accent-soft: rgba(79, 123, 255, 0.12);
  --accent-glow: rgba(79, 123, 255, 0.4);
  --success: #22d3a0;
  --warning: #f59e0b;
  --danger: #f43f5e;
  --conflict: #ff6b35;
  --conflict-soft: rgba(255, 107, 53, 0.1);
  --resolved: #22d3a0;
  --resolved-soft: rgba(34, 211, 160, 0.1);
  --font-sans: "Geist", "DM Sans", system-ui, sans-serif;
  --font-mono: "Geist Mono", "JetBrains Mono", monospace;
  --radius: 10px;
  --radius-sm: 6px;
  --radius-lg: 16px;
  --shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
  --shadow-lg: 0 8px 48px rgba(0, 0, 0, 0.6);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg-base);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 15px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}
```

---

### 3.4 App Layout & Router

**`frontend/src/App.tsx`:**
```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from '@/components/layout/Sidebar'
import { ChatPage } from '@/pages/ChatPage'
import { DocumentsPage } from '@/pages/DocumentsPage'
import { EvalPage } from '@/pages/EvalPage'
import { SettingsPage } from '@/pages/SettingsPage'

const qc = new QueryClient()

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg-base)' }}>
          <Sidebar />
          <main className="flex-1 overflow-hidden">
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/chat/:sessionId" element={<ChatPage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/eval" element={<EvalPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
```

---

### 3.5 Sidebar Component

**`frontend/src/components/layout/Sidebar.tsx`:**
```tsx
import { NavLink, useNavigate } from 'react-router-dom'
import { MessageSquare, FileText, BarChart2, Settings, Plus, Zap } from 'lucide-react'
import { useSessions } from '@/hooks/useSessions'
import { useSettingsStore } from '@/stores/settingsStore'

export function Sidebar() {
  const { sessions, createSession } = useSessions()
  const navigate = useNavigate()
  const { mode } = useSettingsStore()

  const handleNewChat = async () => {
    const sessionId = await createSession()
    navigate(`/chat/${sessionId}`)
  }

  const navItems = [
    { to: '/', icon: MessageSquare, label: 'Chat' },
    { to: '/documents', icon: FileText, label: 'Knowledge Base' },
    { to: '/eval', icon: BarChart2, label: 'Evaluation' },
    { to: '/settings', icon: Settings, label: 'Settings' },
  ]

  return (
    <aside style={{
      width: '260px', background: 'var(--bg-surface)',
      borderRight: '1px solid var(--border)', display: 'flex',
      flexDirection: 'column', height: '100vh',
    }}>
      <div style={{ padding: '20px', borderBottom: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '8px',
            background: 'linear-gradient(135deg, var(--accent), #7c4dff)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Zap size={16} color="white" fill="white" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '14px', letterSpacing: '-0.3px' }}>RAG Research</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Multi-Agent System</div>
          </div>
        </div>
      </div>

      <div style={{ padding: '12px' }}>
        <button onClick={handleNewChat} style={{
          width: '100%', padding: '9px 14px', borderRadius: 'var(--radius)',
          background: 'var(--accent-soft)', border: '1px solid rgba(79,123,255,0.2)',
          color: 'var(--accent)', fontSize: '13px', fontWeight: 600,
          cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px',
          transition: 'all 0.15s',
        }}>
          <Plus size={15} /> New Chat
        </button>
      </div>

      <nav style={{ padding: '0 8px' }}>
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} style={({ isActive }) => ({
            display: 'flex', alignItems: 'center', gap: '10px',
            padding: '9px 12px', borderRadius: 'var(--radius-sm)',
            marginBottom: '2px', textDecoration: 'none', fontSize: '13px',
            fontWeight: isActive ? 600 : 400,
            color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
            background: isActive ? 'var(--bg-elevated)' : 'transparent',
            transition: 'all 0.15s',
          })}>
            <Icon size={15} />{label}
          </NavLink>
        ))}
      </nav>

      <div style={{ flex: 1, overflow: 'auto', padding: '16px 8px 8px' }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)',
          textTransform: 'uppercase', letterSpacing: '0.8px', padding: '0 8px 8px' }}>
          Recent Chats
        </div>
        {sessions.map(session => (
          <NavLink key={session.id} to={`/chat/${session.id}`} style={({ isActive }) => ({
            display: 'block', padding: '8px 12px', borderRadius: 'var(--radius-sm)',
            marginBottom: '2px', textDecoration: 'none', fontSize: '12px',
            color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
            background: isActive ? 'var(--bg-elevated)' : 'transparent',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            transition: 'all 0.15s',
          })}>
            {session.preview || 'New conversation'}
          </NavLink>
        ))}
      </div>

      <div style={{ padding: '12px', borderTop: '1px solid var(--border-subtle)' }}>
        <div style={{
          padding: '8px 12px', borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-elevated)', fontSize: '11px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{ color: 'var(--text-muted)' }}>Mode</span>
          <span style={{ color: mode === 'multi_agent' ? 'var(--success)' : 'var(--accent)', fontWeight: 600 }}>
            {mode === 'multi_agent' ? 'Multi-Agent' : 'Baseline'}
          </span>
        </div>
      </div>
    </aside>
  )
}
```

---

### 3.6 Chat Page (Main Interface)

**`frontend/src/pages/ChatPage.tsx`:**
```tsx
import { useState, useRef, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Send, Loader2, Sparkles } from 'lucide-react'
import { MessageBubble } from '@/components/chat/MessageBubble'
import { SourcePanel } from '@/components/chat/SourcePanel'
import { StreamingIndicator } from '@/components/chat/StreamingIndicator'
import { useChatStore } from '@/stores/chatStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { streamChat } from '@/api/chat'

export function ChatPage() {
  const { sessionId } = useParams()
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingStep, setStreamingStep] = useState('')
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { messages, addMessage, updateMessage } = useChatStore()
  const { mode, topK } = useSettingsStore()
  const sessionMessages = messages[sessionId || ''] || []
  const selectedMessage = sessionMessages.find(m => m.id === selectedMessageId)

  const handleSubmit = async () => {
    if (!input.trim() || isStreaming) return
    const query = input.trim()
    setInput('')
    setIsStreaming(true)
    const userMsgId = addMessage(sessionId!, { role: 'user', content: query })
    const assistantMsgId = addMessage(sessionId!, { role: 'assistant', content: '', isStreaming: true })
    let accumulated = ''
    await streamChat({
      query, sessionId, mode, topK,
      onStatus: (step) => setStreamingStep(step),
      onToken: (token) => {
        accumulated += token
        updateMessage(sessionId!, assistantMsgId, { content: accumulated })
      },
      onDone: (meta) => {
        updateMessage(sessionId!, assistantMsgId, {
          content: accumulated, isStreaming: false,
          sources: meta.sources, conflictInfo: meta.conflict_info, confidence: meta.confidence,
        })
        setSelectedMessageId(assistantMsgId)
        setIsStreaming(false)
        setStreamingStep('')
      },
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() }
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [sessionMessages])

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: selectedMessage ? '1px solid var(--border)' : 'none' }}>
        <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--bg-surface)' }}>
          <Sparkles size={16} style={{ color: 'var(--accent)' }} />
          <span style={{ fontWeight: 600, fontSize: '14px' }}>{mode === 'multi_agent' ? 'Multi-Agent RAG' : 'Baseline RAG'}</span>
          {sessionMessages.length > 0 && (
            <span style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-muted)', background: 'var(--bg-elevated)', padding: '3px 8px', borderRadius: '20px' }}>
              {sessionMessages.length} messages
            </span>
          )}
        </div>

        <div style={{ flex: 1, overflow: 'auto', padding: '24px' }}>
          {sessionMessages.length === 0 && <EmptyState />}
          <AnimatePresence>
            {sessionMessages.map((msg) => (
              <motion.div key={msg.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
                <MessageBubble
                  message={msg}
                  isSelected={msg.id === selectedMessageId}
                  onClick={() => msg.role === 'assistant' && setSelectedMessageId(selectedMessageId === msg.id ? null : msg.id)}
                />
              </motion.div>
            ))}
          </AnimatePresence>
          {isStreaming && <StreamingIndicator step={streamingStep} />}
          <div ref={messagesEndRef} />
        </div>

        <div style={{ padding: '16px 24px', borderTop: '1px solid var(--border-subtle)', background: 'var(--bg-surface)' }}>
          <div style={{ background: 'var(--bg-elevated)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)', display: 'flex', gap: '8px', padding: '10px 14px', alignItems: 'flex-end' }}>
            <textarea
              value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown}
              placeholder="Ask a question about your knowledge base..." rows={1} disabled={isStreaming}
              style={{ flex: 1, resize: 'none', background: 'transparent', border: 'none', outline: 'none', color: 'var(--text-primary)', fontSize: '14px', fontFamily: 'var(--font-sans)', lineHeight: 1.6, maxHeight: '200px', overflow: 'auto' }}
            />
            <button onClick={handleSubmit} disabled={!input.trim() || isStreaming} style={{ width: '36px', height: '36px', borderRadius: '8px', background: input.trim() && !isStreaming ? 'var(--accent)' : 'var(--bg-overlay)', border: 'none', cursor: input.trim() && !isStreaming ? 'pointer' : 'default', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.15s', flexShrink: 0 }}>
              {isStreaming ? <Loader2 size={16} style={{ color: 'var(--text-muted)', animation: 'spin 1s linear infinite' }} /> : <Send size={15} style={{ color: input.trim() ? 'white' : 'var(--text-muted)' }} />}
            </button>
          </div>
          <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center' }}>Press Enter to send · Shift+Enter for new line</div>
        </div>
      </div>

      <AnimatePresence>
        {selectedMessage && (
          <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 380, opacity: 1 }} exit={{ width: 0, opacity: 0 }} transition={{ duration: 0.2 }} style={{ overflow: 'hidden', flexShrink: 0, background: 'var(--bg-surface)' }}>
            <SourcePanel message={selectedMessage} onClose={() => setSelectedMessageId(null)} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function EmptyState() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '16px', opacity: 0.6 }}>
      <div style={{ width: '56px', height: '56px', borderRadius: '16px', background: 'var(--accent-soft)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Sparkles size={24} style={{ color: 'var(--accent)' }} />
      </div>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '16px', fontWeight: 600, marginBottom: '6px' }}>Ask your knowledge base</div>
        <div style={{ fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '320px' }}>
          The multi-agent system will retrieve relevant information, detect contradictions, and resolve conflicts automatically.
        </div>
      </div>
    </div>
  )
}
```

---

### 3.7 Message Bubble Component

**`frontend/src/components/chat/MessageBubble.tsx`:**
```tsx
import { User, Bot, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export function MessageBubble({ message, isSelected, onClick }) {
  const isUser = message.role === 'user'
  return (
    <div onClick={isUser ? undefined : onClick} style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexDirection: isUser ? 'row-reverse' : 'row', cursor: !isUser ? 'pointer' : 'default' }}>
      <div style={{ width: '32px', height: '32px', borderRadius: '8px', flexShrink: 0, background: isUser ? 'var(--accent-soft)' : 'var(--bg-elevated)', border: `1px solid ${isUser ? 'rgba(79,123,255,0.2)' : 'var(--border)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {isUser ? <User size={15} style={{ color: 'var(--accent)' }} /> : <Bot size={15} style={{ color: 'var(--text-secondary)' }} />}
      </div>
      <div style={{ maxWidth: '75%' }}>
        <div style={{ padding: '12px 16px', borderRadius: isUser ? '16px 4px 16px 16px' : '4px 16px 16px 16px', background: isUser ? 'var(--accent)' : 'var(--bg-elevated)', border: isUser ? 'none' : `1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`, boxShadow: isSelected ? '0 0 0 1px var(--accent), 0 0 16px var(--accent-glow)' : 'none', transition: 'all 0.15s', fontSize: '14px', lineHeight: 1.65, color: isUser ? 'white' : 'var(--text-primary)' }}>
          {message.isStreaming && !message.content ? (
            <div style={{ display: 'flex', gap: '4px', alignItems: 'center', padding: '2px 0' }}>
              {[0,1,2].map(i => <div key={i} style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--text-muted)', animation: `bounce 1.2s ${i*0.2}s ease-in-out infinite` }} />)}
              <style>{`@keyframes bounce { 0%,60%,100% { transform:translateY(0) } 30% { transform:translateY(-4px) } }`}</style>
            </div>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ p: ({children}) => <p style={{marginBottom:'8px'}}>{children}</p> }}>
              {message.content}
            </ReactMarkdown>
          )}
        </div>
        {!isUser && !message.isStreaming && message.confidence && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '6px', paddingLeft: '4px' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: message.confidence.overall >= 0.85 ? 'var(--success)' : message.confidence.overall >= 0.65 ? 'var(--accent)' : 'var(--warning)' }}>
              {message.confidence.label} confidence ({Math.round(message.confidence.overall * 100)}%)
            </span>
            {message.conflictInfo?.has_conflict && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: 'var(--conflict)' }}>
                <AlertCircle size={11} /> Conflict detected
              </div>
            )}
            {message.sources && <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{message.sources.length} sources</div>}
          </div>
        )}
      </div>
    </div>
  )
}
```

---

### 3.8 Source Panel, Documents Page, Eval Dashboard, Settings

See sections 3.8–3.11 in the full implementation above. Each page follows the same dark design system with `var(--bg-surface)`, `var(--accent)`, `var(--border)` CSS variables.

Key components to implement:
- **`SourcePanel.tsx`**: Slide-in right panel showing confidence breakdown (retrieval/faithfulness/conflict), conflict cluster cards (resolved/unresolved), and source cards with relevance bars
- **`DocumentsPage.tsx`**: Drag-and-drop upload zone + stats cards (total docs, indexed chunks, store size) + document list with delete
- **`EvalPage.tsx`**: KPI cards + Recharts BarChart (multi-agent vs baseline) + RadarChart (capability profile) + "Run Evaluation" button
- **`SettingsPage.tsx`**: Toggle switches for Hybrid Retrieval / HyDE / Hallucination Check + model selector + Top-K slider

---

### 3.9 API Client

**`frontend/src/api/chat.ts`:**
```typescript
export async function streamChat({ query, sessionId, mode = 'multi_agent', topK = 5, onStatus, onToken, onDone }) {
  const response = await fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId, mode, top_k: topK }),
  })
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const chunk = decoder.decode(value)
    for (const line of chunk.split('\n').filter(l => l.startsWith('data: '))) {
      try {
        const data = JSON.parse(line.slice(6))
        if (data.type === 'status') onStatus(data.value)
        else if (data.type === 'token') onToken(data.value)
        else if (data.type === 'done') onDone(data)
      } catch {}
    }
  }
}
```

---

### 3.10 State Management

**`frontend/src/stores/chatStore.ts`** — Zustand + persist middleware for chat history across page refreshes.

**`frontend/src/stores/settingsStore.ts`** — Persisted settings: mode, topK, model, useHyde, useHybridRetrieval, useHallucinationCheck.

---

### 3.11 Streaming Indicator

**`frontend/src/components/chat/StreamingIndicator.tsx`** — Shows animated status pill for each pipeline stage: "Searching knowledge base..." → "Detecting contradictions..." → "Resolving conflicts..." → "Generating answer..."

---

## 4. COMPLETE PROJECT STRUCTURE AFTER FULL IMPLEMENTATION

```
RAG-gurkirat/
├── src/
│   ├── agents/
│   │   ├── answer_generator.py
│   │   ├── contradiction_detector.py
│   │   ├── resolution.py
│   │   ├── retriever.py
│   │   ├── schemas.py
│   │   ├── query_processor.py         ← NEW
│   │   ├── hallucination_detector.py  ← NEW
│   │   └── confidence_scorer.py       ← NEW
│   ├── api/
│   │   ├── main.py                    ← NEW
│   │   ├── schemas.py                 ← NEW
│   │   └── routers/
│   │       ├── chat.py                ← NEW (SSE streaming)
│   │       ├── documents.py           ← NEW
│   │       ├── sessions.py            ← NEW
│   │       ├── eval.py                ← NEW
│   │       └── health.py              ← NEW
│   ├── cache/
│   │   └── query_cache.py             ← NEW
│   ├── config/config.py               (extended)
│   ├── data_prep/
│   │   ├── build_kb.py
│   │   └── document_loader.py         ← NEW
│   ├── evaluation/
│   │   ├── metrics.py                 (extended: RAGAS, BLEU, ROUGE, MRR, NDCG)
│   │   └── run_eval.py
│   ├── graphs/
│   │   ├── baseline_rag.py
│   │   └── multi_agent_graph.py       (extended 7-stage pipeline)
│   ├── knowledge_graph/
│   │   └── extractor.py               ← NEW
│   ├── models/
│   │   ├── embeddings.py
│   │   └── llm.py
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   └── hybrid_retriever.py        ← NEW
│   ├── session/
│   │   └── session_manager.py         ← NEW
│   └── utils/
│       └── logging.py                 ← NEW
├── frontend/                          ← NEW (entire React UI)
│   ├── src/
│   │   ├── App.tsx
│   │   ├── index.css
│   │   ├── api/chat.ts
│   │   ├── components/
│   │   │   ├── chat/MessageBubble.tsx
│   │   │   ├── chat/SourcePanel.tsx
│   │   │   ├── chat/StreamingIndicator.tsx
│   │   │   └── layout/Sidebar.tsx
│   │   ├── pages/
│   │   │   ├── ChatPage.tsx
│   │   │   ├── DocumentsPage.tsx
│   │   │   ├── EvalPage.tsx
│   │   │   └── SettingsPage.tsx
│   │   └── stores/
│   │       ├── chatStore.ts
│   │       └── settingsStore.ts
│   ├── vite.config.ts
│   └── package.json
├── data/
│   ├── index/chroma_db/
│   ├── raw/
│   ├── contradictions/
│   ├── eval/
│   ├── sessions.db                    ← NEW
│   └── query_cache.db                 ← NEW
├── logs/                              ← NEW
├── start_api.py                       ← NEW
└── pyproject.toml                     (extended)
```

---

## 5. START SCRIPTS

**`start_api.py`:**
```python
import uvicorn
if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
```

**Complete startup:**
```bash
# Terminal 1: Backend
cd C:\Users\BEST LAPTOP\Desktop\RAG-gurkirat
uv run python start_api.py

# Terminal 2: Frontend
cd C:\Users\BEST LAPTOP\Desktop\RAG-gurkirat\frontend
npm run dev

# Open: http://localhost:5173
```

---

## 6. UPDATED pyproject.toml

```toml
[project]
name = "rag-gurkirat"
version = "2.0.0"
description = "MSc Self-Correcting Multi-Agent RAG with Contradiction Detection"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "chromadb>=1.5.9",
    "fastapi>=0.136.3",
    "langchain>=1.3.4",
    "langchain-community>=0.4.2",
    "langchain-groq>=1.1.2",
    "langchain-huggingface>=1.2.2",
    "langchain-openai>=1.2.2",
    "langgraph>=1.2.4",
    "numpy>=2.4.6",
    "openai>=2.41.0",
    "pandas>=3.0.3",
    "pydantic>=2.13.4",
    "pydantic-settings>=2.14.1",
    "python-dotenv>=1.2.2",
    "rich>=15.0.0",
    "sentence-transformers>=5.5.1",
    "tiktoken>=0.13.0",
    "uvicorn>=0.49.0",
    "python-multipart>=0.0.9",
    "httpx>=0.27.0",
    "rank-bm25>=0.2.2",
    "pypdf>=5.0.0",
    "python-docx>=1.1.0",
    "unstructured>=0.16.0",
    "beautifulsoup4>=4.12.0",
    "rouge-score>=0.1.2",
    "sacrebleu>=2.4.0",
    "spacy>=3.7.0",
    "networkx>=3.3",
    "loguru>=0.7.2",
]
```

---

## 7. FEATURE PRIORITY TABLE

| Priority | Feature | Effort | MSc Impact |
|----------|---------|--------|-----------|
| 🔴 P0 | FastAPI REST API + streaming SSE | 2 days | Essential |
| 🔴 P0 | React UI (full frontend) | 3 days | Essential |
| 🔴 P0 | Session management + chat history | 1 day | Essential |
| 🟠 P1 | Document upload (PDF/DOCX) | 1 day | High |
| 🟠 P1 | Hallucination detection agent | 1 day | High |
| 🟠 P1 | Confidence scoring system | 0.5 day | High |
| 🟡 P2 | Hybrid retrieval (BM25 + rerank) | 1.5 days | Medium |
| 🟡 P2 | HyDE query expansion | 0.5 day | Medium |
| 🟡 P2 | Evaluation dashboard (charts) | 1 day | Medium |
| 🟡 P2 | Query cache (SQLite) | 0.5 day | Medium |
| 🟢 P3 | Advanced RAGAS metrics | 1 day | Research |
| 🟢 P3 | Knowledge graph (GraphRAG) | 2 days | Research |
| 🟢 P3 | Structured logging + tracing | 0.5 day | Production |

---

## 8. MSc RESEARCH CONTRIBUTIONS (For Dissertation)

1. **Novel 7-Stage Pipeline**: query-expand → hybrid-retrieve → detect → resolve → generate → verify → score — vs 2-stage baseline
2. **Hybrid Conflict Detection**: Rule-based numeric heuristics + LLM semantic analysis = higher recall than either alone
3. **Self-Correcting Loop**: Faithfulness-gated regeneration = measurable reduction in hallucination rate
4. **Multi-Dimensional Confidence**: 3-factor scoring (retrieval × faithfulness × conflict penalty) — novel composite metric
5. **Comparative Evaluation**: Multi-agent vs. baseline on 6 metrics (accuracy, conflict detection, resolution quality, faithfulness, ROUGE-L, latency)

---

## 9. ENVIRONMENT VARIABLES (.env)

```env
# Required
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional overrides
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
VECTOR_STORE_DIR=data/index/chroma_db
COLLECTION_NAME=multi_agent_rag
TOP_K=5
API_HOST=0.0.0.0
API_PORT=8000
CACHE_TTL_HOURS=24
```

---

*Generated by analyzing every source file in RAG-gurkirat. All code targets Python 3.13 + uv + LangGraph + Groq + ChromaDB + React + Vite + TypeScript.*
