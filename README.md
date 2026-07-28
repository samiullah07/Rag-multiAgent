# 🧠 Multi-Agent RAG Lab

**Self-correcting Retrieval-Augmented Generation with automatic contradiction detection, resolution, and citation grounding**

> MSc Computer Science Project  
> Stack: Python 3.13 · LangGraph · ChromaDB · Groq (Llama 3.3-70B) · HuggingFace Embeddings · Streamlit

---

## What This Project Does

Most RAG systems retrieve documents and generate an answer in two steps. This project adds three additional intelligent agents on top of that baseline:

1. **Contradiction Detector** — identifies when retrieved documents contain conflicting facts (e.g. two sources giving different completion years for the Human Genome Project)
2. **Resolution Agent** — resolves detected contradictions using a configurable strategy: prefer the most recent source, prefer the most authoritative source, or present both sides
3. **Citation-Grounding Verifier** — checks whether the final answer is genuinely grounded in the retrieved documents, catching fabricated citations and answers that silently rely on outside knowledge

The result is a system that doesn't just answer — it knows *when* it's uncertain, *why* sources disagree, and *whether* its own answer can be trusted.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│  Retriever  │  ── Semantic search against ChromaDB (HuggingFace BAAI/bge-small-en-v1.5)
└──────┬──────┘     Optional: session-scoped upload collection merged first
       │
       ▼
┌──────────────────────────┐
│ Contradiction Detector   │  ── Rule-based numeric overlap + LLM semantic refinement
└──────────┬───────────────┘
           │
    ┌──────▼──────┐
    │  Conflict?  │
    └──────┬──────┘
     No ───┤─── Yes ──► ┌─────────────────┐
           │             │ Resolution Agent │  ── most_recent / most_authoritative / explain_both
           │             └────────┬────────┘
           │                      │
           ▼                      ▼
    ┌──────────────────────────────────┐
    │      Answer Generation           │  ── Groq Llama 3.3-70B via LangChain
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │  Citation-Grounding Verifier     │  ── Deterministic citation check + LLM faithfulness
    └──────────────────────────────────┘
```

Baseline comparison: a standard 2-stage RAG (retrieve → generate) is included for direct benchmarking.

---

## Key Results (Verified Empirically)

All metrics evaluated on a 20-question benchmark (10 non-conflict, 10 with deliberate contradictions) using `data/eval/questions.jsonl`:

| Metric | Multi-Agent | Baseline |
|--------|-------------|----------|
| Answer Accuracy (fuzzy match) | **1.0** (20/20) | 0.85 (17/20) |
| Conflict Detection Recall | **100%** (10/10) | 0% |
| Conflict Detection Precision | 55.6% | — |
| Conflict Detection F1 | **71.4%** | — |
| Resolution Quality | **1.0** (10/10) | — |

**Notable findings from evaluation:**
- The multi-agent system correctly identified all 10 true contradictions (100% recall), with false-positive rate of 8/10 on non-conflict queries — a documented limitation of the rule-based numeric-overlap heuristic, which over-flags when multiple numeric facts appear in retrieved documents about the same topic
- The `most_recent` resolution strategy outperforms naive selection: fixing a `publication_date` ingestion bug raised resolution quality from 0.7 → 1.0 (a concrete, traceable improvement)
- A real hallucination case was captured during testing: the system fabricated three entirely fake documents (`astronomy_01`, `astronomy_02`, `astronomy_03`) with realistic formatting to justify an answer not in the knowledge base — this directly motivated the citation-grounding verifier

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-agent pipeline** | 5-node LangGraph graph: retriever → detector → resolver → generator → verifier |
| **Contradiction detection** | Rule-based + LLM refinement, with precision/recall/F1 metrics |
| **Configurable resolution** | 3 strategies selectable in the UI without code changes |
| **Citation-grounding verifier** | Deterministic fabrication check (free) + LLM semantic grounding check |
| **Web search fallback** | Tavily-powered opt-in internet search when KB answer isn't grounded — always asks permission first, never automatic |
| **Document upload** | Upload PDF/DOCX/TXT in the UI; indexed into a session-scoped collection, never touching the eval benchmark data |
| **Temporal Belief-Revision Timeline** | `pages/2_⏳_Belief_Timeline.py` — shows how the system's answer changes as documents with later publication dates are progressively added, grounded in AGM belief revision theory (Alchourrón, Gärdenfors & Makinson, 1985) |
| **Colorful Streamlit UI** | Custom CSS theme with gradient bubbles, glass-card doc panels, KPI badges, conflict/faithfulness indicators |
| **Downloadable session report** | Full conversation + retrieved docs + metrics exported as Markdown |
| **Baseline comparison** | 2-stage naive RAG included for direct A/B comparison |

---

## Project Structure

```
RAG-gurkirat/
├── app.py                          # Streamlit UI (main page)
├── pages/
│   └── 1_📚_Knowledge_Base.py     # Document upload page
├── src/
│   ├── agents/
│   │   ├── retriever.py            # Semantic retrieval agent
│   │   ├── contradiction_detector.py  # Conflict detection (rule + LLM)
│   │   ├── resolution.py           # Conflict resolution strategies
│   │   ├── answer_generator.py     # LLM answer generation
│   │   ├── grounding_verifier.py   # Citation-grounding verifier (NEW)
│   │   ├── web_search_agent.py     # Tavily web search fallback (NEW)
│   │   └── schemas.py              # Shared Pydantic models
│   ├── graphs/
│   │   ├── multi_agent_graph.py    # 5-node LangGraph pipeline
│   │   └── baseline_rag.py         # 2-stage baseline for comparison
│   ├── retrieval/
│   │   └── vector_store.py         # ChromaDB wrapper
│   ├── models/
│   │   ├── llm.py                  # Groq LLM client with retry wrapper
│   │   └── embeddings.py           # HuggingFace embeddings
│   ├── data_prep/
│   │   └── build_kb.py             # KB ingestion (raw + contradiction docs)
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, conflict P/R/F1, resolution quality
│   │   └── run_eval.py             # Full evaluation runner
│   ├── config/
│   │   └── config.py               # Pydantic settings (loads .env)
│   └── session/
│       └── session_manager.py      # Session management
├── data/
│   ├── raw/                        # 50 source documents (5 topics × 10 files)
│   ├── contradictions/             # 18 deliberate contradiction pairs (36 docs)
│   ├── eval/                       # 20-question benchmark + saved results
│   └── index/chroma_db/            # Vector store index
├── main_multi_agent.py             # CLI: multi-agent pipeline
├── main_baseline.py                # CLI: baseline pipeline
├── CLAUDE.md                       # Full project spec and implementation history
├── TESTING.md                      # Manual test checklist with known-answer queries
└── pyproject.toml                  # Dependencies (uv-managed)
```

---

## Setup & Running

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Groq API key](https://console.groq.com) — free tier, 100k tokens/day
- [Tavily API key](https://tavily.com) — free tier, 1000 credits/month (optional, for web search)

### Installation

```bash
git clone <repo-url>
cd RAG-gurkirat

# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env.example .env   # then edit .env with your keys
```

### `.env` file

```env
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
VECTOR_STORE_DIR=data/index/chroma_db
COLLECTION_NAME=multi_agent_rag
TOP_K=5
TAVILY_API_KEY=tvly-...   # optional — only needed for web search fallback
```

### Build the knowledge base

```bash
uv run python -m src.data_prep.build_kb
```

This indexes 86 documents total (50 source docs + 36 contradiction documents) into ChromaDB.

### Run the UI

```bash
uv run streamlit run app.py
```

Open http://localhost:8501 — the 📚 Knowledge Base page appears in the sidebar for document upload.

### Run from CLI (no UI)

```bash
uv run python main_multi_agent.py --query "What is the speed of light?"
uv run python main_baseline.py --query "What is the speed of light?"
```

### Run the full evaluation

```bash
uv run python -m src.evaluation.run_eval
```

Expected output (verified benchmark):

```
Multi-agent accuracy: 1.0
Conflict detection — precision: 0.556, recall: 1.0, F1: 0.714
Resolution quality: 1.0
```

---

## Knowledge Base Design

The KB was deliberately designed to include contradictions for evaluation purposes:

- **50 source documents** across 5 topics (science, history, technology, health, geography) in `data/raw/`
- **18 contradiction pairs** (36 documents) in `data/contradictions/` — each pair contains a "correct" encyclopedia-sourced version and a deliberately wrong "blog post" version, with `publication_date` metadata so the `most_recent` resolution strategy can distinguish them
- **20 evaluation questions** in `data/eval/questions.jsonl` — 10 non-conflict, 10 with known correct answers requiring contradiction resolution

Example contradiction pair (speed of light):
```json
{"id": "science_contr_01a", "text": "The speed of light in a vacuum is exactly 299,792,458 meters per second.", "source": "encyclopedia", "publication_date": "2024-01-01"}
{"id": "science_contr_01b", "text": "The speed of light in a vacuum is approximately 300,000,000 meters per second.", "source": "blog_post",  "publication_date": "2015-06-01"}
```

---

## Research Contributions

This project makes the following contributions relative to a standard 2-stage RAG baseline:

1. **Self-correcting multi-agent pipeline** — demonstrated +15 percentage point accuracy improvement on a contradictory knowledge base (1.0 vs 0.85)
2. **Empirically validated contradiction resolution** — `most_recent` strategy achieves 1.0 resolution quality; a concrete ingestion bug (`publication_date` not stored in Chroma metadata) was found, fixed, and verified to improve resolution quality from 0.7 → 1.0
3. **Fabricated citation detection** — a real, captured hallucination (the system inventing `astronomy_01/02/03` with realistic document formatting) motivated the citation-grounding verifier; the deterministic check catches this at zero additional LLM cost
4. **Honest uncertainty handling** — the system admits when information isn't in the KB and offers opt-in internet search rather than silently hallucinating, with explicit source labelling ("🌐 From the internet, not the knowledge base")
5. **Precision-recall tradeoff analysis** — the 55.6% precision / 100% recall result on conflict detection is a documented, quantified limitation of the rule-based heuristic, not a hidden failure — the over-flagging pattern and its latency cost (~23s for false positives vs ~12s for true conflicts) were empirically measured

---

## Limitations

- **Conflict detection precision** — the rule-based numeric-overlap heuristic over-flags non-conflict queries (55.6% precision). This is a known, quantified limitation documented in the evaluation. The LLM-based refinement step improves precision but doesn't eliminate false positives entirely.
- **Static knowledge base** — the system cannot answer questions about information not in the indexed documents unless the web search fallback is explicitly triggered by the user.
- **Token limits** — the Groq free tier (100k tokens/day, 6k tokens/minute for the 8B model) constrains throughput. The pipeline adds 3 LLM calls per query (detector + generator + verifier) vs 1 for the baseline.
- **Evaluation set size** — 20 questions is small; results should be interpreted as directional rather than statistically conclusive.

---

## Testing

See `TESTING.md` for a complete manual test checklist with known-correct answers.

Quick smoke test (no LLM calls needed):

```bash
# Confirm KB is indexed
uv run python -c "
from src.retrieval.vector_store import get_vector_store
vs = get_vector_store()
print('Collection size:', vs._collection.count())
"
# Expected: Collection size: 86
```

---

## Dependencies

Key packages (see `pyproject.toml` for full list with versions):

| Package | Purpose |
|---------|---------|
| `langgraph` | Multi-agent graph orchestration |
| `langchain-groq` | Groq LLM integration |
| `langchain-huggingface` | HuggingFace embeddings |
| `chromadb` | Vector store |
| `streamlit` | UI |
| `tavily-python` | Web search fallback |
| `pypdf` / `python-docx` | Document upload parsing |
| `sentence-transformers` | Embedding model |

---

## Acknowledgements

- Knowledge base construction used deliberately synthetic contradiction pairs to enable controlled evaluation of the resolution agent
- The contradiction detection approach draws on ideas from Truth Maintenance Systems (Doyle, 1979) applied to a modern RAG context
- Evaluation metrics follow conventions from the RAG evaluation literature (recall, precision, faithfulness)
