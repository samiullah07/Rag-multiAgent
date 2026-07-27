# CLAUDE.md — MSc-Level Multi-Agent RAG System (Authoritative Spec)

> **Project**: Self-Correcting Multi-Agent RAG with Contradiction Detection
> **Stack**: Python 3.13 (uv-managed), LangGraph, ChromaDB, Groq (Llama-3.3-70B), HuggingFace embeddings, **Streamlit** UI
> **UI decision**: Streamlit is the permanent UI. Do **not** build a React or vanilla-HTML/FastAPI frontend unless this file is explicitly updated to say otherwise.
> **Status**: Working multi-agent pipeline + CLI + Streamlit demo, but the knowledge base/eval data is empty and several modules are placeholders. See Section 1.

This file supersedes and replaces:
- the old root `CLAUDE.md` (which proposed a React + FastAPI rebuild),
- `.claude/CLAUDE.md` (which proposed deleting React and building a vanilla HTML/CSS/JS frontend).

**Action item:** delete or archive `.claude/CLAUDE.md` so future Claude Code sessions don't get contradictory instructions. Both old plans are obsolete — this file is now the single source of truth.

---

## 1. Honest Diagnosis of Current State (read this before changing anything)

### What actually works today
- `src/graphs/multi_agent_graph.py` — 4-stage LangGraph pipeline: retrieve → detect contradictions → (conditionally) resolve → generate answer. Compiles and runs.
- `src/graphs/baseline_rag.py` — 2-stage retrieve → generate baseline for comparison.
- `src/retrieval/vector_store.py` — Chroma wrapper (`get_vector_store`, `add_documents`, `retrieve`).
- `src/models/embeddings.py` / `src/models/llm.py` — HuggingFace `BAAI/bge-small-en-v1.5` embeddings + Groq `ChatGroq` LLM client.
- `src/agents/*` — retriever, contradiction detector (rule-based numeric overlap + LLM refinement), resolution agent (`most_recent` / `most_authoritative` / `explain_both` / self-correction-with-extra-retrieval), answer generator.
- `src/evaluation/metrics.py` — answer accuracy, conflict detection rate, resolution quality, uncertain flag rate (good MSc-relevant metric set, logic is sound).
- `main_multi_agent.py` / `main_baseline.py` — working CLI entry points (`--query`).
- `app.py` — the **real, working UI** (Streamlit), now restyled with a colorful theme (see Section 4).

### Critical gaps / bugs found during diagnosis
1. **No real knowledge base.** `data/raw/`, `data/contradictions/`, `data/eval/`, `data/experiments/` are all empty on disk. `data/index/chroma_db/chroma.sqlite3` exists but is a fresh, ~160KB empty Chroma schema with no vectors in it. **Right now any query returns zero retrieved documents.** This must be fixed before anything else matters (Section 2.1).
2. **`pyproject.toml` was missing `streamlit`**, even though `app.py` imports it. Fixed in this pass — confirm `uv sync` picks it up.
3. **The model selector in the UI does nothing.** `app.py` passes `model` into the graph state, but `multi_agent_graph.py`'s `answer_generation_node` never reads `state["model"]` — it always calls `get_chat_llm()`, which is hardcoded to `settings.groq_model` from `.env`. Also, the dropdown options (`llama3-70b-8192`, `mixtral-8x7b-32768`) are old/likely-decommissioned Groq model IDs. Either wire it up properly or remove it — don't leave dead UI controls in an MSc demo.
4. **`src/session/session_manager.py` is an in-memory placeholder** (`SessionManager` class) that nothing imports or calls. `app.py` has no concept of multi-session history — it's single-session via `st.session_state` only.
5. **`src/cache/`, `src/knowledge_graph/`, `src/utils/` are empty directories.** Aspirational, currently dead weight.
6. **Contradiction detection is naive.** `_find_obvious_conflicts` extracts every number/year across *all* retrieved docs and flags a conflict if there's more than one distinct numeric token total — it doesn't check whether the numbers relate to the *same claim*. This will over-flag conflicts on any KB with more than one numeric fact. Fine as a v1 heuristic, but call this out explicitly as a "Limitation" in any dissertation write-up, and prioritize the LLM-refinement step (which is already there and more reliable) over the rule-based one.
7. **`main.py` is a dead stub** (`print("Hello from rag-gurkirat!")`). Either delete it or repurpose it as the project's real entry point (e.g. `uv run streamlit run app.py` launcher).
8. **`frontend/css/main.css` is an orphan.** There is no `index.html` or JS to load it — it was left over from an earlier, abandoned vanilla-JS frontend attempt. Since you've chosen to keep Streamlit, delete the `frontend/` directory entirely to avoid confusing future sessions.
9. **No tests.** No `tests/` directory, no `pytest` dependency. For an MSc submission, untested agent/graph logic is a real weakness an examiner will notice.
10. **`README.md` is essentially empty** (just a placeholder line). This is usually the first thing a marker/supervisor opens.

None of this is fatal — the architecture is sound and the multi-agent contradiction-resolution idea is a genuinely good MSc topic. The work needed is filling in real data, evaluation, and a few missing agents, not a rewrite.

---

## 2. Feature Roadmap (priority order)

| Priority | Feature | Why it matters for MSc | Effort |
|---|---|---|---|
| 🔴 P0 | Real knowledge base + contradiction dataset | Nothing else can be evaluated without real data | 0.5–1 day |
| 🔴 P0 | Evaluation question set (`data/eval/questions.jsonl`) | Needed to produce any quantitative results chapter | 0.5 day |
| 🔴 P0 | Wire up session persistence (SQLite, not in-memory) | Reproducible multi-turn demo | 0.5 day |
| ~~🟠 P1~~ ✅ | ~~Hallucination / faithfulness detector node~~ Citation-grounding verifier | Adds a genuine self-correction contribution | **DONE** — see §5.1 |
| 🟠 P1 | Confidence scoring (multi-factor) | Turns "answer" into "answer + calibrated trust signal" | 0.5 day |
| 🟠 P1 | Document upload inside Streamlit (PDF/DOCX/CSV/TXT) | Live demo-ability, examiners love uploading their own doc | 1 day |
| 🟠 P1 | Experiment logging (`data/experiments/runs.jsonl`) + history page | Lets you show *every* run you ever made, with timings | 0.5 day |
| 🟡 P2 | Hybrid retrieval: BM25 + dense + cross-encoder rerank | Classic "retrieval research depth" MSc requirement | 1.5 days |
| 🟡 P2 | Query intelligence: HyDE, decomposition, expansion | Easy, well-cited techniques to demonstrate breadth | 1 day |
| 🟡 P2 | Evaluation dashboard page (charts: multi-agent vs baseline) | Visual proof your method beats the baseline | 1 day |
| 🟡 P2 | Advanced metrics: MRR, NDCG@k, latency percentiles, ROUGE-L | Standard IR/NLG metrics expected in a dissertation | 1 day |
| 🟢 P3 | Knowledge-graph extraction (spaCy + NetworkX, GraphRAG-lite) | Strong "novelty" section if you have time | 2 days |
| 🟢 P3 | Optional thin FastAPI layer exposing the same graphs | Nice appendix ("also exposes a REST API"), not required | 1 day |
| 🟢 P3 | Dockerfile + `tests/` with pytest | Production polish, reproducibility | 1 day |

Do P0 first. The whole project is currently unusable for evaluation without it.

---

## 3. P0 — Fill the Knowledge Base & Eval Set (do this first)

### 3.1 Knowledge base content
Put real or synthetic documents into:
- `data/raw/*.txt` — "ground truth" documents (loaded with `source="real_world"` by `build_kb.py`).
- `data/contradictions/*.jsonl` — synthetic contradiction pairs, one JSON object per line:
```json
{"id": "doc_017", "text": "The Eiffel Tower was completed in 1887.", "source": "wikipedia", "label": "contradicts:doc_018"}
{"id": "doc_018", "text": "The Eiffel Tower was completed in 1889.", "source": "synthetic_contradiction", "label": "contradicts:doc_017"}
```
Aim for **at least 30–50 source docs and 15–20 deliberate contradiction pairs** (numeric, temporal, and opinion-based) so the contradiction detector and resolution agent have something real to demonstrate on.

Also add `publication_date` to metadata for documents you want `most_recent` resolution to work on — `resolution.py`'s `_choose_most_recent` already reads `metadata.get("publication_date")`, it just has nothing to read today.

Then run:
```bash
uv run python -m src.data_prep.build_kb
```

### 3.2 Evaluation question set
Create `data/eval/questions.jsonl`:
```json
{"id": "q1", "query": "When was the Eiffel Tower completed?", "ground_truth": "1889", "has_conflict": true, "should_flag_uncertain": false, "correct_doc_ids": ["doc_018"]}
{"id": "q2", "query": "What is the capital of Australia?", "ground_truth": "Canberra", "has_conflict": false}
```
Match the fields `compute_conflict_detection_rate`, `compute_resolution_quality`, and `compute_uncertain_flag_rate` in `src/evaluation/metrics.py` already expect (`has_conflict`, `should_flag_uncertain`, `correct_doc_ids`) — these functions are written but have never had real data to run against.

#### Labelling rule for `has_conflict`
**A question has `has_conflict: true` if and only if the knowledge base contains two documents that make incompatible claims ABOUT THE FACT THE QUESTION ASKS FOR.** Retrieval pulling in an unrelated contradiction pair does not make a question conflicted. The label describes the ground-truth state of the KB relative to the query, not what the system happens to retrieve or predict. Labels must never be changed to match system output — that is circular validation.

#### False-positive failure taxonomy
When the system flags a conflict that the label says shouldn't exist, the root cause falls into one of four categories:

1. **Adjacent-topic retrieval** — The retriever surfaces a contradiction pair about a semantically related but different fact. Example: query "When did WW1 begin?" retrieves the Treaty of Versailles pair (1919 vs 1920), which is about WW1's *end*, not its start.

2. **Interpretive disagreement** — The pair agrees on the factual answer to the question but disagrees on framing, originality, or attribution. Example: query "What year did Jenner develop the first vaccine?" retrieves pair 23a/23b — both say 1796, but they disagree on whether the technique was truly original.

3. **Partial-overlap pair** — Two documents agree on one claim and conflict on another, where the label depends on which claim the question targets. Example: geography_contr_29a says the Great Barrier Reef is *"2,300 kilometres"* long and *"clearly visible from space"*; 29b says it is *"1,430 miles"* (≈2,301 km) long but *"not visible to the naked eye from orbit."* The length claims agree after unit conversion; the visibility claims directly contradict. A question asking "How long is the Great Barrier Reef?" has `has_conflict: false` (no incompatible claim on length). A question asking "Is the Great Barrier Reef visible from space?" has `has_conflict: true` (direct contradiction on the asked fact).

4. **Unit mismatch (apparent only)** — Documents state the same measurement in different unit systems (km vs miles, Celsius vs Fahrenheit). After conversion they agree. This is NOT a genuine contradiction but an NLI model may flag it due to different surface-level numbers.

Then:
```bash
uv run python -m src.evaluation.run_eval
```

#### Benchmark construction history (for methodology section)

The evaluation benchmark was built iteratively, not in a single pass:

1. **Initial set (20 questions)**: 10 conflict, 10 non-conflict. Labels assigned by manual inspection of the KB before any system run.
2. **Expansion to 50 questions**: 30 additional questions added (18 conflict targeting the new harder contradiction pairs 19–34, 5 out-of-KB, 7 additional non-conflict). Labels assigned before running the system against them.
3. **Labelling rule formalised** (after a circular-validation incident where 8 labels were changed to match system predictions, then reverted): the rule in §3.2 above was written and applied retrospectively to all 50 items.
4. **Label corrections under the rule (5 total)**:
   - `eval_nonconflict_18` → `has_conflict: true`. Reason: science_contr_03a says "37.2 trillion" cells, 03b says "75 trillion" — incompatible claims about the asked fact (cell count).
   - `eval_nonconflict_13` → `has_conflict: true`. Reason: technology_contr_07a says Google founded "in 1998", 07b says "in 1996" — incompatible claims about the asked fact (founding year).
   - `eval_conflict_12` → `has_conflict: false`. Reason: geography_contr_20a says "21,196 kilometres", 20b says "13,171 miles" (= 21,196 km). No incompatible claim about the asked fact (length). This is a partial-overlap pair — both docs also claim visibility from space, agreeing on that too.
   - `eval_conflict_21` → `has_conflict: false`. Reason: geography_contr_29a says "2,300 kilometres", 29b says "1,430 miles" (≈ 2,301 km). No incompatible claim about the asked fact (length). The visibility claims DO conflict but the question asks about length.
   - `eval_nonconflict_03`: removed `out_of_kb: true`. Reason: the fact ("149.6 million kilometers") exists verbatim in science_01 through science_10.
5. **Two questions added (50 → 52)**: `eval_conflict_28` ("Is the Great Barrier Reef visible from space?", `has_conflict: true`) and `eval_nonconflict_19` ("Is the Great Wall of China visible from space?", `has_conflict: false`) — added to ensure the genuine visibility contradiction in pair 29a/29b is measured.

**Final composition**: 52 questions — 28 conflict, 24 non-conflict, 6 out-of-KB.

The rule was applied *before* seeing system output for newly-added questions, and corrections to existing labels were justified solely from KB document text, never from system predictions. The one exception (eval_nonconflict_13) was identified via the FP analysis output — transparency requires noting this, though the label change is independently justified by the rule and the quoted document text.

### 3.3 Persistent session storage
Replace the in-memory `SessionManager` placeholder with a SQLite-backed one (see `src/session/session_manager.py`):
```python
# src/session/session_manager.py
import sqlite3, uuid, json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/sessions.db")

class SessionManager:
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY, created_at TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS messages(
                id TEXT PRIMARY KEY, session_id TEXT, role TEXT,
                content TEXT, metadata TEXT, timestamp TEXT)""")

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO sessions VALUES (?, ?)", (sid, datetime.utcnow().isoformat()))
        return sid

    def add_message(self, session_id: str, role: str, content: str, metadata: dict | None = None):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), session_id, role, content, json.dumps(metadata or {}), datetime.utcnow().isoformat()),
            )

    def get_history(self, session_id: str, last_n: int = 20) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT role, content, metadata, timestamp FROM messages WHERE session_id=? ORDER BY timestamp ASC LIMIT ?",
                (session_id, last_n),
            ).fetchall()
        return [{"role": r[0], "content": r[1], "metadata": json.loads(r[2]), "timestamp": r[3]} for r in rows]
```
Wire it into `app.py`: create a session on first load (`st.session_state.setdefault("session_id", manager.create_session())`), call `add_message` after every user/assistant turn, and offer a "Past sessions" picker in the sidebar.

---

## 4. Streamlit Colorful UI (already applied to `app.py`)

`app.py` now uses a custom CSS theme injected via `st.markdown(..., unsafe_allow_html=True)`:

- **Background**: dark base (`#0b0f1a`) with a soft pink/indigo/teal "aurora" radial-gradient glow — colorful but still readable.
- **Hero header**: gradient text title (`🧠 Multi-Agent RAG Lab`) using a 5-color `linear-gradient` + `background-clip: text`.
- **Chat bubbles**: custom HTML bubbles (not `st.chat_message`) — user = pink→orange gradient pill, assistant = indigo/teal glass card — replacing the previous plain `st.chat_message` calls.
- **Badges**: pill badges for conflict/faithfulness status (`badge-ok` green/teal, `badge-warn` amber, `badge-conflict` red/pink).
- **KPI cards**: colorful gradient cards for Recall/Precision instead of plain `st.metric`.
- **Document cards**: glass cards with a green glow border for "chosen" docs and a red glow for unresolved-conflict docs.
- **Buttons**: gradient pill buttons with a hover lift + glow.

Helper functions added at the top of `app.py`: `esc()`, `badge()`, `kpi_card()`, `chat_bubble()` — reuse these for any new page/feature so the visual language stays consistent.

### 4.1 Next step: convert to a multi-page Streamlit app
Streamlit supports native multipage apps via a `pages/` directory next to `app.py`. Recommended pages:

```
RAG-gurkirat/
├── app.py                      # "Chat" — keep as the main/home page
└── pages/
    ├── 1_📚_Knowledge_Base.py   # upload docs, view KB stats, re-index button
    ├── 2_📊_Evaluation.py       # run_eval trigger + colorful charts (multi-agent vs baseline)
    ├── 3_🕘_Experiment_History.py  # table from data/experiments/runs.jsonl, with filters
    └── 4_ℹ️_About.py            # architecture diagram + MSc research contributions
```
Each page should `import` the same CSS block (factor it into `src/utils/theme.py` returning the `<style>` string, so you only maintain one copy) and reuse `badge()` / `kpi_card()` from a shared module, e.g. `src/utils/ui_components.py`, instead of duplicating them per page.

### 4.2 Charts for the Evaluation page
Use Plotly (colorful, interactive, good for a dissertation screenshot) with a vivid qualitative palette:
```python
import plotly.express as px
fig = px.bar(
    df, x="metric", y="score", color="system",
    color_discrete_map={"multi_agent": "#8b5cf6", "baseline": "#14b8a6"},
    barmode="group", template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)
```

---

## 5. P1 — Self-Correction & Trust Signals

### 5.1 Hallucination / faithfulness detector — ✅ IMPLEMENTED (citation-grounding verifier)

**Status: done.** Implemented as `src/agents/grounding_verifier.py`, wired into
`src/graphs/multi_agent_graph.py` as a new terminal node
(`grounding_verification`) running after `answer_generation`, before `END`.

This went further than the original plan below: instead of a generic
faithfulness re-generation loop, it's a **citation-grounding verifier**
motivated directly by a real, captured failure in this project's own eval
data — `eval_nonconflict_03` in `data/eval/multi_agent_records_explain_both.json`,
where the answer fabricated entire fake documents (`Doc id=astronomy_01/02/03`)
with realistic formatting to justify an answer not actually grounded in the
real retrieved context.

How it works:
1. **Deterministic check (free, instant)**: extracts every `Doc id=X` the
   final answer cites via regex and compares against the IDs that were
   *actually* retrieved for that query. Any cited ID that doesn't exist in
   the real set is a fabricated citation — this alone catches the
   astronomy_01/02/03 pattern with zero extra LLM calls.
2. **Semantic grounding check (one LLM call)**: only runs if step 1 passes —
   verifies the answer's claims are genuinely supported by the retrieved
   text rather than outside/parametric knowledge.

Results populate `state["faithful"]`, `state["faithfulness_notes"]` (already
read by `app.py`'s insights panel — no UI plumbing was needed for the basic
badge), and `state["fabricated_citations"]` (surfaced as its own 🚨 badge in
`app.py` and in the downloadable session report).

**Cost note**: this adds one extra LLM call per query (skipped only when a
fabrication is already caught deterministically) — a full 20-question eval
run now uses noticeably more Groq quota than before. Pace accordingly.

**Known limitation (by design)**: the citation regex only recognizes the
`Doc id=X` / `Document Id: X` format — the exact format the answer-generation
prompt uses, and the exact format of the proven fabrication case. It does
*not* catch doc IDs cited in other styles (e.g. `('science_contr_01a')`
parenthetical style, which real answers sometimes use). This is a deliberate
high-precision/lower-recall tradeoff — broadening the regex risks false
positives on legitimate prose. Worth revisiting if a second real fabrication
example surfaces in a different citation style.

**Original plan (superseded by the above, kept for reference):**
New file `src/agents/hallucination_detector.py`:
```python
from typing import List
import json
from src.models.llm import get_chat_llm
from src.agents.schemas import RetrievedDoc

class HallucinationDetector:
    def check_faithfulness(self, query: str, answer: str, docs: List[RetrievedDoc]) -> dict:
        context = "\n\n".join(f"[{d.id}]: {d.text}" for d in docs)
        prompt = f"""Verify whether the answer is faithful to the context.
Question: {query}
Context: {context}
Answer: {answer}
Return ONLY JSON: {{"is_faithful": true/false, "faithfulness_score": 0.0-1.0,
"unfaithful_claims": [...], "grounded_claims": [...]}}"""
        resp = get_chat_llm().invoke(prompt)
        try:
            return json.loads(resp.content)
        except Exception:
            return {"is_faithful": True, "faithfulness_score": 1.0, "unfaithful_claims": [], "grounded_claims": []}

    def regenerate_if_unfaithful(self, query, answer, docs, max_attempts: int = 2):
        for _ in range(max_attempts):
            result = self.check_faithfulness(query, answer, docs)
            if result["faithfulness_score"] >= 0.85:
                return answer, result["faithfulness_score"]
            context = "\n\n".join(d.text for d in docs)
            prompt = (f"Previous answer had ungrounded claims: {result['unfaithful_claims']}\n"
                      f"Rewrite using ONLY the context below.\nQuestion: {query}\nContext: {context}\nFaithful answer:")
            answer = get_chat_llm().invoke(prompt).content
        return answer, result["faithfulness_score"]
```
Add it as a new terminal node in `src/graphs/multi_agent_graph.py` after `answer_generation`, populating `state["faithful"]` and `state["faithfulness_notes"]` — `app.py`'s insights panel already reads exactly those two keys, so the UI will light up automatically once this node exists.

### 5.2 Confidence scoring
New file `src/agents/confidence_scorer.py` — combine retrieval similarity, faithfulness score, and conflict-resolution penalty into one 0–1 "confidence" with a HIGH/MEDIUM/LOW/VERY_LOW label (formula and code identical in spirit to a standard weighted composite — see the project's evaluation metrics for the existing scoring conventions). Surface it in `app.py` as another colorful badge next to the faithfulness badge.

### 5.3 Document upload
Add a page (`pages/1_📚_Knowledge_Base.py`) using `st.file_uploader(accept_multiple_files=True, type=["txt","pdf","docx","csv"])`. For non-`.txt` files, add a small loader module `src/data_prep/document_loader.py` using `pypdf` (PDF) and `python-docx` (DOCX), chunk with `langchain.text_splitter.RecursiveCharacterTextSplitter`, then call the existing `add_documents()` in `src/retrieval/vector_store.py` — no need to touch the vector store code.

---

## 6. P2 — Retrieval & Query Quality

### 6.1 Hybrid retrieval (BM25 + dense + rerank)
New file `src/retrieval/hybrid_retriever.py` using `rank-bm25` for sparse retrieval over the same corpus, Reciprocal Rank Fusion to merge with the existing dense Chroma results, and `sentence-transformers` `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` to rerank the merged top-20 down to top-k. Expose a `use_hybrid: bool` toggle in the Streamlit sidebar.

### 6.2 Query intelligence (HyDE / decomposition / expansion)
New file `src/agents/query_processor.py` with three small LLM-prompted methods: `hyde_query()`, `decompose_query()`, `expand_query()`. Add sidebar toggles in `app.py` for each — purely additive, don't change default behaviour unless toggled on.

### 6.3 Advanced evaluation metrics
Extend `src/evaluation/metrics.py` with `compute_mean_reciprocal_rank`, `compute_ndcg`, `compute_latency_percentiles`, and (optional, heavier dependency) `compute_bleu_rouge` using `rouge-score` / `sacrebleu`. Surface all of these on the new Evaluation page as colorful KPI cards + a Plotly radar chart comparing baseline vs multi-agent across every metric at once — this single chart is usually the most-cited figure in an MSc results chapter.

---

## 7. P3 — Stretch / Research Novelty

- **Knowledge graph (GraphRAG-lite)**: `src/knowledge_graph/extractor.py` using `spacy` for entity extraction + an LLM prompt for `(subject, relation, object)` triples, stored in a `networkx.DiGraph`. Add a graph-browsing page that renders the ego-graph of an entity (Plotly or `pyvis` both render fine inside Streamlit via `st.components.v1.html`).
- **Optional FastAPI layer**: only if you want to demonstrate the pipeline is "productionizable" beyond Streamlit. Wrap `build_multi_agent_app()` / `build_baseline_app()` behind two endpoints (`/query/multi-agent`, `/query/baseline`) and a `/health` check. This is genuinely optional — Streamlit alone is a complete, demoable system.
- **Tests**: add `tests/` with `pytest`, covering at minimum: `contradiction_detection_agent` on a known conflicting pair, `resolution_agent` for each strategy, and `compute_*` metric functions against hand-built fixtures. Add `pytest>=8.0.0` to `pyproject.toml`.
- **Docker**: a simple `Dockerfile` (`uv sync --frozen` + `CMD ["uv","run","streamlit","run","app.py"]`) is enough for reproducibility appendix purposes.

---

## 8. Updated `pyproject.toml` dependencies

```toml
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
    "streamlit>=1.38.0",
    "plotly>=5.24.0",
    "rank-bm25>=0.2.2",
    "pypdf>=5.0.0",
    "python-docx>=1.1.0",
    "loguru>=0.7.2",
    "networkx>=3.3",
    "spacy>=3.7.0",
    "rouge-score>=0.1.2",
    "sacrebleu>=2.4.0",
    "pytest>=8.0.0",
]
```
Add P2/P3 packages only when you actually start that section, to keep `uv sync` fast while iterating on P0/P1.

---

## 9. Implementation Rules

1. Never break the existing LangGraph graphs or CLI entry points while adding new nodes — add new terminal/optional nodes, don't rewrite the existing 4-stage pipeline.
2. All new config (feature toggles, TTLs, model names) goes in `src/config/config.py` + `.env`, never hard-coded.
3. Reuse `esc()`, `badge()`, `kpi_card()`, `chat_bubble()` from `app.py` (factor them into `src/utils/ui_components.py` once you add a second page) so every page shares the same colorful visual language.
4. Every new agent function should have a docstring stating its algorithm/source (e.g. "RRF fusion, Cormack et al. 2009") — dissertations need citations, code comments are the easiest place to keep them attached to the implementation.
5. Delete dead code as you go: `main.py` stub, `frontend/` directory, the unused `model` dropdown (or wire it up — don't leave it cosmetic).

---

## 10. Environment Variables (`.env`)

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
VECTOR_STORE_DIR=data/index/chroma_db
COLLECTION_NAME=multi_agent_rag
TOP_K=5
FAITHFULNESS_ENABLED=true
```

---

## 11. Run Instructions

```bash
cd "C:\Users\BEST LAPTOP\Desktop\RAG-gurkirat"
uv sync
uv run python -m src.data_prep.build_kb        # P0: index data/raw + data/contradictions
uv run python -m src.evaluation.run_eval       # P0: run eval set, print metrics
uv run streamlit run app.py                    # launch the colorful UI
```

---

## 12. MSc Research Contributions (for the dissertation)

1. **Self-correcting multi-agent pipeline**: retrieve → detect → resolve → generate (→ verify → score, once Section 5 is implemented), vs a 2-stage baseline — a clear, measurable architectural contribution.
2. **Hybrid contradiction detection**: rule-based numeric heuristics + LLM semantic analysis — discuss the over-flagging limitation found in Section 1 (#6) honestly; that's a legitimate "limitations and future work" point.
3. **Configurable conflict-resolution strategies** (`most_recent` / `most_authoritative` / `explain_both`) as an explicit, comparable design space rather than one fixed policy.
4. **Faithfulness-gated self-correction loop** (Section 5.1) — measurable reduction in hallucination rate, a popular and well-cited MSc contribution pattern.
5. **Comparative evaluation**: multi-agent vs baseline across accuracy, conflict detection rate, resolution quality, uncertainty flag rate, and (once added) MRR/NDCG/latency — your `metrics.py` already defines four of these; you just need real data to run them on.

---

## 13. Feature Expansion: Web Search Fallback, Document Upload, Agent Trace

**Context**: the project owner asked for unrestricted "knowledge of everything"
and always-current information. That was deliberately scoped down in
discussion with Claude (chat assistant) before implementation, because it
directly conflicts with the controlled, bounded-knowledge-base design this
project's evaluation results depend on (resolution quality, conflict
detection precision/recall are all measured against a known corpus with
known correct answers — "the agent can answer literally anything from the
open internet" makes those numbers meaningless). The resolution: keep the
bounded KB as the rigorous default mode, and add internet search as an
explicit, opt-in, clearly-labeled **fallback** — never silent, never default.

### 13.1 Web search fallback (opt-in, permission-gated)

**Library**: `tavily-python` (`pip`/`uv` package `tavily-python`, import as
`from tavily import TavilyClient`). Free tier: **1,000 API credits/month, no
credit card required** (basic search = 1 credit/request). Requires signup at
https://tavily.com and a `TAVILY_API_KEY` added to `.env`. Chosen over
DuckDuckGo because Tavily returns clean, structured, LLM-ready content
(summaries + citations) rather than raw search-result links, which fits
this project's existing citation-grounding pattern much better.

**Flow**:
1. The grounding verifier (`src/agents/grounding_verifier.py`) already
   detects when an answer isn't grounded in retrieved documents. Reuse that
   signal: if `faithful == False` or `fabricated_citations` is non-empty
   AND the answer has no real supporting doc, do NOT silently answer from
   parametric knowledge.
2. Instead, set a new state field `state["web_search_offer"] = True` with a
   message like: *"This isn't in my knowledge base. Want me to search the
   internet for it?"* Surface this as a yes/no prompt in the Streamlit UI
   (e.g. `st.button("🌐 Search the internet")`), not an automatic action.
3. Only if the user clicks yes, call a new `src/agents/web_search_agent.py`:
   ```python
   import os
   from tavily import TavilyClient

   def web_search(query: str, max_results: int = 5) -> list[dict]:
       client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
       response = client.search(query=query, max_results=max_results)
       # response["results"] is a list of dicts with title/url/content/score
       return response.get("results", [])
   ```
   Add `tavily-python` to `pyproject.toml` and `TAVILY_API_KEY=tvly-...` to
   `.env` / `.env.example`. Do NOT commit a real key.
4. Pass the search results back through the SAME answer-generation +
   grounding-verification pattern, but tag the answer clearly:
   `"🌐 From the internet (not the knowledge base):"` prefix, and tag the
   source as `source: "web_search"` in any saved record — never blend web
   results into the knowledge-base provenance fields (`chosen_doc_ids`,
   `correct_doc_ids` etc.), since those are used by the eval metrics and
   must stay meaningful for the bounded-KB evaluation.
5. **Browsing animation**: while `web_search()` runs, show a simple
   Streamlit spinner (`st.spinner("🌐 Searching the web...")` is sufficient —
   don't over-engineer this into a fake browser-chrome animation, it adds
   complexity for no real benefit over Streamlit's built-in spinner).
6. **Cost awareness**: each search call costs 1 Tavily credit (1,000/month
   free). This is generous relative to the Groq token limits this project
   has repeatedly hit, but it's still a real ceiling — don't call
   web_search() in a loop or as part of routine eval runs; it should only
   ever fire on an explicit user click.

**Do not** wire this into the default `multi_agent_graph.py` pipeline as an
automatic step. It must remain a manually-triggered, separate path so the
core evaluation pipeline (used for all the accuracy/precision/recall numbers
in this document) stays unchanged and re-runnable exactly as before.

### 13.2 Document upload + cross-questioning (refines §5.3)

Add `pages/1_📚_Knowledge_Base.py` (Streamlit native multipage, per §4.1):
- `st.file_uploader(accept_multiple_files=True, type=["txt","pdf","docx"])`
- For PDF: `pypdf.PdfReader`. For DOCX: `python-docx`. Chunk with
  `langchain.text_splitter.RecursiveCharacterTextSplitter`.
- **Important distinction for cross-questioning**: tag uploaded-doc chunks
  with `metadata={"source": "user_upload", "upload_session_id": <uuid>}`.
  When a user has an active upload in their session, bias retrieval toward
  `user_upload` chunks for that session (e.g. a metadata filter in the
  Chroma query) so follow-up questions naturally stay grounded in the
  document they just uploaded, without polluting the permanent knowledge
  base used by the eval suite.
- Do not persist uploaded documents into `data/raw/` or re-index them into
  the permanent `data/index/chroma_db` collection used by `run_eval.py` —
  use a separate, session-scoped collection (or an in-memory Chroma
  collection) so uploads can never silently change the eval benchmark.

### 13.3 Agent execution trace (addresses "verify all agents ran")

New `src/utils/trace.py`: a simple list-based trace collector. Each node in
`multi_agent_graph.py` appends one entry:
```python
state.setdefault("trace", []).append({
    "agent": "retriever",
    "summary": f"Retrieved {len(docs)} documents",
})
```
Do this in every node (`retriever_node`, `contradiction_detection_node`,
`resolution_node`, `answer_generation_node`, `grounding_verification_node`).
In `app.py`, add a new expander "🧩 Agent trace" showing the ordered list —
this is the literal answer to "verify all agents ran and show which one
did what," and it's nearly free to build since every agent's relevant
output already exists in `state` — this just narrates it in order.

### 13.4 "Always show latest information" — scope clarification

Within the bounded KB, "latest" already means: use `publication_date`
metadata and the `most_recent` resolution strategy (already implemented,
see §3.3/§6). For genuinely current real-world information, that is the
web search fallback in §13.1, not a property of the core RAG system — a
static indexed corpus cannot be "always current" by definition.

