# Claude Instructions — Self-Correcting Multi-Agent RAG for Contradictory Knowledge Bases

## Scope Summary

This project implements a self-correcting multi-agent RAG system over contradictory knowledge bases with:

- A FastAPI backend wrapping existing LangGraph graphs (multi-agent + baseline).
- Rich evaluation signals: retrieval precision/recall, faithfulness, runtime, conflict metadata.
- A pure HTML/CSS/vanilla JS frontend served by FastAPI (no Next.js/React).
- Research-grade UX: compare-both mode, experiment history, conflict/faithfulness visualization.

You must follow this file as the authoritative project specification.

---

## Existing Stack (do not modify)

- **Language / framework:** Python, LangChain, LangGraph
- **LLM backend:** Groq (`ChatGroq`)
- **Embeddings:** HuggingFace `BAAI/bge-small-en-v1.5`
- **Vector store:** Chroma
- **Agents:** retriever → contradiction detector → resolution agent → answer generator
- **Baseline:** single-pass RAG graph
- **Evaluation pipeline metrics:** answer accuracy, conflict detection rate, resolution quality, uncertain flag rate, response time

---

## Phase 0 — Plan First, Then Execute (MANDATORY)

Before writing a single line of code, you must:

1. **Print a numbered execution plan** in the terminal covering every step you will take in this session.
   The plan must explicitly list:
   - which Next.js / React files and directories will be deleted,
   - which new files will be created (with paths),
   - which existing Python files will be touched and why.
2. **Wait for user confirmation** (`y` / `yes`) before proceeding.
3. Only after confirmation, execute the plan step by step, printing a short status line (`✓ done` / `✗ failed`) for each step.

---

## Phase 1 — Remove Next.js / React Frontend

The project previously had a Next.js frontend. Remove it completely before building the new HTML/CSS/JS frontend.

### Step 1 — Audit

Run the following and print the output so the user can see what will be deleted:

```bash
# List everything that belongs to the old Next.js setup
find . \( \
  -name "package.json" -o \
  -name "package-lock.json" -o \
  -name "yarn.lock" -o \
  -name "pnpm-lock.yaml" -o \
  -name "next.config.*" -o \
  -name "next-env.d.ts" -o \
  -name "tsconfig.json" -o \
  -name "postcss.config.*" -o \
  -name "tailwind.config.*" -o \
  -name ".eslintrc*" \
\) -not -path "*/node_modules/*" | sort

# Also list these directories if they exist
for d in frontend node_modules .next out; do
  [ -d "$d" ] && echo "DIR: $d"
done
```

### Step 2 — Delete

Delete **only** the items surfaced in Step 1. Do not touch `src/`, `data/`, or any Python file.

```bash
# Directories
rm -rf frontend/ node_modules/ .next/ out/

# Root-level JS/TS config files (adjust list to match what Step 1 found)
rm -f package.json package-lock.json yarn.lock pnpm-lock.yaml \
      next.config.js next.config.ts next.config.mjs \
      next-env.d.ts tsconfig.json \
      postcss.config.js postcss.config.mjs \
      tailwind.config.js tailwind.config.ts \
      .eslintrc .eslintrc.js .eslintrc.json .eslintrc.yaml
```

### Step 3 — Verify

Confirm nothing Python-related was deleted:

```bash
# These must still exist after cleanup
ls src/ data/ README.md 2>/dev/null && echo "✓ Python project intact"
```

---

## Implementation Rules (apply to every task below)

1. **Never** rewrite or break existing LangGraph graphs, agents, or evaluation scripts.
2. Prefer adding new modules / files; minimise edits to existing ones.
3. All configuration goes in `src/config/config.py` and `.env`; no hard-coded values.
4. The frontend must talk **only** to FastAPI endpoints — no direct Python imports.
5. Keep commits small and focused; comment every section that touches LangGraph or evaluation code.
6. Handle errors explicitly: return appropriate HTTP status codes and JSON error bodies instead of crashing.

---

## Backend API File Layout

Use this structure for clarity:

- `src/api/main.py`: FastAPI app factory, mounts routers and static files.
- `src/api/routes/query.py`: multi-agent and baseline query endpoints (sync + streaming).
- `src/api/routes/eval.py`: `/api/eval/results`, `/api/experiments`, `/api/health`.
- `src/api/routes/documents.py`: document upload and listing endpoints.
- `src/api/utils/experiments.py`: helper functions for appending/reading `runs.jsonl`.
- `src/api/utils/retrieval_metrics.py`: helper to compute recall/precision from eval data.

---

## Task 1 — FastAPI Backend Layer

Create or update `src/api/main.py` and route modules. Wrap existing graphs behind these endpoints.

### Core Endpoints & JSON Contracts

#### `POST /api/query/multi-agent`

Request:

```json
{
  "query": "string",
  "strategy": "most_recent | most_authoritative | explain_both",
  "top_k": 5,
  "model": "llama3-70b-8192"
}
```

Response:

```json
{
  "answer": "string",
  "has_conflict": true,
  "conflict_type": "numeric | temporal | opinion | other | null",
  "chosen_doc_ids": ["doc1"],
  "flagged_uncertain": true,
  "retrieved_docs": [
    { "id": "doc1", "text": "...", "metadata": { "source": "...", "topic_id": "..." } }
  ],
  "retriever_recall": 0.8,
  "retriever_precision": 0.6,
  "faithful": true,
  "faithfulness_notes": "string",
  "runtime_ms": 1240
}
```

Notes:

- `retriever_recall` / `retriever_precision`:
  - Populated only when the query matches an entry in `data/eval/questions.jsonl`, else `null`.
- `faithful` / `faithfulness_notes`:
  - Populated only when the faithfulness checker is enabled in config, else `null`.

Error responses:

- `422` – validation error (FastAPI default).
- `500` – graph execution error (include `{ "error": "string" }`).
- `503` – LLM or vector store unavailable (include `{ "error": "backend unavailable" }`).

#### `POST /api/query/baseline`

Request:

```json
{
  "query": "string",
  "top_k": 5,
  "model": "llama3-70b-8192"
}
```

Response:

```json
{
  "answer": "string",
  "retrieved_docs": [
    { "id": "doc1", "text": "...", "metadata": {} }
  ],
  "retriever_recall": null,
  "retriever_precision": null,
  "runtime_ms": 430
}
```

Notes:

- `retriever_recall` / `retriever_precision`:
  - If eval data is available and applicable, you may compute them, otherwise leave `null`.
- Baseline endpoint must ignore any `strategy` field (frontend must not send it).

Error responses same as multi-agent.

#### `POST /api/query/stream` (Server-Sent Events)

Streaming endpoint for multi-agent responses.

- Request body:

```json
{
  "query": "string",
  "strategy": "most_recent | most_authoritative | explain_both",
  "top_k": 5,
  "model": "llama3-70b-8192"
}
```

- Response: `Content-Type: text/event-stream`.

SSE events:

- Token events:

```text
data: {"type": "token", "value": "partial text..."}
```

- Final event:

```text
data: {
  "type": "done",
  "answer": "full answer",
  "has_conflict": true,
  "conflict_type": "numeric | temporal | opinion | other | null",
  "chosen_doc_ids": ["doc1"],
  "flagged_uncertain": true,
  "retrieved_docs": [...],
  "retriever_recall": 0.8,
  "retriever_precision": 0.6,
  "faithful": true,
  "faithfulness_notes": "string",
  "runtime_ms": 1240
}
```

Frontend must use `EventSource` (or equivalent) to stream tokens into the assistant bubble; skeleton shimmer is visible only until the first token arrives.

#### `GET /api/eval/results`

Returns the aggregated metrics from the existing evaluation pipeline (read-only; do not re-run it).

Example response:

```json
{
  "answer_accuracy": 0.89,
  "conflict_detection_rate": 0.72,
  "resolution_quality": 0.91,
  "uncertainty_flag_rate": 0.15,
  "avg_response_time": 250
}
```

#### `GET /api/experiments`

Serves the contents of `data/experiments/runs.jsonl` as a JSON array for the Experiments UI view.

- If `data/experiments/runs.jsonl` does not exist yet, return an empty array `[]`.

Example response:

```json
[
  {
    "timestamp": "2026-06-09T12:34:56Z",
    "query": "string",
    "system_type": "baseline",
    "strategy": null,
    "model": "llama3-70b-8192",
    "answer": "string",
    "has_conflict": false,
    "conflict_type": null,
    "chosen_doc_ids": [],
    "flagged_uncertain": false,
    "faithful": null,
    "retriever_recall": null,
    "retriever_precision": null,
    "runtime_ms": 430
  }
]
```

#### `GET /api/health`

Health check endpoint.

Example response (200):

```json
{
  "status": "ok",
  "llm_backend": "up",
  "vector_store": "up",
  "version": "1.0.0"
}
```

If any core component is unavailable, return `503` with:

```json
{
  "status": "degraded",
  "error": "LLM backend unavailable"
}
```

Use this endpoint for frontend preflight checks.

#### `POST /api/documents/upload`

Enables document upload for MSc-level demos.

- `Content-Type: multipart/form-data`.
- Fields:
  - `file`: uploaded document (PDF, TXT, etc.).
  - `source`: optional string, default `"user_upload"`.

Response example:

```json
{
  "status": "ok",
  "doc_id": "string",
  "message": "Indexed successfully"
}
```

#### `GET /api/documents`

Lists indexed documents:

```json
[
  {
    "id": "string",
    "source": "user_upload",
    "created_at": "ISO-8601"
  }
]
```

---

### App Mounting and Static Files

After defining all API routes, mount the frontend as static:

```python
from fastapi.staticfiles import StaticFiles

# Mount last so API routes take priority
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
```

- Visiting `http://localhost:8000/` serves `frontend/index.html`.
- Visiting `http://localhost:8000/experiments` serves `frontend/experiments.html`.

Add a startup check to ensure APIs are available:

```python
@app.on_event("startup")
async def verify_routes():
    routes = [r.path for r in app.routes]
    assert "/api/query/multi-agent" in routes
```

### CORS (if needed)

If the frontend is served by FastAPI from the same origin, CORS is not required.  
If during development you serve the frontend from another origin, configure:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Task 2 — Conflict-Handling Strategy (backend)

Extend the multi-agent graph’s `resolution_agent` to read a `strategy` field from graph state.

| Strategy           | Behaviour                                                                 |
|--------------------|---------------------------------------------------------------------------|
| `most_recent`      | Prefer docs with the largest `publication_date` in metadata              |
| `most_authoritative` | Prefer docs whose `source` appears earliest in `["real_world", "wikipedia", "synthetic"]` |
| `explain_both`     | Leave conflict unresolved; describe both sides in the answer             |

Rules:

- `strategy` is passed in via `/api/query/multi-agent` and via the streaming endpoint, stored in the initial graph state.
- If `publication_date` is missing for all documents, `most_recent` should fall back to `explain_both` (do not pick arbitrarily).
- For `most_authoritative`, if `source` is missing, treat it as lowest priority.
- When `strategy = "explain_both"`, resolution agent must:
  - Mark conflicts as unresolved,
  - Ensure `flagged_uncertain = true`,
  - Include both sides explicitly in the final answer.

The baseline graph must ignore `strategy`. The frontend must not send `strategy` in baseline mode.

---

## Task 3 — Faithfulness / Hallucination Checker (backend)

Controlled by `FAITHFULNESS_ENABLED=true/false` in config (`src/config/config.py`).

When enabled, add a **terminal node** to the multi-agent graph that:

1. Receives:
   - `final_answer`,
   - `retrieved_evidence_texts` (concatenation or list of retrieved docs).
2. Calls an LLM once more with a dedicated prompt asking for:

```json
{
  "faithful": true,
  "notes": "string"
}
```

3. Merges the result into graph state:

- `faithful`: boolean or `null`,
- `faithfulness_notes`: string or `null`.

Rules:

- Implement as an extra node or a small wrapper function only; do not modify existing nodes.
- When `FAITHFULNESS_ENABLED=false`, ensure:
  - API responses return `faithful: null` and `faithfulness_notes: null`.
  - Experiment logs also store `faithful: null`.

---

## Task 4 — Retrieval Diagnostics (backend)

In the API layer (not inside the evaluation pipeline), for each query:

1. Check if `query` matches an entry in `data/eval/questions.jsonl`.
   - Matching by exact string is sufficient; if needed, you may normalize whitespace.
2. If found, compute:

   - `retriever_recall`:
     - `1` if any retrieved doc’s `topic_id` or `id` is among the ground-truth `correct_doc_ids`.
     - `0` otherwise.
   - `retriever_precision`:
     - `(# retrieved docs whose topic_id/id is in correct_doc_ids) / (total retrieved docs)`.

3. If not found, return `null` for both fields.

Implementation detail:

- Place this logic in `src/api/utils/retrieval_metrics.py` and call it from the query endpoints.

---

## Task 5 — Experiment Logging (backend)

After every API call to either query endpoint (multi-agent, baseline, and streaming completion), append one JSONL record to `data/experiments/runs.jsonl`:

```json
{
  "timestamp": "ISO-8601",
  "query": "string",
  "system_type": "baseline | multi_agent",
  "strategy": "most_recent | most_authoritative | explain_both | null",
  "model": "llama3-70b-8192",
  "answer": "string",
  "has_conflict": true,
  "conflict_type": "numeric | temporal | opinion | other | null",
  "chosen_doc_ids": ["doc1"],
  "flagged_uncertain": false,
  "faithful": true | false | null,
  "retriever_recall": 0.8,
  "retriever_precision": 0.6,
  "runtime_ms": 1240
}
```

Notes:

- When `FAITHFULNESS_ENABLED=false`, `faithful` must be `null`.
- For baseline runs, `has_conflict` should generally be `false`, `conflict_type` `null`, `chosen_doc_ids` empty.
- Ensure the file is created if it does not exist; append one JSON object per line.

---

## Task 6 — HTML / CSS / JS Frontend (`frontend/`)

No framework. Pure HTML5 + CSS3 + Vanilla JS. FastAPI serves the static files directly — no separate dev server.

### Serving Static Files via FastAPI

In `src/api/main.py`, mount the frontend folder **after** all API routes:

```python
from fastapi.staticfiles import StaticFiles

# Mount last so API routes take priority
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
```

- `http://localhost:8000/` → `frontend/index.html`.
- `http://localhost:8000/experiments` → `frontend/experiments.html`.

---

### File Structure

```text
frontend/
├── index.html          # Main chat + evidence page
├── experiments.html    # Experiment history page
├── css/
│   ├── theme.css       # CSS custom properties (gradient palette, typography, spacing)
│   └── main.css        # All component styles — imports theme.css
└── js/
    ├── api.js          # All fetch() calls to FastAPI endpoints (one function per endpoint)
    ├── chat.js         # Chat panel logic (render bubbles, handle send, compare mode, streaming)
    ├── evidence.js     # Evidence panel logic (render doc cards, badges, progress bars)
    └── experiments.js  # Experiments page (fetch, render table, client-side filters)
```

Use ES modules (`type="module"`) and avoid global variables except a single `window.RAG_APP` namespace if needed.

---

### Gradient Design System (`css/theme.css`)

Define everything as CSS custom properties so the whole theme is changed in one file:

```css
:root {
  /* --- Gradient palette --- */
  --grad-primary:   linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
  --grad-accent:    linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
  --grad-success:   linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
  --grad-surface:   linear-gradient(160deg, #1e1b4b 0%, #0f172a 100%);

  /* --- Solid colours derived from the palette --- */
  --color-primary:   #7c3aed;
  --color-accent:    #06b6d4;
  --color-success:   #10b981;
  --color-warning:   #f59e0b;
  --color-danger:    #ef4444;

  /* --- Backgrounds --- */
  --bg-page:       #0a0a1a;
  --bg-surface:    rgba(30, 27, 75, 0.6);
  --bg-input:      rgba(255,255,255,0.05);

  /* --- Text --- */
  --text-primary:  #f1f5f9;
  --text-muted:    #94a3b8;

  /* --- Borders --- */
  --border-default:  rgba(255,255,255,0.08);
  --border-chosen:   #10b981;
  --border-conflict: #ef4444;

  /* --- Blur / glass --- */
  --glass-blur: blur(16px);

  /* --- Typography --- */
  --font-sans: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

  /* --- Spacing scale (8-point grid) --- */
  --space-1: 4px;  --space-2: 8px;  --space-3: 12px;
  --space-4: 16px; --space-6: 24px; --space-8: 32px;

  /* --- Border radius --- */
  --radius-sm: 8px; --radius-md: 12px; --radius-lg: 20px; --radius-pill: 9999px;

  /* --- Transitions --- */
  --transition: 0.2s ease;
}
```

- Load **Inter** and **JetBrains Mono** from Google Fonts in `<head>`.
- Dark mode is default.
- `.light` class on `<body>` overrides only background and text colours; gradients remain the same.

---

### Layout (`index.html`)

Overall layout:

```text
┌────────────────────────────────────────────────────────────┐
│  Header — gradient logo + nav tabs + theme toggle         │
├──────────────────────────────┬────────────────────────────┤
│  Chat Panel (flex-grow)      │ Evidence & Analysis Panel │
│  [message bubbles]           │ [status badges]           │
│                               │ [doc cards]              │
│  [Controls toolbar]          │                           │
└──────────────────────────────┴────────────────────────────┘
```

- Header: full-width bar with `background: var(--grad-primary)`, white text, link to Experiments page, theme toggle button.
- Chat Panel: left, ~60% width, scrollable.
- Evidence Panel: right, ~40% width, glassmorphism card with independent scroll.
- On mobile (< 768 px): panels stack vertically; evidence panel moves below chat panel.

---

### Chat Panel — Message Bubbles

**User bubble** (right-aligned):

- Background: `var(--bg-input)` with `border: 1px solid var(--border-default)`.
- Rounded corners (`border-radius: var(--radius-md)`).

**Assistant bubble** (left-aligned):

- Background: `var(--grad-primary)`.
- White text, subtle drop shadow.
- When `has_conflict: true` → include:

```html
<span class="badge badge-conflict">⚡ Conflict</span>
```

- When `flagged_uncertain: true` → include:

```html
<span class="badge badge-warn">⚠ Uncertain</span>
```

- Timestamp in muted text below (`var(--text-muted)`).

**Skeleton loader**:

- While waiting for first token or full response, show a shimmering placeholder bubble with CSS keyframes.

---

### Controls Toolbar (bottom of Chat Panel)

Sticky toolbar at the bottom:

```text
[ Text input (flex-grow) ] [ Send ▶ ]
[ Mode ▾ ] [ Strategy ▾ ] [ top_k: 5 ] [ Model ▾ ]
```

Implementation details:

- Send button: pill shape, `background: var(--grad-primary)`, white text.
- Dropdowns:
  - Use native `<select>` styled with `var(--bg-input)` and `var(--border-default)`.
- Mode options: `Multi-agent`, `Baseline`, `Compare Both`.
- Strategy options: `Most Recent`, `Most Authoritative`, `Explain Both`.
  - When Mode = Baseline, the Strategy dropdown:
    - is visually disabled (`opacity: 0.4; pointer-events: none`),
    - and **must not** send `strategy` in the request body.
- `top_k`: `<input type="number" min="1" max="20" value="5">`.
- `Model` dropdown: `llama3-70b-8192`, `llama3-8b-8192`, `mixtral-8x7b-32768`.

---

### Session History (frontend-only)

To avoid losing chat on refresh:

- On first load, `chat.js` generates a `session_id` (UUID) and stores it in `localStorage["rag-session-id"]`.
- Messages are stored in `localStorage["rag-session-" + session_id]` as a JSON array of message objects.
- On page load, `chat.js`:
  - Reads `session_id`,
  - Loads the previous messages array,
  - Re-renders the chat bubbles.
- No backend session storage is required.

---

### Evidence & Analysis Panel

#### Status Badges (top of panel)

Use gradient badges based on CSS classes:

| Class             | Style                              | Usage                                            |
|-------------------|------------------------------------|--------------------------------------------------|
| `.badge-ok`       | `background: var(--grad-success)` | Conflict No, Resolution OK, Faithful OK          |
| `.badge-conflict` | `background: var(--grad-accent)`  | Conflict Yes, Unresolved                         |
| `.badge-warn`     | `background: var(--color-warning)`| Uncertain, Possible hallucination                |
| `.badge-neutral`  | `background: var(--bg-input)`     | Baseline mode / neutral info                     |

Display:

- **Conflict detected**: Yes (`.badge-conflict`) / No (`.badge-ok`).
- **Conflict type**: pill with `numeric` / `temporal` / `opinion` / `other`; hide if `null`.
- **Resolution**: Resolved (`.badge-ok`) / Unresolved (`.badge-conflict`).
- **Uncertainty**: Flagged (`.badge-warn`) / Not flagged (`.badge-ok`).
- **Faithfulness**:
  - If `faithful === true` → `✓ Faithful` (`.badge-ok`).
  - If `faithful === false` → `⚠ Possible hallucination` (`.badge-warn`), hover to show `faithfulness_notes` in a tooltip.
  - If `faithful === null` → hide badge.
- **Retriever Recall / Precision**:
  - Use `<progress>` elements with `accent-color: var(--color-accent)` and numeric labels.
  - Hide the entire section when values are `null`.

#### Document Cards

One card per `retrieved_doc`:

```css
.doc-card {
  background: var(--bg-surface);
  backdrop-filter: var(--glass-blur);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-default);
  padding: var(--space-4);
}

.doc-card.chosen {
  border-color: var(--border-chosen);
  box-shadow: 0 0 12px rgba(16,185,129,0.25);
}

.doc-card.conflict {
  border-color: var(--border-conflict);
  box-shadow: 0 0 12px rgba(239,68,68,0.25);
}
```

Each card shows:

- Document ID,
- `metadata.source`,
- A short excerpt of `text` (truncated by default),
- A “Show more” toggle to expand.

---

### Compare Both Mode

When Mode = **Compare Both**, `chat.js` must:

1. Build a shared payload `{ query, top_k, model, strategy }`.
2. Make both calls in parallel:

```js
const [baselineRes, multiRes] = await Promise.all([
  api.queryBaseline(payload),
  api.queryMultiAgent(payload)
]);
```

3. Render a side-by-side comparison block inside the chat timeline:

```text
┌───────────────────────────┬────────────────────────────┐
│  Baseline                 │  Multi-agent              │
│  [answer text]            │  [answer text]            │
│                           │  [conflict badges]        │
└───────────────────────────┴────────────────────────────┘
```

On desktop, use equal-width columns; on mobile, stack vertically (multi-agent first).

#### Diff Highlighting Algorithm

To highlight differences (numbers and dates):

1. For each answer, extract:
   - Numbers: `/\b\d+(\.\d+)?\b/`.
   - Years: `/\b(19|20)\d{2}\b/`.
2. Create sets:
   - `numsBaseline`, `numsMulti`.
3. When rendering each answer string:
   - Tokenize on whitespace/punctuation.
   - For each token:
     - If it is a number/year and appears **only** in one set:
       - Wrap it in `<mark class="diff-highlight">token</mark>`.
4. CSS:

```css
.diff-highlight {
  background: rgba(245,158,11,0.25);
  border-radius: 3px;
  padding: 0 2px;
}
```

If eval data exists for the query, add a ✓ / ✗ indicator per card based on whether the answer matches the ground truth.

---

### Experiments Page (`experiments.html`)

On load:

- Call `GET /api/experiments`.
- Render a styled table with columns:
  - Timestamp
  - Query (truncated)
  - System
  - Strategy
  - Conflict Type
  - Faithful
  - Runtime (ms)

#### Filter Bar

Above the table:

- Date-from / date-to: `<input type="date">`.
- System `<select>`: `All | Baseline | Multi-agent`.
- Conflict type `<select>`: `All | numeric | temporal | opinion | other | none`.

All filters are applied client-side (no extra API calls).

#### Table Style

- Header row: `background: var(--grad-primary)`, white text.
- Striped rows via `tr:nth-child(even)`.
- Use `var(--font-mono)` for timestamps and runtime.

---

### UX Rules

- **Loading**:
  - Show skeleton shimmer in both chat and evidence panels while any fetch is in flight.
  - For streaming, skeleton is replaced as soon as the first token is received.
- **Errors**:
  - On any non-2xx response or network failure, show a dismissible error banner at the top:
    - `background: var(--grad-accent)`, white text, ✕ close button.
  - Do not swallow errors silently; log to `console.error`.
- **Theme toggle**:
  - Clicking the toggle adds/removes `.light` class on `<body>`.
  - Store preference in `localStorage["rag-theme"]`.
  - On load, read `rag-theme` and apply.
- **Animations**:
  - Chat bubbles: `animation: slideIn 0.2s ease`.
  - Badges: `animation: fadeScale 0.15s ease`.
  - Keep all animations under 300 ms for responsiveness.

---

## Accessibility and Misc

- Add `aria-label` or `title` attributes for key controls (send button, theme toggle, filters).
- Ensure sufficient contrast between text and backgrounds according to WCAG AA guidelines.
- Use semantic HTML elements (`<header>`, `<main>`, `<section>`, `<nav>`, `<table>`) for better accessibility.

---

Follow this specification exactly. If you need to deviate for technical reasons, explain the deviation in the terminal before implementing it.