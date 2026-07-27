"""
Temporal Belief-Revision Timeline

Motivation: the core RAG pipeline resolves contradictions by picking the
most recently published source. This page makes that process *visible* by
replaying what the system's answer would have been at each point in time
as documents arrived, showing how beliefs revise as newer evidence is
indexed.

Conceptually grounded in AGM belief revision theory (Alchourron, Gardenfors
& Makinson, 1985) — the idea that a rational agent should update beliefs
minimally when confronted with new evidence. Here we apply it empirically:
given a query whose documents have known publication dates, we show the
belief state at each distinct date rather than just the final answer.
"""

from __future__ import annotations

import json
from datetime import datetime, MINYEAR
from pathlib import Path
from typing import Any

import streamlit as st

from src.config.config import settings
from src.graphs.multi_agent_graph import build_multi_agent_app

st.set_page_config(
    page_title="Temporal Belief-Revision Timeline",
    page_icon="⏳",
    layout="wide",
)

# ── same CSS variables as app.py so the visual language is consistent ─────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap');
html,body,[class*="css"]{ font-family:'Poppins',sans-serif; }
.stApp{
  background:
    radial-gradient(circle at 8% 0%,  rgba(236,72,153,0.18), transparent 45%),
    radial-gradient(circle at 92% 12%, rgba(99,102,241,0.20), transparent 45%),
    radial-gradient(circle at 50% 100%,rgba(20,184,166,0.15), transparent 55%),
    #0b0f1a;
}
/* Timeline track */
.tl-track{
  position:relative; padding-left:36px; margin:0 0 0 8px;
}
.tl-track::before{
  content:""; position:absolute; left:11px; top:0; bottom:0;
  width:2px; background:linear-gradient(180deg,#8b5cf6,#14b8a6);
  border-radius:2px;
}
/* Single timeline node */
.tl-node{
  position:relative; margin-bottom:22px;
  background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10);
  border-radius:14px; padding:14px 16px;
}
.tl-node::before{
  content:""; position:absolute; left:-29px; top:18px;
  width:12px; height:12px; border-radius:50%;
  background:linear-gradient(135deg,#8b5cf6,#14b8a6);
  box-shadow:0 0 8px rgba(139,92,246,0.6);
}
.tl-node.belief-changed{ border-color:#ec4899; box-shadow:0 0 14px rgba(236,72,153,0.25); }
.tl-node.belief-unchanged{ border-color:#14b8a6; box-shadow:0 0 8px rgba(20,184,166,0.15); }
.tl-date{
  font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#8b5cf6;
  font-weight:600; margin-bottom:4px;
}
.tl-doc-label{
  font-size:0.8rem; color:#a8b0c4; margin-bottom:6px;
}
.tl-belief{
  font-size:0.95rem; color:#eef0fa; font-weight:600; margin-bottom:6px;
}
.tl-change-badge{
  display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.7rem;
  font-weight:700; margin-left:8px;
}
.badge-revised{ background:linear-gradient(135deg,#ec4899,#8b5cf6); color:white; }
.badge-stable{ background:rgba(20,184,166,0.2); color:#14b8a6; border:1px solid rgba(20,184,166,0.4); }
.badge-initial{ background:rgba(99,102,241,0.25); color:#818cf8; border:1px solid rgba(99,102,241,0.4); }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="background:linear-gradient(90deg,#ec4899,#8b5cf6,#14b8a6);'
    '-webkit-background-clip:text;background-clip:text;color:transparent;'
    'font-weight:800;font-size:1.9rem">⏳ Temporal Belief-Revision Timeline</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    "Shows how the system's answer to a query **would have changed** as "
    "documents with later publication dates were progressively added to the "
    "knowledge base — visualising belief revision in the spirit of AGM theory "
    "(Alchourrón, Gärdenfors & Makinson, 1985).",
    unsafe_allow_html=True,
)
st.write("")

# ── Load contradiction pairs from disk ────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_eval_questions() -> list[dict]:
    """Load questions.jsonl for matching scenarios to real benchmark queries."""
    qpath = Path("data/eval/questions.jsonl")
    if not qpath.exists():
        return []
    items = []
    for line in qpath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


@st.cache_data(show_spinner=False)
def load_contradiction_pairs() -> list[dict[str, Any]]:
    """
    Reads all JSONL files from data/contradictions/ and groups them by their
    'label' cross-reference into pairs. Each pair becomes one timeline scenario
    the user can select.
    """
    contr_dir = Path("data/contradictions")
    if not contr_dir.exists():
        return []

    raw: list[dict] = []
    for f in sorted(contr_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Group into pairs by label cross-reference
    by_id = {d["id"]: d for d in raw}
    seen_pairs: set[frozenset] = set()
    pairs: list[dict] = []

    for doc in raw:
        label = doc.get("label", "")
        if not label.startswith("contradicts:"):
            continue
        partner_id = label[len("contradicts:"):]
        partner = by_id.get(partner_id)
        if not partner:
            continue
        key = frozenset([doc["id"], partner_id])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        # Sort so older doc is first
        both = sorted(
            [doc, partner],
            key=lambda d: _parse_date_str(d.get("publication_date", "")),
        )
        pairs.append({"older": both[0], "newer": both[1]})

    return pairs


def _default_query_for_pair(pair: dict) -> str:
    """
    Find a real benchmark question whose correct_doc_ids reference either doc
    in this pair. Falls back to a generic string if no match.
    """
    questions = _load_eval_questions()
    pair_ids = {pair["older"]["id"], pair["newer"]["id"]}
    for q in questions:
        correct = set(q.get("correct_doc_ids") or [])
        if correct & pair_ids:
            return q["query"]
    # Fallback: generic (should rarely fire now)
    topic = pair["older"]["id"].rsplit("_contr_", 1)[0].replace("_", " ")
    return f"What is the {topic} value?"


def _ground_truth_for_pair(pair: dict) -> str | None:
    """
    Return the ground_truth from questions.jsonl for a question targeting this
    pair, or None if no match.
    """
    questions = _load_eval_questions()
    pair_ids = {pair["older"]["id"], pair["newer"]["id"]}
    for q in questions:
        correct = set(q.get("correct_doc_ids") or [])
        if correct & pair_ids:
            return q.get("ground_truth")
    return None


def _parse_date_str(value: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try:
            return datetime.strptime(str(value), fmt)
        except (ValueError, TypeError):
            continue
    return datetime(MINYEAR, 1, 1)


def _friendly_date(value: str) -> str:
    d = _parse_date_str(value)
    if d.year == MINYEAR:
        return "Unknown date"
    return d.strftime("%B %Y")


# ── Simulate belief at a given cutoff date ────────────────────────────────────

def simulate_belief(query: str, docs_available: list[dict], app) -> str:
    """
    Build a minimal graph invocation using only the documents that would have
    been available at or before the cutoff date, returning the answer text.

    We bypass the actual retriever (which queries the live Chroma index) and
    instead inject the time-filtered docs directly into the graph state as
    pre-retrieved documents, so the contradiction/resolution/generation chain
    still runs normally but on a controlled document set.
    """
    from src.agents.schemas import RetrievedDoc

    retrieved = [
        RetrievedDoc(
            id=d["id"],
            text=d.get("text", ""),
            metadata={
                "source": d.get("source", "unknown"),
                "publication_date": d.get("publication_date", ""),
            },
        )
        for d in docs_available
    ]

    # Build a minimal initial state that skips retrieval (retrieved_docs
    # already populated) — we jump straight into contradiction detection
    from src.graphs.multi_agent_graph import (
        contradiction_detection_node,
        resolution_node,
        answer_generation_node,
    )

    state = {
        "query": query,
        "retrieved_docs": retrieved,
        "strategy": "most_recent",
        "top_k": 5,
        "upload_session_id": None,
        "trace": [],
    }
    state = contradiction_detection_node(state)
    if state.get("has_conflict"):
        state = resolution_node(state)
    state = answer_generation_node(state)
    return state.get("answer", "No answer generated.")


# ── Main UI ───────────────────────────────────────────────────────────────────

pairs = load_contradiction_pairs()

if not pairs:
    st.warning(
        "No contradiction pairs found in `data/contradictions/`. "
        "Run `uv run python -m src.data_prep.build_kb` first."
    )
    st.stop()

# Build human-readable labels for the dropdown
def pair_label(p: dict) -> str:
    older, newer = p["older"], p["newer"]
    topic = older["id"].rsplit("_contr_", 1)[0].replace("_", " ").title()
    return f"{topic} — {_friendly_date(older['publication_date'])} vs {_friendly_date(newer['publication_date'])}"


col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### ⚙️ Configuration")
    selected_idx = st.selectbox(
        "Choose a contradiction scenario",
        options=list(range(len(pairs))),
        format_func=lambda i: pair_label(pairs[i]),
    )
    pair = pairs[selected_idx]

    custom_query = st.text_input(
        "Query to trace",
        value=_default_query_for_pair(pair),
        help="What question should the system try to answer at each time step?",
    )

    run_btn = st.button("▶ Generate Timeline", use_container_width=True)

with col_right:
    st.markdown("### 📋 Scenario Documents")
    older, newer = pair["older"], pair["newer"]

    for doc, label, color in [
        (older, "Older source", "#f59e0b"),
        (newer, "Newer source", "#8b5cf6"),
    ]:
        st.markdown(
            f'<div style="border:1px solid {color};border-radius:12px;'
            f'padding:10px 14px;margin-bottom:10px;background:rgba(255,255,255,0.03)">'
            f'<div style="font-size:0.7rem;color:{color};font-weight:700;'
            f'text-transform:uppercase;margin-bottom:4px">{label}</div>'
            f'<div style="font-size:0.8rem;color:#a8b0c4;font-family:\'JetBrains Mono\',monospace">'
            f'ID: {doc["id"]} · {_friendly_date(doc.get("publication_date",""))} · {doc.get("source","?")}'
            f'</div><div style="font-size:0.9rem;color:#eef0fa;margin-top:6px">{doc.get("text","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Timeline generation ───────────────────────────────────────────────────────

if run_btn and custom_query.strip():
    st.write("")
    st.markdown("---")
    st.markdown("### ⏳ Belief Revision Timeline")
    st.markdown(
        "Each node shows the system's answer using only the documents available "
        "**up to that date**. A 🔄 badge means the answer changed from the previous step."
    )
    st.write("")

    app = build_multi_agent_app()

    # Three time slices:
    # 1. Only the older document available
    # 2. Only the newer document available (counterfactual — if we had skipped the older)
    # 3. Both documents available (reality — the system sees the conflict and resolves it)
    time_slices = [
        {
            "label": "Step 1 — Only older source available",
            "cutoff": _friendly_date(older.get("publication_date", "")),
            "docs": [older],
            "description": f'Before the newer source was published ({_friendly_date(newer.get("publication_date",""))}), '
                           f'the system would only have seen the older document.',
        },
        {
            "label": "Step 2 — Newer source published (both available)",
            "cutoff": _friendly_date(newer.get("publication_date", "")),
            "docs": [older, newer],
            "description": "With both documents in the KB, the contradiction detector fires. "
                           "The resolution agent picks the newer source using the `most_recent` strategy.",
        },
        {
            "label": "Step 3 — Final resolved belief",
            "cutoff": "Present",
            "docs": [older, newer],
            "description": "The system's final, stable belief — grounded in the most recently published source.",
        },
    ]

    prev_answer = None

    ground_truth = _ground_truth_for_pair(pair)

    def _belief_changed(prev: str, curr: str) -> bool:
        """
        Compare two answers for substantive change. If a ground-truth value is
        known and present in both answers, the belief is unchanged regardless
        of surrounding wording differences.
        """
        if prev is None:
            return False
        if ground_truth and ground_truth.lower() in prev.lower() and ground_truth.lower() in curr.lower():
            return False
        return curr.strip() != prev.strip()

    with st.spinner("⏳ Simulating beliefs across time..."):
        results = []
        for step in time_slices:
            answer = simulate_belief(custom_query, step["docs"], app)
            changed = _belief_changed(prev_answer, answer)
            results.append({**step, "answer": answer, "changed": changed, "first": prev_answer is None})
            prev_answer = answer

    st.markdown('<div class="tl-track">', unsafe_allow_html=True)

    for r in results:
        node_class = "tl-node belief-changed" if r["changed"] else "tl-node belief-unchanged"
        if r["changed"]:
            badge = '<span class="tl-change-badge badge-revised">🔄 Belief Revised</span>'
        elif r["first"]:
            badge = '<span class="tl-change-badge badge-initial">📌 Initial Belief</span>'
        else:
            badge = '<span class="tl-change-badge badge-stable">✓ Unchanged</span>'

        st.markdown(
            f'<div class="{node_class}">'
            f'<div class="tl-date">📅 {r["cutoff"]}</div>'
            f'<div style="font-weight:700;font-size:0.9rem;color:#cbd5e1;margin-bottom:4px">'
            f'{r["label"]}{badge}</div>'
            f'<div class="tl-doc-label">{r["description"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="margin:-10px 0 16px 4px;padding:10px 14px;'
            f'background:rgba(99,102,241,0.08);border-left:3px solid #8b5cf6;'
            f'border-radius:0 10px 10px 0;font-size:0.9rem;color:#eef0fa">'
            f'{r["answer"][:600]}{"..." if len(r["answer"]) > 600 else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Summary panel
    n_revisions = sum(1 for r in results if r["changed"])
    st.markdown(
        f'<div style="margin-top:16px;padding:14px 18px;border-radius:14px;'
        f'background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.3)">'
        f'<b>Summary:</b> The system revised its belief '
        f'<b>{n_revisions} time{"s" if n_revisions != 1 else ""}</b> across '
        f'{len(results)} time steps for this scenario. '
        f'{"The newer source caused a belief update — this is the contradiction-resolution mechanism in action." if n_revisions > 0 else "The answer remained stable across all available evidence."}'
        f'</div>',
        unsafe_allow_html=True,
    )

elif run_btn:
    st.warning("Please enter a query above before generating the timeline.")
