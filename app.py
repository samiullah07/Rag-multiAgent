import html
import time
from typing import Any, Dict

import streamlit as st

from src.graphs.multi_agent_graph import build_multi_agent_app

# Load the LangGraph app once
_multi_agent_app = build_multi_agent_app()

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent RAG Lab",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Colorful theme (custom CSS injection)
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --c-pink:#ec4899; --c-purple:#8b5cf6; --c-indigo:#6366f1; --c-blue:#3b82f6;
  --c-teal:#14b8a6; --c-green:#22c55e; --c-amber:#f59e0b; --c-orange:#fb923c; --c-red:#ef4444;
  --bg-card: rgba(255,255,255,0.05);
  --border-card: rgba(255,255,255,0.12);
}

html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

/* App background: soft colorful aurora glow over a dark base */
.stApp{
  background:
    radial-gradient(circle at 8%   0%, rgba(236,72,153,0.20), transparent 45%),
    radial-gradient(circle at 92% 12%, rgba(99,102,241,0.22), transparent 45%),
    radial-gradient(circle at 50% 100%,rgba(20,184,166,0.18), transparent 55%),
    #0b0f1a;
  background-attachment: fixed;
}

/* Hero header */
.hero-title{
  font-weight:800; font-size:2.3rem; letter-spacing:-0.5px; margin-bottom:0;
  background: linear-gradient(90deg, var(--c-pink), var(--c-orange), var(--c-amber), var(--c-teal), var(--c-indigo));
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero-sub{ color:#a8b0c4; font-size:0.95rem; margin-top:-4px; margin-bottom:6px; }
.hero-pill{
  display:inline-block; padding:3px 12px; border-radius:999px; font-size:0.72rem; font-weight:600;
  background: rgba(255,255,255,0.06); border:1px solid var(--border-card); color:#cbd5e1; margin-right:6px;
}

/* Sidebar / left column styling */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(99,102,241,0.10), rgba(20,184,166,0.05));
  border-right:1px solid var(--border-card);
}

/* Expanders look like glass cards */
div[data-testid="stExpander"]{
  border-radius:14px; border:1px solid var(--border-card);
  background: rgba(255,255,255,0.03);
}
div[data-testid="stExpander"] summary{ font-weight:600; }

/* Chat bubbles */
.bubble-row{ display:flex; margin-bottom:14px; }
.bubble-row.user{ justify-content:flex-end; }
.bubble{
  max-width:82%; padding:12px 16px; border-radius:18px; font-size:0.94rem; line-height:1.55;
  box-shadow:0 4px 18px rgba(0,0,0,0.28); white-space:pre-wrap;
}
.bubble.user{
  background: linear-gradient(135deg, var(--c-pink), var(--c-orange));
  color:white; border-radius:18px 4px 18px 18px;
}
.bubble.assistant{
  background: linear-gradient(160deg, rgba(99,102,241,0.20), rgba(20,184,166,0.10));
  border:1px solid var(--border-card); color:#eef0fa;
  border-radius:4px 18px 18px 18px;
}
.bubble-label{ font-size:0.7rem; color:#9aa3bd; margin:0 4px 4px; }

/* Badges */
.badge{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.72rem; font-weight:700; margin-right:6px; margin-bottom:4px;}
.badge-ok{ background:linear-gradient(135deg,#22c55e,#14b8a6); color:#06281c; }
.badge-warn{ background:linear-gradient(135deg,#f59e0b,#fb923c); color:#2a1700; }
.badge-conflict{ background:linear-gradient(135deg,#ef4444,#ec4899); color:white; }
.badge-neutral{ background:rgba(255,255,255,0.08); color:#cbd5e1; border:1px solid var(--border-card); }

/* KPI cards */
.kpi-row{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:6px; }
.kpi-card{ flex:1; min-width:110px; border-radius:14px; padding:12px 14px; color:white; box-shadow:0 6px 18px rgba(0,0,0,0.28); }
.kpi-label{ font-size:0.68rem; opacity:0.9; text-transform:uppercase; letter-spacing:0.5px; }
.kpi-value{ font-size:1.45rem; font-weight:800; }

/* Document cards */
.doc-card{ border-radius:12px; padding:10px 12px; margin-bottom:8px; background:rgba(255,255,255,0.04); border:1px solid var(--border-card); }
.doc-card.chosen{ border-color:var(--c-green); box-shadow:0 0 14px rgba(34,197,94,0.25); }
.doc-card.conflict{ border-color:var(--c-red); box-shadow:0 0 14px rgba(239,68,68,0.25); }
.doc-card.upload{ border-color:var(--c-purple); box-shadow:0 0 14px rgba(139,92,246,0.3); background:rgba(139,92,246,0.07); }
.doc-meta{ font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:#9aa3bd; }

/* Buttons */
.stButton>button, .stDownloadButton>button{
  background: linear-gradient(135deg, var(--c-pink), var(--c-purple), var(--c-indigo));
  color:white; border:none; border-radius:10px; font-weight:600;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform: translateY(-2px); box-shadow:0 8px 20px rgba(139,92,246,0.45);
}

/* Chat input */
[data-testid="stChatInput"]{ border-radius:14px; }

/* Captions (e.g. faithfulness notes) — rendered as our own controlled
   block rather than st.caption(), so we don't depend on guessing
   Streamlit's internal class/testid names across versions. display:block
   plus clear:both prevents any inline-flow overlap with badges above it. */
.faithfulness-note{
  display:block;
  clear:both;
  font-size:0.78rem;
  color:#9aa3bd;
  line-height:1.5;
  margin:8px 0 16px 0;
  word-wrap:break-word;
}

.download-section { margin-top: 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Small render helpers
# ----------------------------------------------------------------------------

def esc(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def badge(label: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{esc(label)}</span>'


def kpi_card(label: str, value: str, gradient: str) -> str:
    return (
        f'<div class="kpi-card" style="background:{gradient}">'
        f'<div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div></div>'
    )


def chat_bubble(role: str, content: str) -> str:
    cls = "user" if role == "user" else "assistant"
    label = "You" if role == "user" else "Assistant"
    return (
        f'<div class="bubble-row {cls}">'
        f'<div><div class="bubble-label">{label}</div>'
        f'<div class="bubble {cls}">{esc(content)}</div></div></div>'
    )

# ----------------------------------------------------------------------------
# Query-intent routing: don't force chit-chat/meta questions through the RAG
# pipeline. Found via manual testing (TESTING.md) — asking "what can you do
# for me" was previously force-retrieved against the knowledge base and
# answered with an unrelated fact (blood volume) pulled back almost at
# random. This is a cheap, deterministic pre-check, not an LLM call, so it
# doesn't add cost or latency to real factual queries.
# ----------------------------------------------------------------------------
import re as _re

_CHITCHAT_PATTERN = _re.compile(
    r"^\s*(hi|hello|hey|yo|sup|thanks|thank you|good\s+(morning|afternoon|evening)|"
    r"who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
    r"how\s+does\s+this\s+work|help\b|what\s+is\s+this(\s+app|\s+tool|\s+project)?|"
    r"how\s+can\s+you\s+help|tell\s+me\s+about\s+yourself)",
    _re.IGNORECASE,
)

_CAPABILITIES_RESPONSE = (
    "I'm a self-correcting multi-agent RAG system, not a general chatbot — "
    "I work best with specific factual questions I can look up, rather than "
    "open-ended chat. Here's what I can do:\n\n"
    "- Answer factual questions from my knowledge base — try something specific, "
    "e.g. \"What is the speed of light?\" or \"When did World War II end?\"\n"
    "- Automatically detect when my sources contradict each other\n"
    "- Resolve those contradictions using a strategy you pick on the left "
    "(most recent, most authoritative, or explain both sides)\n"
    "- Check my own answers for fabricated citations or claims not actually "
    "supported by the retrieved documents\n\n"
    "Ask me a real factual question and I'll show you exactly which documents "
    "I used, whether I found a conflict, and how confident I am in the answer."
)


def is_chitchat(query: str) -> bool:
    """Cheap, deterministic check for greetings/meta questions that shouldn't
    be force-retrieved against the knowledge base."""
    q = (query or "").strip()
    if not q:
        return True
    return bool(_CHITCHAT_PATTERN.match(q))

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-title">🧠 Multi-Agent RAG Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Self-correcting multi-agent retrieval-augmented generation '
    'with automatic contradiction detection &amp; resolution</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="hero-pill">⚡ Groq · Llama 3.3</span>'
    '<span class="hero-pill">🗂️ ChromaDB</span>'
    '<span class="hero-pill">🕸️ LangGraph</span>',
    unsafe_allow_html=True,
)
st.write("")

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_result" not in st.session_state:
    st.session_state.graph_result = None

# Layout
col1, col2, col3 = st.columns([1, 4.5, 1])


def build_report_text() -> str:
    result = st.session_state.get("graph_result")
    messages = st.session_state.get("messages", [])

    lines: list[str] = ["# Multi-Agent RAG Session Report", ""]

    lines.append("## 💬 Conversation")
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"**{role}**: {m['content']}")
    lines.append("")

    if result and "answer" in result:
        lines.append("## ✅ Final Answer")
        lines.append(result["answer"])
        lines.append("")

    lines.append("## 📚 Retrieved Documents")
    if result:
        for doc in result.get("retrieved_docs", []):
            lines.append(f"### Doc {doc.id}")
            if getattr(doc, "metadata", None) and "source" in doc.metadata:
                lines.append(f"Source: {doc.metadata['source']}")
            txt = getattr(doc, "text", "")
            snippet = txt[:500]
            lines.append(f"Content: {snippet}{'...' if len(txt) > 500 else ''}")
            lines.append("")
    else:
        lines.append("_No retrieved documents available yet._")
        lines.append("")

    lines.append("## ⚠️ Conflicts")
    if result and result.get("has_conflict"):
        clusters = result.get("conflict_clusters", [])
        for cluster in clusters:
            cid = getattr(cluster, "cluster_id", f"cluster_{clusters.index(cluster)}")
            desc = getattr(cluster, "description", str(cluster))
            lines.append(f"### Cluster {cid}")
            lines.append(desc)
            resolved = result.get("resolved")
            if resolved:
                res_dict = getattr(resolved, "resolutions", {})
                if cid in res_dict:
                    r = res_dict[cid]
                    chosen = getattr(r, "chosen_doc_ids", [])
                    if chosen:
                        lines.append(f"- Chosen docs: {', '.join(chosen)}")
            lines.append("")
    else:
        lines.append("_No conflicts detected._")
        lines.append("")

    lines.append("## 📊 Evaluation Metrics")
    if result:
        if result.get("retriever_recall") is not None:
            lines.append(f"Retrieval Recall: {result['retriever_recall']:.2f}")
        if result.get("retriever_precision") is not None:
            lines.append(f"Retrieval Precision: {result['retriever_precision']:.2f}")
        if result.get("faithful") is not None:
            lines.append(f"Faithfulness: {'Yes' if result['faithful'] else 'No'}")
        if result.get("faithfulness_notes"):
            lines.append(f"Faithfulness Notes: {result['faithfulness_notes']}")
        if result.get("fabricated_citations"):
            lines.append(f"Fabricated Citations Detected: {', '.join(result['fabricated_citations'])}")
    else:
        lines.append("_No evaluation metrics available yet._")

    lines.append("")
    lines.append(f"*Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    return "\n".join(lines)

# ----------------------------------------------------------------------------
# Left column – controls
# ----------------------------------------------------------------------------
with col1:
    with st.expander("⚙️ Settings", expanded=True):
        if st.button("🧹 Clear chat"):
            st.session_state.messages = []
            st.session_state.graph_result = None
            st.rerun()

        top_k = st.number_input(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of documents to retrieve",
        )

        strategy = st.selectbox(
            "Strategy",
            ["most_recent", "most_authoritative", "explain_both"],
            help="How to select evidence",
        )

        model = st.selectbox(
            "Model",
            ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
            help="LLM backend name",
        )

# ----------------------------------------------------------------------------
# Middle column – chat
# ----------------------------------------------------------------------------
with col2:
    for msg in st.session_state.messages:
        st.markdown(chat_bubble(msg["role"], msg["content"]), unsafe_allow_html=True)

    if user_input := st.chat_input("Ask a question..."):
        # Clear any stale pending web-search offer from a previous turn —
        # asking something new shouldn't leave an old offer lingering.
        st.session_state.pop("pending_web_search", None)

        st.session_state.messages.append({"role": "user", "content": user_input})

        if is_chitchat(user_input):
            assistant_content = _CAPABILITIES_RESPONSE
        else:
            # Normal RAG pipeline
            graph_state: Dict[str, Any] = {
                "query": user_input,
                "top_k": int(top_k),
                "strategy": strategy,
                "model": model,
                "upload_session_id": st.session_state.get("upload_session_id"),
            }

            result = _multi_agent_app.invoke(graph_state)
            st.session_state.graph_result = result

            if not result.get("faithful") or result.get("fabricated_citations"):
                # Don't silently answer from outside knowledge. Store the
                # query and offer a button — rendered separately BELOW,
                # outside this block. A button click triggers its own
                # rerun, on which st.chat_input() returns empty and this
                # entire block is skipped — so the button can't live here.
                st.session_state.pending_web_search = user_input
                assistant_content = (
                    "I couldn't find a reliable answer in the knowledge base. "
                    "Want me to search the internet? Use the button below."
                )
            else:
                assistant_content = result.get("answer", "")

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content}
        )
        st.rerun()

    # ------------------------------------------------------------------------
    # Pending web-search offer — rendered OUTSIDE the chat_input block above
    # so it survives the rerun a button click triggers (on that rerun,
    # chat_input() is empty and the block above never executes at all).
    # ------------------------------------------------------------------------
    pending_query = st.session_state.get("pending_web_search")
    if pending_query:
        if st.button(f'🌐 Search the internet for: "{pending_query}"'):
            with st.spinner("🌐 Searching the web..."):
                from src.agents.web_search_agent import web_search
                web_results = web_search(pending_query)
            if web_results:
                search_answer = "🌐 From the internet (not the knowledge base): " + web_results[0].get("content", "")
            else:
                search_answer = "🌐 No results found online."
            st.session_state.messages.append(
                {"role": "assistant", "content": search_answer}
            )
            del st.session_state["pending_web_search"]
            st.rerun()

# ----------------------------------------------------------------------------
# Right column – insights panel
# ----------------------------------------------------------------------------
with col3:
    result = st.session_state.get("graph_result")
    if result:
        with st.expander("📊 Evaluation metrics", expanded=True):
            recall = result.get("retriever_recall")
            precision = result.get("retriever_precision")

            cards = []
            if recall is not None:
                cards.append(kpi_card("Recall", f"{recall:.2f}", "linear-gradient(135deg,#6366f1,#3b82f6)"))
            if precision is not None:
                cards.append(kpi_card("Precision", f"{precision:.2f}", "linear-gradient(135deg,#14b8a6,#22c55e)"))
            if cards:
                st.markdown(f'<div class="kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)

            faithful = result.get("faithful")
            notes = result.get("faithfulness_notes", "")
            fabricated = result.get("fabricated_citations") or []
            if faithful is not None:
                st.markdown(
                    badge("✓ Faithful" if faithful else "⚠ Not faithful", "ok" if faithful else "conflict"),
                    unsafe_allow_html=True,
                )
            if fabricated:
                st.markdown(
                    badge(f"🚨 Fabricated citation: {', '.join(fabricated)}", "conflict"),
                    unsafe_allow_html=True,
                )
            if notes:
                st.markdown(f'<div class="faithfulness-note">* {esc(notes)}</div>', unsafe_allow_html=True)

        with st.expander("⚡ Conflicts"):
            has_conflict = result.get("has_conflict", False)
            conflict_clusters = result.get("conflict_clusters", [])
            resolved = result.get("resolved")
            if not has_conflict:
                st.markdown(badge("No conflicts detected", "ok"), unsafe_allow_html=True)
            else:
                st.markdown(badge("Conflict detected", "conflict"), unsafe_allow_html=True)
                for cluster in conflict_clusters:
                    cid = getattr(
                        cluster,
                        "cluster_id",
                        f"cluster_{conflict_clusters.index(cluster)}",
                    )
                    desc = getattr(cluster, "description", str(cluster))
                    st.markdown(f"**Cluster `{cid}`**")
                    st.write(desc)
                    if resolved:
                        resolutions = getattr(resolved, "resolutions", {})
                        if cid in resolutions:
                            r = resolutions[cid]
                            chosen = getattr(r, "chosen_doc_ids", [])
                            if chosen:
                                st.markdown(
                                    badge("Resolved", "ok") + f" chosen: {', '.join(chosen)}",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    badge(f"Status: {getattr(r, 'status', 'unknown')}", "warn"),
                                    unsafe_allow_html=True,
                                )

        with st.expander("📚 Retrieved documents"):
            chosen_ids = set()
            if result.get("resolved"):
                for r in getattr(result["resolved"], "resolutions", {}).values():
                    chosen_ids.update(getattr(r, "chosen_doc_ids", []))

            for doc in result.get("retrieved_docs", []):
                css_class = "doc-card"
                is_upload = bool(doc.metadata and doc.metadata.get("source") == "user_upload")
                if is_upload:
                    css_class += " upload"
                elif doc.id in chosen_ids:
                    css_class += " chosen"
                elif result.get("has_conflict"):
                    css_class += " conflict"

                meta_line = ""
                if doc.metadata:
                    meta_line = ", ".join(
                        f"{k}: {v}"
                        for k, v in doc.metadata.items()
                        if k.lower() not in ["text", "content"]
                    )

                snippet = doc.text[:400]
                snippet_html = esc(snippet + ("..." if len(doc.text) > 400 else ""))

                upload_tag = '<span class="badge badge-neutral">📄 From your upload</span>' if is_upload else ""

                st.markdown(
                    f'<div class="{css_class}">'
                    f'{upload_tag}'
                    f'<b>{esc(doc.id)}</b><br>'
                    f'<span class="doc-meta">{esc(meta_line)}</span>'
                    f'<p style="margin-top:6px;font-size:0.85rem;">{snippet_html}</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with st.expander("🧩 Agent trace"):
            trace = result.get("trace", [])
            if not trace:
                st.caption("No trace available for this query.")
            else:
                agent_icons = {
                    "retriever": "🔍",
                    "contradiction_detection": "⚡",
                    "resolution": "⚖️",
                    "answer_generation": "✍️",
                    "grounding_verification": "🛡️",
                }
                for i, entry in enumerate(trace):
                    agent = entry.get("agent", "unknown")
                    summary = entry.get("summary", "")
                    icon = agent_icons.get(agent, "🔹")
                    st.markdown(
                        f'<div style="display:flex;align-items:flex-start;gap:10px;'
                        f'margin-bottom:8px;padding:8px 10px;border-radius:10px;'
                        f'background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08)">'
                        f'<span style="font-size:1.1rem;min-width:22px">{icon}</span>'
                        f'<div><div style="font-size:0.75rem;font-weight:700;color:#8b5cf6;'
                        f'text-transform:uppercase;letter-spacing:0.5px">{esc(agent.replace("_"," "))}</div>'
                        f'<div style="font-size:0.85rem;color:#cbd5e1;margin-top:2px">{esc(summary)}</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("### 📥 Download Session Report")
        report_md = build_report_text()
        st.download_button(
            label="📥 Download as Markdown (.md)",
            data=report_md,
            file_name="rag_session_report.md",
            mime="text/markdown",
        )
    else:
        st.info("Run a query to generate a session report.")
