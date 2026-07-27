# Starting prompt for Claude Code

Paste this to start the next phase of work:

```
Read CLAUDE.md Section 13 in full before doing anything. This section
documents a deliberate scope decision made with the project owner: a
request for "unlimited knowledge" and "always current information" was
reconciled into three separate, bounded features rather than implemented
literally, because unrestricted internet access would break the controlled
evaluation this project's dissertation results depend on. Do not relitigate
that decision — implement what's specified in 13.1-13.4.

Also: I fixed a frontend bug myself in this session (faithfulness notes
text overlapping with the Conflicts/Retrieved-documents panels below it).
The fix replaced st.caption() with a custom .faithfulness-note div in
app.py. Don't revert this back to st.caption().

Work through Section 13 in this order, confirming each step actually works
before moving to the next — this project has had repeated incidents this
session of reported results not matching what was actually saved to disk,
so verify with real command output at each step, not just a description:

1. Add `tavily-python` to pyproject.toml dependencies. Add
   `TAVILY_API_KEY=tvly-your-key-here` to .env (sign up free at
   https://tavily.com if not done already — 1,000 free credits/month, no
   credit card). Run `uv sync` and confirm it installs successfully (paste
   the real output). Confirm the key actually loads — e.g. a one-line
   `print(bool(os.environ.get("TAVILY_API_KEY")))` check — then remove that
   debug line once confirmed.

2. Create src/agents/web_search_agent.py implementing web_search() exactly
   as specified in §13.1, step 3 (uses TavilyClient, not duckduckgo-search).

3. Wire the opt-in flow into app.py: when the grounding verifier flags
   faithful=False or a fabricated citation, show a button offering to
   search the web (don't auto-search). On click, call web_search(), show
   a st.spinner while it runs, then generate an answer from the results
   tagged "🌐 From the internet (not the knowledge base):" — do NOT touch
   chosen_doc_ids, correct_doc_ids, or anything the eval suite reads.

4. Test this manually: ask a question you know isn't in the knowledge base
   (e.g. "what's the current weather in London") and confirm: (a) it does
   NOT silently answer from the LLM's own knowledge, (b) it offers to
   search, (c) clicking yes actually searches and shows a real, current
   answer with the web-source prefix. Paste the actual answer text as
   proof, not a description of what happened.

5. Implement the agent trace (§13.3) — add the trace-collector append to
   each of the five existing graph nodes, plus a new "🧩 Agent trace"
   expander in app.py. Test with one real query and paste the actual
   trace list that gets shown.

6. Implement document upload (§13.2) as a new Streamlit page. Test by
   uploading a short .txt file with a fact NOT in the existing knowledge
   base, asking about it, and confirming the answer comes from the
   uploaded doc specifically (check the metadata tag in the retrieved
   chunk, don't just trust the answer text).

7. After all of the above, re-run `uv run python -m src.evaluation.run_eval`
   once and confirm the existing benchmark numbers (accuracy 1.0/0.85,
   conflict P/R/F1 55.6%/100%/71.4%, resolution quality 1.0 — see
   TESTING.md) are UNCHANGED. If they changed, something in steps 1-6
   leaked into the core pipeline and that's a regression to fix before
   continuing.

Show real, verbatim output at every step — file sizes, command output,
actual answer text — not summaries of what should have happened.
```
