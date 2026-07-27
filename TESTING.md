# Manual Testing Checklist — Multi-Agent RAG Lab

How to verify the **complete system** works end-to-end, from a user's
perspective, using known-correct answers already verified during development
(not guesses). Run through this whenever you've made a change, before a demo,
or before submission.

---

## 0. Prerequisites

```bash
cd "C:\Users\BEST LAPTOP\Desktop\RAG-gurkirat"
uv sync
```

Confirm `.env` has a valid `GROQ_API_KEY` and `GROQ_MODEL` (check current Groq
quota at https://console.groq.com if you've hit limits before — this project
has hit Groq rate limits repeatedly during development).

Confirm the knowledge base is indexed:

```bash
uv run python -m src.data_prep.build_kb
```

Should report indexing ~86 documents (50 from `data/raw`, 36 from
`data/contradictions`). If it reports a very different number, the KB may be
stale — delete `data/index/chroma_db` and re-run.

---

## 1. Launch the UI

```bash
uv run streamlit run app.py
```

Check on load:
- [ ] Gradient hero title "🧠 Multi-Agent RAG Lab" renders
- [ ] Settings panel (left) shows Top K, Strategy, Model controls
- [ ] Chat input box is visible and focused

---

## 2. Known-answer test queries

Run each query below and check the **actual answer** against the **expected
answer** — these are real values verified against the knowledge base during
development, not assumptions.

### 2a. Simple non-conflict fact (sanity check)
**Query:** `When was the first iPhone released?`
**Expected answer contains:** `2007`
**Expected UI:** No conflict badge in the "⚡ Conflicts" panel; "Retrieved
documents" panel shows plain doc cards (no green/red glow border).

### 2b. Clean conflict, should resolve correctly
**Query:** `What is the speed of light in meters per second?`
**Expected answer contains:** `299,792,458`
**Expected UI:**
- "⚡ Conflicts" panel shows "Conflict detected"
- "Retrieved documents" shows `science_contr_01a` with a **green glow**
  (chosen) and `science_contr_01b` with a different look (not chosen)
- Evaluation metrics panel shows a "✓ Faithful" badge

### 2c. Previously-buggy resolution case (regression check)
**Query:** `When was the Human Genome Project completed?`
**Expected answer contains:** `2003` — **not** 2001.
(This exact question previously failed due to a missing `publication_date`
bug in `build_kb.py`. If it answers 2001, that bug has regressed.)

### 2d. Second previously-buggy case (regression check)
**Query:** `What is the average life expectancy in Japan?`
**Expected answer contains:** `84.6` — **not** 81.2.

### 2e. Known false-positive conflict trigger (documented limitation, not a bug)
**Query:** `Who created the periodic table?`
**Expected answer contains:** `Dmitri Mendeleev`
**Expected UI:** This one is *expected* to show "Conflict detected" even
though there's no real contradiction — the rule-based numeric-overlap
detector over-triggers on repeated/duplicate numeric facts. This is a
documented limitation (see CLAUDE.md §1), not something to "fix" by panic —
just confirm the final answer is still correct despite the false alarm.

### 2f. The hallucination test — most important new check
**Query:** `What is the average distance of Earth from the Sun?`
**Watch for:** This question previously caused the system to fabricate fake
documents (`Doc id=astronomy_01/02/03`) that don't exist in the knowledge
base, to justify an answer not actually grounded in retrieved context.
**Expected UI now:** the citation-grounding verifier (added most recently)
should catch this. Look for either:
- A "🚨 Fabricated citation" badge if it fabricates a fake doc ID again, or
- A "⚠ Not faithful" badge if it answers from outside knowledge without a
  fake citation, or
- Ideally: an honest "I don't have this information in the retrieved
  documents" answer with a "✓ Faithful" badge.

**This is the one test you haven't run yet since adding the verifier — it's
the real proof the new feature works, not just that it compiles.**

---

## 3. UI panel checklist (after running 2b above)

- [ ] "📊 Evaluation metrics" — shows Recall/Precision KPI cards, a
      Faithful/Not-faithful badge
- [ ] "⚡ Conflicts" — shows cluster description and which doc was chosen
- [ ] "📚 Retrieved documents" — shows doc cards with source/metadata,
      green border on the chosen doc
- [ ] "📥 Download Session Report" button produces a `.md` file — open it
      and confirm it contains the conversation, answer, conflicts, and
      faithfulness fields

---

## 4. CLI entry points (no UI)

```bash
uv run python main_multi_agent.py --query "What year did World War II end?"
uv run python main_baseline.py --query "What year did World War II end?"
```
Both should print an answer containing `1945`. Compare the two outputs —
the baseline should be faster but more naive; the multi-agent one should
mention resolving a conflict.

---

## 5. Full automated regression run

```bash
uv run python -m src.evaluation.run_eval
```

Compare against the last fully-verified benchmark (recorded here so future
runs have something concrete to regress-test against):

| Metric | Last verified value |
|---|---|
| Multi-agent answer accuracy | 1.0 (20/20) |
| Baseline answer accuracy | 0.85 (17/20) |
| Conflict detection precision / recall / F1 | 55.6% / 100% / 71.4% |
| Resolution quality (most_recent strategy) | 1.0 (10/10) |

If a fresh run differs significantly from these, don't assume the new run is
wrong — but don't assume it's right either. Re-verify by reading
`data/eval/multi_agent_records.json` directly (`Get-Item` for timestamp/size,
then open and check a few records by hand) before trusting any printed
summary. This project has had multiple incidents this session of reported
numbers not matching what was actually saved to disk.

**Note:** this run now costs more Groq quota than before the grounding
verifier was added (one extra LLM call per question). Don't run it
repeatedly without reason.

---

## 6. Edge cases worth trying once

- [ ] Empty query / single character — shouldn't crash
- [ ] A question totally unrelated to the KB (e.g. "What's the capital of
      France?") — check it admits uncertainty rather than hallucinating
- [ ] Switching `Strategy` to `explain_both` vs `most_recent` mid-session and
      re-asking the same conflict question — confirm the answer's tone
      actually changes (hedged vs decisive)

---

*Keep this file updated as new known-good/known-bad cases are discovered.*
