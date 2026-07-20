# Specter RAG Evaluation Suite

Evaluates the Specter RAG pipeline end to end on a real contract, using four
independent techniques: **BLEU**, **ROUGE**, **RAGAS**, and **LLM-as-a-Judge (LaaJ)**.

Everything runs locally against Ollama (`llama3.2` + `nomic-embed-text`). No API keys,
no network calls.

---

## Quick start

```bash
# Ollama must be running, with both models pulled
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text

# Full run — all four techniques
python -m evaluation.run_evaluation
```

Results print to the terminal and are saved to `evaluation/results/`.

### Options

```bash
python -m evaluation.run_evaluation --metrics bleu,rouge   # fast, deterministic, no LLM judging
python -m evaluation.run_evaluation --top-k 8              # retrieve more clauses
python -m evaluation.run_evaluation --no-graph             # isolate pure vector retrieval
python -m evaluation.run_evaluation --skip-ingest          # reuse the loaded ChromaDB
python -m evaluation.run_evaluation --pdf path/to/other.pdf
```

> ⚠️ A full run ingests the PDF, and `ingest_constitution()` calls `VectorStore.clear()` —
> **it wipes whatever is currently in `./data/chroma_db`**. That is the app's existing
> "fresh ingest each time" behaviour, not something this suite added. If you had a document
> loaded through the UI, re-upload it afterwards.

---

## What's in here

```
evaluation/
├── dataset/qa_dataset.json     10 ground-truth Q&A pairs from the Berkshire/Auriemma contract
├── metrics/
│   ├── bleu.py                 BLEU-4, pure Python, smoothed + brevity penalty
│   ├── rouge.py                ROUGE-1 / ROUGE-2 / ROUGE-L, pure Python
│   ├── ragas_local.py          The four core RAGAS metrics, local reimplementation
│   ├── laaj.py                 LLM-as-a-Judge, 4-dimension 1-5 rubric
│   └── _llm.py                 Shared Ollama judge helper + robust JSON parsing
├── rag_runner.py               Drives the real pipeline (mirrors app.py exactly)
├── run_evaluation.py           Orchestrator + reporting
└── results/                    Timestamped runs, plus latest.json
```

The dataset covers the **Endorsement Agreement between Geno Auriemma and Berkshire Bank**
(CUAD v1), spanning parties, term dates, compensation structure, payment terms,
obligations, termination, cure periods, and governing law / arbitration.

---

## The four techniques

They measure genuinely different things. That's the point of running all four.

| | Measures | Needs | Deterministic | Cost |
|---|---|---|---|---|
| **BLEU** | n-gram *precision* vs reference | reference answer | ✅ | free |
| **ROUGE** | n-gram + LCS *recall* vs reference | reference answer | ✅ | free |
| **RAGAS** | retrieval vs generation quality, separately | reference + contexts | ❌ | ~6 LLM calls/question |
| **LaaJ** | holistic correctness, as a user would read it | reference + contexts | ❌ (temp=0) | 1 LLM call/question |

### BLEU — surface precision

Standard BLEU-4 with smoothing method 1 (Chen & Cherry, 2014) and a brevity penalty.
Reports per-order precisions so a low score can be diagnosed rather than just recorded.

**Expect low scores here, and don't chase them.** Specter is explicitly prompted to answer
in plain English rather than quote the contract, so n-gram overlap with a reference answer
stays low even when the answer is perfect. BLEU is included because it's the standard
baseline and because a *sudden drop* between runs is still informative — not because
the absolute number means much for this task.

### ROUGE — reference coverage

ROUGE-1 (content words), ROUGE-2 (local phrasing), ROUGE-L (longest common subsequence,
order-aware). Each as precision / recall / F1.

More useful than BLEU here because of the recall framing: an answer that omits the 30-day
cure period is wrong in a way precision alone won't catch. Still surface-level — it cannot
tell a paraphrase from an error.

### RAGAS — component-level diagnosis

Four metrics, split across the two halves of the pipeline:

| Metric | Half | Question it answers |
|---|---|---|
| `faithfulness` | generation | Is every claim in the answer grounded in the retrieved context? |
| `answer_relevancy` | generation | Does the answer actually address the question? |
| `context_precision` | retrieval | Were the useful chunks ranked highly? |
| `context_recall` | retrieval | Did retrieval surface everything the answer needed? |

This split is what makes RAGAS worth the runtime. If faithfulness is high but
context_recall is low, the prompt is fine and the retriever is starving it — fixing the
wrong half is the usual RAG debugging failure.

**This does not import the `ragas` pip package.** That package defaults to hosted OpenAI
models for both its LLM and its embeddings, and this project is deliberately offline-only.
The metric *definitions* from the RAGAS paper (Es et al., 2023) are reimplemented in
`ragas_local.py` against the local Ollama stack. Formulas match; **the numbers are not
comparable to published RAGAS benchmarks** that used GPT-4 as the backing model.

### LaaJ — LLM-as-a-Judge

`llama3.2` reads the question, retrieved context, generated answer and ground truth, then
scores four dimensions 1-5 with written reasoning:

- **correctness** — do the facts match the ground truth? (the one that matters most; a wrong dollar figure is a wrong answer)
- **completeness** — does it cover everything the reference covers?
- **groundedness** — is every statement traceable to the context?
- **clarity** — is it plain English a non-lawyer can act on? (this is what the product promises)

The rubric spells out what each of 1-5 means per dimension, because unanchored 1-5 scales
produce noise.

---

## Limitations worth knowing before trusting the numbers

These affect how much weight the scores deserve:

1. **The judge is the same model family as the generator.** `llama3.2` evaluating
   `llama3.2` shows self-preference bias — models rate their own phrasing generously.
   RAGAS and LaaJ scores are optimistic in absolute terms. Swap `JUDGE_MODEL` in
   `metrics/_llm.py` to a stronger model to test this.
2. **Small models compress toward the middle of a scale.** LaaJ rarely awards 1 or 5.
   Watch relative ordering across runs, not absolute values.
3. **10 questions is a small sample.** Differences of a few points between runs are
   noise. This suite is built for detecting regressions and comparing configurations,
   not for publishing a benchmark number.
4. **Temperature is pinned to 0** for judging, so runs are reproducible — but
   reproducible is not the same as accurate.
5. **`context_recall` depends on hand-written ground truth.** It measures the retriever
   against what a human decided was the right answer, so it inherits any bias in
   `qa_dataset.json`.

---

## Reading the results

`run_evaluation.py` prints a **READING THE RESULTS** section that flags the common
diagnostic patterns automatically. The rules behind it:

| Pattern | What it means | Where to fix it |
|---|---|---|
| Low BLEU, high LaaJ | Paraphrasing, working as designed | Nothing — don't tune against BLEU |
| Low `context_recall` | Retriever is missing clauses | `--top-k`, or chunking in `clause_extractor.py` |
| High `faithfulness` + low `context_recall` | **Dangerous.** Faithful to the wrong context — confident and incomplete | Retrieval, urgently |
| Low `context_precision` | Useful clauses out-ranked by noise | Try `--no-graph` to isolate graph expansion |
| Low ROUGE recall, decent LaaJ | Terse answers | Prompt in `query/qa.py` |
| High retrieval scores, low `faithfulness` | Right context, model ignoring it | Prompt in `query/qa.py` |

Every run writes full per-question detail to `results/` — the answer, the retrieved
clauses, and each judge's reasoning — so any score can be traced back to what produced it.
`results/latest.json` always points at the most recent run for diffing.

---

## Extending it

**More questions:** add entries to `dataset/qa_dataset.json`. Each needs `id`, `question`,
`ground_truth`, `ground_truth_contexts` (verbatim clause text an ideal retriever should
find) and `category`.

**A different contract:** point `--pdf` at it and write a matching dataset file, then pass
`--dataset`. Any CUAD contract works.

**A different metric:** add a module under `metrics/`, then wire it into `score_records()`
and `aggregate()` in `run_evaluation.py`.

**A stronger judge:** change `JUDGE_MODEL` in `metrics/_llm.py`. This is the single
highest-leverage change for score quality.
