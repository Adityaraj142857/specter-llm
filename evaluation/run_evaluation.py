"""
evaluation/run_evaluation.py

Entry point for the Specter RAG evaluation suite.

Ingests a contract PDF through the real pipeline, answers the ground-truth
question set, then scores every answer with four independent techniques:

    BLEU    n-gram precision vs the reference answer          (surface form)
    ROUGE   n-gram + LCS recall vs the reference answer       (surface form)
    RAGAS   faithfulness / relevancy / precision / recall     (semantic, LLM+embeddings)
    LaaJ    1-5 rubric scored by an LLM judge                 (semantic, holistic)

They are kept separate on purpose. BLEU and ROUGE are cheap, deterministic and
blind to meaning. RAGAS splits the pipeline into its retrieval and generation
halves so a bad score points at the component that caused it. LaaJ is the only
one that reads the answer the way a user would. Agreement across all four is
a strong signal; disagreement tells you which layer to look at.

Usage:
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --top-k 8 --no-graph
    python -m evaluation.run_evaluation --skip-ingest        # reuse the loaded ChromaDB
    python -m evaluation.run_evaluation --metrics bleu,rouge # fast, no LLM judging
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# Make the project root importable when run as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import bleu as bleu_metric
from evaluation.metrics import rouge as rouge_metric
from evaluation.metrics import laaj as laaj_metric
from evaluation.metrics import ragas_local
from evaluation.rag_runner import ingest_pdf, run_rag


DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset" / "qa_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
ALL_METRICS = ("bleu", "rouge", "ragas", "laaj")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(path: Path = DATASET_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_records(
    records: list[dict],
    qa_pairs: list[dict],
    metrics: tuple[str, ...],
    checkpoint_path: Path | None = None,
) -> list[dict]:
    """
    Runs the selected metrics over every answered question.

    Writes a checkpoint after each question when checkpoint_path is given.
    Scoring 10 questions takes ~10 minutes of local LLM calls, and losing all
    of it because the process was interrupted at question 9 is not acceptable
    — partial results are still worth reading.
    """
    scored = []

    for record, qa in zip(records, qa_pairs):
        answer = record["answer"]
        ground_truth = qa["ground_truth"]
        entry = {
            "id": qa["id"],
            "category": qa.get("category", ""),
            "question": qa["question"],
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "retrieved_clauses": record["retrieved_clauses"],
            "latency_seconds": record["latency_seconds"],
            "error": record.get("error"),
            "metrics": {},
        }

        if record.get("error") or not answer.strip():
            entry["metrics"]["note"] = "pipeline failed for this question — not scored"
            scored.append(entry)
            _write_checkpoint(checkpoint_path, scored)
            continue

        if "bleu" in metrics:
            entry["metrics"]["bleu"] = bleu_metric.sentence_bleu(answer, ground_truth)

        if "rouge" in metrics:
            entry["metrics"]["rouge"] = rouge_metric.rouge_scores(answer, ground_truth)

        if "ragas" in metrics:
            print(f"  [ragas] scoring {qa['id']} ...")
            entry["metrics"]["ragas"] = ragas_local.evaluate_ragas(
                question=qa["question"],
                answer=answer,
                contexts=record["contexts"],
                ground_truth=ground_truth,
            )

        if "laaj" in metrics:
            print(f"  [laaj]  judging {qa['id']} ...")
            entry["metrics"]["laaj"] = laaj_metric.judge_answer(
                question=qa["question"],
                answer=answer,
                context=record["context_string"],
                ground_truth=ground_truth,
            )

        scored.append(entry)
        _write_checkpoint(checkpoint_path, scored)

    return scored


def _write_checkpoint(path: Path | None, scored: list[dict]) -> None:
    """Best-effort partial dump. A checkpoint failure must never kill the run."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"partial": True, "scored_so_far": len(scored), "per_question": scored},
                      f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[eval] checkpoint write failed (continuing): {exc}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values: list) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(clean), 4) if clean else None


def aggregate(scored: list[dict], records: list[dict], metrics: tuple[str, ...]) -> dict:
    """Builds the corpus-level summary across all scored questions."""
    usable = [s for s in scored if not s.get("error") and s["generated_answer"].strip()]
    summary = {
        "questions_total": len(scored),
        "questions_answered": len(usable),
        "mean_latency_seconds": _mean([r["latency_seconds"] for r in records]),
    }

    if not usable:
        summary["note"] = "no questions produced an answer — nothing to score"
        return summary

    if "bleu" in metrics:
        # Corpus BLEU pools n-gram counts before dividing; that is the correct
        # aggregate, and it will not equal the mean of the per-question scores.
        summary["bleu"] = {
            "corpus_bleu": bleu_metric.corpus_bleu(
                [s["generated_answer"] for s in usable],
                [s["ground_truth"] for s in usable],
            ),
            "mean_sentence_bleu": _mean([s["metrics"]["bleu"]["bleu"] for s in usable]),
        }

    if "rouge" in metrics:
        summary["rouge"] = rouge_metric.aggregate_rouge([s["metrics"]["rouge"] for s in usable])

    if "ragas" in metrics:
        ragas_entries = [s["metrics"]["ragas"] for s in usable]
        summary["ragas"] = {
            name: _mean([e[name].get("score") for e in ragas_entries])
            for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        }
        summary["ragas"]["ragas_score"] = _mean([e.get("ragas_score") for e in ragas_entries])

    if "laaj" in metrics:
        summary["laaj"] = laaj_metric.aggregate_laaj([s["metrics"]["laaj"] for s in usable])

    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def print_report(summary: dict, scored: list[dict], metrics: tuple[str, ...]) -> None:
    line = "=" * 74
    print(f"\n{line}\n  SPECTER RAG EVALUATION — SUMMARY\n{line}")
    print(f"  Questions answered : {summary['questions_answered']}/{summary['questions_total']}")
    print(f"  Mean latency       : {_fmt(summary.get('mean_latency_seconds'))}s")

    if "bleu" in metrics and "bleu" in summary:
        print(f"\n  BLEU  — n-gram precision vs reference (surface overlap)")
        print(f"    corpus BLEU-4      : {_fmt(summary['bleu']['corpus_bleu']['bleu'])}")
        print(f"    mean sentence BLEU : {_fmt(summary['bleu']['mean_sentence_bleu'])}")
        precisions = summary["bleu"]["corpus_bleu"]["precisions"]
        print(f"    precisions 1-4     : {', '.join(_fmt(p) for p in precisions)}")
        print(f"    brevity penalty    : {_fmt(summary['bleu']['corpus_bleu']['brevity_penalty'])}")

    if "rouge" in metrics and "rouge" in summary:
        print(f"\n  ROUGE — reference coverage (recall-oriented)")
        for variant in ("rouge1", "rouge2", "rougeL"):
            scores = summary["rouge"][variant]
            print(f"    {variant:<8} P {_fmt(scores['precision'])}  "
                  f"R {_fmt(scores['recall'])}  F1 {_fmt(scores['f1'])}")

    if "ragas" in metrics and "ragas" in summary:
        print(f"\n  RAGAS — component-level diagnosis")
        print(f"    faithfulness       : {_fmt(summary['ragas']['faithfulness'])}   (generation: grounded?)")
        print(f"    answer_relevancy   : {_fmt(summary['ragas']['answer_relevancy'])}   (generation: on-topic?)")
        print(f"    context_precision  : {_fmt(summary['ragas']['context_precision'])}   (retrieval: ranked well?)")
        print(f"    context_recall     : {_fmt(summary['ragas']['context_recall'])}   (retrieval: found it all?)")
        print(f"    ragas_score        : {_fmt(summary['ragas']['ragas_score'])}")

    if "laaj" in metrics and "laaj" in summary and "per_dimension_5" in summary["laaj"]:
        print(f"\n  LaaJ  — LLM judge rubric (1-5)")
        for dimension, score in summary["laaj"]["per_dimension_5"].items():
            print(f"    {dimension:<18} : {score}/5")
        print(f"    mean               : {summary['laaj']['mean_score_5']}/5 "
              f"(normalised {_fmt(summary['laaj']['normalised_score'])})")

    # Per-question table — makes the weak questions obvious at a glance.
    print(f"\n{line}\n  PER-QUESTION\n{line}")
    header = f"  {'ID':<5}{'BLEU':>8}{'R-L F1':>9}{'Faith':>8}{'C-Rec':>8}{'LaaJ':>7}  Question"
    print(header)
    print("  " + "-" * 70)
    for entry in scored:
        m = entry["metrics"]
        bleu_score = m.get("bleu", {}).get("bleu")
        rouge_l = m.get("rouge", {}).get("rougeL", {}).get("f1")
        faith = m.get("ragas", {}).get("faithfulness", {}).get("score")
        recall = m.get("ragas", {}).get("context_recall", {}).get("score")
        judge = m.get("laaj", {}).get("mean_score_5")

        def cell(value, width, suffix=""):
            return f"{value}{suffix}".rjust(width) if isinstance(value, (int, float)) else "—".rjust(width)

        print(f"  {entry['id']:<5}"
              f"{cell(round(bleu_score, 3) if isinstance(bleu_score, float) else bleu_score, 8)}"
              f"{cell(rouge_l, 9)}"
              f"{cell(faith, 8)}"
              f"{cell(recall, 8)}"
              f"{cell(judge, 7)}"
              f"  {entry['question'][:38]}")
    print(line + "\n")


def print_interpretation(summary: dict, metrics: tuple[str, ...]) -> None:
    """
    Turns the numbers into the one thing worth acting on.

    The rules below are heuristics for reading a RAG scorecard, not thresholds
    with any formal basis — they encode which component to suspect when a
    given pair of metrics disagree.
    """
    print("  READING THE RESULTS")
    print("  " + "-" * 70)

    ragas = summary.get("ragas", {}) if "ragas" in metrics else {}
    faith = ragas.get("faithfulness")
    recall = ragas.get("context_recall")
    precision = ragas.get("context_precision")
    bleu_score = summary.get("bleu", {}).get("corpus_bleu", {}).get("bleu") if "bleu" in metrics else None
    rouge_recall = summary.get("rouge", {}).get("rouge1", {}).get("recall") if "rouge" in metrics else None
    judge = summary.get("laaj", {}).get("normalised_score") if "laaj" in metrics else None

    notes = []

    if isinstance(bleu_score, float) and bleu_score < 0.15 and isinstance(judge, float) and judge > 0.6:
        notes.append(
            "BLEU is low while the judge scores well — expected. The system\n"
            "    paraphrases into plain English by design, so exact n-gram overlap\n"
            "    with the reference stays low. Do not tune against BLEU here."
        )

    if isinstance(recall, float) and recall < 0.6:
        notes.append(
            "context_recall is low — the retriever is missing clauses the answer\n"
            "    needs. Raise --top-k, or revisit chunking in clause_extractor.py.\n"
            "    No prompt change will fix this."
        )

    if isinstance(faith, float) and isinstance(recall, float) and faith > 0.8 and recall < 0.6:
        notes.append(
            "High faithfulness with low context_recall is the dangerous pattern:\n"
            "    answers are faithful to the wrong context. They will read as\n"
            "    confident and still be incomplete."
        )

    if isinstance(precision, float) and precision < 0.5:
        notes.append(
            "context_precision is low — useful clauses are being out-ranked by\n"
            "    noise. Check the graph expansion (--no-graph to isolate it)."
        )

    if isinstance(rouge_recall, float) and rouge_recall < 0.4:
        notes.append(
            "ROUGE-1 recall is low — answers cover little of the reference.\n"
            "    Usually terseness rather than error; check completeness in LaaJ."
        )

    if not notes:
        notes.append("No systematic weakness stands out. Inspect the lowest per-question\n    rows above for individual failures.")

    for note in notes:
        print(f"  • {note}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Specter RAG pipeline.")
    parser.add_argument("--pdf", type=str, default=None,
                        help="Path to the PDF (defaults to the one named in the dataset).")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH),
                        help="Path to the QA dataset JSON.")
    parser.add_argument("--top-k", type=int, default=5, help="Clauses to retrieve per question.")
    parser.add_argument("--no-graph", action="store_true",
                        help="Disable graph expansion, isolating pure vector retrieval.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Reuse the ChromaDB collection as-is. Graph expansion is "
                             "unavailable in this mode since the graph is built at ingest.")
    parser.add_argument("--metrics", type=str, default=",".join(ALL_METRICS),
                        help=f"Comma-separated subset of: {', '.join(ALL_METRICS)}")
    parser.add_argument("--from-answers", type=str, default=None,
                        help="Skip ingest and generation entirely, scoring a cached "
                             "answers file from a previous run (see results/answers_cache.json). "
                             "Lets you re-score without paying for generation again.")
    args = parser.parse_args()

    metrics = tuple(m.strip().lower() for m in args.metrics.split(",") if m.strip())
    unknown = [m for m in metrics if m not in ALL_METRICS]
    if unknown:
        parser.error(f"unknown metric(s): {', '.join(unknown)}. Valid: {', '.join(ALL_METRICS)}")

    dataset = load_dataset(Path(args.dataset))
    qa_pairs = dataset["qa_pairs"]
    pdf_path = args.pdf or str(PROJECT_ROOT / dataset["document_path"])

    print("=" * 74)
    print("  SPECTER RAG EVALUATION")
    print("=" * 74)
    print(f"  Document : {Path(pdf_path).name}")
    print(f"  Questions: {len(qa_pairs)}")
    print(f"  Metrics  : {', '.join(metrics)}")
    print(f"  Retrieval: top_k={args.top_k}, graph={'off' if args.no_graph else 'on'}")
    print("=" * 74 + "\n")

    run_start = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    answers_cache = RESULTS_DIR / "answers_cache.json"

    if args.from_answers:
        cache_path = Path(args.from_answers)
        with open(cache_path, encoding="utf-8") as f:
            records = json.load(f)["records"]
        print(f"[eval] --from-answers: scoring {len(records)} cached answers "
              f"from {cache_path.name}, skipping ingest and generation.\n")
        if len(records) != len(qa_pairs):
            parser.error(
                f"cached answers ({len(records)}) do not match the dataset "
                f"({len(qa_pairs)} questions) — they must correspond one to one"
            )
    else:
        graph = None
        if args.skip_ingest:
            print("[eval] --skip-ingest: reusing existing ChromaDB, graph expansion disabled.\n")
        else:
            _, graph = ingest_pdf(pdf_path)

        records = run_rag(
            questions=[qa["question"] for qa in qa_pairs],
            graph=graph,
            top_k=args.top_k,
            use_graph=not args.no_graph,
        )

        # Persist answers before scoring — generation is the expensive half, and
        # an interruption during scoring should never cost it.
        with open(answers_cache, "w", encoding="utf-8") as f:
            json.dump({"document": Path(pdf_path).name, "records": records},
                      f, indent=2, ensure_ascii=False)
        print(f"\n[eval] Answers cached to {answers_cache.relative_to(PROJECT_ROOT)}")
        print("[eval] Re-score without regenerating via --from-answers.")

    print("\n[eval] Scoring answers...")
    scored = score_records(records, qa_pairs, metrics,
                           checkpoint_path=RESULTS_DIR / "_checkpoint.json")
    summary = aggregate(scored, records, metrics)

    print_report(summary, scored, metrics)
    print_interpretation(summary, metrics)

    # Persist the full run — per-question detail included, so a score can
    # always be traced back to the answer and the clauses that produced it.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"

    payload = {
        "run": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "document": Path(pdf_path).name,
            "metrics": list(metrics),
            "top_k": args.top_k,
            "graph_expansion": not args.no_graph,
            "skip_ingest": args.skip_ingest,
            "total_runtime_seconds": round(time.time() - run_start, 1),
        },
        "summary": summary,
        "per_question": scored,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Stable pointer to the most recent run, for diffing between runs.
    latest_path = RESULTS_DIR / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # The run completed, so the partial checkpoint is now redundant.
    checkpoint = RESULTS_DIR / "_checkpoint.json"
    if checkpoint.exists():
        checkpoint.unlink()

    print(f"  Full results : {output_path.relative_to(PROJECT_ROOT)}")
    print(f"  Latest run   : {latest_path.relative_to(PROJECT_ROOT)}")
    print(f"  Total runtime: {payload['run']['total_runtime_seconds']}s\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
