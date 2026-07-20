"""
evaluation/rag_runner.py

Drives the real Specter RAG pipeline over a PDF and captures, per question,
exactly what the app would have produced: the retrieved clauses and the
generated answer.

This deliberately mirrors app.py rather than reimplementing retrieval — same
ingest path, same embed_query → VectorStore.query → graph expansion → context
string → answer_question chain. If app.py changes, this must change with it,
or the evaluation stops measuring the shipped system.

WARNING: ingest_constitution() calls VectorStore.clear() on every run, which
deletes whatever is currently in ./data/chroma_db. That is the app's existing
"fresh ingest each time" behaviour, not something added here — but running an
evaluation will wipe a constitution you previously ingested through the UI.
Re-upload it afterwards.
"""

import time
from pathlib import Path

from ingestion.pdf_reader import ingest_constitution
from ingestion.embedder import embed_query
from storage.vector_store import VectorStore
from query.qa import answer_question


DEFAULT_TOP_K = 5
DEFAULT_USE_GRAPH = True


def ingest_pdf(pdf_path: str | Path) -> tuple[list[dict], object]:
    """
    Runs the app's ingest pipeline on a PDF file.

    Returns (clauses, graph). Embedding every clause through local Ollama is
    the slow part — expect a minute or two for a contract of this size.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()

    print(f"[runner] Ingesting {pdf_path.name} ...")
    print("[runner] NOTE: this clears the existing ChromaDB collection.")
    start = time.time()
    clauses, graph = ingest_constitution(pdf_bytes)
    print(f"[runner] Ingested {len(clauses)} clauses in {time.time() - start:.1f}s")

    return clauses, graph


def retrieve(
    question: str,
    graph=None,
    top_k: int = DEFAULT_TOP_K,
    use_graph: bool = DEFAULT_USE_GRAPH,
) -> list[dict]:
    """
    Vector retrieval plus graph expansion — identical to the app's
    "Ask a Question" page, including the graph-expanded 0.5 placeholder
    distance for neighbours that were not similarity-ranked.
    """
    vs = VectorStore()
    query_embedding = embed_query(question)
    results = vs.query(query_embedding, top_k=top_k)

    if use_graph and graph is not None:
        extra_ids = set()
        for result in results:
            clause_id = result["id"]
            if graph.has_node(clause_id):
                for _, target, data in graph.out_edges(clause_id, data=True):
                    if data.get("type") in ("DEPENDS_ON", "CONFLICTS_WITH"):
                        extra_ids.add(target)

        existing_ids = {r["id"] for r in results}
        for extra_id in extra_ids:
            if extra_id not in existing_ids:
                node = graph.nodes.get(extra_id, {})
                results.append({
                    "id": extra_id,
                    "heading": node.get("heading", ""),
                    "text": node.get("text", ""),
                    "section": node.get("section", ""),
                    "distance": 0.5,          # graph-expanded, not similarity-ranked
                    "graph_expanded": True,
                })

    return results


def build_context(results: list[dict]) -> str:
    """The exact context string format app.py hands to the LLM."""
    return "\n\n---\n\n".join(f"[{r['heading']}]\n{r['text']}" for r in results)


def run_rag(
    questions: list[str],
    graph=None,
    top_k: int = DEFAULT_TOP_K,
    use_graph: bool = DEFAULT_USE_GRAPH,
) -> list[dict]:
    """
    Answers each question through the full pipeline.

    Returns one record per question containing the answer, the raw retrieved
    clauses, the assembled context string and the latency — everything the
    four evaluators need downstream.
    """
    records = []

    for i, question in enumerate(questions, start=1):
        print(f"[runner] ({i}/{len(questions)}) {question[:70]}...")
        start = time.time()

        try:
            results = retrieve(question, graph=graph, top_k=top_k, use_graph=use_graph)
            context = build_context(results)
            answer = answer_question(question, context)
            error = None
        except Exception as exc:
            # One failed question should not abandon the whole run — record it
            # and let the metrics treat it as a failure rather than crashing.
            print(f"[runner] ERROR on question {i}: {exc}")
            results, context, answer, error = [], "", "", str(exc)

        records.append({
            "question": question,
            "answer": answer,
            "contexts": [r["text"] for r in results],
            "retrieved_clauses": [
                {
                    "id": r["id"],
                    "heading": r.get("heading", ""),
                    "section": r.get("section", ""),
                    "distance": r.get("distance"),
                    "graph_expanded": r.get("graph_expanded", False),
                    "text_preview": r["text"][:200],
                }
                for r in results
            ],
            "context_string": context,
            "latency_seconds": round(time.time() - start, 2),
            "error": error,
        })

    return records
