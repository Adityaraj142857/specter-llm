"""
query/impact.py

The new-clause impact engine.

Given a new clause (text), this module:
1. Embeds it using nomic-embed-text
2. Finds similar existing clauses via ChromaDB
3. Adds it to the graph with classified edges
4. Traverses the graph to produce a full impact report

Usage:
    from query.impact import analyse_new_clause

    report = analyse_new_clause(
        new_clause_text="The president shall serve for a term of three years.",
        new_clause_heading="5.3 Presidential Term",
        commit_to_graph=False,   # True = permanently add to graph
    )
"""

import json
import re
import requests
from ingestion.embedder import embed_text, embed_query
from storage.vector_store import VectorStore
from graph.builder import load_graph, add_clause_to_graph, save_graph
from graph.traversal import get_impact_subgraph
from ingestion.clause_extractor import _slugify


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
TOP_K_SIMILAR = 8   # how many similar clauses to retrieve for analysis


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_new_clause(
    new_clause_text: str,
    new_clause_heading: str = "New Clause",
    commit_to_graph: bool = False,
) -> dict:
    """
    Full impact analysis pipeline for a new clause.

    Args:
        new_clause_text    : the full text of the new clause
        new_clause_heading : a title for the new clause
        commit_to_graph    : if True, permanently adds the clause to the graph

    Returns a report dict:
    {
        "clause_id":       str,
        "heading":         str,
        "similar":         [...],
        "dependencies":    [...],
        "dependents":      [...],
        "conflicts":       [...],
        "overrides":       {...},
        "change_required": [...],
        "llm_summary":     str,    ← plain-English impact summary from LLM
        "summary":         {...},
        "committed":       bool
    }
    """
    # ── 1. Load existing graph ──────────────────────────────────────────────
    G = load_graph()
    if G is None:
        return {
            "error": "No graph found. Please ingest the constitution first.",
            "committed": False,
        }

    vs = VectorStore()
    if vs.count() == 0:
        return {
            "error": "Vector store is empty. Please ingest the constitution first.",
            "committed": False,
        }

    # ── 2. Build clause dict ────────────────────────────────────────────────
    clause_id = _slugify(f"new_{new_clause_heading}")
    # Avoid ID collisions with existing nodes
    base_id = clause_id
    counter = 1
    while G.has_node(clause_id):
        clause_id = f"{base_id}_{counter}"
        counter += 1

    new_clause = {
        "id":      clause_id,
        "heading": new_clause_heading,
        "text":    new_clause_text,
        "section": "new",
        "index":   -1,
    }

    # ── 3. Embed and find similar ───────────────────────────────────────────
    embedding = embed_text(f"search_document: {new_clause_heading}\n\n{new_clause_text}")
    new_clause["embedding"] = embedding

    similar_clauses = vs.query(embedding, top_k=TOP_K_SIMILAR)

    # ── 4. Add to graph (in-memory, optionally committed) ──────────────────
    G_working = add_clause_to_graph(G, new_clause, similar_clauses)

    # ── 5. Traverse for impact ─────────────────────────────────────────────
    impact = get_impact_subgraph(G_working, clause_id, max_hops=2)

    # ── 6. LLM plain-English summary ───────────────────────────────────────
    impact["llm_summary"] = _generate_llm_summary(new_clause_heading, new_clause_text, impact)

    # ── 7. Commit or rollback ──────────────────────────────────────────────
    if commit_to_graph:
        vs.add_single_clause(new_clause)
        save_graph(G_working)
        impact["committed"] = True
    else:
        # Don't persist — analysis only
        impact["committed"] = False

    return impact


# ---------------------------------------------------------------------------
# LLM plain-English summary
# ---------------------------------------------------------------------------

def _generate_llm_summary(
    heading: str,
    text: str,
    impact: dict,
) -> str:
    """
    Asks the LLM to write a 3-5 sentence plain-English summary of the
    impact analysis results, suitable for showing to a non-technical user.
    """
    conflicts_list = "\n".join(
        f"  - {c['heading']} ({c['label']} conflict)" for c in impact.get("conflicts", [])
    ) or "  None found."

    dependents_list = "\n".join(
        f"  - {d['heading']} (hop {d['hop']}, {d['label']})" for d in impact.get("dependents", [])
    ) or "  None found."

    changes_list = "\n".join(
        f"  - {ch}" for ch in impact.get("change_required", [])
    ) or "  No changes required."

    prompt = f"""You are a governance document analyst. A new clause is being considered for addition to a college constitution.

New clause heading: "{heading}"
New clause text: "{text}"

Impact analysis found:
Conflicts with existing clauses:
{conflicts_list}

Clauses that would be affected (dependents):
{dependents_list}

Required changes:
{changes_list}

Write a clear 3-5 sentence plain-English summary of:
1. What this clause does
2. Whether it's safe to add (based on conflicts)
3. What other parts of the constitution would need updating

Write for a college student council member, not a lawyer. Be direct. No bullet points.
"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"Summary unavailable: {e}"
