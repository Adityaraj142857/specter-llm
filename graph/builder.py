"""
graph/builder.py

Builds and persists the NetworkX clause graph.

Nodes  → each clause (id, heading, text, section, index)
Edges  → directed relationships between clauses:
    - SIMILAR_TO      (undirected, cosine score)
    - DEPENDS_ON      (A cannot exist without B)
    - CONFLICTS_WITH  (A contradicts B)
    - OVERRIDES       (A supersedes B)

Edge attributes:
    type  : str   — one of the four types above
    score : float — 0.0 to 1.0 (strength of relationship)

Graph is saved/loaded from ./data/clause_graph.pkl using pickle.
"""

import os
import json
import pickle
import re
import requests
import networkx as nx
from itertools import combinations
from typing import Optional
from graph.definitions import add_definition_edges


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "llama3.2"
GRAPH_PATH      = "./data/clause_graph.pkl"
SIMILARITY_THRESHOLD = 0.25   # cosine distance below this → add SIMILAR_TO edge
                               # (ChromaDB distance: 0 = identical, 1 = orthogonal)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(clauses: list[dict], similarity_results: dict[str, list[dict]]) -> nx.DiGraph:
    """
    Builds a fresh graph from a list of clause dicts.

    Args:
        clauses           : list of clause dicts (must have id, heading, text, section, index)
        similarity_results: mapping of clause_id → list of similar clause dicts from ChromaDB
                            (each with 'id' and 'distance')

    Returns:
        A NetworkX DiGraph.  Also saves it to GRAPH_PATH.
    """
    G = nx.DiGraph()

    # Add nodes
    for clause in clauses:
        G.add_node(
            clause["id"],
            heading=clause.get("heading", ""),
            text=clause.get("text", ""),
            section=clause.get("section", ""),
            index=clause.get("index", -1),
        )

    # Add SIMILAR_TO edges from vector search results
    _add_similarity_edges(G, similarity_results)

    # Add semantic edges (DEPENDS_ON, CONFLICTS_WITH, OVERRIDES) via LLM
    # Only run LLM on pairs already connected by similarity to bound API calls
    _add_semantic_edges(G, clauses)

    # Add DEFINES / USES edges from terminology detection
    add_definition_edges(G, clauses)

    save_graph(G)
    print(f"[graph_builder] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def add_clause_to_graph(
    G: nx.DiGraph,
    new_clause: dict,
    similar_clauses: list[dict],
) -> nx.DiGraph:
    """
    Adds a single new clause node and its edges to an existing graph.
    Used by the impact engine when a new clause arrives.

    Args:
        G              : existing graph (modified in place)
        new_clause     : clause dict with id, heading, text, section, index
        similar_clauses: top-k results from vector store for this clause

    Returns:
        Updated graph (also saved to disk).
    """
    # Add node
    G.add_node(
        new_clause["id"],
        heading=new_clause.get("heading", ""),
        text=new_clause.get("text", ""),
        section=new_clause.get("section", ""),
        index=new_clause.get("index", -1),
    )

    # Similarity edges
    for sim in similar_clauses:
        if sim["distance"] <= SIMILARITY_THRESHOLD and sim["id"] != new_clause["id"]:
            score = round(1.0 - sim["distance"], 4)
            G.add_edge(new_clause["id"], sim["id"], type="SIMILAR_TO", score=score)
            G.add_edge(sim["id"], new_clause["id"], type="SIMILAR_TO", score=score)

    # Semantic edges: classify against each similar clause
    neighbors = [c["id"] for c in similar_clauses if c["id"] != new_clause["id"]]
    for neighbor_id in neighbors:
        if not G.has_node(neighbor_id):
            continue
        neighbor_text    = G.nodes[neighbor_id].get("text", "")
        neighbor_heading = G.nodes[neighbor_id].get("heading", "")
        edges = _classify_relationship(
            new_clause["id"], new_clause["heading"], new_clause["text"],
            neighbor_id,      neighbor_heading,      neighbor_text,
        )
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], type=edge["type"], score=edge["score"])

    # Check if new clause defines or uses any terms
    add_definition_edges(G, [new_clause])

    save_graph(G)
    return G


def load_graph() -> Optional[nx.DiGraph]:
    """Loads the persisted graph from disk. Returns None if not found."""
    if not os.path.exists(GRAPH_PATH):
        return None
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def save_graph(G: nx.DiGraph) -> None:
    """Saves the graph to disk."""
    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)


# ---------------------------------------------------------------------------
# Internal — similarity edges
# ---------------------------------------------------------------------------

def _add_similarity_edges(G: nx.DiGraph, similarity_results: dict[str, list[dict]]) -> None:
    """
    Adds undirected SIMILAR_TO edges from ChromaDB query results.
    Skips pairs already connected and self-loops.
    """
    added = set()
    for clause_id, similars in similarity_results.items():
        for sim in similars:
            target_id = sim["id"]
            if target_id == clause_id:
                continue
            if sim["distance"] > SIMILARITY_THRESHOLD:
                continue
            pair = tuple(sorted([clause_id, target_id]))
            if pair in added:
                continue
            score = round(1.0 - sim["distance"], 4)
            G.add_edge(clause_id, target_id, type="SIMILAR_TO", score=score)
            G.add_edge(target_id, clause_id, type="SIMILAR_TO", score=score)
            added.add(pair)


# ---------------------------------------------------------------------------
# Internal — semantic edges via LLM
# ---------------------------------------------------------------------------

def _add_semantic_edges(G: nx.DiGraph, clauses: list[dict]) -> None:
    """
    For each SIMILAR_TO edge pair, asks the LLM to classify the deeper
    relationship (DEPENDS_ON, CONFLICTS_WITH, OVERRIDES, or none).
    Adds edges only when a meaningful relationship is found.
    """
    clause_map = {c["id"]: c for c in clauses}

    # Only check pairs already connected by similarity
    checked = set()
    for u, v, data in list(G.edges(data=True)):
        if data.get("type") != "SIMILAR_TO":
            continue
        pair = tuple(sorted([u, v]))
        if pair in checked:
            continue
        checked.add(pair)

        if u not in clause_map or v not in clause_map:
            continue

        cu = clause_map[u]
        cv = clause_map[v]

        edges = _classify_relationship(
            cu["id"], cu["heading"], cu["text"],
            cv["id"], cv["heading"], cv["text"],
        )
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], type=edge["type"], score=edge["score"])


def _classify_relationship(
    id_a: str, heading_a: str, text_a: str,
    id_b: str, heading_b: str, text_b: str,
) -> list[dict]:
    """
    Calls llama3.2 to determine the semantic relationship between two clauses.
    Returns a list of edge dicts (may be empty if no strong relationship found).

    Each edge dict: { source, target, type, score }
    """
    prompt = f"""You are a governance document analyst reviewing two clauses from a college constitution.

CLAUSE A — "{heading_a}":
{text_a}

CLAUSE B — "{heading_b}":
{text_b}

Determine the relationship between these two clauses. Respond ONLY with a JSON array.
Each item in the array must be one relationship and have these exact keys:
  - "source": "A" or "B"
  - "target": "A" or "B"
  - "type": one of ["DEPENDS_ON", "CONFLICTS_WITH", "OVERRIDES", "DEFINES", "USES"]
  - "score": a float from 0.0 to 1.0 indicating relationship strength
  - "reason": one sentence explanation

Rules:
- DEPENDS_ON: source clause cannot function without target clause existing
- CONFLICTS_WITH: source clause directly contradicts target clause
- OVERRIDES: source clause supersedes or replaces target clause
- DEFINES: source clause formally defines a term used in target clause
- USES: source clause uses a term formally defined in target clause
- Only include relationships with score >= 0.5
- If no meaningful relationship exists beyond similarity, return []
- No markdown, no extra text. Only the JSON array.
"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
        # Extract only the JSON array — ignore any text before or after
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        raw = match.group(0)
        # Fix common LLM JSON mistakes
        raw = re.sub(r",\s*([}\]])", r"\1", raw)   # trailing commas
        raw = re.sub(r"'", '"', raw)                # single quotes → double
        raw = re.sub(r"(\w+):", r'"\1":', raw)      # unquoted keys → quoted
        raw = re.sub(r'""(\w)', r'"\1', raw)        # fix double-quote artifacts

        items = json.loads(raw)
        edges = []
        for item in items:
            source_id = id_a if item.get("source") == "A" else id_b
            target_id = id_b if item.get("target") == "B" else id_a
            if source_id == target_id:
                continue
            edges.append({
                "source": source_id,
                "target": target_id,
                "type":   item["type"],
                "score":  float(item.get("score", 0.5)),
                "reason": item.get("reason", ""),
            })
        return edges

    except Exception as e:
        print(f"[graph_builder] LLM edge classification failed: {e}")
        return []
