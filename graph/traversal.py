"""
graph/traversal.py

Graph traversal utilities for the impact engine.

Key functions:
    get_dependencies(G, clause_id)   → all clauses this one depends on
    get_dependents(G, clause_id)     → all clauses that depend on this one
    get_conflicts(G, clause_id)      → all clauses that conflict
    get_impact_subgraph(G, clause_id)→ full neighbourhood for impact report
    score_label(score)               → "strong" / "moderate" / "weak"
"""

import networkx as nx
from typing import Optional


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def score_label(score: float) -> str:
    """Converts a numeric edge score to a human-readable strength label."""
    if score >= 0.8:
        return "strong"
    elif score >= 0.5:
        return "moderate"
    else:
        return "weak"


def score_emoji(score: float) -> str:
    if score >= 0.8:
        return "🔴"
    elif score >= 0.5:
        return "🟡"
    else:
        return "🟢"


# ---------------------------------------------------------------------------
# Traversal functions
# ---------------------------------------------------------------------------

def get_dependencies(
    G: nx.DiGraph,
    clause_id: str,
    max_hops: int = 3,
) -> list[dict]:
    """
    Returns all clauses that `clause_id` DEPENDS_ON, up to max_hops deep.
    Results are sorted by hop distance then score descending.

    Each result dict:
        id, heading, text, section, hop, score, label
    """
    return _bfs_edges(G, clause_id, "DEPENDS_ON", direction="successors", max_hops=max_hops)


def get_dependents(
    G: nx.DiGraph,
    clause_id: str,
    max_hops: int = 3,
) -> list[dict]:
    """
    Returns all clauses that DEPEND_ON `clause_id` (reverse direction).
    These are clauses that would be broken if clause_id changed.
    """
    return _bfs_edges(G, clause_id, "DEPENDS_ON", direction="predecessors", max_hops=max_hops)


def get_conflicts(
    G: nx.DiGraph,
    clause_id: str,
) -> list[dict]:
    """
    Returns all clauses that directly CONFLICTS_WITH `clause_id`.
    Conflicts are bidirectional so we check both directions.
    """
    conflicts = []
    seen = set()

    for neighbor in list(G.successors(clause_id)) + list(G.predecessors(clause_id)):
        if neighbor in seen:
            continue
        seen.add(neighbor)

        # Check both edge directions for CONFLICTS_WITH
        for u, v in [(clause_id, neighbor), (neighbor, clause_id)]:
            if G.has_edge(u, v):
                data = G[u][v]
                if data.get("type") == "CONFLICTS_WITH":
                    node_data = G.nodes.get(neighbor, {})
                    conflicts.append({
                        "id":      neighbor,
                        "heading": node_data.get("heading", ""),
                        "text":    node_data.get("text", ""),
                        "section": node_data.get("section", ""),
                        "score":   data.get("score", 0.5),
                        "label":   score_label(data.get("score", 0.5)),
                        "reason":  data.get("reason", ""),
                    })
                    break

    return sorted(conflicts, key=lambda x: x["score"], reverse=True)


def get_overrides(
    G: nx.DiGraph,
    clause_id: str,
) -> dict:
    """
    Returns:
        overrides   : clauses that clause_id overrides
        overridden_by: clauses that override clause_id
    """
    overrides = []
    overridden_by = []

    for _, v, data in G.out_edges(clause_id, data=True):
        if data.get("type") == "OVERRIDES":
            node_data = G.nodes.get(v, {})
            overrides.append({
                "id":      v,
                "heading": node_data.get("heading", ""),
                "score":   data.get("score", 0.5),
                "label":   score_label(data.get("score", 0.5)),
                "reason":  data.get("reason", ""),
            })

    for u, _, data in G.in_edges(clause_id, data=True):
        if data.get("type") == "OVERRIDES":
            node_data = G.nodes.get(u, {})
            overridden_by.append({
                "id":      u,
                "heading": node_data.get("heading", ""),
                "score":   data.get("score", 0.5),
                "label":   score_label(data.get("score", 0.5)),
                "reason":  data.get("reason", ""),
            })

    return {"overrides": overrides, "overridden_by": overridden_by}


def get_similar(
    G: nx.DiGraph,
    clause_id: str,
    min_score: float = 0.6,
) -> list[dict]:
    """
    Returns clauses with a SIMILAR_TO edge above the minimum score threshold.
    """
    results = []
    for _, v, data in G.out_edges(clause_id, data=True):
        if data.get("type") == "SIMILAR_TO" and data.get("score", 0) >= min_score:
            node_data = G.nodes.get(v, {})
            results.append({
                "id":      v,
                "heading": node_data.get("heading", ""),
                "text":    node_data.get("text", ""),
                "section": node_data.get("section", ""),
                "score":   data.get("score", 0.5),
                "label":   score_label(data.get("score", 0.5)),
            })
    return sorted(results, key=lambda x: x["score"], reverse=True)


# ---------------------------------------------------------------------------
# Full impact report
# ---------------------------------------------------------------------------

def get_impact_subgraph(
    G: nx.DiGraph,
    clause_id: str,
    max_hops: int = 2,
) -> dict:
    """
    Master function used by the impact engine.
    Returns a full impact dict for a given clause:

    {
        "clause_id":    str,
        "heading":      str,
        "dependencies": [...],   # what this clause needs
        "dependents":   [...],   # what would break if this clause changes
        "conflicts":    [...],   # direct contradictions
        "overrides":    {...},   # override relationships
        "similar":      [...],   # semantically related clauses
        "change_required": [...] # human-readable list of required amendments
    }
    """
    if not G.has_node(clause_id):
        return {"error": f"Clause '{clause_id}' not found in graph."}

    node_data    = G.nodes[clause_id]
    dependencies = get_dependencies(G, clause_id, max_hops)
    dependents   = get_dependents(G, clause_id, max_hops)
    conflicts    = get_conflicts(G, clause_id)
    overrides    = get_overrides(G, clause_id)
    similar      = get_similar(G, clause_id)

    change_required = _generate_change_list(
        clause_id, dependencies, dependents, conflicts, overrides
    )

    return {
        "clause_id":       clause_id,
        "heading":         node_data.get("heading", ""),
        "section":         node_data.get("section", ""),
        "dependencies":    dependencies,
        "dependents":      dependents,
        "conflicts":       conflicts,
        "overrides":       overrides,
        "similar":         similar,
        "change_required": change_required,
        "summary": {
            "total_affected":   len(dependents) + len(conflicts),
            "conflict_count":   len(conflicts),
            "dependent_count":  len(dependents),
            "dependency_count": len(dependencies),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bfs_edges(
    G: nx.DiGraph,
    start: str,
    edge_type: str,
    direction: str,  # "successors" or "predecessors"
    max_hops: int,
) -> list[dict]:
    """
    BFS traversal following only edges of a specific type.
    Returns list of reached nodes with hop distance and edge score.
    """
    visited = {start}
    queue   = [(start, 0)]
    results = []

    while queue:
        node, hop = queue.pop(0)
        if hop >= max_hops:
            continue

        neighbors_fn = G.successors if direction == "successors" else G.predecessors
        for neighbor in neighbors_fn(node):
            if neighbor in visited:
                continue

            # Check edge type in correct direction
            u, v = (node, neighbor) if direction == "successors" else (neighbor, node)
            if not G.has_edge(u, v):
                continue
            edge_data = G[u][v]
            if edge_data.get("type") != edge_type:
                continue

            visited.add(neighbor)
            queue.append((neighbor, hop + 1))

            node_data = G.nodes.get(neighbor, {})
            score = edge_data.get("score", 0.5)
            results.append({
                "id":      neighbor,
                "heading": node_data.get("heading", ""),
                "text":    node_data.get("text", ""),
                "section": node_data.get("section", ""),
                "hop":     hop + 1,
                "score":   score,
                "label":   score_label(score),
                "reason":  edge_data.get("reason", ""),
            })

    return sorted(results, key=lambda x: (x["hop"], -x["score"]))


def _generate_change_list(
    clause_id: str,
    dependencies: list[dict],
    dependents: list[dict],
    conflicts: list[dict],
    overrides: dict,
) -> list[str]:
    """
    Produces a human-readable list of changes required when this clause
    is added or modified.
    """
    changes = []

    for dep in dependencies:
        changes.append(
            f"Ensure '{dep['heading']}' exists and is consistent "
            f"({dep['label']} dependency, hop {dep['hop']})"
        )

    for dep in dependents:
        changes.append(
            f"Review '{dep['heading']}' — it depends on this clause "
            f"and may need updating ({dep['label']} link)"
        )

    for conflict in conflicts:
        changes.append(
            f"CONFLICT: '{conflict['heading']}' directly contradicts this clause "
            f"— one must be amended ({conflict['label']} conflict)"
        )

    for ov in overrides.get("overridden_by", []):
        changes.append(
            f"WARNING: '{ov['heading']}' already overrides this clause — "
            f"check precedence rules ({ov['label']} override)"
        )

    return changes
