"""
graph/definitions.py

Detects DEFINES and USES edges between clauses in a college constitution.

Strategy (looser than contract parsing — constitutions don't always have
a formal definitions section):

DEFINES detection:
    - Explicit patterns: "means", "shall mean", "is defined as",
      "hereinafter referred to as", "hereinafter called"
    - Quoted or capitalised multi-word terms e.g. "General Body", 'Quorum'
    - LLM fallback for any clause that looks definitional but doesn't
      match patterns

USES detection:
    - Once defined terms are extracted, scan every other clause for
      occurrences of those exact terms (case-insensitive)
    - Draw a USES edge from the using clause → the defining clause

Edge attributes added:
    type  : "DEFINES" or "USES"
    score : 1.0 for pattern-matched, 0.8 for LLM-detected
    term  : the defined term (string)
"""

import re
import json
import requests
import networkx as nx


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# Patterns that signal a clause is defining a term
DEFINE_PATTERNS = [
    r'["\u2018\u2019\u201c\u201d]([A-Z][^\'"]{2,40})["\u2018\u2019\u201c\u201d]\s+(?:means|shall mean|refers to|is defined as)',
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:means|shall mean|refers to|is defined as)',
    r'(?:hereinafter referred to as|hereinafter called)\s+["\u201c\u2018]?([A-Z][^\'"]{2,40})["\u201d\u2019]?',
    r'(?:the term|the phrase|the word)\s+["\u201c\u2018]([^\'"]{2,40})["\u201d\u2019]\s+(?:means|shall mean|refers to)',
    r'["\u201c]([A-Z][^\'"]{2,40})["\u201d]\s+(?:means|shall mean|refers to|is defined)',
]

# Minimum term length to avoid false positives like "The", "A", "An"
MIN_TERM_LENGTH = 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_definition_edges(G: nx.DiGraph, clauses: list[dict]) -> nx.DiGraph:
    """
    Main entry point. Detects defined terms across all clauses,
    then draws DEFINES and USES edges in the graph.

    Modifies G in place and returns it.
    """
    # Step 1 — extract all defined terms per clause
    definitions = _extract_all_definitions(clauses)
    # definitions: { clause_id: [term1, term2, ...] }

    if not definitions:
        print("[definitions] No defined terms found in constitution.")
        return G

    total_terms = sum(len(v) for v in definitions.items())
    print(f"[definitions] Found defined terms in {len(definitions)} clauses.")

    # Step 2 — add DEFINES edges (clause → itself is the definer)
    # Store terms on the node for USE scanning
    for clause_id, terms in definitions.items():
        if G.has_node(clause_id):
            existing = G.nodes[clause_id].get("defined_terms", [])
            G.nodes[clause_id]["defined_terms"] = list(set(existing + terms))

    # Step 3 — scan all clauses for USES edges
    _add_uses_edges(G, clauses, definitions)

    # Count new edges added
    def_count  = sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "DEFINES")
    uses_count = sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "USES")
    print(f"[definitions] Added {def_count} DEFINES edges, {uses_count} USES edges.")

    return G


def extract_defined_terms(clause: dict) -> list[str]:
    """
    Public helper — extracts defined terms from a single clause dict.
    Used by the impact engine when analysing a new clause.
    """
    return _extract_terms_from_clause(clause)


# ---------------------------------------------------------------------------
# Internal — term extraction
# ---------------------------------------------------------------------------

def _extract_all_definitions(clauses: list[dict]) -> dict[str, list[str]]:
    """
    Runs pattern matching on every clause.
    Falls back to LLM for clauses that look definitional but didn't match.
    Returns { clause_id: [terms] }
    """
    results = {}

    for clause in clauses:
        terms = _extract_terms_from_clause(clause)
        if terms:
            results[clause["id"]] = terms

    return results


def _extract_terms_from_clause(clause: dict) -> list[str]:
    """
    Pattern-first extraction for a single clause.
    Falls back to LLM if the clause looks definitional but no pattern matched.
    """
    text  = clause.get("text", "")
    terms = _pattern_match_terms(text)

    if not terms and _looks_definitional(text):
        terms = _llm_extract_terms(clause)

    # Deduplicate and filter short terms
    terms = list({t.strip() for t in terms if len(t.strip()) >= MIN_TERM_LENGTH})
    return terms


def _pattern_match_terms(text: str) -> list[str]:
    """Runs all regex patterns against clause text."""
    terms = []
    for pattern in DEFINE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        terms.extend(matches)
    return terms


def _looks_definitional(text: str) -> bool:
    """
    Heuristic: does this clause look like it might define terms
    even if no pattern matched?
    """
    triggers = [
        "means ", "shall mean", "defined as", "referred to as",
        "hereinafter", "for the purpose", "for purposes of",
        "in this constitution", "as used in"
    ]
    text_lower = text.lower()
    return any(t in text_lower for t in triggers)


def _llm_extract_terms(clause: dict) -> list[str]:
    """
    LLM fallback — asks llama3.2 to extract defined terms from a clause
    that looks definitional but didn't match any pattern.
    """
    prompt = f"""You are parsing a college constitution clause to extract defined terms.

CLAUSE: "{clause.get('heading', '')}"
TEXT: {clause.get('text', '')}

List every term that this clause formally defines or gives a specific meaning to.
These are words or phrases that this clause says "means X" or "refers to X" or
"hereinafter called X".

Respond ONLY with a JSON array of strings (the defined terms).
If no terms are defined, return [].
No markdown, no explanation. Only the JSON array.

Example: ["General Body", "Office Bearer", "Quorum"]
"""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
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
        terms = json.loads(raw)
        if isinstance(terms, list):
            return [str(t) for t in terms if t]
        return []
    except Exception as e:
        print(f"[definitions] LLM term extraction failed for '{clause.get('heading')}': {e}")
        return []


# ---------------------------------------------------------------------------
# Internal — USES edge detection
# ---------------------------------------------------------------------------

def _add_uses_edges(
    G: nx.DiGraph,
    clauses: list[dict],
    definitions: dict[str, list[str]],
) -> None:
    """
    For every defined term, scans all other clauses for usage.
    Adds USES edge: using_clause_id → defining_clause_id
    Skips self-loops (a clause doesn't 'use' its own definitions).
    """
    # Build flat lookup: term (lowercase) → defining clause_id
    term_to_definer: dict[str, str] = {}
    for clause_id, terms in definitions.items():
        for term in terms:
            term_lower = term.lower().strip()
            if term_lower:
                term_to_definer[term_lower] = clause_id

    if not term_to_definer:
        return

    # Scan each clause for term occurrences
    for clause in clauses:
        clause_id = clause["id"]
        text_lower = clause.get("text", "").lower()

        for term_lower, definer_id in term_to_definer.items():
            if clause_id == definer_id:
                continue  # skip self

            # Word-boundary match to avoid partial matches
            pattern = r'\b' + re.escape(term_lower) + r'\b'
            if re.search(pattern, text_lower):
                # Add USES edge if not already present
                if not G.has_edge(clause_id, definer_id):
                    G.add_edge(
                        clause_id,
                        definer_id,
                        type="USES",
                        score=1.0,
                        term=term_lower,
                        reason=f"Uses the term '{term_lower}' defined in this clause",
                    )
                else:
                    # Edge exists — only upgrade to USES if it was SIMILAR_TO
                    existing = G[clause_id][definer_id]
                    if existing.get("type") == "SIMILAR_TO":
                        G[clause_id][definer_id].update(
                            type="USES", score=1.0, term=term_lower
                        )
