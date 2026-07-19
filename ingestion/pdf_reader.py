"""
ingestion/pdf_reader.py

Replaces the old generic text chunker.

Responsibilities:
    1. Extract raw text from a PDF (PyMuPDF)
    2. Clean and normalize the text
    3. Hand off to clause_extractor for clause-level splitting
    4. Embed all clauses via embedder
    5. Store embeddings in ChromaDB (vector_store)
    6. Build the NetworkX graph (graph/builder)
    7. Return (clauses, graph) ready for the UI

Entry point used by app.py:
    from ingestion.pdf_reader import ingest_constitution
    clauses, G = ingest_constitution(pdf_bytes)
"""

import io
import re
import fitz                         # PyMuPDF
from ingestion.clause_extractor import extract_clauses
from ingestion.embedder import embed_clauses, embed_text
from storage.vector_store import VectorStore
from graph.builder import build_graph, load_graph, save_graph


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Minimum clause body length to keep (filter out empty / heading-only blocks)
MIN_CLAUSE_LENGTH = 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_constitution(pdf_bytes: bytes) -> tuple[list[dict], object]:
    """
    Full ingestion pipeline for a college constitution PDF.

    Args:
        pdf_bytes : raw bytes of the uploaded PDF

    Returns:
        clauses : list of clause dicts (id, heading, text, section, index, embedding)
        G       : NetworkX DiGraph with all clause nodes and edges

    Side effects:
        - Persists embeddings to ChromaDB at ./data/chroma_db/
        - Persists graph to ./data/clause_graph.pkl
    """
    # Step 1 — Extract + clean raw text
    raw_text = _extract_text(pdf_bytes)
    clean_text = _clean_text(raw_text)

    # Step 2 — Split into clauses (hybrid rule + LLM)
    clauses = extract_clauses(clean_text)

    # Step 3 — Filter out empty/tiny blocks
    clauses = [c for c in clauses if len(c.get("text", "").strip()) >= MIN_CLAUSE_LENGTH]

    # Step 4 — Embed all clauses
    clauses = embed_clauses(clauses)

    # Step 5 — Store in ChromaDB
    vs = VectorStore()
    vs.clear()                          # fresh ingest each time
    vs.add_clauses(clauses)

    # Step 6 — Build similarity map for graph builder
    #           For each clause, find its top-5 similar neighbours
    similarity_results = _build_similarity_map(clauses, vs, top_k=5)

    # Step 7 — Build graph
    G = build_graph(clauses, similarity_results)

    return clauses, G


def extract_text_only(pdf_bytes: bytes) -> str:
    """
    Lightweight helper — just returns cleaned text without ingesting.
    Used by query/qa.py for RAG context window building.
    """
    return _clean_text(_extract_text(pdf_bytes))


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(pdf_bytes: bytes) -> str:
    """
    Uses PyMuPDF to extract text page by page.
    Preserves line breaks between pages with a clear separator.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Normalises raw PDF text for clause extraction.

    Fixes:
    - Ligature characters (ﬁ → fi, ﬂ → fl)
    - Hyphenated line-break joins (word-\nword → word word)
    - Excessive blank lines (collapse to double newline)
    - Trailing whitespace per line
    - Page numbers / header/footer noise (common PDF artifacts)
    """
    # Fix common ligatures
    ligature_map = {
        "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff",
        "\ufb03": "ffi", "\ufb04": "ffl",
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
    }
    for char, replacement in ligature_map.items():
        text = text.replace(char, replacement)

    # Join hyphenated line breaks: "constitu-\ntion" → "constitution"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove lone page numbers (a line that's just a number)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


# ---------------------------------------------------------------------------
# Similarity map builder
# ---------------------------------------------------------------------------

def _build_similarity_map(
    clauses: list[dict],
    vs: VectorStore,
    top_k: int = 5,
) -> dict[str, list[dict]]:
    """
    For each clause, queries ChromaDB for its top-k similar neighbours.
    Returns a dict: { clause_id → [similar_clause_dicts] }

    This is pre-computed once during ingest so the graph builder
    doesn't need to hit the vector store repeatedly.
    """
    similarity_map = {}
    for clause in clauses:
        embedding = clause.get("embedding")
        if not embedding:
            continue
        # top_k+1 because the clause itself appears in results
        results = vs.query(embedding, top_k=top_k + 1)
        # Remove self from results
        results = [r for r in results if r["id"] != clause["id"]]
        similarity_map[clause["id"]] = results[:top_k]
    return similarity_map


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

def split_into_chunks(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[str]:
    """
    COMPATIBILITY SHIM — kept so existing query/qa.py continues to import
    without changes while the rest of the codebase migrates to clause-level
    splitting via clause_extractor.py.

    Splits text into overlapping chunks, preferring sentence boundaries.
    Do NOT use this for new code — use extract_clauses() instead.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break at a sentence boundary within the last 200 chars of the window
        if end < text_len:
            search_start = max(start, end - 200)
            boundary = -1
            for punct in (".", "!", "?"):
                pos = text.rfind(punct, search_start, end)
                if pos > boundary:
                    boundary = pos
            if boundary != -1:
                end = boundary + 1   # include the punctuation

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance, stepping back by overlap so adjacent chunks share context
        start = end - overlap if end - overlap > start else end

    return chunks
