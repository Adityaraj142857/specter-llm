"""
ingestion/clause_extractor.py

Splits a college constitution into individual clause nodes using a
hybrid strategy: regex first, LLM fallback for ambiguous boundaries.

Each clause is returned as a dict:
    {
        "id":      "clause_1_2",       # slugified from heading
        "heading": "1.2 Membership",   # original heading text
        "text":    "...",              # full clause body
        "section": "1",               # top-level section number
        "index":   3                   # position in document
    }
"""

import re
import json
import requests
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# Matches headings like:  1.  /  1.2  /  1.2.3  /  Article I  /  Section 2
HEADING_PATTERN = re.compile(
    r"^(?:"
    r"(?:Article|Section|Part|Chapter)\s+[IVXLCDM\d]+"   # named headings
    r"|"
    r"\d+(?:\.\d+)*\.?"                                    # numeric 1 / 1.2 / 1.2.3
    r")"
    r"\s+\S",                                              # must have text after
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_clauses(text: str) -> list[dict]:
    """
    Main entry point.  Returns a list of clause dicts sorted by position.
    Uses rule-based splitting first; falls back to LLM for any block that
    looks too large to be a single clause (> 800 chars and no sub-heading).
    """
    raw_blocks = _rule_based_split(text)

    clauses = []
    index = 0
    for block in raw_blocks:
        if _needs_llm_split(block["text"]):
            sub = _llm_split(block["text"], block["heading"])
            for s in sub:
                s["index"] = index
                s["section"] = block["section"]
                clauses.append(s)
                index += 1
        else:
            block["index"] = index
            clauses.append(block)
            index += 1

    seen = {}
    for clause in clauses:
        seen[clause["id"]] = clause
    return list(seen.values())


# ---------------------------------------------------------------------------
# Rule-based splitting
# ---------------------------------------------------------------------------

def _rule_based_split(text: str) -> list[dict]:
    """
    Splits text on detected headings.  Returns raw blocks before LLM review.
    """
    lines = text.splitlines()
    blocks: list[dict] = []
    current_heading = "Preamble"
    current_lines: list[str] = []
    current_section = "0"

    for line in lines:
        stripped = line.strip()
        if HEADING_PATTERN.match(stripped):
            # Save previous block
            body = "\n".join(current_lines).strip()
            if body:
                blocks.append({
                    "id": _slugify(current_heading),
                    "heading": current_heading,
                    "text": body,
                    "section": current_section,
                    "index": -1,
                })
            current_heading = stripped
            current_section = _extract_section(stripped)
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last block
    body = "\n".join(current_lines).strip()
    if body:
        blocks.append({
            "id": _slugify(current_heading),
            "heading": current_heading,
            "text": body,
            "section": current_section,
            "index": -1,
        })

    return blocks


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

def _needs_llm_split(text: str) -> bool:
    """
    Heuristic: if a block is very long and has no internal headings,
    the rule-based pass may have merged multiple clauses.
    """
    return len(text) > 800 and not HEADING_PATTERN.search(text)


def _llm_split(text: str, parent_heading: str) -> list[dict]:
    """
    Asks llama3.2 to identify clause boundaries within a large block.
    Returns list of clause dicts (without index/section — caller fills those).
    Falls back to returning the block as-is on any failure.
    """
    prompt = f"""You are a legal document parser. The following text is from a college constitution under the heading "{parent_heading}".

Split this text into individual clauses. Each clause should be a self-contained rule or provision.

Return ONLY a JSON array. Each item must have:
  - "heading": short descriptive title you infer (string)
  - "text": the full clause text (string)

No extra explanation, no markdown fences. Just the JSON array.

TEXT:
{text}
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
        return [
            {
                "id": _slugify(f"{parent_heading}_{item['heading']}"),
                "heading": item["heading"],
                "text": item["text"].strip(),
                "section": "",   # caller fills
                "index": -1,
            }
            for item in items
            if item.get("text", "").strip()
        ]

    except Exception as e:
        # Fallback: return the block unchanged
        print(f"[clause_extractor] LLM fallback failed for '{parent_heading}': {e}")
        return [
            {
                "id": _slugify(parent_heading),
                "heading": parent_heading,
                "text": text.strip(),
                "section": "",
                "index": -1,
            }
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Converts heading text to a safe node ID."""
    import hashlib
    original = text.strip()
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    slug = text[:80]
    suffix = hashlib.md5(original.encode()).hexdigest()[:8]
    return f"{slug}_{suffix}"


def _extract_section(heading: str) -> str:
    """Pulls the top-level section number from a heading string."""
    m = re.match(r"(\d+)", heading)
    if m:
        return m.group(1)
    m = re.match(r"(?:Article|Section|Part|Chapter)\s+([IVXLCDM\d]+)", heading, re.I)
    if m:
        return m.group(1)
    return "0"
