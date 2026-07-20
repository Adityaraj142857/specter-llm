"""
evaluation/metrics/_llm.py

Shared Ollama helper for the two model-based evaluators (RAGAS and LaaJ).

Everything in this project runs locally, so the judge is llama3.2 via Ollama
— the same family as the model being evaluated. That is a known limitation
(self-preference bias: models rate their own style generously), and it is
called out in evaluation/README.md. The judge temperature is pinned to 0 so
repeated runs are reproducible.
"""

import json
import re

import ollama


JUDGE_MODEL = "llama3.2"
JUDGE_OPTIONS = {"temperature": 0.0}


def judge_chat(prompt: str, model: str = JUDGE_MODEL) -> str:
    """Single-turn call to the local judge model. Returns raw text."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=JUDGE_OPTIONS,
    )
    return response["message"]["content"].strip()


def judge_json(prompt: str, model: str = JUDGE_MODEL, retries: int = 2) -> dict | list | None:
    """
    Calls the judge and parses its reply as JSON.

    Small local models wrap JSON in prose or markdown fences more often than
    hosted ones, so this strips fences and falls back to extracting the first
    balanced {...} or [...] block before giving up. Returns None if every
    attempt fails, and callers treat None as "metric unavailable" rather than
    silently scoring it zero — a parse failure is not evidence of a bad answer.
    """
    for _ in range(retries + 1):
        raw = judge_chat(prompt, model=model)
        parsed = _parse_json(raw)
        if parsed is not None:
            return parsed
    return None


def _parse_json(raw: str) -> dict | list | None:
    text = raw.strip()

    # Strip ```json ... ``` fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced JSON object or array in the text.
    # Order the candidates by where they actually appear, not by bracket type:
    # a reply like 'Here you go: [{"claim": ...}]' must parse as the array, not
    # as the first object nested inside it.
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        position = text.find(opener)
        if position != -1:
            candidates.append((position, opener, closer))

    for start, opener, closer in sorted(candidates):
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences for claim-level metrics.

    Legal text is full of abbreviations and enumerations ('Paragraph 4 (b)-(c)',
    'i.e.', '$480,000.00') that naive '.' splitting shreds, so this protects
    decimals, single-letter initials and common legal abbreviations first.
    """
    protected = text
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)          # 1.5, $30,000.00
    protected = re.sub(r"\b([A-Za-z])\.", r"\1<DOT>", protected)        # i.e., e.g., initials
    protected = re.sub(r"\b(No|Inc|Ltd|Corp|Co|LLP|LLC|St|Mr|Ms|Dr)\.", r"\1<DOT>", protected)

    parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for part in parts:
        restored = part.replace("<DOT>", ".").strip()
        # Drop fragments too short to carry a checkable claim.
        if len(restored) > 10:
            sentences.append(restored)
    return sentences
