"""
evaluation/metrics/rouge.py

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) — Lin, 2004.

Where BLEU asks "how much of what the model said appears in the reference",
ROUGE asks the reverse: "how much of the reference did the model cover".
For contract Q&A that recall framing matters — an answer that omits the
30-day cure period is wrong in a way precision alone will not catch.

Implemented here, pure Python:
    ROUGE-1  unigram overlap        → content word coverage
    ROUGE-2  bigram overlap         → local phrasing / fluency
    ROUGE-L  longest common subseq  → sequence-level overlap, order-aware
                                      but tolerant of insertions

Each is reported as precision / recall / F1.
"""

import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word/number tokens, dropping punctuation."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _prf(matches: int, cand_total: int, ref_total: int) -> dict:
    """Builds a precision/recall/F1 triple from raw match counts."""
    precision = matches / cand_total if cand_total else 0.0
    recall = matches / ref_total if ref_total else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidate: str, reference: str, n: int = 1) -> dict:
    """ROUGE-N: overlapping n-gram counts, clipped at reference frequency."""
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    cand_ngrams = _ngrams(cand_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)

    overlap = sum((cand_ngrams & ref_ngrams).values())
    return _prf(overlap, sum(cand_ngrams.values()), sum(ref_ngrams.values()))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """
    Longest common subsequence length, computed with a rolling two-row DP
    so memory stays O(min(len(a), len(b))) rather than O(n*m).
    """
    if not a or not b:
        return 0
    # Iterate over the longer sequence in the outer loop, keeping rows small.
    if len(a) < len(b):
        a, b = b, a

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[len(b)]


def rouge_l(candidate: str, reference: str) -> dict:
    """
    ROUGE-L: F1 over the longest common subsequence.

    Order-aware — 'Berkshire pays Auriemma' and 'Auriemma pays Berkshire'
    score identically under ROUGE-1 but differently here.
    """
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    lcs = _lcs_length(cand_tokens, ref_tokens)
    return _prf(lcs, len(cand_tokens), len(ref_tokens))


def rouge_scores(candidate: str, reference: str) -> dict:
    """All three ROUGE variants for one (candidate, reference) pair."""
    return {
        "rouge1": rouge_n(candidate, reference, n=1),
        "rouge2": rouge_n(candidate, reference, n=2),
        "rougeL": rouge_l(candidate, reference),
    }


def aggregate_rouge(all_scores: list[dict]) -> dict:
    """
    Averages per-question ROUGE scores across the test set.

    Unlike BLEU, averaging ROUGE F1 across examples is the conventional
    reporting method, so a plain mean is correct here.
    """
    if not all_scores:
        return {}

    aggregated = {}
    for variant in ("rouge1", "rouge2", "rougeL"):
        for field in ("precision", "recall", "f1"):
            values = [s[variant][field] for s in all_scores]
            aggregated.setdefault(variant, {})[field] = round(sum(values) / len(values), 4)
    return aggregated
