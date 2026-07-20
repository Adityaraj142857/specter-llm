"""
evaluation/metrics/bleu.py

BLEU (Bilingual Evaluation Understudy) — Papineni et al., 2002.

Measures n-gram precision of the generated answer against the reference
answer, with a brevity penalty so short answers cannot game precision.

Pure Python, no external dependencies — the standard corpus/sentence BLEU
formula with smoothing method 1 (Chen & Cherry, 2014) so a single sentence
with zero 4-gram matches does not collapse the whole score to 0.

What a BLEU score means here:
    High  → the answer reuses the reference's exact wording
    Low   → the answer may still be correct but phrased differently

BLEU is a *surface* metric. A RAG answer can be perfectly correct and score
low simply because it paraphrases. Read it alongside RAGAS and LaaJ.
"""

import math
import re
from collections import Counter


MAX_N = 4          # BLEU-4 (standard)
SMOOTH_EPSILON = 0.1


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word/number tokens, dropping punctuation."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(candidate: list[str], reference: list[str], n: int) -> tuple[int, int]:
    """
    Returns (clipped_matches, total_candidate_ngrams) for order n.
    Clipping caps each n-gram's count at how often it appears in the reference,
    which is what stops 'the the the the' from scoring well.
    """
    cand_ngrams = _ngrams(candidate, n)
    ref_ngrams = _ngrams(reference, n)
    total = sum(cand_ngrams.values())
    if total == 0:
        return 0, 0
    clipped = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
    return clipped, total


def _brevity_penalty(cand_len: int, ref_len: int) -> float:
    """Penalises candidates shorter than the reference. No bonus for longer."""
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        return 1.0
    return math.exp(1.0 - ref_len / cand_len)


def sentence_bleu(candidate: str, reference: str, max_n: int = MAX_N) -> dict:
    """
    BLEU for a single (candidate, reference) pair.

    Returns a dict with the aggregate 'bleu' plus the per-order precisions
    and the brevity penalty, so a low score can be diagnosed rather than
    just reported.
    """
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return {
            "bleu": 0.0,
            "precisions": [0.0] * max_n,
            "brevity_penalty": 0.0,
            "candidate_length": len(cand_tokens),
            "reference_length": len(ref_tokens),
        }

    precisions = []
    for n in range(1, max_n + 1):
        clipped, total = _modified_precision(cand_tokens, ref_tokens, n)
        if total == 0:
            # Candidate is shorter than n tokens — no n-grams of this order exist.
            precisions.append(0.0)
        elif clipped == 0:
            # Smoothing method 1: replace a zero numerator with a small epsilon
            # instead of letting the geometric mean go to zero.
            precisions.append(SMOOTH_EPSILON / total)
        else:
            precisions.append(clipped / total)

    # Geometric mean of the precisions, with uniform weights.
    if all(p > 0 for p in precisions):
        log_mean = sum(math.log(p) for p in precisions) / max_n
        geo_mean = math.exp(log_mean)
    else:
        geo_mean = 0.0

    bp = _brevity_penalty(len(cand_tokens), len(ref_tokens))

    return {
        "bleu": round(bp * geo_mean, 4),
        "precisions": [round(p, 4) for p in precisions],
        "brevity_penalty": round(bp, 4),
        "candidate_length": len(cand_tokens),
        "reference_length": len(ref_tokens),
    }


def corpus_bleu(candidates: list[str], references: list[str], max_n: int = MAX_N) -> dict:
    """
    Corpus-level BLEU — aggregates n-gram counts across all pairs *before*
    dividing. This is the statistically correct way to score a whole test set
    and is not the same as averaging the per-sentence scores.
    """
    if len(candidates) != len(references):
        raise ValueError("candidates and references must be the same length")

    clipped_totals = [0] * max_n
    ngram_totals = [0] * max_n
    cand_len_total = 0
    ref_len_total = 0

    for cand, ref in zip(candidates, references):
        cand_tokens = tokenize(cand)
        ref_tokens = tokenize(ref)
        cand_len_total += len(cand_tokens)
        ref_len_total += len(ref_tokens)
        for n in range(1, max_n + 1):
            clipped, total = _modified_precision(cand_tokens, ref_tokens, n)
            clipped_totals[n - 1] += clipped
            ngram_totals[n - 1] += total

    precisions = []
    for i in range(max_n):
        if ngram_totals[i] == 0:
            precisions.append(0.0)
        elif clipped_totals[i] == 0:
            precisions.append(SMOOTH_EPSILON / ngram_totals[i])
        else:
            precisions.append(clipped_totals[i] / ngram_totals[i])

    if all(p > 0 for p in precisions):
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    else:
        geo_mean = 0.0

    bp = _brevity_penalty(cand_len_total, ref_len_total)

    return {
        "bleu": round(bp * geo_mean, 4),
        "precisions": [round(p, 4) for p in precisions],
        "brevity_penalty": round(bp, 4),
        "candidate_length": cand_len_total,
        "reference_length": ref_len_total,
    }
