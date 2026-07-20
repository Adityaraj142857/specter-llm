"""
evaluation/metrics/ragas_local.py

RAGAS metrics — a local reimplementation of the four core metrics from the
RAGAS framework (Es et al., 2023, "RAGAS: Automated Evaluation of Retrieval
Augmented Generation").

IMPORTANT — this does not import the `ragas` pip package. That package
defaults to hosted OpenAI models for both its LLM and embeddings, and this
project is deliberately offline-only (llama3.2 + nomic-embed-text via
Ollama). So the metric *definitions* are reimplemented here against the local
stack. Scores follow the same formulas but are not numerically comparable to
published RAGAS benchmarks that used GPT-4 as the backing model.

The four metrics, and what each one isolates:

    faithfulness       Is the answer grounded in the retrieved context?
                       Decompose the answer into atomic claims, check each
                       against the context. Catches hallucination.
                       → a generation-side metric

    answer_relevancy   Does the answer actually address the question asked?
                       Reverse-generate questions from the answer, embed them,
                       compare to the real question. Catches evasive or
                       off-topic answers that are technically grounded.
                       → a generation-side metric

    context_precision  Did the retriever rank useful chunks highly?
                       Judge each retrieved chunk for relevance, then weight
                       by rank position. Catches noisy retrieval.
                       → a retrieval-side metric

    context_recall     Did the retriever find everything needed?
                       Split the ground truth into claims, check each is
                       attributable to the retrieved context. Catches misses.
                       → a retrieval-side metric

Splitting generation from retrieval is the whole point: if faithfulness is
high but context_recall is low, the prompt is fine and the retriever is
starving it. Fixing the wrong half is the usual RAG debugging failure.
"""

import math

from ingestion.embedder import embed_text
from evaluation.metrics._llm import judge_json, split_sentences


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_PROMPT = """Break the following ANSWER into a list of atomic factual claims.
Each claim must be a single, self-contained statement that can be independently verified.
Do not add any information that is not in the answer.

ANSWER:
{answer}

Reply with ONLY a JSON array of strings, nothing else. Example:
["The contract ends on May 31, 2016.", "The payment is $90,000."]
"""

FAITHFULNESS_PROMPT = """You are verifying whether claims are supported by a source context.

CONTEXT:
{context}

For each claim below, decide if it can be directly inferred from the CONTEXT.
Answer 1 if the context supports the claim, 0 if it does not or if the context is silent.

CLAIMS:
{claims}

Reply with ONLY a JSON array of objects, one per claim, in the same order:
[{{"claim": "...", "supported": 1, "reason": "..."}}]
"""

QUESTION_GENERATION_PROMPT = """Read the ANSWER below and write {n} different questions that this answer would be a direct and complete response to.

ANSWER:
{answer}

Reply with ONLY a JSON array of {n} question strings, nothing else.
"""

CONTEXT_PRECISION_PROMPT = """You are judging whether a retrieved document chunk was useful.

QUESTION:
{question}

CORRECT ANSWER:
{ground_truth}

RETRIEVED CHUNK:
{chunk}

Was this chunk useful for arriving at the correct answer?
Reply with ONLY JSON: {{"useful": 1, "reason": "..."}} or {{"useful": 0, "reason": "..."}}
"""

CONTEXT_RECALL_PROMPT = """You are checking whether each sentence of a correct answer could have been derived from the retrieved context.

RETRIEVED CONTEXT:
{context}

For each sentence below, answer 1 if the retrieved context contains the information needed to support it, 0 if that information is missing from the context.

SENTENCES:
{sentences}

Reply with ONLY a JSON array of objects, one per sentence, in the same order:
[{{"sentence": "...", "attributed": 1, "reason": "..."}}]
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _number_list(items: list[str]) -> str:
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _coerce_binary(value) -> int:
    """Judges return 1/0, true/false, or 'yes'/'no' depending on the run."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 0.5 else 0
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "yes", "true", "supported", "y"} else 0
    return 0


# ---------------------------------------------------------------------------
# Metric 1 — Faithfulness (generation side)
# ---------------------------------------------------------------------------

def faithfulness(answer: str, contexts: list[str]) -> dict:
    """
    faithfulness = (# answer claims supported by context) / (# answer claims)

    A score of 1.0 means every statement in the answer traces back to a
    retrieved clause. Anything below that is the model adding information the
    contract did not give it.
    """
    context_text = "\n\n".join(contexts)
    if not answer.strip() or not context_text.strip():
        return {"score": None, "reason": "empty answer or context", "claims": []}

    claims = judge_json(CLAIM_EXTRACTION_PROMPT.format(answer=answer))
    if not isinstance(claims, list) or not claims:
        # Fall back to sentence splitting if the model would not produce JSON.
        claims = split_sentences(answer)
    claims = [str(c) for c in claims if str(c).strip()][:12]

    if not claims:
        return {"score": None, "reason": "no claims extracted", "claims": []}

    verdicts = judge_json(
        FAITHFULNESS_PROMPT.format(context=context_text, claims=_number_list(claims))
    )
    if not isinstance(verdicts, list) or not verdicts:
        return {"score": None, "reason": "judge did not return parseable verdicts", "claims": claims}

    supported = [_coerce_binary(v.get("supported")) for v in verdicts if isinstance(v, dict)]
    if not supported:
        return {"score": None, "reason": "no usable verdicts", "claims": claims}

    return {
        "score": round(sum(supported) / len(supported), 4),
        "supported_claims": sum(supported),
        "total_claims": len(supported),
        "claims": claims,
        "verdicts": verdicts,
    }


# ---------------------------------------------------------------------------
# Metric 2 — Answer relevancy (generation side)
# ---------------------------------------------------------------------------

def answer_relevancy(question: str, answer: str, n_questions: int = 3) -> dict:
    """
    answer_relevancy = mean cosine similarity between the original question
    and n questions reverse-generated from the answer.

    The intuition: if you can read the answer and recover the question that
    was asked, the answer was on point. Answers that dodge ("I could not find
    this in the contract") or wander into unrelated clauses produce
    reverse-generated questions that sit far from the original in embedding
    space.

    Note this is measured with nomic-embed-text — the same embedding model
    the retriever uses.
    """
    if not answer.strip():
        return {"score": None, "reason": "empty answer"}

    generated = judge_json(QUESTION_GENERATION_PROMPT.format(answer=answer, n=n_questions))
    if not isinstance(generated, list) or not generated:
        return {"score": None, "reason": "judge did not return parseable questions"}

    generated = [str(q) for q in generated if str(q).strip()][:n_questions]
    if not generated:
        return {"score": None, "reason": "no questions generated"}

    try:
        question_vec = embed_text(question)
        similarities = [_cosine(question_vec, embed_text(q)) for q in generated]
    except Exception as exc:
        return {"score": None, "reason": f"embedding failed: {exc}"}

    return {
        "score": round(sum(similarities) / len(similarities), 4),
        "generated_questions": generated,
        "similarities": [round(s, 4) for s in similarities],
    }


# ---------------------------------------------------------------------------
# Metric 3 — Context precision (retrieval side)
# ---------------------------------------------------------------------------

def context_precision(question: str, ground_truth: str, contexts: list[str]) -> dict:
    """
    Rank-weighted precision@k over the retrieved chunks:

        context_precision = Σ (precision@k × useful_k) / (total useful chunks)

    The rank weighting is what makes this different from plain precision — a
    useful chunk at position 1 counts for more than the same chunk at position
    5, because the generator reads top-down and a long noisy prefix pushes the
    real evidence out of effective attention.
    """
    if not contexts:
        return {"score": None, "reason": "no contexts retrieved"}

    usefulness = []
    reasons = []
    for chunk in contexts:
        verdict = judge_json(
            CONTEXT_PRECISION_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                chunk=chunk[:4000],
            )
        )
        if isinstance(verdict, dict):
            usefulness.append(_coerce_binary(verdict.get("useful")))
            reasons.append(str(verdict.get("reason", ""))[:200])
        else:
            usefulness.append(0)
            reasons.append("unparseable judge response")

    total_useful = sum(usefulness)
    if total_useful == 0:
        return {
            "score": 0.0,
            "useful_flags": usefulness,
            "reasons": reasons,
            "note": "no retrieved chunk judged useful",
        }

    weighted_sum = 0.0
    running_useful = 0
    for k, is_useful in enumerate(usefulness, start=1):
        running_useful += is_useful
        if is_useful:
            weighted_sum += running_useful / k   # precision@k at this position

    return {
        "score": round(weighted_sum / total_useful, 4),
        "useful_flags": usefulness,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Metric 4 — Context recall (retrieval side)
# ---------------------------------------------------------------------------

def context_recall(ground_truth: str, contexts: list[str]) -> dict:
    """
    context_recall = (# ground-truth sentences attributable to the retrieved
                      context) / (# ground-truth sentences)

    This is the only one of the four that needs a human-written ground truth,
    and it is the one that catches the most damaging RAG failure: the answer
    looks confident and faithful, but the retriever never surfaced the clause
    that would have contradicted it.
    """
    context_text = "\n\n".join(contexts)
    if not context_text.strip():
        return {"score": 0.0, "reason": "no contexts retrieved"}

    sentences = split_sentences(ground_truth)
    if not sentences:
        sentences = [ground_truth]

    verdicts = judge_json(
        CONTEXT_RECALL_PROMPT.format(
            context=context_text,
            sentences=_number_list(sentences),
        )
    )
    if not isinstance(verdicts, list) or not verdicts:
        return {"score": None, "reason": "judge did not return parseable verdicts"}

    attributed = [_coerce_binary(v.get("attributed")) for v in verdicts if isinstance(v, dict)]
    if not attributed:
        return {"score": None, "reason": "no usable verdicts"}

    return {
        "score": round(sum(attributed) / len(attributed), 4),
        "attributed_sentences": sum(attributed),
        "total_sentences": len(attributed),
        "verdicts": verdicts,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_ragas(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    """Runs all four RAGAS metrics for a single QA sample."""
    results = {
        "faithfulness": faithfulness(answer, contexts),
        "answer_relevancy": answer_relevancy(question, answer),
        "context_precision": context_precision(question, ground_truth, contexts),
        "context_recall": context_recall(ground_truth, contexts),
    }

    # The harmonic-mean-style summary RAGAS calls the "ragas score", computed
    # only over metrics that actually produced a number this run.
    scores = [r["score"] for r in results.values() if isinstance(r.get("score"), (int, float))]
    if scores and all(s > 0 for s in scores):
        results["ragas_score"] = round(len(scores) / sum(1 / s for s in scores), 4)
    elif scores:
        results["ragas_score"] = 0.0
    else:
        results["ragas_score"] = None

    return results
