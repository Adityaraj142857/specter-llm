import ollama

def ask_llm(question: str, context_clauses: list[dict]) -> dict:
    """
    Given a question and retrieved clauses, ask Llama to answer
    using only the provided clauses as context.
    """
    if not context_clauses:
        return {
            "answer": "No relevant clauses found.",
            "confidence": 0.0,
            "escalated": True,
        }

    context_text = "\n\n".join([
        f"[Clause {i+1} | Type: {c['clause_type'][:80]}]\n{c['text']}"
        for i, c in enumerate(context_clauses)
    ])

    prompt = f"""You are a legal document assistant helping non-legal employees understand contracts.
Answer the question using ONLY the clauses provided below. Do not use outside knowledge.

If the clauses do not contain enough information to answer confidently, reply with exactly:
INSUFFICIENT_CONTEXT

After your answer, on a new line write:
CONFIDENCE: <a number from 0.0 to 1.0>

Clauses:
{context_text}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response["message"]["content"].strip()

    # Parse out confidence score
    confidence = 0.5
    answer = raw
    if "CONFIDENCE:" in raw:
        parts = raw.rsplit("CONFIDENCE:", 1)
        answer = parts[0].strip()
        try:
            confidence = float(parts[1].strip())
        except ValueError:
            confidence = 0.5

    escalated = "INSUFFICIENT_CONTEXT" in answer or confidence < 0.5

    return {
        "answer": answer,
        "confidence": confidence,
        "escalated": escalated,
    }
