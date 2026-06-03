import ollama
from ingestion.red_flag_detector import summarise_chunk
from ingestion.pdf_reader import split_into_chunks

QA_PROMPT = """You are a legal document assistant.
Here is a summary of the full contract for context:
{summary}

Here is the relevant section of the contract:
{context}

Answer this question using the contract. Use plain simple English. No legal jargon.
If the answer is not in the contract say: "I could not find this in the contract."

Question: {question}

Answer:"""

def build_full_summary(full_text: str) -> str:
    """Build a running summary of the entire contract."""
    chunks = split_into_chunks(full_text)
    summary = ""
    for chunk in chunks[:10]:
        summary = summarise_chunk(chunk, summary)
    return summary

def answer_question(question: str, full_text: str, summary: str = "") -> str:
    """Answer a question using contract text and its summary for context."""
    words = full_text.split()
    context = " ".join(words[:3000])

    prompt = QA_PROMPT.format(
        summary=summary if summary else "No summary available.",
        context=context,
        question=question
    )
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()
