import ollama

SUMMARY_PROMPT = """You are reading a legal contract section by section.
Here is what you have read so far:
{previous_summary}

Here is the next section:
{chunk}

Update the summary to include any important definitions, parties, obligations, or terms introduced in this section.
Keep the summary under 200 words. Be factual, no opinions.

Updated summary:"""

RED_FLAG_PROMPT = """You are a legal risk analyst reading a contract.

Here is a summary of the contract so far (for context):
{summary}

Here is the current section to analyse:
{chunk}

Identify RED FLAGS in this section. Use the summary for context where needed.
Red flags are clauses that are risky, one-sided, or important to understand before signing:
- One-sided termination rights
- Broad IP ownership grabs
- Auto-renewal traps
- Unlimited liability
- Non-compete restrictions
- Unilateral amendment rights
- Vague payment terms

For each red flag found, respond in this exact format:
FLAG: <short title>
CLAUSE: <the problematic text, max 2 sentences>
WHY: <plain English explanation of why this is risky>
SEVERITY: <High / Medium / Low>
---

If no red flags found in this section write: NO FLAGS FOUND"""

def summarise_chunk(chunk: str, previous_summary: str) -> str:
    """Ask Llama to update the running summary with new chunk."""
    prompt = SUMMARY_PROMPT.format(
        previous_summary=previous_summary if previous_summary else "Nothing read yet.",
        chunk=chunk
    )
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

def detect_red_flags(chunks: list[str]) -> list[dict]:
    """
    Read contract chunks with a running summary for context.
    Each chunk is analysed knowing what came before it.
    """
    all_flags = []
    running_summary = ""

    for i, chunk in enumerate(chunks[:10]):
        print(f"Processing chunk {i+1}/{min(len(chunks), 10)}...")

        # Step 1 — update running summary with this chunk
        running_summary = summarise_chunk(chunk, running_summary)

        # Step 2 — detect red flags using chunk + summary as context
        prompt = RED_FLAG_PROMPT.format(
            summary=running_summary,
            chunk=chunk
        )
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"].strip()

        if "NO FLAGS FOUND" in raw:
            continue

        # Step 3 — parse the response
        entries = raw.split("---")
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            flag = {}
            for line in entry.split("\n"):
                if line.startswith("FLAG:"):
                    flag["title"] = line.replace("FLAG:", "").strip()
                elif line.startswith("CLAUSE:"):
                    flag["clause"] = line.replace("CLAUSE:", "").strip()
                elif line.startswith("WHY:"):
                    flag["why"] = line.replace("WHY:", "").strip()
                elif line.startswith("SEVERITY:"):
                    flag["severity"] = line.replace("SEVERITY:", "").strip()
            if "title" in flag and "why" in flag:
                all_flags.append(flag)

    return all_flags
