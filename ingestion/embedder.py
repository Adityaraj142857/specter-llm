"""
ingestion/embedder.py

Generates embeddings for clause text using nomic-embed-text running
locally via Ollama.  Provides both single and batch embedding helpers
used by the vector store and the graph builder.
"""

import requests
import time
from typing import Union


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/embed"   # Current Ollama embedding endpoint
EMBED_MODEL  = "nomic-embed-text"
MAX_INPUT_LENGTH = 8000  # Characters; nomic-embed-text has ~8k token context limit
BATCH_DELAY  = 0.05   # seconds between batch requests (be kind to local Ollama)
MAX_RETRIES  = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """
    Returns a single embedding vector for the given text.
    Raises RuntimeError if Ollama is unreachable after retries.
    """
    return _request_embedding(text)


def embed_clauses(clauses: list[dict]) -> list[dict]:
    """
    Adds an 'embedding' key to each clause dict in-place and returns the list.

    Input clause dicts must have at least 'id' and 'text' keys.
    The embedding is computed over:  heading + "\\n\\n" + text
    (heading gives the model context about what kind of clause this is)
    """
    for i, clause in enumerate(clauses):
        content = _clause_to_embed_input(clause)
        clause["embedding"] = _request_embedding(content)
        if i > 0 and i % 10 == 0:
            print(f"[embedder] Embedded {i}/{len(clauses)} clauses...")
        time.sleep(BATCH_DELAY)

    print(f"[embedder] Done — {len(clauses)} clauses embedded.")
    return clauses


def embed_query(query: str) -> list[float]:
    """
    Embeds a user query for similarity search.
    Prefixes with 'search_query: ' as recommended by nomic-embed-text docs.
    """
    return _request_embedding(f"search_query: {query}")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _clause_to_embed_input(clause: dict) -> str:
    """
    Builds the string that gets embedded.
    nomic-embed-text benefits from a 'search_document: ' prefix for
    passages that will be retrieved (vs queries).
    """
    heading = clause.get("heading", "")
    text    = clause.get("text", "")
    return f"search_document: {heading}\n\n{text}"


def _request_embedding(text: str) -> list[float]:
    """
    Makes the Ollama embedding API call with retry logic.
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"Invalid text for embedding: {type(text)}")
    
    text = text.strip()
    if not text:
        raise ValueError("Empty text cannot be embedded")
    
    # Truncate to max input length to avoid "context length exceeds" errors
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            payload = {
                "model": EMBED_MODEL,
                "input": text
            }
            resp = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            # /api/embed returns {"embeddings": [[...float...]]}  (list of lists)
            data = resp.json()
            embeddings = data.get("embeddings")
            if not embeddings or not embeddings[0]:
                raise ValueError(f"Empty embedding returned by Ollama. Response: {data}")
            return embeddings[0]

        except requests.exceptions.ConnectionError:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Cannot connect to Ollama. "
                    "Make sure Ollama is running: `ollama serve`"
                )
            time.sleep(1 * attempt)

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Embedding failed after {MAX_RETRIES} attempts: {error_msg}\n"
                    f"Make sure the model '{EMBED_MODEL}' is installed: `ollama pull {EMBED_MODEL}`"
                )
            print(f"[embedder] Attempt {attempt}/{MAX_RETRIES} failed: {error_msg}")
            time.sleep(1 * attempt)

        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts: {e}")
            print(f"[embedder] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(1 * attempt)

    # Should never reach here
    raise RuntimeError("Embedding failed.")
