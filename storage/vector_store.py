"""
storage/vector_store.py

ChromaDB wrapper for storing and querying clause embeddings.
All data is persisted locally under ./data/chroma_db/

Usage:
    from storage.vector_store import VectorStore

    vs = VectorStore()
    vs.add_clauses(clauses)          # list of clause dicts with 'embedding'
    results = vs.query("quorum rules", top_k=5)
"""

import chromadb
from chromadb.config import Settings
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_PATH       = "./data/chroma_db"
COLLECTION_NAME   = "constitution_clauses"


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, path: str = CHROMA_PATH):
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_clauses(self, clauses: list[dict]) -> None:
        """
        Upserts clauses into ChromaDB.
        Each clause dict must have: id, text, heading, section, embedding.
        Uses upsert so re-ingesting the same constitution is safe.
        """
        if not clauses:
            return

        seen = {}
        for c in clauses:
            seen[c["id"]] = c
        clauses = list(seen.values())

        ids        = [c["id"] for c in clauses]
        embeddings = [c["embedding"] for c in clauses]
        documents  = [c["text"] for c in clauses]
        metadatas  = [
            {
                "heading": c.get("heading", ""),
                "section": c.get("section", ""),
                "index":   str(c.get("index", -1)),
            }
            for c in clauses
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"[vector_store] Upserted {len(clauses)} clauses into ChromaDB.")

    def add_single_clause(self, clause: dict) -> None:
        """Convenience wrapper for adding one clause (e.g. a new incoming clause)."""
        self.add_clauses([clause])

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Returns top_k most similar clauses to the query embedding.

        Each result dict contains:
            id, text (document), heading, section, index, distance (0=identical)
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id":       doc_id,
                "text":     results["documents"][0][i],
                "heading":  results["metadatas"][0][i].get("heading", ""),
                "section":  results["metadatas"][0][i].get("section", ""),
                "index":    int(results["metadatas"][0][i].get("index", -1)),
                "distance": results["distances"][0][i],
            })

        return output

    def get_all_ids(self) -> list[str]:
        """Returns all stored clause IDs."""
        return self._collection.get(include=[])["ids"]

    def get_clause(self, clause_id: str) -> Optional[dict]:
        """Fetches a single clause by ID."""
        result = self._collection.get(
            ids=[clause_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return {
            "id":      result["ids"][0],
            "text":    result["documents"][0],
            "heading": result["metadatas"][0].get("heading", ""),
            "section": result["metadatas"][0].get("section", ""),
            "index":   int(result["metadatas"][0].get("index", -1)),
        }

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Deletes and recreates the collection. Use carefully."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[vector_store] Collection cleared.")
