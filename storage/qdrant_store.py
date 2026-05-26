from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchAny
)
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL
from models.document import Clause
import uuid

client = QdrantClient(path="./qdrant_storage")
embedder = SentenceTransformer(EMBEDDING_MODEL)

COLLECTION_NAME = "clauses"

def init_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")

def upsert_clauses(clauses: list[Clause]):
    texts = [c.text for c in clauses]
    vectors = embedder.encode(texts, show_progress_bar=True)
    points = []
    for clause, vector in zip(clauses, vectors):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "clause_id": clause.clause_id,
                "document_id": clause.document_id,
                "clause_type": clause.clause_type,
                "text": clause.text,
                "roles": clause.roles,
                "approved": clause.approved,
            }
        ))
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(points)} clauses in Qdrant.")

def search_clauses(query: str, role: str, top_k: int = 5) -> list[dict]:
    vector = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="roles",
                    match=MatchAny(any=[role])
                )
            ]
        ),
        limit=top_k,
    )
    return [
        {**r.payload, "score": r.score}
        for r in results.points
    ]
