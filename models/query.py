from pydantic import BaseModel
from typing import List, Optional
from models.document import Clause

class QueryRequest(BaseModel):
    user_id: str
    role: str
    question: str

class CitedAnswer(BaseModel):
    answer: str
    source_clauses: List[Clause]
    confidence: float
    escalated: bool = False
