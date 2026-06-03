from pydantic import BaseModel
from typing import List, Optional

class Clause(BaseModel):
    clause_id: str
    document_id: str
    clause_type: str
    text: str
    roles: List[str]  # which roles can see this clause
    approved: bool = False

class Document(BaseModel):
    document_id: str
    title: str
    source: str  # "cuad", "sec_edgar", etc.
    clauses: List[Clause] = []
