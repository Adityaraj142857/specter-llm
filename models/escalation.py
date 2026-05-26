from pydantic import BaseModel
from typing import Optional

class Escalation(BaseModel):
    escalation_id: str
    user_id: str
    question: str
    document_id: Optional[str] = None
    clause_text: Optional[str] = None
    reason: str
    resolved: bool = False
    legal_answer: Optional[str] = None
