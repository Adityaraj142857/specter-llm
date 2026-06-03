from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    user_id: str
    name: str
    role: str  # "hr", "sde", "external", "legal"
    company: Optional[str] = None
