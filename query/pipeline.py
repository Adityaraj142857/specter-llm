from storage.qdrant_store import search_clauses
from query.llm_client import ask_llm
from query.confidence_gate import should_escalate
from models.query import QueryRequest, CitedAnswer
from models.document import Clause

def run_query(request: QueryRequest) -> CitedAnswer:
    """
    Full pipeline:
    1. Search Qdrant for relevant clauses (role-scoped)
    2. Send clauses + question to LLM
    3. Check confidence — escalate if too low
    4. Return CitedAnswer
    """
    # Step 1 — retrieve relevant clauses
    raw_results = search_clauses(
        query=request.question,
        role=request.role,
        top_k=5
    )

    # Step 2 — ask LLM
    llm_result = ask_llm(
        question=request.question,
        context_clauses=raw_results
    )

    # Step 3 — build source clause objects
    source_clauses = [
        Clause(
            clause_id=r["clause_id"],
            document_id=r["document_id"],
            clause_type=r["clause_type"],
            text=r["text"],
            roles=r["roles"],
            approved=r["approved"],
        )
        for r in raw_results
    ]

    return CitedAnswer(
        answer=llm_result["answer"],
        source_clauses=source_clauses,
        confidence=llm_result["confidence"],
        escalated=llm_result["escalated"],
    )
