import json
import uuid
from models.document import Document, Clause
from config.settings import CUAD_JSON

# These clause types matter for HR
HR_CLAUSE_TYPES = [
    "Non-Compete/Non-Solicit",
    "Termination For Cause",
    "Non-Disparagement",
    "IP Ownership Assignment",
    "Anti-Assignment",
]

# These clause types matter for SDE
SDE_CLAUSE_TYPES = [
    "IP Ownership Assignment",
    "License Grant",
    "Source Code Escrow",
    "Non-Compete/Non-Solicit",
    "Data Processing",
]

# These clause types matter for external partners
EXTERNAL_CLAUSE_TYPES = [
    "Revenue/Profit Sharing",
    "Non-Compete/Non-Solicit",
    "Audit Rights",
    "Most Favored Nation",
    "Termination For Convenience",
    "License Grant",
]

def assign_roles(clause_type: str) -> list:
    """Given a clause type, return which roles should see it."""
    roles = ["legal"]  # legal always sees everything
    if clause_type in HR_CLAUSE_TYPES:
        roles.append("hr")
    if clause_type in SDE_CLAUSE_TYPES:
        roles.append("sde")
    if clause_type in EXTERNAL_CLAUSE_TYPES:
        roles.append("external")
    if len(roles) == 1:
        # clause type not in any specific list — show to all
        roles = ["hr", "sde", "external", "legal"]
    return roles

def load_cuad(max_documents: int = 50) -> list[Document]:
    """
    Load CUAD JSON and return a list of Document objects.
    max_documents limits how many contracts we load (start small).
    """
    print(f"Loading CUAD from {CUAD_JSON} ...")
    with open(CUAD_JSON, "r") as f:
        raw = json.load(f)

    documents = []
    contract_data = raw["data"]

    for i, contract in enumerate(contract_data[:max_documents]):
        title = contract["title"]
        doc_id = f"cuad_{i}"
        clauses = []

        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]  # the actual contract text
            for qa in paragraph["qas"]:
                clause_type = qa["question"]  # CUAD uses questions as clause type labels
                if not qa["answers"]:
                    continue  # skip unanswered / not present clauses
                for answer in qa["answers"]:
                    clause_text = answer["text"].strip()
                    if not clause_text:
                        continue
                    roles = assign_roles(clause_type)
                    clause = Clause(
                        clause_id=str(uuid.uuid4()),
                        document_id=doc_id,
                        clause_type=clause_type,
                        text=clause_text,
                        roles=roles,
                        approved=True,  # CUAD data is pre-approved for testing
                    )
                    clauses.append(clause)

        doc = Document(
            document_id=doc_id,
            title=title,
            source="cuad",
            clauses=clauses,
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} documents, "
          f"{sum(len(d.clauses) for d in documents)} clauses total.")
    return documents
