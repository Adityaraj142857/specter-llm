import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = "data/CUAD_v1"
CUAD_JSON = "data/CUAD_v1/CUAD_v1.json"
CONTRACTS_TXT_DIR = "data/CUAD_v1/full_contract_txt"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Confidence threshold — below this score, escalate instead of answering
CONFIDENCE_THRESHOLD = 0.5

# Roles
ROLES = ["hr", "sde", "external", "legal"]
