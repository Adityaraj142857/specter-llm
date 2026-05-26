# Specter LLM

An AI-powered legal contract analyser that runs fully locally on your machine.

## What it does
- Upload any legal contract PDF
- Detects red flags — risky, one-sided, or unusual clauses explained in plain English
- Ask any question about the contract and get a plain English answer
- Saves full history of all documents analysed, flags found, and questions asked

## Tech stack
- Llama 3.2 via Ollama — runs locally, no API key needed
- Streamlit — web interface
- SQLite — document history
- PyMuPDF — PDF text extraction
- Python 3.13

## How to run
1. Install Ollama from https://ollama.com and pull llama3.2
2. Clone this repo
3. Create a virtual environment and install requirements
4. Run: streamlit run app.py
