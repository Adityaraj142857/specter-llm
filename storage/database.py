import sqlite3
import json
from datetime import datetime

DB_PATH = "specter.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            word_count INTEGER,
            summary TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            title TEXT,
            clause TEXT,
            why TEXT,
            severity TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            asked_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    conn.commit()
    conn.close()
    print("Database ready.")

def save_document(filename: str, word_count: int, summary: str) -> int:
    """Save a document and return its ID."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO documents (filename, uploaded_at, word_count, summary) VALUES (?, ?, ?, ?)",
        (filename, datetime.now().isoformat(), word_count, summary)
    )
    doc_id = c.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def save_flags(document_id: int, flags: list[dict]):
    """Save red flags for a document."""
    conn = get_connection()
    c = conn.cursor()
    for flag in flags:
        c.execute(
            "INSERT INTO flags (document_id, title, clause, why, severity) VALUES (?, ?, ?, ?, ?)",
            (document_id, flag.get("title"), flag.get("clause"), flag.get("why"), flag.get("severity"))
        )
    conn.commit()
    conn.close()

def save_question(document_id: int, question: str, answer: str):
    """Save a question and answer."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO questions (document_id, question, answer, asked_at) VALUES (?, ?, ?, ?)",
        (document_id, question, answer, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_all_documents() -> list[dict]:
    """Get all uploaded documents."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, filename, uploaded_at, word_count FROM documents ORDER BY uploaded_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "uploaded_at": r[2], "word_count": r[3]} for r in rows]

def get_flags_for_document(document_id: int) -> list[dict]:
    """Get all flags for a document."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT title, clause, why, severity FROM flags WHERE document_id = ?", (document_id,))
    rows = c.fetchall()
    conn.close()
    return [{"title": r[0], "clause": r[1], "why": r[2], "severity": r[3]} for r in rows]

def get_questions_for_document(document_id: int) -> list[dict]:
    """Get all questions asked about a document."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT question, answer, asked_at FROM questions WHERE document_id = ? ORDER BY asked_at DESC", (document_id,))
    rows = c.fetchall()
    conn.close()
    return [{"question": r[0], "answer": r[1], "asked_at": r[2]} for r in rows]
