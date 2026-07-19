"""
query/risk.py

Governance risk engine for college constitutions.

Unlike the old red_flag_detector.py (which looked for legal contract risks),
this module identifies governance violations — structural problems that could
make a constitution unfair, unenforceable, or open to abuse.

Risk categories (with severity HIGH / MEDIUM / LOW):
    - POWER_CONCENTRATION   : one body/person has unchecked authority
    - NO_APPEAL             : decisions can't be challenged
    - VAGUE_ENFORCEMENT     : penalties/consequences are undefined
    - MISSING_QUORUM        : voting rules don't specify minimum attendance
    - MISSING_TERM_LIMITS   : no limits on how long someone holds a role
    - AMENDMENT_LOCK        : constitution is too hard or too easy to amend
    - CONFLICT_OF_INTEREST  : no rules around bias or self-interest
    - OPAQUE_PROCESS        : key decisions happen without defined process

Usage:
    from query.risk import analyse_risks

    risks = analyse_risks(clauses)   # list of clause dicts
"""

import json
import re
import requests
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

RISK_CATEGORIES = [
    "POWER_CONCENTRATION",
    "NO_APPEAL",
    "VAGUE_ENFORCEMENT",
    "MISSING_QUORUM",
    "MISSING_TERM_LIMITS",
    "AMENDMENT_LOCK",
    "CONFLICT_OF_INTEREST",
    "OPAQUE_PROCESS",
]

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_risks(clauses: list[dict]) -> list[dict]:
    """
    Analyses all clauses for governance risks.

    Args:
        clauses: list of clause dicts (must have id, heading, text)

    Returns:
        List of risk dicts, sorted by severity then score:
        {
            clause_id   : str,
            heading     : str,
            category    : str,   # one of RISK_CATEGORIES
            severity    : str,   # HIGH / MEDIUM / LOW
            score       : float, # 0.0–1.0
            reason      : str,   # one sentence explanation
            suggestion  : str,   # how to fix it
        }
    """
    all_risks = []
    total = len(clauses)

    for i, clause in enumerate(clauses):
        if i % 5 == 0:
            print(f"[risk] Analysing clause {i+1}/{total}: {clause.get('heading', '')}")
        risks = _analyse_single_clause(clause)
        all_risks.extend(risks)

    # Sort: HIGH first, then by score descending
    all_risks.sort(key=lambda r: (SEVERITY_ORDER.get(r["severity"], 3), -r.get("score", 0)))
    print(f"[risk] Found {len(all_risks)} risk flags across {total} clauses.")
    return all_risks


def analyse_single_clause_risk(clause: dict) -> list[dict]:
    """
    Public wrapper for single-clause risk analysis.
    Used by the impact engine to check a new clause before committing.
    """
    return _analyse_single_clause(clause)


def risk_summary(risks: list[dict]) -> dict:
    """
    Returns a summary dict with counts per severity and category.
    """
    summary = {
        "total": len(risks),
        "HIGH": 0, "MEDIUM": 0, "LOW": 0,
        "by_category": {cat: 0 for cat in RISK_CATEGORIES},
    }
    for risk in risks:
        sev = risk.get("severity", "LOW")
        summary[sev] = summary.get(sev, 0) + 1
        cat = risk.get("category", "")
        if cat in summary["by_category"]:
            summary["by_category"][cat] += 1
    return summary


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _analyse_single_clause(clause: dict) -> list[dict]:
    """
    Sends one clause to the LLM for governance risk analysis.
    Returns list of risk dicts (empty if no risks found).
    """
    heading = clause.get("heading", "Untitled")
    text    = clause.get("text", "")

    if len(text.strip()) < 20:
        return []

    categories_str = "\n".join(f"  - {c}" for c in RISK_CATEGORIES)

    prompt = f"""You are a governance expert reviewing a college constitution clause for structural risks.

CLAUSE: "{heading}"
TEXT: "{text}"

Check for these governance risk categories:
{categories_str}

Definitions:
- POWER_CONCENTRATION: one person/body has unchecked, unilateral authority
- NO_APPEAL: a decision process has no mechanism for challenge or review
- VAGUE_ENFORCEMENT: consequences for violations are undefined or purely discretionary
- MISSING_QUORUM: a vote or meeting has no minimum attendance requirement
- MISSING_TERM_LIMITS: a role has no defined maximum tenure
- AMENDMENT_LOCK: clause is written such that it cannot be changed, or can be changed by one party alone
- CONFLICT_OF_INTEREST: no recusal rules for people with personal stake in a decision
- OPAQUE_PROCESS: a major decision happens without any defined procedure or transparency requirement

Respond ONLY with a JSON array. Each item:
  - "category": one of the categories above
  - "severity": "HIGH", "MEDIUM", or "LOW"
  - "score": float 0.0–1.0 (how serious is this risk)
  - "reason": one sentence explaining the risk
  - "suggestion": one sentence on how to fix it

Only include genuine risks with score >= 0.5.
If no risks found, return [].
No markdown. No extra text. Only the JSON array.
"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()

        items = json.loads(raw)
        risks = []
        for item in items:
            if item.get("score", 0) < 0.5:
                continue
            if item.get("category") not in RISK_CATEGORIES:
                continue
            risks.append({
                "clause_id":  clause["id"],
                "heading":    heading,
                "category":   item["category"],
                "severity":   item.get("severity", "MEDIUM").upper(),
                "score":      float(item.get("score", 0.5)),
                "reason":     item.get("reason", ""),
                "suggestion": item.get("suggestion", ""),
            })
        return risks

    except Exception as e:
        print(f"[risk] Analysis failed for '{heading}': {e}")
        return []
