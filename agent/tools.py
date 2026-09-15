"""
tools.py
--------
The concrete "tools" the agent can call. Each function:
  - takes plain, explicit parameters (never invents data itself)
  - returns a plain dict with a `success` flag so calling nodes can
    handle failures gracefully instead of crashing the graph.

These are intentionally framework-agnostic (no LangGraph / LLM
imports here) so they can be unit-tested in isolation.
"""

import re
from typing import Any, Dict, List, Optional

from . import db
from .validation import canonical_category

STOP_WORDS = {
    "a", "an", "the", "is", "are", "my", "i", "to", "for", "of", "on", "in",
    "how", "do", "does", "can", "please", "help", "with", "and", "not",
    "it", "me", "you", "your", "this", "that", "there", "have", "has",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS]


# ---------------------------------------------------------------------------
# Tool 1: Knowledge Search
# ---------------------------------------------------------------------------

def search_knowledge_base(query: str, top_k: int = 2) -> Dict[str, Any]:
    """Simple keyword-overlap scoring across title/tags/content.

    Kept deliberately simple (no external embedding/vector-DB call)
    so the project runs fully offline on a local machine, per the
    assignment's 'intentionally realistic and achievable' guidance.
    """
    if not query or not query.strip():
        return {"success": False, "error": "Empty search query.", "results": []}

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return {"success": False, "error": "Query had no searchable keywords.", "results": []}

    scored = []
    for article in db.get_knowledge_base():
        haystack_tokens = set(_tokenize(article["title"])) | set(
            t.lower() for t in article.get("tags", [])
        ) | set(_tokenize(article["content"]))
        overlap = query_tokens & haystack_tokens
        if overlap:
            scored.append((len(overlap), article))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_matches = [article for _, article in scored[:top_k]]

    return {
        "success": True,
        "results": top_matches,
        "match_count": len(scored),
    }


# ---------------------------------------------------------------------------
# Tool 2: Ticket Lookup
# ---------------------------------------------------------------------------

def lookup_tickets(employee_id: Optional[str] = None, ticket_id: Optional[str] = None) -> Dict[str, Any]:
    if not employee_id:
        return {"success": False, "error": "An employee ID is required for ticket lookup.", "results": []}

    # Employee-based lookup: employee must exist
    employee = db.find_employee(employee_id)
    if not employee:
        return {
            "success": False,
            "error": f"No employee found with ID '{employee_id}'. Please double-check the ID.",
            "results": [],
        }

    results = db.find_tickets(employee_id=employee_id, ticket_id=ticket_id)
    return {"success": True, "results": results, "employee": employee}


# ---------------------------------------------------------------------------
# Tool 3: Ticket Creation
# ---------------------------------------------------------------------------

REQUIRED_TICKET_FIELDS = ["employee_id", "category", "description"]
VALID_PRIORITIES = {"low", "medium", "high"}


def validate_ticket_draft(draft: Dict[str, Any]) -> List[str]:
    """Returns a list of missing/invalid required fields (empty list = valid)."""
    problems = []
    for field in REQUIRED_TICKET_FIELDS:
        if not draft.get(field) or not str(draft.get(field)).strip():
            problems.append(field)
    if draft.get("category") and not canonical_category(draft["category"]):
        problems.append("category")
    return problems


def create_ticket(employee_id: str, category: str, description: str, priority: Optional[str] = None) -> Dict[str, Any]:
    # Never trust unvalidated input into the "database" -- re-validate here too,
    # even if the graph already validated it, since this function may be
    # called directly (e.g. in tests).
    missing = validate_ticket_draft(
        {"employee_id": employee_id, "category": category, "description": description}
    )
    if missing:
        return {"success": False, "error": f"Missing required fields: {', '.join(missing)}", "ticket": None}

    employee = db.find_employee(employee_id)
    if not employee:
        return {"success": False, "error": f"Cannot create ticket: unknown employee ID '{employee_id}'.", "ticket": None}

    if not priority or priority.strip().lower() not in VALID_PRIORITIES:
        priority = "Medium"

    category = canonical_category(category)

    # Duplicate prevention
    existing = db.find_open_ticket_for_category(employee_id, category)
    if existing:
        return {
            "success": True,
            "duplicate": True,
            "ticket": existing,
            "message": (
                f"An open ticket ({existing['ticket_id']}) already exists for "
                f"'{category}'. Skipped creating a duplicate."
            ),
        }

    ticket = db.create_ticket(employee_id, category, description, priority)
    return {"success": True, "duplicate": False, "ticket": ticket}


# ---------------------------------------------------------------------------
# Helper tool: Employee verification
# ---------------------------------------------------------------------------

def verify_employee(employee_id: str) -> Dict[str, Any]:
    employee = db.find_employee(employee_id)
    if not employee:
        return {"success": False, "error": f"No employee found with ID '{employee_id}'.", "employee": None}
    return {"success": True, "employee": employee}


# ---------------------------------------------------------------------------
# Bonus tool: System status
# ---------------------------------------------------------------------------

def check_system_status(system_name: Optional[str] = None) -> Dict[str, Any]:
    statuses = db.get_system_status()
    if system_name:
        tokens = set(_tokenize(system_name))
        matches = [
            s for s in statuses
            if tokens & set(_tokenize(s["system"]))
        ]
        if not matches:
            return {"success": False, "error": f"No status information for '{system_name}'.", "results": []}
        return {"success": True, "results": matches}
    return {"success": True, "results": statuses}
