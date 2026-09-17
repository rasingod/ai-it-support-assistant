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

STOP_WORDS = {
    "a", "an", "the", "is", "are", "my", "i", "to", "for", "of", "on", "in",
    "how", "do", "does", "can", "please", "help", "with", "and", "not",
    "it", "me", "you", "your", "this", "that", "there", "have", "has",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS]


# ---------------------------------------------------------------------------
# Shared: category normalization
# ---------------------------------------------------------------------------
# Constraining category to a known list (rather than storing whatever raw
# text the user typed) fixes two related problems: a free-form reply that
# crams in extra detail no longer gets stored verbatim as the "category"
# (which broke duplicate-detection, since two tickets about the same VPN
# issue could end up with different literal category strings), and every
# displayed/stored category value is guaranteed to be one of a small,
# predictable set.

ALLOWED_CATEGORIES = [
    "VPN", "Laptop", "Email", "Printer", "Software", "Internet",
    "Mobile", "Account Access", "Hardware", "Network",
]
_CATEGORY_ALIASES = {
    "wifi": "Network", "wi-fi": "Network", "wi fi": "Network",
    "phone": "Mobile", "cell phone": "Mobile", "cellphone": "Mobile",
    "pc": "Laptop", "computer": "Laptop", "desktop": "Laptop", "notebook": "Laptop",
    "outlook": "Email", "mail": "Email", "e-mail": "Email",
    "app": "Software", "application": "Software", "program": "Software",
    "mfa": "Account Access", "2fa": "Account Access", "login": "Account Access",
    "password": "Account Access",
}


def normalize_category(raw_text: str) -> Dict[str, Optional[str]]:
    """Extracts a known category from free-form text instead of storing the
    raw text verbatim. Returns {"category": <one of ALLOWED_CATEGORIES or
    "Other">, "leftover": <remaining text that looks like a description, or
    None>}.

    This lets a combined reply like "Printer, my printer prints blank pages"
    (answering a single "what category?" question) split cleanly into a
    valid category plus a separate description, instead of the entire
    sentence becoming the category.
    """
    if not raw_text or not raw_text.strip():
        return {"category": "Other", "leftover": None}

    text = raw_text.strip()
    text_lower = text.lower()
    token_set = set(_tokenize(text))

    matched = None
    matched_phrase = None
    for cat in ALLOWED_CATEGORIES:
        cat_tokens = set(_tokenize(cat))
        if cat_tokens and cat_tokens.issubset(token_set):
            matched = cat
            matched_phrase = cat
            break
    if not matched:
        for alias, cat in _CATEGORY_ALIASES.items():
            if alias in text_lower:
                matched = cat
                matched_phrase = alias
                break

    if not matched:
        # Couldn't confidently identify a known category -- use the safe
        # generic bucket rather than storing arbitrary free text as the
        # category, but keep the user's own words as the description so
        # nothing they said is silently discarded.
        return {"category": "Other", "leftover": text}

    leftover = re.sub(re.escape(matched_phrase), "", text, count=1, flags=re.IGNORECASE).strip(" ,.-:;")
    leftover = re.sub(r"\s+", " ", leftover).strip()  # collapse any double-space left by the removal
    if len(leftover) < 4:  # not enough left over to be a meaningful description
        leftover = None

    return {"category": matched, "leftover": leftover}


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
    if not employee_id and not ticket_id:
        return {
            "success": False,
            "error": "Need either an employee ID or a ticket ID to look up tickets.",
            "results": [],
        }

    if ticket_id:
        results = db.find_tickets(ticket_id=ticket_id)
        if not results:
            return {"success": False, "error": f"No ticket found with ID '{ticket_id}'.", "results": []}

        # Ownership check: if we know which employee is asking, the ticket
        # must belong to them. Return the SAME "not found" message either
        # way (don't first reveal the ticket exists/belongs to someone else
        # and only then warn -- that's disclosure regardless of the wording
        # attached to it).
        if employee_id and results[0]["employee_id"].upper() != employee_id.strip().upper():
            return {"success": False, "error": f"No ticket found with ID '{ticket_id}'.", "results": []}

        return {"success": True, "results": results}

    # Employee-based lookup: employee must exist
    employee = db.find_employee(employee_id)
    if not employee:
        return {
            "success": False,
            "error": f"No employee found with ID '{employee_id}'. Please double-check the ID.",
            "results": [],
        }

    results = db.find_tickets(employee_id=employee_id)
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

    # Constrain category to the known set here too, not just in the graph's
    # slot-filling step -- this function is the single point every ticket
    # write goes through, so it's the most reliable place to guarantee a
    # consistent, validated category regardless of how it was captured.
    category = normalize_category(category)["category"]

    if not priority or priority.strip().lower() not in VALID_PRIORITIES:
        priority = "Medium"

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
# Tool 4: Ticket Escalation
# ---------------------------------------------------------------------------

NON_ESCALATABLE_STATUSES = {"resolved", "closed"}


def escalate_ticket(ticket_id: str, reason: str, requesting_employee_id: Optional[str] = None) -> Dict[str, Any]:
    """Escalates an existing ticket: bumps priority to High and records why.

    Guardrails (mirroring create_ticket's validation style):
      - Ticket must exist -- never invents one.
      - If a requesting employee ID is given, the ticket must belong to them
        (an employee shouldn't be able to escalate someone else's ticket).
      - A resolved/closed ticket can't be escalated.
      - Escalating an already-escalated ticket is a no-op, not a duplicate
        action -- mirrors create_ticket's duplicate-prevention pattern.
    """
    if not ticket_id or not str(ticket_id).strip():
        return {"success": False, "error": "Missing ticket ID to escalate.", "ticket": None}
    if not reason or not str(reason).strip():
        return {"success": False, "error": "Missing a reason for the escalation.", "ticket": None}

    matches = db.find_tickets(ticket_id=ticket_id)
    if not matches:
        return {"success": False, "error": f"No ticket found with ID '{ticket_id}'.", "ticket": None}
    ticket = matches[0]

    if requesting_employee_id and ticket["employee_id"].upper() != requesting_employee_id.strip().upper():
        return {
            "success": False,
            "error": f"Ticket {ticket['ticket_id']} doesn't belong to employee '{requesting_employee_id}', so it can't be escalated from here.",
            "ticket": None,
        }

    if ticket.get("status", "").lower() in NON_ESCALATABLE_STATUSES:
        return {
            "success": False,
            "error": f"Ticket {ticket['ticket_id']} is already {ticket['status']} and can't be escalated.",
            "ticket": ticket,
        }

    if ticket.get("escalated"):
        return {
            "success": True,
            "already_escalated": True,
            "ticket": ticket,
            "message": f"Ticket {ticket['ticket_id']} was already escalated on {ticket.get('escalated_at', 'an earlier date')}.",
        }

    updated = db.escalate_ticket(ticket_id, reason)
    return {"success": True, "already_escalated": False, "ticket": updated}


# ---------------------------------------------------------------------------
# Bonus tool: System status
# ---------------------------------------------------------------------------

_STATUS_FRESHNESS_NOTE = (
    "This reflects the last recorded status update in the system-status "
    "data source, not a live real-time check performed just now."
)


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
        return {"success": True, "results": matches, "note": _STATUS_FRESHNESS_NOTE}
    return {"success": True, "results": statuses, "note": _STATUS_FRESHNESS_NOTE}
