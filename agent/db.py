"""
db.py
-----
Thin data-access layer around the local JSON "database" files.

Keeping all file I/O in one module means every other part of the
application (tools.py, graph.py) only ever talks to plain Python
objects (lists / dicts), never touches file paths directly. This
keeps the code modular and makes it trivial to swap JSON files for
SQLite later without changing any calling code.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

EMPLOYEES_FILE = os.path.join(DATA_DIR, "employees.json")
TICKETS_FILE = os.path.join(DATA_DIR, "tickets.json")
KNOWLEDGE_BASE_FILE = os.path.join(DATA_DIR, "knowledge_base.json")
SYSTEM_STATUS_FILE = os.path.join(DATA_DIR, "system_status.json")


def _read_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------

def get_employees() -> List[Dict[str, Any]]:
    return _read_json(EMPLOYEES_FILE)


def find_employee(employee_id: str) -> Optional[Dict[str, Any]]:
    if not employee_id:
        return None
    employee_id = employee_id.strip().upper()
    for emp in get_employees():
        if emp["employee_id"].upper() == employee_id:
            return emp
    return None


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------

def get_tickets() -> List[Dict[str, Any]]:
    return _read_json(TICKETS_FILE)


def find_tickets(employee_id: Optional[str] = None, ticket_id: Optional[str] = None) -> List[Dict[str, Any]]:
    tickets = get_tickets()
    results = tickets

    if ticket_id:
        ticket_id = ticket_id.strip().upper()
        results = [t for t in results if t["ticket_id"].upper() == ticket_id]

    if employee_id:
        employee_id = employee_id.strip().upper()
        results = [t for t in results if t["employee_id"].upper() == employee_id]

    # Most recent first
    results = sorted(results, key=lambda t: t.get("updated_at", ""), reverse=True)
    return results


def find_open_ticket_for_category(employee_id: str, category: str) -> Optional[Dict[str, Any]]:
    """Used for duplicate-ticket prevention."""
    for t in get_tickets():
        if (
            t["employee_id"].upper() == employee_id.upper()
            and t["category"].lower() == category.lower()
            and t["status"] in ("Open", "In Progress")
        ):
            return t
    return None


def _next_ticket_id(tickets: List[Dict[str, Any]]) -> str:
    max_num = 1000
    for t in tickets:
        try:
            num = int(t["ticket_id"].split("-")[1])
            max_num = max(max_num, num)
        except (IndexError, ValueError):
            continue
    return f"TCK-{max_num + 1}"


def create_ticket(employee_id: str, category: str, description: str, priority: str = "Medium") -> Dict[str, Any]:
    tickets = get_tickets()
    now = datetime.now().isoformat(timespec="seconds")
    new_ticket = {
        "ticket_id": _next_ticket_id(tickets),
        "employee_id": employee_id.strip().upper(),
        "category": category.strip().title(),
        "description": description.strip(),
        "priority": priority.strip().title() if priority else "Medium",
        "status": "Open",
        "created_at": now,
        "updated_at": now,
    }
    tickets.append(new_ticket)
    _write_json(TICKETS_FILE, tickets)
    return new_ticket


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

def get_knowledge_base() -> List[Dict[str, Any]]:
    return _read_json(KNOWLEDGE_BASE_FILE)


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------

def get_system_status() -> List[Dict[str, Any]]:
    return _read_json(SYSTEM_STATUS_FILE)
