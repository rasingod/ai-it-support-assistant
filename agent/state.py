"""
state.py
--------
Defines the shared state object that flows through every node of the
LangGraph workflow. Using a single TypedDict keeps state management
explicit: every node declares exactly what it reads and writes.
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- Conversation ---
    messages: List[Dict[str, str]]     # full chat history: [{"role": "user"/"assistant", "content": str}]
    user_input: str                    # latest raw user message being processed this turn

    # --- Routing / intent ---
    intent: Optional[str]              # "knowledge_search" | "ticket_lookup" | "ticket_creation" | "escalation" | "system_status" | "general_chat"
    pending_intent: Optional[str]      # intent we were mid-way through when a clarifying question was asked

    # --- Slot-filling memory (persists across turns) ---
    employee_id: Optional[str]
    employee_verified: bool
    ticket_id: Optional[str]
    ticket_draft: Dict[str, Any]       # {category, description, priority}
    escalation_reason: Optional[str]   # reason given for escalating ticket_id
    search_query: Optional[str]

    # --- Conditional-routing control ---
    awaiting_field: Optional[str]      # e.g. "employee_id" | "category" | "description" -- set when the agent must ask the user something before continuing

    # --- Tool execution results (transient, per turn) ---
    tool_name: Optional[str]
    tool_result: Optional[Dict[str, Any]]

    # --- Output ---
    final_response: str
    error: Optional[str]


def new_state() -> AgentState:
    """Factory for a fresh conversation state (used at session start)."""
    return AgentState(
        messages=[],
        user_input="",
        intent=None,
        pending_intent=None,
        employee_id=None,
        employee_verified=False,
        ticket_id=None,
        ticket_draft={},
        escalation_reason=None,
        search_query=None,
        awaiting_field=None,
        tool_name=None,
        tool_result=None,
        final_response="",
        error=None,
    )
