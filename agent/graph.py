"""
graph.py
--------
LangGraph workflow definition for the AI IT Support Assistant.

    START
      |
      v
  entry_router --(awaiting_field set)--> fill_slot ----+
      |                                                 |
      | (no pending field)                              |
      v                                                 |
  classify_intent                                        |
      |                                                 |
      +----------------- route_by_intent <---------------+
      |            |            |            |            |
      v            v            v            v            v
  knowledge_   ticket_      ticket_      escalation   general_chat
  search       lookup       creation                       |
      |            |            |            |              v
      +----- needs_clarification? -----------+             END
      |                       |
      v                       v
  generate_response          END  (a clarifying question was already set)
      |
      v
     END

Each node is a plain function: (AgentState) -> partial AgentState update.
Conditional edges are plain functions: (AgentState) -> str (name of next node).
"""

import re
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from . import llm_client, tools
from .state import AgentState
from .validation import CATEGORIES, canonical_category, category_answer

VALID_INTENTS = {"knowledge_search", "ticket_lookup", "ticket_creation", "escalation", "system_status", "general_chat"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _conversation_context(state: AgentState, max_turns: int = 6) -> str:
    history = state.get("messages", [])[-max_turns:]
    lines = [f"{m['role']}: {m['content']}" for m in history]
    return "\n".join(lines)


def _ask(state: AgentState, question: str, awaiting_field: str, pending_intent: str) -> Dict[str, Any]:
    """Helper to build the 'need more info' partial state update."""
    return {
        "final_response": question,
        "awaiting_field": awaiting_field,
        "pending_intent": pending_intent,
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def entry_router(state: AgentState) -> Dict[str, Any]:
    """Clear per-turn output and intercept cancellation before slot collection."""
    update = {"creation_confirmed": False, "intent": None, "tool_name": None, "tool_result": None, "error": None, "final_response": ""}
    if state["user_input"].strip().lower().strip(" .!") in CANCEL_PHRASES | {"reset"}:
        update.update(ticket_draft={}, awaiting_field=None, pending_intent=None,
                      intent="cancelled", ticket_id=None, search_query=None, escalation_reason=None,
                      final_response="Pending request discarded. No saved tickets were changed.")
    return update


CANCEL_PHRASES = {
    "cancel", "stop", "nevermind", "never mind", "quit", "abort",
    "forget it", "forget about it", "cancel that", "cancel it", "no thanks",
}


def fill_slot(state: AgentState) -> Dict[str, Any]:
    """Consumes the user's answer to a previously-asked clarifying
    question and stores it in the right place, WITHOUT calling the LLM.
    This directly demonstrates state retention across conversation turns."""
    field = state.get("awaiting_field")
    value = state["user_input"].strip()

    ticket_draft = dict(state.get("ticket_draft", {}))
    update: Dict[str, Any] = {"awaiting_field": None, "intent": state.get("pending_intent")}

    if field == "employee_id":
        match = re.search(r"\bEMP\d+\b", value, re.I)
        update["employee_id"] = match.group().upper() if match else value
        update["employee_verified"] = False
    elif field == "ticket_id":
        update["ticket_id"] = value
    elif field == "search_query":
        update["search_query"] = value
    elif field == "escalation_reason":
        update["escalation_reason"] = value
    elif field == "confirmation":
        if value.lower() != "confirm":
            return {"awaiting_field": "confirmation", "intent": "ticket_creation"}
        update["creation_confirmed"] = True
    elif field == "category":
        category, description = category_answer(value)
        ticket_draft["category"] = category
        if description:
            ticket_draft["description"] = description
        update["ticket_draft"] = ticket_draft
    elif field in ("description", "priority"):
        ticket_draft[field] = value
        update["ticket_draft"] = ticket_draft
    else:
        # Unknown field somehow -- fall back to treating this as fresh input
        update["intent"] = None

    return update


def classify_intent(state: AgentState) -> Dict[str, Any]:
    """LLM call #1: decide which tool/branch is needed and extract slots.
    The model is never trusted to invent data -- it only ever echoes back
    what the user said, or null."""
    context = _conversation_context(state)
    try:
        route = llm_client.classify_request(state["user_input"], context)
    except Exception as exc:  # graceful failure per assignment requirements
        return {
            "intent": "general_chat",
            "error": "Intent classification unavailable or invalid.",
            "final_response": "I'm having trouble understanding that request right now. Could you rephrase it?",
        }

    intent = route.get("intent") if route.get("intent") in VALID_INTENTS else "general_chat"

    update: Dict[str, Any] = {"intent": intent, "error": None}

    # Identity is established from explicit input, never changed by model output.
    explicit_id = re.search(r"\bEMP\d+\b", state["user_input"], re.I)
    if not state.get("employee_id") and explicit_id:
        update["employee_id"] = explicit_id.group().upper()
        update["employee_verified"] = False

    # Unlike employee_id (which should stay "sticky" for the whole session so
    # the user doesn't have to repeat it), ticket_id must NOT persist across
    # turns. Otherwise a ticket mentioned earlier in the conversation would
    # silently narrow a later "list my tickets" down to just that one ticket
    # instead of showing all of them. Same reasoning applies to escalation:
    # each escalation request should target exactly the ticket mentioned
    # THIS turn (or none, prompting a fresh ask), never a stale one.
    if intent in ("ticket_lookup", "escalation"):
        update["ticket_id"] = route.get("ticket_id") or None
    elif route.get("ticket_id"):
        update["ticket_id"] = route["ticket_id"]

    if route.get("search_query"):
        update["search_query"] = route["search_query"]
    elif intent == "knowledge_search":
        update["search_query"] = state["user_input"]

    if intent == "escalation":
        update["escalation_reason"] = route.get("escalation_reason") or None

    if intent == "ticket_creation":
        ticket_draft = dict(state.get("ticket_draft", {}))
        if route.get("category"):
            category, description = category_answer(route["category"])
            ticket_draft["category"] = category
            if description:
                ticket_draft["description"] = description
        for field in ("description", "priority"):
            if route.get(field):
                ticket_draft[field] = route[field]
        update["ticket_draft"] = ticket_draft
    if intent == "system_status":
        update["search_query"] = route.get("search_query") or state["user_input"]

    return update


def knowledge_search_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("search_query") or state["user_input"]
    if not query or not query.strip():
        return _ask(state, "What would you like help with? (e.g. 'reset VPN password')", "search_query", "knowledge_search")

    result = tools.search_knowledge_base(query)
    return {"tool_name": "search_knowledge_base", "tool_result": result}


def ticket_lookup_node(state: AgentState) -> Dict[str, Any]:
    employee_id = state.get("employee_id")
    ticket_id = state.get("ticket_id")

    if not employee_id:
        return _ask(
            state,
            "Sure -- what's your employee ID so I can look that up?",
            "employee_id",
            "ticket_lookup",
        )

    result = tools.lookup_tickets(employee_id=employee_id, ticket_id=ticket_id)

    if not result["success"] and not tools.verify_employee(employee_id)["success"]:
        # Unknown employee ID -- ask them to re-confirm rather than guessing
        return _ask(
            state,
            f"I couldn't find an employee with ID '{employee_id}'. Could you double-check your employee ID?",
            "employee_id",
            "ticket_lookup",
        )

    update: Dict[str, Any] = {"tool_name": "lookup_tickets", "tool_result": result}
    if result.get("success") and result.get("employee"):
        update["employee_verified"] = True
    return update


def ticket_creation_node(state: AgentState) -> Dict[str, Any]:
    employee_id = state.get("employee_id")
    draft = state.get("ticket_draft", {})

    # Step 1: employee ID
    if not employee_id:
        return _ask(state, "Happy to raise a ticket. What's your employee ID?", "employee_id", "ticket_creation")

    # Step 1b: verify the employee actually exists (never create a ticket
    # for an ID we can't confirm)
    verify = tools.verify_employee(employee_id)
    if not verify["success"]:
        return {**_ask(state, "Please enter a valid demo employee ID.", "employee_id", "ticket_creation"),
                "employee_verified": False}

    # Step 2: category
    if not canonical_category(draft.get("category")):
        return _ask(
            state,
            "Choose a category: " + ", ".join(CATEGORIES) + ". You can add a description after a period.",
            "category",
            "ticket_creation",
        )

    # Step 3: description
    if not draft.get("description"):
        return _ask(state, "Could you briefly describe the issue?", "description", "ticket_creation")

    # All required fields present -- validate once more, then create
    missing = tools.validate_ticket_draft(
        {"employee_id": employee_id, "category": draft.get("category"), "description": draft.get("description")}
    )
    if missing:
        return _ask(state, f"I still need your {missing[0].replace('_', ' ')}.", missing[0], "ticket_creation")

    draft = {**draft, "category": canonical_category(draft["category"]),
             "description": draft["description"].strip(),
             "priority": draft.get("priority") if str(draft.get("priority", "")).lower() in tools.VALID_PRIORITIES else "Medium"}
    if not state.get("creation_confirmed"):
        return {**_ask(state,
            f"Review ticket for {employee_id}:\n\nCategory: {draft['category']}\n\nDescription: {draft['description']}\n\nPriority: {draft['priority']}\n\nType confirm to create or cancel to discard.",
            "confirmation", "ticket_creation"), "ticket_draft": draft, "employee_verified": True}

    result = tools.create_ticket(
        employee_id=employee_id,
        category=draft["category"],
        description=draft["description"],
        priority=draft.get("priority"),
    )

    update: Dict[str, Any] = {
        "tool_name": "create_ticket",
        "tool_result": result,
        "employee_verified": True,
    }
    if result.get("success"):
        # Clear the draft so a follow-up message doesn't accidentally reuse it
        update["ticket_draft"] = {}
        update["pending_intent"] = None
        update["awaiting_field"] = None
    return update


def escalate_ticket_node(state: AgentState) -> Dict[str, Any]:
    """Escalation flow mirrors ticket_creation_node's step-by-step slot
    filling: confirm who's asking, which ticket, and why -- in that order --
    before ever touching the data layer."""
    employee_id = state.get("employee_id")

    # Step 1: employee ID (needed to confirm ticket ownership below)
    if not employee_id:
        return _ask(
            state,
            "Sure -- what's your employee ID? I'll use it to confirm the ticket is yours before escalating.",
            "employee_id",
            "escalation",
        )

    if not state.get("employee_verified"):
        verify = tools.verify_employee(employee_id)
        if not verify["success"]:
            return _ask(
                state,
                f"I couldn't find employee ID '{employee_id}'. Could you re-check and re-enter it?",
                "employee_id",
                "escalation",
            )

    # Step 2: which ticket
    ticket_id = state.get("ticket_id")
    if not ticket_id:
        return _ask(state, "Which ticket would you like to escalate? Please share the ticket ID.", "ticket_id", "escalation")

    # Step 3: why
    reason = state.get("escalation_reason")
    if not reason:
        return _ask(
            state,
            "What's the reason for escalating this ticket? (e.g. unresolved for too long, urgent business impact)",
            "escalation_reason",
            "escalation",
        )

    result = tools.escalate_ticket(ticket_id=ticket_id, reason=reason, requesting_employee_id=employee_id)

    update: Dict[str, Any] = {
        "tool_name": "escalate_ticket",
        "tool_result": result,
        "employee_verified": True,
    }
    if result.get("success"):
        # Clear so a later, unrelated message doesn't accidentally reuse
        # this ticket/reason (same rationale as ticket_draft after creation).
        update["ticket_id"] = None
        update["escalation_reason"] = None
        update["pending_intent"] = None
        update["awaiting_field"] = None
    return update


def system_status_node(state: AgentState) -> Dict[str, Any]:
    system_name = state.get("search_query") or state.get("user_input")
    result = tools.check_system_status(system_name)
    return {"tool_name": "check_system_status", "tool_result": result}


def general_chat_node(state: AgentState) -> Dict[str, Any]:
    if state.get("error"):
        return {}
    return {"final_response": "I can search IT guidance, look up tickets for your demo employee profile, prepare a ticket for confirmation, escalate an existing ticket to High priority, or show sample system status. I cannot edit ticket details or cancel saved tickets, add comments, or send notifications."}


def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """Render actual tool fields; the LLM classifies but cannot invent result facts."""
    return {"final_response": _fallback_response(state.get("tool_name"), state.get("tool_result") or {})}


def _fallback_response(tool_name: str, result: Dict[str, Any]) -> str:
    """Render stored facts without a second generative call."""
    if not result.get("success", True):
        return f"Sorry, I ran into an issue: {result.get('error', 'unknown error')}"

    if tool_name == "search_knowledge_base":
        articles = result.get("results", [])
        if not articles:
            return "I couldn't find a knowledge-base article matching that. Could you rephrase, or would you like me to raise a ticket?"
        lines = [f"- {a['title']} ({a['article_id']}): {a['content']}" for a in articles]
        return "Here's what I found:\n" + "\n".join(lines)

    if tool_name == "lookup_tickets":
        t = result.get("results", [])
        if not t:
            return "No tickets found for that ID."
        lines = [f"- {x['ticket_id']} [{x['status']}] {x['category']}: {x['description']}" for x in t]
        return "Here are the tickets I found:\n" + "\n".join(lines)

    if tool_name == "create_ticket":
        ticket = result.get("ticket")
        if result.get("duplicate"):
            return result.get("message", "An open ticket already exists for this issue.")
        if ticket:
            return f"Done! Ticket {ticket['ticket_id']} has been raised ({ticket['category']}, priority: {ticket['priority']}, status: {ticket['status']}).\n{ticket['description']}"
        return "Sorry, I wasn't able to create the ticket."

    if tool_name == "escalate_ticket":
        ticket = result.get("ticket")
        if result.get("already_escalated"):
            return result.get("message", "This ticket was already escalated.")
        if ticket:
            return f"Ticket {ticket['ticket_id']} has been escalated and its priority raised to {ticket['priority']}."
        return "Sorry, I wasn't able to escalate that ticket."

    if tool_name == "check_system_status":
        rows = result.get("results", [])
        lines = [f"- {r['system']}: {r['status']} ({r['notes']}) — Last incident: {r.get('last_incident') or 'not recorded'}" for r in rows]
        return "Sample system status (local data, not a live availability check; refresh time unknown):\n" + "\n".join(lines)

    return "Done."


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_after_entry(state: AgentState) -> str:
    return END if state.get("intent") == "cancelled" else ("fill_slot" if state.get("awaiting_field") else "classify_intent")


def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent") or "general_chat"
    return {
        "knowledge_search": "knowledge_search",
        "ticket_lookup": "ticket_lookup",
        "ticket_creation": "ticket_creation",
        "escalation": "escalation",
        "system_status": "system_status",
        "general_chat": "general_chat",
    }.get(intent, "general_chat")


def needs_clarification(state: AgentState) -> str:
    return END if state.get("awaiting_field") else "generate_response"


def route_after_fill_slot(state: AgentState) -> str:
    """Like route_by_intent, but first checks for the 'cancelled' sentinel
    fill_slot sets when the user's reply was a cancellation rather than an
    answer to the pending question -- in that case final_response is
    already set, so this routes straight to END instead of running a tool
    node (which would try to act on the now-cleared draft)."""
    if state.get("intent") == "cancelled":
        return END
    return route_by_intent(state)


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("entry_router", entry_router)
    graph.add_node("fill_slot", fill_slot)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("knowledge_search", knowledge_search_node)
    graph.add_node("ticket_lookup", ticket_lookup_node)
    graph.add_node("ticket_creation", ticket_creation_node)
    graph.add_node("escalation", escalate_ticket_node)
    graph.add_node("system_status", system_status_node)
    graph.add_node("general_chat", general_chat_node)
    graph.add_node("generate_response", generate_response_node)

    graph.set_entry_point("entry_router")

    graph.add_conditional_edges(
        "entry_router",
        route_after_entry,
        {END: END, "fill_slot": "fill_slot", "classify_intent": "classify_intent"},
    )

    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "knowledge_search": "knowledge_search",
            "ticket_lookup": "ticket_lookup",
            "ticket_creation": "ticket_creation",
            "escalation": "escalation",
            "system_status": "system_status",
            "general_chat": "general_chat",
        },
    )

    graph.add_conditional_edges(
        "fill_slot",
        route_after_fill_slot,
        {
            "knowledge_search": "knowledge_search",
            "ticket_lookup": "ticket_lookup",
            "ticket_creation": "ticket_creation",
            "escalation": "escalation",
            "system_status": "system_status",
            "general_chat": "general_chat",
            END: END,
        },
    )

    for tool_node in ("knowledge_search", "ticket_lookup", "ticket_creation", "escalation", "system_status"):
        graph.add_conditional_edges(
            tool_node,
            needs_clarification,
            {END: END, "generate_response": "generate_response"},
        )

    graph.add_edge("general_chat", END)
    graph.add_edge("generate_response", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
