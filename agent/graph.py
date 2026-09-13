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
      |            |            |            |
      v            v            v            v
  knowledge_   ticket_      ticket_       general_chat
  search       lookup       creation           |
      |            |            |              v
      +----- needs_clarification? -----+      END
      |                       |
      v                       v
  generate_response          END  (a clarifying question was already set)
      |
      v
     END

Each node is a plain function: (AgentState) -> partial AgentState update.
Conditional edges are plain functions: (AgentState) -> str (name of next node).
"""

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from . import llm_client, tools
from .state import AgentState

VALID_INTENTS = {"knowledge_search", "ticket_lookup", "ticket_creation", "system_status", "general_chat"}


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
    """Pass-through node. Its only purpose is to exist as a clean START
    target so the conditional edge below can inspect state before any
    LLM call is made (saves a token spend when we're just filling a slot)."""
    return {}


def fill_slot(state: AgentState) -> Dict[str, Any]:
    """Consumes the user's answer to a previously-asked clarifying
    question and stores it in the right place, WITHOUT calling the LLM.
    This directly demonstrates state retention across conversation turns."""
    field = state.get("awaiting_field")
    value = state["user_input"].strip()
    ticket_draft = dict(state.get("ticket_draft", {}))

    update: Dict[str, Any] = {"awaiting_field": None, "intent": state.get("pending_intent")}

    if field == "employee_id":
        update["employee_id"] = value
    elif field == "ticket_id":
        update["ticket_id"] = value
    elif field == "search_query":
        update["search_query"] = value
    elif field in ("category", "description", "priority"):
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
            "error": f"Intent classification failed: {exc}",
            "final_response": "I'm having trouble understanding that request right now. Could you rephrase it?",
        }

    intent = route.get("intent") if route.get("intent") in VALID_INTENTS else "general_chat"

    update: Dict[str, Any] = {"intent": intent, "error": None}

    if route.get("employee_id"):
        update["employee_id"] = route["employee_id"]

    # Unlike employee_id (which should stay "sticky" for the whole session so
    # the user doesn't have to repeat it), ticket_id must NOT persist across
    # turns. Otherwise a ticket mentioned earlier in the conversation would
    # silently narrow a later "list my tickets" down to just that one ticket
    # instead of showing all of them. So for ticket_lookup we always sync
    # ticket_id to exactly what (if anything) was mentioned THIS turn.
    if intent == "ticket_lookup":
        update["ticket_id"] = route.get("ticket_id") or None
    elif route.get("ticket_id"):
        update["ticket_id"] = route["ticket_id"]

    if route.get("search_query"):
        update["search_query"] = route["search_query"]
    elif intent == "knowledge_search":
        update["search_query"] = state["user_input"]

    ticket_draft = dict(state.get("ticket_draft", {}))
    if route.get("category"):
        ticket_draft["category"] = route["category"]
    if route.get("description"):
        ticket_draft["description"] = route["description"]
    if route.get("priority"):
        ticket_draft["priority"] = route["priority"]
    if ticket_draft:
        update["ticket_draft"] = ticket_draft

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

    if not employee_id and not ticket_id:
        return _ask(
            state,
            "Sure -- what's your employee ID (or the ticket ID) so I can look that up?",
            "employee_id",
            "ticket_lookup",
        )

    result = tools.lookup_tickets(employee_id=employee_id, ticket_id=ticket_id)

    if not result["success"] and employee_id and not ticket_id:
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
    if not state.get("employee_verified"):
        verify = tools.verify_employee(employee_id)
        if not verify["success"]:
            return _ask(
                state,
                f"I couldn't find employee ID '{employee_id}'. Could you re-check and re-enter it?",
                "employee_id",
                "ticket_creation",
            )

    # Step 2: category
    if not draft.get("category"):
        return _ask(
            state,
            "What category best describes the issue? (e.g. VPN, Laptop, Email, Printer, Software)",
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
    return update


def system_status_node(state: AgentState) -> Dict[str, Any]:
    system_name = state.get("search_query") or state.get("user_input")
    result = tools.check_system_status(system_name)
    return {"tool_name": "check_system_status", "tool_result": result}


def general_chat_node(state: AgentState) -> Dict[str, Any]:
    system_prompt = (
        "You are a friendly, concise IT Support Assistant for a fictional company. "
        "The user's message is not a specific IT request (e.g. it's a greeting or "
        "thanks). Reply briefly and, if relevant, mention you can help with "
        "knowledge-base questions, ticket status checks, or raising new tickets."
    )
    try:
        reply = llm_client.generate_reply(system_prompt, [{"role": "user", "content": state["user_input"]}])
    except Exception as exc:
        reply = "Hi! I can help you search IT help articles, check ticket status, or raise a new ticket. What do you need?"
        return {"final_response": reply, "error": f"General chat generation failed: {exc}"}
    return {"final_response": reply}


def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """LLM call #2: turn a tool's raw result into a clear, user-friendly
    answer. The LLM is only asked to *phrase* the already-retrieved data,
    never to add new facts."""
    tool_name = state.get("tool_name")
    result = state.get("tool_result") or {}

    system_prompt = (
        "You are an IT Support Assistant. You will be given the name of a tool "
        "that was just executed and its JSON result. Write a short, clear, "
        "friendly response to the user based ONLY on that JSON data. "
        "Do not invent any information (ticket IDs, statuses, article steps, "
        "employee names, etc.) that is not present in the JSON. "
        "If the JSON indicates an error or empty results, say so plainly and, "
        "for ticket creation, never claim a ticket was created if it wasn't."
    )

    user_message = (
        f"Tool executed: {tool_name}\n"
        f"Tool result (JSON):\n{result}\n\n"
        f"Original user message: {state.get('user_input')}"
    )

    try:
        reply = llm_client.generate_reply(system_prompt, [{"role": "user", "content": user_message}])
    except Exception as exc:
        reply = _fallback_response(tool_name, result)
        return {"final_response": reply, "error": f"Response generation failed: {exc}"}

    return {"final_response": reply}


def _fallback_response(tool_name: str, result: Dict[str, Any]) -> str:
    """Deterministic, template-based fallback used only if the LLM call
    itself fails -- keeps the app usable even without API access, and
    satisfies the 'handle tool failures gracefully' requirement."""
    if not result.get("success", True):
        return f"Sorry, I ran into an issue: {result.get('error', 'unknown error')}"

    if tool_name == "search_knowledge_base":
        articles = result.get("results", [])
        if not articles:
            return "I couldn't find a knowledge-base article matching that. Could you rephrase, or would you like me to raise a ticket?"
        lines = [f"- {a['title']} ({a['article_id']}): {a['content'][:150]}..." for a in articles]
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
            return f"Done! Ticket {ticket['ticket_id']} has been raised ({ticket['category']}, priority: {ticket['priority']})."
        return "Sorry, I wasn't able to create the ticket."

    if tool_name == "check_system_status":
        rows = result.get("results", [])
        lines = [f"- {r['system']}: {r['status']} ({r['notes']})" for r in rows]
        return "Current system status:\n" + "\n".join(lines)

    return "Done."


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_after_entry(state: AgentState) -> str:
    return "fill_slot" if state.get("awaiting_field") else "classify_intent"


def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent") or "general_chat"
    return {
        "knowledge_search": "knowledge_search",
        "ticket_lookup": "ticket_lookup",
        "ticket_creation": "ticket_creation",
        "system_status": "system_status",
        "general_chat": "general_chat",
    }.get(intent, "general_chat")


def needs_clarification(state: AgentState) -> str:
    return END if state.get("awaiting_field") else "generate_response"


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
    graph.add_node("system_status", system_status_node)
    graph.add_node("general_chat", general_chat_node)
    graph.add_node("generate_response", generate_response_node)

    graph.set_entry_point("entry_router")

    graph.add_conditional_edges(
        "entry_router",
        route_after_entry,
        {"fill_slot": "fill_slot", "classify_intent": "classify_intent"},
    )

    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "knowledge_search": "knowledge_search",
            "ticket_lookup": "ticket_lookup",
            "ticket_creation": "ticket_creation",
            "system_status": "system_status",
            "general_chat": "general_chat",
        },
    )

    graph.add_conditional_edges(
        "fill_slot",
        route_by_intent,
        {
            "knowledge_search": "knowledge_search",
            "ticket_lookup": "ticket_lookup",
            "ticket_creation": "ticket_creation",
            "system_status": "system_status",
            "general_chat": "general_chat",
        },
    )

    for tool_node in ("knowledge_search", "ticket_lookup", "ticket_creation", "system_status"):
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
