"""
llm_client.py
-------------
Thin wrapper around OpenRouter's OpenAI-compatible chat completions API
(https://openrouter.ai/api/v1). OpenRouter lets you call many different
models (including Claude, GPT, Llama, etc.) through a single API key,
so this project can run with just an OPENROUTER_API_KEY instead of a
provider-specific key.

Two capabilities are exposed:
  1. classify_request()  -> forces the model to call a single
     structured "route_request" tool, so we get reliable JSON back
     for intent + slot extraction (this is the GenAI "decide which
     tool is needed / extract parameters" step of the assignment).
  2. generate_reply()    -> a normal free-text completion, used to
     phrase the final, user-friendly answer from retrieved data.

The model is NEVER allowed to invent ticket data: classify_request
only ever returns what the user actually said (or null), and the
graph is responsible for treating nulls as "missing information"
rather than asking the LLM to guess.
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Any model slug available on https://openrouter.ai/models works here.
# Defaults to Claude via OpenRouter; override with OPENROUTER_MODEL.
MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

# Optional but recommended by OpenRouter for analytics/rate-limit attribution.
# https://openrouter.ai/docs#requests
SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "")
SITE_NAME = os.environ.get("OPENROUTER_SITE_NAME", "AI IT Support Assistant")

_client: Optional[OpenAI] = None


# Values that mean "not actually configured" -- catches the common case of
# copying .env.example to .env without editing in a real key.
_PLACEHOLDER_KEYS = {"", "your-openrouter-api-key-here"}


def get_client() -> OpenAI:
    global _client
    if _client is None:
        # Strip whitespace and any accidental surrounding quotes (a common
        # .env authoring mistake: OPENROUTER_API_KEY="sk-or-..." keeps the
        # literal quote characters if a parser doesn't strip them).
        api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip().strip('"').strip("'")
        if api_key in _PLACEHOLDER_KEYS:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing or still has the placeholder value from "
                ".env.example. Get a real key from https://openrouter.ai/keys, put it in "
                ".env (local) or your deployment's Secrets panel (Streamlit Cloud), then "
                "restart the app -- editing .env while the app is already running has no "
                "effect until it's restarted, since the key is only read once at startup."
            )
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _client


def _extra_headers() -> Dict[str, str]:
    headers = {}
    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    if SITE_NAME:
        headers["X-Title"] = SITE_NAME
    return headers


# OpenAI/OpenRouter "tools" format (function-calling schema).
ROUTE_REQUEST_TOOL = {
    "type": "function",
    "function": {
        "name": "route_request",
        "description": (
            "Classify an IT support request and extract only the information the "
            "user explicitly stated. Never guess or invent a value -- use null for "
            "anything not clearly provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "knowledge_search",
                        "ticket_lookup",
                        "ticket_creation",
                        "escalation",
                        "system_status",
                        "general_chat",
                    ],
                    "description": (
                        "knowledge_search: user wants to know HOW to do/fix something. "
                        "ticket_lookup: user wants the STATUS of one existing ticket, OR wants to "
                        "list/view/see ALL of their tickets (e.g. 'list my tickets', 'what tickets do "
                        "I have', 'show my open tickets', 'any tickets under my name'). "
                        "ticket_creation: user wants to REPORT a new issue / raise a ticket. "
                        "escalation: user wants to ESCALATE an existing ticket -- raise its priority / "
                        "get it urgent attention (e.g. 'escalate it', 'escalate ticket TCK-1005', 'this "
                        "needs urgent attention', 'please bump the priority on my ticket'). "
                        "system_status: user is asking if a system/service is currently down. "
                        "general_chat: greetings, thanks, or anything not IT-support related."
                    ),
                },
                "employee_id": {
                    "type": ["string", "null"],
                    "description": "Employee ID exactly as stated by the user, e.g. 'EMP1024'. Null if not mentioned.",
                },
                "ticket_id": {
                    "type": ["string", "null"],
                    "description": "Ticket ID exactly as stated by the user, e.g. 'TCK-1002'. Null if not mentioned.",
                },
                "category": {
                    "type": ["string", "null"],
                    "description": "ONLY for ticket_creation intent -- short issue category, e.g. 'VPN', 'Laptop', 'Email', 'Printer'. Null for every other intent, including system_status and knowledge_search, even if the message mentions a similar-sounding system.",
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "The user's own description of the problem, in their words. Null if not a ticket-creation request.",
                },
                "priority": {
                    "type": ["string", "null"],
                    "enum": ["Low", "Medium", "High", None],
                    "description": "Only set if the user explicitly stated urgency. Otherwise null.",
                },
                "search_query": {
                    "type": ["string", "null"],
                    "description": "For knowledge_search intent: the concise topic/question to search the knowledge base for.",
                },
                "escalation_reason": {
                    "type": ["string", "null"],
                    "description": "For escalation intent: the user's own reason for escalating (e.g. 'unresolved for too long', 'urgent business impact'). Null if not stated.",
                },
            },
            "required": ["intent"],
        },
    },
}


def classify_request(user_input: str, conversation_context: str = "") -> Dict[str, Any]:
    """Calls the LLM with a forced tool_choice to get structured routing output."""
    client = get_client()

    system_prompt = (
        "You are the routing brain of an IT Support Assistant. Read the latest "
        "user message (and brief context if given) and call the route_request "
        "tool with your classification. Only extract information the user "
        "actually stated -- never fabricate employee IDs, ticket IDs, or "
        "descriptions."
    )

    user_message = user_input
    if conversation_context:
        user_message = f"Conversation so far:\n{conversation_context}\n\nLatest user message:\n{user_input}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=500,
        extra_headers=_extra_headers(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        tools=[ROUTE_REQUEST_TOOL],
        tool_choice={"type": "function", "function": {"name": "route_request"}},
    )

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            if call.function.name == "route_request":
                try:
                    return json.loads(call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    break

    # Should not happen given forced tool_choice, but fail safe.
    return {"intent": "general_chat"}


def generate_reply(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    """Plain text generation used to phrase the final answer to the user."""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=600,
        extra_headers=_extra_headers(),
        messages=[{"role": "system", "content": system_prompt}, *messages],
    )
    return (response.choices[0].message.content or "").strip()
