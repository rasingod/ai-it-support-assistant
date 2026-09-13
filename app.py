"""
app.py
------
Streamlit front-end for the AI IT Support Assistant.

Run with:  streamlit run app.py

Note on rerun behavior: `st.chat_input` already triggers a full script
rerun on submit, so we deliberately do NOT call `st.rerun()` ourselves
after processing a message. Calling it again after a slow operation
(our LLM calls) can start a second script execution for the same
session before Streamlit finishes flushing the first one's output --
this is what caused the "Tried to use SessionInfo before it was
initialized" error. Instead, we compute the new state first, store it
in st.session_state, and let the normal top-to-bottom render show it
in the same run.
"""

import os

import streamlit as st
from dotenv import load_dotenv

# st.set_page_config() must be the very first Streamlit command executed --
# including before any st.secrets access below, since reading st.secrets
# when no secrets.toml exists triggers an internal Streamlit notice that
# itself counts as a "command" and would otherwise violate that rule.
st.set_page_config(page_title="IT Support Assistant", page_icon="🛠️", layout="centered")

load_dotenv()

# When deployed on Streamlit Community Cloud, secrets are set via the app's
# "Secrets" panel and exposed through st.secrets -- NOT as real environment
# variables. Our agent code reads os.environ (so it works the same way
# locally via .env), so we bridge any matching secrets into os.environ here.
# Wrapped in try/except because st.secrets can raise if no secrets.toml
# exists at all (the normal case for local development) -- in that case
# there's simply nothing to bridge, and .env (loaded above) already covers it.
try:
    for _key in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_SITE_URL", "OPENROUTER_SITE_NAME"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

from agent.graph import get_graph
from agent.state import new_state

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

if "agent_state" not in st.session_state:
    st.session_state.agent_state = new_state()

if "graph" not in st.session_state:
    st.session_state.graph = get_graph()

# ---------------------------------------------------------------------------
# Handle new input BEFORE rendering anything, so the sidebar and chat
# history below reflect the latest state in this same script run.
# ---------------------------------------------------------------------------

user_input = st.chat_input("Type your IT question or issue...")

if user_input:
    state = st.session_state.agent_state
    state["user_input"] = user_input
    state.setdefault("messages", []).append({"role": "user", "content": user_input})

    try:
        with st.spinner("Thinking..."):
            result_state = st.session_state.graph.invoke(state)
    except Exception as exc:
        result_state = dict(state)
        result_state["final_response"] = (
            "Something went wrong while processing that request. Please try again. "
            f"(Details: {exc})"
        )
        result_state["error"] = str(exc)

    answer = result_state.get("final_response") or "Sorry, I don't have a response for that."
    result_state.setdefault("messages", state["messages"])
    result_state["messages"].append({"role": "assistant", "content": answer})

    st.session_state.agent_state = result_state

# ---------------------------------------------------------------------------
# Sidebar (reflects post-turn state)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🛠️ IT Support Assistant")
    st.caption("Agentic AI demo built with LangGraph")

    _api_key = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
    if _api_key in ("", "your-openrouter-api-key-here"):
        st.warning(
            "OPENROUTER_API_KEY is missing or still has the placeholder value. "
            "Copy `.env.example` to `.env`, add your real key from "
            "[openrouter.ai/keys](https://openrouter.ai/keys), then restart the app."
        )

    state = st.session_state.agent_state
    st.subheader("Session state")
    st.write("**Employee ID:**", state.get("employee_id") or "—")
    st.write("**Verified:**", "✅" if state.get("employee_verified") else "—")
    st.write("**Awaiting:**", state.get("awaiting_field") or "—")
    ticket_draft = state.get("ticket_draft") or {}
    if ticket_draft:
        st.write("**Ticket draft:**")
        st.json(ticket_draft)

    if state.get("error"):
        st.caption(f"⚠️ Last error: {state['error']}")

    if state.get("tool_result") is not None:
        with st.expander("Last tool result (debug)"):
            st.json(state["tool_result"])

    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.agent_state = new_state()
        st.rerun()

    st.divider()
    st.caption("Try asking:")
    st.code("How do I reset my VPN password?", language=None)
    st.code("What's the status of my laptop issue? EMP1024", language=None)
    st.code("My VPN is not working, please raise a ticket.", language=None)
    st.code("Is email down right now?", language=None)

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("IT Support Assistant")
st.caption("Ask about IT help articles, check ticket status, or raise a new ticket.")

for msg in st.session_state.agent_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
