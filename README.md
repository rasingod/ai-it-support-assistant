# AI IT Support Assistant

An agentic AI IT-support chatbot built with **LangGraph**, **openrouter**, and **Streamlit**. It understands an employee's request, decides which tool it needs, executes that tool against local sample data, and returns a clear, grounded response — while holding onto context (like an employee ID) across multiple turns of conversation.

---

## 1. Problem Statement

Employees at a fictional organization ("FictionalCorp") raise repetitive IT requests — "how do I reset my VPN password", "what's the status of my ticket", "my laptop is overheating, please log a ticket." Handling these manually (or through a rigid keyword-matching bot) is slow and inconsistent. We need an assistant that can:

- Understand free-form natural-language requests
- Decide *which* internal system/tool is relevant (knowledge base vs. ticket system)
- Ask for missing information instead of guessing
- Take action (create a ticket) safely, without inventing data or duplicating existing tickets
- Carry conversation context across multiple turns

## 2. Solution Overview

The assistant is built as a **LangGraph** state machine with four capabilities exposed as tools. LLM calls are routed through **OpenRouter** (an OpenAI-compatible API that can serve Claude, GPT, Llama, and other models through a single key), so the project only needs one `OPENROUTER_API_KEY` regardless of which underlying model you pick.

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Searches a local IT knowledge base for how-to articles |
| `lookup_tickets` | Looks up existing tickets by employee ID or ticket ID |
| `create_ticket` | Creates a new ticket after validating required fields and checking for duplicates |
| `check_system_status` *(bonus)* | Reports current status of internal systems (VPN, Email, Wi-Fi, Printing) |

An LLM call (Claude, via a forced tool call) classifies the user's intent and extracts only the information they explicitly stated. The graph then routes to the right tool node, executes it against local JSON data, and a second LLM call phrases the tool's raw result into a friendly reply. If required information is missing, the graph asks a clarifying question and remembers the answer on the next turn instead of restarting the conversation.

## 3. Architecture Diagram

```mermaid
flowchart TD
    START([User message]) --> ENTRY[entry_router]
    ENTRY -->|awaiting_field set| FILL[fill_slot<br/>fills remembered slot, no LLM call]
    ENTRY -->|no pending field| CLASSIFY[classify_intent<br/>LLM tool-call: intent + slot extraction]

    FILL --> ROUTE{route_by_intent}
    CLASSIFY --> ROUTE

    ROUTE -->|knowledge_search| KB[knowledge_search_node<br/>Tool: search_knowledge_base]
    ROUTE -->|ticket_lookup| LOOKUP[ticket_lookup_node<br/>Tool: lookup_tickets]
    ROUTE -->|ticket_creation| CREATE[ticket_creation_node<br/>Tool: verify_employee + create_ticket]
    ROUTE -->|system_status| STATUS[system_status_node<br/>Tool: check_system_status]
    ROUTE -->|general_chat| CHAT[general_chat_node<br/>LLM free-text reply]

    KB --> CLAR1{needs_clarification?}
    LOOKUP --> CLAR2{needs_clarification?}
    CREATE --> CLAR3{needs_clarification?}
    STATUS --> CLAR4{needs_clarification?}

    CLAR1 -->|yes: ask user| END1([END - clarifying question])
    CLAR1 -->|no| GEN[generate_response_node<br/>LLM phrases tool result]
    CLAR2 -->|yes| END1
    CLAR2 -->|no| GEN
    CLAR3 -->|yes| END1
    CLAR3 -->|no| GEN
    CLAR4 -->|yes| END1
    CLAR4 -->|no| GEN

    GEN --> END2([END - final answer])
    CHAT --> END2
```

**Key design point:** clarifying questions (e.g. "what's your employee ID?") are answered by `fill_slot` on the *next* turn without re-running the LLM classifier — the state (`awaiting_field`, `pending_intent`, `ticket_draft`) persists in Streamlit's session and is fed back into the graph each turn, which is how multi-turn memory is achieved without a database-backed checkpointer.

## 4. Technology Stack

- **Python 3.10+**
- **LangGraph** — workflow orchestration (state graph, conditional routing)
- **OpenRouter API** (OpenAI-compatible `chat.completions`) — intent classification (via forced tool/function-calling) and response generation; defaults to `anthropic/claude-3.5-sonnet` but any OpenRouter model slug works
- **Streamlit** — chat UI
- **Local JSON files** — employees, tickets, knowledge base, system status (no external DB needed)

## 5. Project Structure

```
ai-it-support-assistant/
├── app.py                     # Streamlit chat UI
├── agent/
│   ├── __init__.py
│   ├── state.py                # AgentState TypedDict (shared graph state)
│   ├── graph.py                 # LangGraph nodes, conditional edges, graph assembly
│   ├── tools.py                 # The 3 required tools + verify_employee + bonus status tool
│   ├── llm_client.py            # OpenRouter API wrapper (tool-forced classification + text generation)
│   └── db.py                    # Local JSON data access layer
├── data/
│   ├── employees.json
│   ├── tickets.json
│   ├── knowledge_base.json
│   └── system_status.json
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 6. Setup Instructions

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ai-it-support-assistant

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# then edit .env and paste your OPENROUTER_API_KEY
```

> **Optional:** locally, Streamlit may print a harmless "No secrets found" console
> message the first time the app checks for Streamlit Cloud-style secrets (used only
> when deployed). It doesn't affect functionality -- `.env` above is all you need
> locally. To silence the message, copy `.streamlit/secrets.toml.example` to
> `.streamlit/secrets.toml` (even empty is fine); `.gitignore` already excludes the
> real file from version control.

## 7. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key. Get one at https://openrouter.ai/keys |
| `OPENROUTER_MODEL` | No | Overrides the default model (`anthropic/claude-3.5-sonnet`). Browse available slugs at https://openrouter.ai/models — pick any model that supports tool/function calling. |
| `OPENROUTER_SITE_URL` | No | Optional; sent as the `HTTP-Referer` header for OpenRouter analytics/rate-limit attribution. |
| `OPENROUTER_SITE_NAME` | No | Optional; sent as the `X-Title` header, shown in your OpenRouter dashboard. |

## 8. How to Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

Use the sidebar's **🔄 Reset conversation** button to clear state and start over, and expand **Last tool result (debug)** to see the raw tool output behind each answer.

## 9. Sample Inputs & Outputs

**Knowledge search**
> User: `How do I reset my VPN password?`
> Assistant: *Retrieves KB-001 and explains the 5-step reset process, ending with what to do if the reset email doesn't arrive.*

**Ticket lookup (multi-turn, matches the assignment's example flow)**
> User: `I have a VPN issue.`
> Assistant: `What's your employee ID?`
> User: `EMP1024`
> Assistant: *Looks up EMP1024, confirms the profile, and reports existing tickets.*

**Ticket creation with validation**
> User: `My VPN is not working. Please raise a ticket.`
> Assistant: `What's your employee ID?` → `EMP1002` → `What category best describes the issue?` → `VPN` → `Could you briefly describe the issue?` → *user describes it* →
> Assistant: `Done! Ticket TCK-1005 has been raised (VPN, priority: Medium).`

**Duplicate prevention**
> User (as EMP1024, already has an open "Laptop" ticket): `My laptop is still overheating, raise another ticket.`
> Assistant: `An open ticket (TCK-1001) already exists for 'Laptop'. I haven't created a duplicate — would you like me to check its status instead?`

**Unknown employee ID (graceful validation)**
> User: `What's the status of my ticket? My ID is EMP9999.`
> Assistant: `I couldn't find an employee with ID 'EMP9999'. Could you double-check your employee ID?`

## 10. Key Design Decisions

- **Structured tool-calling for routing.** Intent classification forces the model to call a single `route_request` function with a strict JSON schema, rather than free-text parsing — this is far more reliable than asking the LLM to "reply in JSON" and demonstrates genuine function-calling rather than prompt-based guessing.
- **OpenRouter instead of a single provider's SDK.** The app talks to OpenRouter's OpenAI-compatible `chat.completions` endpoint via the `openai` Python package pointed at a different `base_url`. This means the model is swappable via one environment variable (`OPENROUTER_MODEL`) without touching any code — useful for comparing providers or falling back to a cheaper/faster model.
- **The LLM never invents ticket data.** The classifier is explicitly instructed to return `null` for anything the user didn't state, and `ticket_creation_node` re-validates required fields in code (not via the LLM) before ever calling `create_ticket`.
- **State-driven slot filling instead of giant prompts.** Rather than stuffing the whole conversation into every LLM call and hoping it remembers the employee ID, the graph tracks `awaiting_field` / `pending_intent` / `ticket_draft` explicitly in state, so `fill_slot` can resume a partially-completed ticket without even calling the LLM.
- **Duplicate-ticket prevention lives in the tool layer**, not the prompt, so it can't be bypassed by a differently-phrased request — `create_ticket()` always checks for an existing open ticket in the same category before writing a new one.
- **Two-stage LLM usage.** One call decides *what* to do (classification), a second call decides *how to phrase* the already-retrieved result. This keeps the "brain" (decision-making) and the "voice" (user-facing phrasing) separate and makes each easier to reason about and test independently of the other.
- **Deterministic fallback responses.** `_fallback_response()` in `graph.py` provides a template-based answer if the phrasing LLM call fails, so a transient API error degrades the experience rather than crashing it.
- **JSON files instead of a real database.** Keeps the project runnable anywhere with zero setup, per the assignment's "achievable on a local machine" guidance. `db.py` isolates all file I/O, so swapping in SQLite later only requires changing that one module.

## 11. Limitations

- Knowledge-base search uses simple keyword-overlap scoring, not embeddings/semantic search — good enough for a small local KB, but won't scale to a large, diverse article set.
- No authentication — employee ID is trusted based on lookup only, mirroring a low-stakes internal tool, not a production identity system.
- `st.session_state` (single browser session) is the only persistence layer; the app doesn't yet use LangGraph's built-in checkpointer, so state won't survive a server restart or work across multiple concurrent users.
- Ticket priority defaults to "Medium" unless the user explicitly states urgency; the assistant does not infer priority from issue severity.
- Intent classification quality depends on the underlying LLM; ambiguous multi-intent messages (e.g. "check my ticket and also raise a new one") are handled as a single intent per turn.

## 12. Deploying for Free (Shareable Link)

The fastest free option is **Streamlit Community Cloud** (the official host, made by the Streamlit team):

1. Push this project to a **public** (or private, if your account supports it) GitHub repository. `.gitignore` already excludes `.env`, so your API key won't be committed.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick the repository/branch, and set the main file path to `app.py`.
4. Before deploying, open **Advanced settings → Secrets** and paste:
   ```toml
   OPENROUTER_API_KEY = "sk-or-your-real-key"
   OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"
   ```
5. Click **Deploy**. You'll get a link like `https://your-app-name.streamlit.app` to share with evaluators.

Streamlit Cloud exposes secrets via `st.secrets`, not real environment variables, so `app.py` bridges any matching secret into `os.environ` at startup — the rest of the code (which reads `os.environ`) needs no changes and behaves identically locally and when deployed.

**Alternatives** if you outgrow the free tier or want a different host: Hugging Face Spaces (select the "Streamlit" SDK when creating a Space) or Render's free web service tier both work with this project unmodified, using the same `requirements.txt` and the same environment variables set through each platform's own secrets/environment settings.

## 13. Technical Skills Demonstrated

Python · LangGraph (State, Nodes, Edges, Conditional Routing) · Agentic AI · Tool/Function Calling · State Management · Local JSON Data Integration · Prompt Engineering · Streamlit · Error Handling · Modular Architecture
