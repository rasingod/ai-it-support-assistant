import json
from pathlib import Path
import pytest
from agent import db, tools, graph, llm_client
from agent.state import new_state

@pytest.fixture
def session(tmp_path, monkeypatch):
    target = tmp_path / "tickets.json"
    target.write_text(Path(db.TICKETS_FILE).read_text())
    monkeypatch.setattr(db, "TICKETS_FILE", str(target))
    state = new_state()
    compiled = graph.build_graph()
    def chat(message, route=None):
        nonlocal state
        monkeypatch.setattr(llm_client, "classify_request", lambda *args: route or {"intent": "general_chat"})
        state = compiled.invoke({**state, "user_input": message})
        return state
    return chat


def test_scope_at_tool_boundary():
    assert tools.lookup_tickets(ticket_id="TCK-1003")["success"] is False
    assert tools.lookup_tickets("EMP1024", "TCK-1003")["results"] == []
    assert tools.lookup_tickets("EMP1003", "TCK-1003")["results"][0]["employee_id"] == "EMP1003"


def test_identity_sticky_and_missing_employee(session):
    s = session("Show TCK-1003", {"intent":"ticket_lookup", "ticket_id":"TCK-1003"})
    assert s["awaiting_field"] == "employee_id"
    s = session("EMP1024")
    assert not s["tool_result"]["results"]
    s = session("show TCK-1003 EMP1003", {"intent":"ticket_lookup", "employee_id":"EMP1003", "ticket_id":"TCK-1003"})
    assert s["employee_id"] == "EMP1024"
    assert not s["tool_result"]["results"]


@pytest.mark.parametrize("stage", ["employee_id", "category", "description", "confirmation"])
def test_cancel_each_stage(session, stage):
    before = db.get_tickets()
    s = session("raise ticket", {"intent":"ticket_creation"})
    if stage != "employee_id": s = session("EMP1024")
    if stage in {"description", "confirmation"}: s = session("Printer")
    if stage == "confirmation": s = session("Printer prints blank pages")
    assert s["awaiting_field"] == stage
    s = session("cancel")
    assert s["ticket_draft"] == {} and s["awaiting_field"] is None
    assert db.get_tickets() == before
    assert session("hello")["intent"] == "general_chat"


def test_confirm_combined_input_and_duplicate(session):
    before = len(db.get_tickets())
    session("Raise a ticket EMP1024", {"intent":"ticket_creation"})
    s = session("Printer. My printer prints blank pages.")
    assert s["awaiting_field"] == "confirmation"
    draft = dict(s["ticket_draft"])
    assert draft["category"] == "Printer" and draft["description"] == "My printer prints blank pages."
    assert len(db.get_tickets()) == before
    assert session("yes")["awaiting_field"] == "confirmation"
    s = session("confirm")
    assert len(db.get_tickets()) == before + 1
    ticket = s["tool_result"]["ticket"]
    assert ticket["description"] == draft["description"]
    session("Raise same ticket", {"intent":"ticket_creation", **draft})
    s = session("confirm")
    assert s["tool_result"]["duplicate"] and s["tool_result"]["ticket"]["ticket_id"] == ticket["ticket_id"]
    assert len(db.get_tickets()) == before + 1
    assert tools.lookup_tickets("EMP1024", ticket["ticket_id"])["results"]


def test_invalid_category_at_all_boundaries(session):
    assert not tools.create_ticket("EMP1024", "Printer. Extra prose", "Blank pages")["success"]
    with pytest.raises(ValueError): db.create_ticket("EMP1024", "cancel", "Blank pages")
    session("raise ticket EMP1024", {"intent":"ticket_creation"})
    assert session("unknown category")["awaiting_field"] == "category"


def test_status_read_only_grounded_and_query_fresh(session):
    s = session("VPN status", {"intent":"system_status", "search_query":"VPN", "category":"VPN"})
    s = session("Is email down?", {"intent":"system_status", "category":"Email"})
    assert s["ticket_draft"] == {}
    assert "not a live" in s["final_response"]
    assert "unaffected" not in s["final_response"]
    assert all(r["system"] == "Email" for r in s["tool_result"]["results"])
    assert "2026-08-27" in s["final_response"]


def test_kb_and_stale_debug_clear(session):
    s = session("VPN password", {"intent":"knowledge_search", "search_query":"VPN password"})
    assert "KB-001" in s["final_response"]
    assert session("hello")["tool_result"] is None


def test_failure_no_write(session, monkeypatch):
    before = db.get_tickets()
    s = new_state()
    def fail(*args): raise RuntimeError("private provider detail")
    monkeypatch.setattr(llm_client, "classify_request", fail)
    result = graph.build_graph().invoke({**s, "user_input":"raise a ticket"})
    assert "private provider detail" not in str(result)
    assert db.get_tickets() == before


@pytest.mark.parametrize("payload", ['[]', '{"intent":"bogus"}', '{"intent":"ticket_creation","category":{}}', 'not json'])
def test_invalid_llm_routes(payload, monkeypatch):
    from types import SimpleNamespace as NS
    call = NS(function=NS(name="route_request", arguments=payload))
    client = NS(chat=NS(completions=NS(create=lambda **kwargs: NS(choices=[NS(message=NS(tool_calls=[call]))]))))
    monkeypatch.setattr(llm_client, "get_client", lambda: client)
    with pytest.raises(ValueError): llm_client.classify_request("raise ticket")


def test_unknown_employee_and_list_after_id_lookup(session):
    s = session("show tickets EMP9999", {"intent":"ticket_lookup"})
    assert s["awaiting_field"] == "employee_id"
    s = session("EMP1024")
    assert s["employee_verified"]
    s = session("Show TCK-1001", {"intent":"ticket_lookup", "ticket_id":"TCK-1001"})
    assert len(s["tool_result"]["results"]) == 1
    s = session("list tickets", {"intent":"ticket_lookup"})
    assert s["ticket_id"] is None


def test_confirmation_does_not_use_llm(session, monkeypatch):
    session("raise printer ticket EMP1024", {"intent":"ticket_creation", "category":"Printer", "description":"Blank pages"})
    def fail(*args): raise AssertionError("Confirmation must not classify again")
    monkeypatch.setattr(llm_client, "classify_request", fail)
    state = new_state()
    state.update(employee_id="EMP1024", ticket_draft={"category":"Printer", "description":"Blank pages"},
                 awaiting_field="confirmation", pending_intent="ticket_creation", user_input="confirm")
    result = graph.build_graph().invoke(state)
    assert result["tool_result"]["success"]


def test_streamlit_creation_reset_and_lookup(session, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(llm_client, "classify_request", lambda *args: {"intent":"ticket_creation"})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    app.chat_input[0].set_value("raise ticket EMP1024").run()
    assert app.session_state.agent_state["awaiting_field"] == "category"
    app.chat_input[0].set_value("Printer. Blank pages.").run()
    assert app.session_state.agent_state["awaiting_field"] == "confirmation"
    app.chat_input[0].set_value("confirm").run()
    ticket_id = app.session_state.agent_state["tool_result"]["ticket"]["ticket_id"]
    assert not app.exception
    app.button[0].click().run()
    assert app.session_state.agent_state["employee_id"] is None
    assert app.session_state.agent_state["messages"] == []
    assert tools.lookup_tickets("EMP1024", ticket_id)["results"]

