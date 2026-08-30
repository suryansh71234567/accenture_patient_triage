"""
test_chat_hospital_scoping.py
--------------------------------
Phase 9: chat must be hospital-scoped, not silently default-hospital.

Covers two links in the chain:
1. AgentRuntime.run_tool() auto-fills hospital_id from AgentState.hospital_id
   for any tool whose schema accepts it, when the caller (LLM) didn't supply
   one — this is what makes every tool call inside a chat turn hospital-aware
   without relying on the LLM to remember to pass hospital_id itself.
2. POST /api/chat stamps the request's hospital_id onto the session's
   AgentState before running the turn, so switching the hospital selector
   takes effect on the very next message.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_agent.runtime.agent_runtime import AgentRuntime
from triageguard_agent.state.agent_state import AgentState
from triageguard_agent.hospital.hospital_state_service import HospitalStateService


@pytest.fixture(autouse=True)
def reset_hospital_singleton():
    HospitalStateService.reset_instance()
    yield
    HospitalStateService.reset_instance()


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Two isolated hospitals with different department sets — never touches
    the real repo's data/hospitals/."""
    from triageguard_agent.hospital import hospital_registry as hr

    hr.reset_default_registry()
    test_registry = hr.HospitalRegistry(manifest_path=tmp_path / "hospitals" / "registry.json")
    monkeypatch.setattr(hr, "_default_registry", test_registry)

    test_registry.register(
        "hosp_b", "Hospital B",
        config_dict={"departments": {
            "ICU": {"capacity": 4, "occupied": 1, "status": "OPEN"},
            "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
        }},
    )
    yield test_registry
    hr.reset_default_registry()


class TestRunToolAutoFillsHospitalId:
    def test_hospital_scoped_tool_resolves_selected_hospital_without_explicit_kwarg(self, sandbox):
        runtime = AgentRuntime(auto_register=True)
        state = AgentState(session_id="s1", hospital_id="hosp_b")

        result = runtime.run_tool("get_hospital_state", {}, agent_state=state)

        assert result.success
        depts = result.data["departments"]
        assert "ICU" in depts and depts["ICU"]["capacity"] == 4
        # Confirms it did NOT silently resolve to the default hospital's config.
        assert "CICU" not in depts

    def test_no_hospital_id_on_session_falls_back_to_default_unchanged(self, sandbox):
        runtime = AgentRuntime(auto_register=True)
        state = AgentState(session_id="s2")  # hospital_id=None

        result = runtime.run_tool("get_hospital_state", {}, agent_state=state)

        assert result.success
        # Default hospital's config (untouched — pre-existing behavior).
        assert "ICU" in result.data["departments"]

    def test_explicit_kwarg_hospital_id_is_never_overridden(self, sandbox):
        runtime = AgentRuntime(auto_register=True)
        state = AgentState(session_id="s3", hospital_id="hosp_b")

        # Explicitly asks for the default hospital's state — session's
        # selected hospital must not silently override an explicit kwarg.
        result = runtime.run_tool("get_hospital_state", {"hospital_id": "default"}, agent_state=state)

        assert result.success
        assert "CICU" in result.data["departments"]  # default hospital has CICU, hosp_b doesn't


class TestChatEndpointStampsHospitalId:
    def test_chat_stamps_hospital_id_onto_session_before_running_turn(self, monkeypatch, sandbox):
        import api_server
        from fastapi.testclient import TestClient
        from triageguard_agent.schemas.agent_response import AgentResponse

        captured = {}

        def fake_process_turn(message, state):
            captured["hospital_id"] = state.hospital_id
            return AgentResponse.error_response("stubbed — not exercising the real LLM in this test")

        monkeypatch.setattr(api_server.RUNTIME, "process_turn", fake_process_turn)

        client = TestClient(api_server.app)
        session = client.post("/api/session", json={"role": "nurse"}).json()
        r = client.post(
            "/api/chat",
            json={"session_id": session["session_id"], "message": "hi", "hospital_id": "hosp_b"},
        )
        assert r.status_code == 200
        assert captured["hospital_id"] == "hosp_b"

    def test_chat_without_hospital_id_leaves_session_hospital_id_unset(self, monkeypatch, sandbox):
        import api_server
        from fastapi.testclient import TestClient
        from triageguard_agent.schemas.agent_response import AgentResponse

        captured = {}

        def fake_process_turn(message, state):
            captured["hospital_id"] = state.hospital_id
            return AgentResponse.error_response("stubbed")

        monkeypatch.setattr(api_server.RUNTIME, "process_turn", fake_process_turn)

        client = TestClient(api_server.app)
        session = client.post("/api/session", json={"role": "nurse"}).json()
        client.post("/api/chat", json={"session_id": session["session_id"], "message": "hi"})
        assert captured["hospital_id"] is None
