"""
test_resource_check_policy_aware.py
-------------------------------------
Phase 6: api_server._resource_check() (the file-based patient "resource
check" display hint) must prefer the hospital-calibrated routing policy's
already-computed operational_department/resource_constraint
(assessment["policy_applied"], set by run_triage_assessment) over
recomputing its own cruder ICU/CICU/ADMITTED_GEN-only threshold check —
mirroring the same fix Phase 5 made for HospitalSimulator.triage_patient().
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import api_server
from triageguard_agent.state.agent_state import AgentState


class _FakeToolResult:
    def __init__(self, data):
        self.success = True
        self.data = data
        self.error = None


def _hospital_state(capacity: int, occupied: int) -> _FakeToolResult:
    return _FakeToolResult({"state": {"capacity": capacity, "occupied": occupied, "available": capacity - occupied}})


class TestResourceCheckUsesCalibratedPolicy:
    def test_policy_driven_constraint_overrides_threshold_check(self, monkeypatch):
        """ICU has open beds (a bed-count-only check would say 'not
        constrained'), but the calibrated policy already found ICU
        resource-constrained and stepped down to ADMITTED_GEN. The file-based
        patient resource check must reflect the policy's decision, not
        recompute its own from raw bed counts."""
        monkeypatch.setattr(
            api_server.RUNTIME, "run_tool",
            lambda name, kwargs, agent_state=None: _hospital_state(capacity=10, occupied=2),
        )
        assessment = {
            "department": "ICU",
            "policy_applied": True,
            "operational_department": "ADMITTED_GEN",
            "resource_constraint": True,
            "human_review_recommended": False,
        }
        check = api_server._resource_check(assessment, AgentState(session_id="s1"))

        assert check["preferred_department"] == "ICU"
        assert check["allocated_department"] == "ADMITTED_GEN"
        assert check["resource_constrained"] is True

    def test_policy_direct_allocation_not_flagged_constrained(self, monkeypatch):
        monkeypatch.setattr(
            api_server.RUNTIME, "run_tool",
            lambda name, kwargs, agent_state=None: _hospital_state(capacity=10, occupied=2),
        )
        assessment = {
            "department": "ICU",
            "policy_applied": True,
            "operational_department": "ICU",
            "resource_constraint": False,
            "human_review_recommended": False,
        }
        check = api_server._resource_check(assessment, AgentState(session_id="s1"))

        assert check["allocated_department"] == "ICU"
        assert check["resource_constrained"] is False

    def test_no_calibrated_policy_keeps_legacy_threshold_behavior(self, monkeypatch):
        """policy_applied=False (no calibrated policy for this hospital) must
        behave exactly as before this change: the ICU/CICU/ADMITTED_GEN-only
        threshold check, driven purely by bed counts."""
        monkeypatch.setattr(
            api_server.RUNTIME, "run_tool",
            lambda name, kwargs, agent_state=None: _hospital_state(capacity=10, occupied=10),
        )
        assessment = {
            "department": "ICU",
            "policy_applied": False,
            "operational_department": "ICU",
            "resource_constraint": False,
            "human_review_recommended": False,
        }
        check = api_server._resource_check(assessment, AgentState(session_id="s1"))

        assert check["resource_constrained"] is True
        assert check["allocated_department"] == "ED_OBS"

    def test_policy_no_feasible_department_falls_back_to_threshold_check(self, monkeypatch):
        """Policy found no feasible department at all (operational_department
        is None) — a genuine resource conflict. Must never surface that None
        as allocated_department; falls back to the threshold check's own
        (non-None) answer instead."""
        monkeypatch.setattr(
            api_server.RUNTIME, "run_tool",
            lambda name, kwargs, agent_state=None: _hospital_state(capacity=10, occupied=3),
        )
        assessment = {
            "department": "ICU",
            "policy_applied": True,
            "operational_department": None,
            "resource_constraint": True,
            "human_review_recommended": True,
        }
        check = api_server._resource_check(assessment, AgentState(session_id="s1"))

        assert check["allocated_department"] is not None
