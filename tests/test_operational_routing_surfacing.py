"""
test_operational_routing_surfacing.py
---------------------------------------
Step 6.5: assessment_tools.run_triage_assessment() must surface the
hospital-aware allocation (hospital_routing) to the agent/nurse as
"operational_department", without ever renaming/removing "department"
(the clinical preference).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from triageguard_agent.tools import assessment_tools


class _FakePipeline:
    def __init__(self, result):
        self._result = result

    def run(self, patient_data):
        return self._result


def _run(monkeypatch, result):
    monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: _FakePipeline(result))
    return assessment_tools.run_triage_assessment({"patient_id": "1"})


def _base_result(**overrides):
    result = {
        "department": "ICU",
        "department_reasoning": "clinical reasoning",
        "acuity_tier": 1,
        "reconciled_admission_risk": 0.9,
        "reconciled_icu_risk": 0.8,
        "branches_agree": True,
        "confidence_note": "note",
        "top_diagnoses": [],
        "red_flags": [],
        "structured_output": {},
        "rag_response": "",
        "xgb": {},
        "hospital_routing": None,
    }
    result.update(overrides)
    return result


class TestNoCalibratedPolicy:
    def test_falls_back_to_department_when_hospital_routing_is_none(self, monkeypatch):
        result = _run(monkeypatch, _base_result(department="ADMITTED_GEN", hospital_routing=None))
        assert result.success
        assert result.data["department"] == "ADMITTED_GEN"
        assert result.data["operational_department"] == "ADMITTED_GEN"
        assert result.data["resource_constraint"] is False
        assert result.data["human_review_recommended"] is False


class TestHospitalAwareAllocationSurfaced:
    def test_resource_constrained_allocation_differs_from_clinical_department(self, monkeypatch):
        hospital_routing = {
            "routing": {
                "preferred_department": "ICU",
                "allocated_department": "ADMITTED_GEN",  # ICU was full, stepped down
                "resource_constraint": True,
                "human_review_recommended": False,
            }
        }
        result = _run(monkeypatch, _base_result(department="ICU", hospital_routing=hospital_routing))
        assert result.success
        # Clinical preference is preserved, unrenamed, unremoved.
        assert result.data["department"] == "ICU"
        # Operational recommendation reflects the hospital-aware allocation.
        assert result.data["operational_department"] == "ADMITTED_GEN"
        assert result.data["resource_constraint"] is True
        assert result.data["human_review_recommended"] is False

    def test_matching_allocation_when_preferred_department_is_available(self, monkeypatch):
        hospital_routing = {
            "routing": {
                "preferred_department": "ICU",
                "allocated_department": "ICU",
                "resource_constraint": False,
                "human_review_recommended": False,
            }
        }
        result = _run(monkeypatch, _base_result(department="ICU", hospital_routing=hospital_routing))
        assert result.data["department"] == "ICU"
        assert result.data["operational_department"] == "ICU"
        assert result.data["resource_constraint"] is False


class TestInfeasibleDepartmentNeverSurfaced:
    def test_resource_conflict_never_falls_back_to_infeasible_department(self, monkeypatch):
        hospital_routing = {
            "routing": {
                "preferred_department": "ICU",
                "allocated_department": None,   # no feasible department exists at all
                "resource_constraint": True,
                "human_review_recommended": True,
            }
        }
        result = _run(monkeypatch, _base_result(department="ICU", hospital_routing=hospital_routing))
        # "department" (ICU) is known infeasible here — it must NEVER be
        # substituted in as the operational answer.
        assert result.data["department"] == "ICU"
        assert result.data["operational_department"] is None
        assert result.data["human_review_recommended"] is True
