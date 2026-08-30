"""
test_simulator_policy_routing.py
----------------------------------
Phase 5: HospitalSimulator.triage_patient() must consume a hospital's
calibrated Bayesian/RL routing decision (operational_department /
resource_constraint, surfaced via run_triage_assessment's "policy_applied"
flag) instead of silently discarding it and recomputing its own cruder,
single-threshold bed-count check. Without this, the live simulated-ED-queue
flow (LiveHospital) never actually exercises per-hospital-calibrated routing
behavior — only raw bed counts, identical across any two hospitals with the
same occupancy regardless of how differently their nurses calibrated them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_agent.hospital.hospital_state_service import HospitalStateService
from triageguard_agent.simulation.hospital_simulator import HospitalSimulator
from triageguard_agent.simulation.patient_flow import PatientStatus, SimulatedPatient
from triageguard_agent.tools import assessment_tools


@pytest.fixture(autouse=True)
def reset_singleton():
    HospitalStateService.reset_instance()
    yield
    HospitalStateService.reset_instance()


def _patient(patient_id: str = "P1", acuity: int = 1) -> SimulatedPatient:
    return SimulatedPatient(
        patient_id=patient_id,
        age=60,
        sex="M",
        chief_complaint="chest pain",
        vitals={"hr": 130, "spo2": 90},
        acuity=acuity,
        arrival_time_min=0,
        expected_los_min=60,
        status=PatientStatus.ARRIVED,
    )


def _fake_pipeline(hospital_routing):
    class _Fake:
        def run(self, patient_data):
            return {
                "department": "ICU",
                "department_reasoning": "high risk",
                "acuity_tier": 1,
                "reconciled_admission_risk": 0.9,
                "reconciled_icu_risk": 0.85,
                "branches_agree": True,
                "confidence_note": "note",
                "top_diagnoses": [],
                "red_flags": [],
                "structured_output": {},
                "rag_response": "",
                "xgb": {},
                "hospital_routing": hospital_routing,
            }

    return _Fake()


class TestSimulatorConsumesCalibratedPolicy:
    def test_policy_decision_overrides_crude_threshold_check(self, monkeypatch):
        """ICU has open beds (a bed-count-only check would say 'direct
        admission'), but this hospital's calibrated policy says ICU is
        resource-constrained and steps down to ADMITTED_GEN. The simulator
        must trust the policy's decision, not silently recompute its own —
        proving hospital-specific CALIBRATED routing behavior, not just raw
        bed math, actually drives the live queue."""
        hospital_routing = {
            "routing": {
                "allocated_department": "ADMITTED_GEN",
                "resource_constraint": True,
                "human_review_recommended": False,
            }
        }
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: _fake_pipeline(hospital_routing))

        sim = HospitalSimulator(scenario="NORMAL_DAY")
        assert sim.state_service.get_state("ICU")["available"] > 0  # crude check would allocate ICU directly

        res = sim.triage_patient(_patient())
        op = res["operational_decision"]

        assert op["clinical_department"] == "ICU"
        assert op["operational_department"] == "ADMITTED_GEN"
        assert op["capacity_warning"] is True
        assert op["confirmation_required"] is True

    def test_policy_direct_allocation_still_surfaced(self, monkeypatch):
        hospital_routing = {
            "routing": {
                "allocated_department": "ICU",
                "resource_constraint": False,
                "human_review_recommended": False,
            }
        }
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: _fake_pipeline(hospital_routing))

        sim = HospitalSimulator(scenario="NORMAL_DAY")
        res = sim.triage_patient(_patient())
        op = res["operational_decision"]

        assert op["operational_department"] == "ICU"
        assert op["capacity_warning"] is False
        assert op["confirmation_required"] is False

    def test_no_feasible_department_never_fabricated_from_policy(self, monkeypatch):
        """Policy found no feasible department at all (allocated_department is
        None — a genuine resource conflict). Must never surface that None as
        the operational_department (breaks the existing API/UI contract);
        falls through to the simulator's own bed-count safety net instead."""
        hospital_routing = {
            "routing": {
                "allocated_department": None,
                "resource_constraint": True,
                "human_review_recommended": True,
            }
        }
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: _fake_pipeline(hospital_routing))

        sim = HospitalSimulator(scenario="NORMAL_DAY")
        res = sim.triage_patient(_patient())
        op = res["operational_decision"]

        assert op["operational_department"] is not None
        assert op["clinical_department"] == "ICU"

    def test_no_calibrated_policy_keeps_legacy_threshold_behavior(self, monkeypatch):
        """hospital_routing=None (no calibrated policy for this hospital) must
        behave exactly as before this change: the simulator's own bed-count
        threshold check, unaffected by the new policy-consuming branch."""
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: _fake_pipeline(None))

        sim = HospitalSimulator(scenario="RESOURCE_CONSTRAINED")
        svc = sim.state_service
        for dept in ["ICU", "CICU", "ADMITTED_GEN"]:
            curr = svc.get_state(dept)
            if curr:
                svc.apply_update(dept, {"occupied": curr["capacity"]})

        res = sim.triage_patient(_patient())
        op = res["operational_decision"]

        assert op["capacity_warning"] is True
        assert op["operational_department"] == "ED_OBS"
