"""
test_queue_management.py
--------------------------
Phase 9: nurse-facing queue management (drag-and-drop reorder within a
department queue, cross-department override) on HospitalSimulator /
PatientFlowManager. All operate on the existing single waiting-queue list —
no parallel per-department state.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_agent.hospital.hospital_state_service import HospitalStateService
from triageguard_agent.simulation.event_engine import EventType
from triageguard_agent.simulation.hospital_simulator import HospitalSimulator
from triageguard_agent.simulation.patient_flow import PatientStatus, SimulatedPatient


@pytest.fixture(autouse=True)
def reset_singleton():
    HospitalStateService.reset_instance()
    yield
    HospitalStateService.reset_instance()


def _triaged_patient(patient_id: str, operational_dept: str, clinical_dept: str = None) -> SimulatedPatient:
    clinical_dept = clinical_dept or operational_dept
    return SimulatedPatient(
        patient_id=patient_id,
        age=55,
        sex="M",
        chief_complaint="test",
        vitals={},
        acuity=2,
        arrival_time_min=0,
        expected_los_min=60,
        status=PatientStatus.TRIAGED,
        clinical_assessment={"department": clinical_dept},
        operational_decision={
            "clinical_department": clinical_dept,
            "operational_department": operational_dept,
            "ai_operational_department": operational_dept,
            "nurse_override": False,
            "override_reason": None,
            "available_beds_in_clinical_dept": 5,
            "operating_mode": "NORMAL",
            "lambda": 0.5,
            "capacity_warning": False,
            "confirmation_required": False,
            "recommendation_summary": "test",
        },
    )


class TestReorderWithinDepartment:
    def test_reorder_within_same_department(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p1 = _triaged_patient("P1", "ADMITTED_GEN")
        p2 = _triaged_patient("P2", "ADMITTED_GEN")
        p3 = _triaged_patient("P3", "ADMITTED_GEN")
        for p in (p1, p2, p3):
            sim.patient_flow.enqueue_patient(p)

        moved = sim.patient_flow.reorder_within_department("P3", "ADMITTED_GEN", 0)
        assert moved is True

        dept_members = [p.patient_id for p in sim.patient_flow.full_waiting_queue if p.patient_id in ("P1", "P2", "P3")]
        assert dept_members == ["P3", "P1", "P2"]

    def test_reorder_does_not_disturb_other_departments(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        icu1 = _triaged_patient("ICU1", "ICU")
        gen1 = _triaged_patient("GEN1", "ADMITTED_GEN")
        icu2 = _triaged_patient("ICU2", "ICU")
        for p in (icu1, gen1, icu2):
            sim.patient_flow.enqueue_patient(p)

        assert sim.patient_flow.reorder_within_department("ICU2", "ICU", 0) is True

        icu_order = [p.patient_id for p in sim.patient_flow.full_waiting_queue if p.patient_id.startswith("ICU")]
        assert icu_order == ["ICU2", "ICU1"]
        # GEN1 untouched / still present
        assert any(p.patient_id == "GEN1" for p in sim.patient_flow.full_waiting_queue)

    def test_reorder_rejects_patient_not_in_that_department(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_triaged_patient("P1", "ICU"))
        assert sim.patient_flow.reorder_within_department("P1", "ADMITTED_GEN", 0) is False

    def test_reorder_unknown_patient_returns_false(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        assert sim.patient_flow.reorder_within_department("GHOST", "ICU", 0) is False


class TestCrossDepartmentOverride:
    def test_override_moves_patient_and_preserves_ai_recommendation(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        patient = _triaged_patient("P1", "ADMITTED_GEN", clinical_dept="ICU")
        sim.patient_flow.enqueue_patient(patient)
        assert sim.state_service.get_state("ICU")["available"] > 0

        result = sim.override_department("P1", "ICU", reason="Family requested, bed manager approved")

        assert result["nurse_override"] is True
        assert result["operational_department"] == "ICU"
        assert result["ai_operational_department"] == "ADMITTED_GEN"
        assert result["previous_department"] == "ADMITTED_GEN"

        decision = patient.operational_decision
        # AI's original recommendation and clinical preference must survive.
        assert decision["ai_operational_department"] == "ADMITTED_GEN"
        assert decision["clinical_department"] == "ICU"
        assert decision["operational_department"] == "ICU"
        assert decision["nurse_override"] is True
        assert decision["override_reason"] == "Family requested, bed manager approved"

    def test_override_emits_staff_override_event(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_triaged_patient("P1", "ADMITTED_GEN"))

        sim.override_department("P1", "ED_OBS", reason="bed management")

        events = sim.events.get_history(event_type=EventType.STAFF_OVERRIDE)
        assert len(events) == 1
        assert events[0].patient_id == "P1"
        assert events[0].data["new_department"] == "ED_OBS"
        assert events[0].data["ai_operational_department"] == "ADMITTED_GEN"

    def test_override_rejects_closed_department(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_triaged_patient("P1", "ADMITTED_GEN"))
        sim.state_service.apply_update("ICU", {"status": "CLOSED"})

        with pytest.raises(ValueError):
            sim.override_department("P1", "ICU")
        # Never fabricated — original operational_department untouched.
        patient = sim.patient_flow.get_patient("P1")
        assert patient.operational_decision["operational_department"] == "ADMITTED_GEN"
        assert patient.operational_decision["nurse_override"] is False

    def test_override_rejects_full_department(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_triaged_patient("P1", "ADMITTED_GEN"))
        curr = sim.state_service.get_state("ICU")
        sim.state_service.apply_update("ICU", {"occupied": curr["capacity"]})

        with pytest.raises(ValueError):
            sim.override_department("P1", "ICU")

    def test_override_rejects_unknown_department(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_triaged_patient("P1", "ADMITTED_GEN"))

        with pytest.raises(ValueError):
            sim.override_department("P1", "NOT_A_REAL_DEPARTMENT")

    def test_override_rejects_unknown_patient(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        with pytest.raises(KeyError):
            sim.override_department("GHOST", "ICU")

    def test_override_rejects_non_triaged_patient(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        patient = _triaged_patient("P1", "ADMITTED_GEN")
        patient.status = PatientStatus.ARRIVED
        sim.patient_flow.enqueue_patient(patient)

        with pytest.raises(ValueError):
            sim.override_department("P1", "ICU")

    def test_overridden_department_becomes_admit_default(self):
        """The existing admit_patient() already prefers operational_decision's
        operational_department — confirms override flows through unchanged."""
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_triaged_patient("P1", "ADMITTED_GEN"))
        sim.override_department("P1", "ICU")

        result = sim.admit_patient("P1")
        assert result["department"] == "ICU"


class TestHospitalIsolation:
    def test_override_in_one_hospital_does_not_affect_another(self, tmp_path, monkeypatch):
        from triageguard_agent.hospital import hospital_registry as hr

        hr.reset_default_registry()
        test_registry = hr.HospitalRegistry(manifest_path=tmp_path / "hospitals" / "registry.json")
        monkeypatch.setattr(hr, "_default_registry", test_registry)

        departments = {
            "ICU": {"capacity": 8, "occupied": 2, "status": "OPEN"},
            "ADMITTED_GEN": {"capacity": 20, "occupied": 5, "status": "OPEN"},
            "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
        }
        test_registry.register("hosp_a", "Hospital A", config_dict={"departments": departments})
        test_registry.register("hosp_b", "Hospital B", config_dict={"departments": departments})

        sim_a = HospitalSimulator(hospital_id="hosp_a", scenario="NORMAL_DAY")
        sim_b = HospitalSimulator(hospital_id="hosp_b", scenario="NORMAL_DAY")

        sim_a.patient_flow.enqueue_patient(_triaged_patient("SAME_ID", "ADMITTED_GEN"))
        sim_b.patient_flow.enqueue_patient(_triaged_patient("SAME_ID", "ADMITTED_GEN"))

        sim_a.override_department("SAME_ID", "ICU", reason="A's decision")

        patient_a = sim_a.patient_flow.get_patient("SAME_ID")
        patient_b = sim_b.patient_flow.get_patient("SAME_ID")
        assert patient_a.operational_decision["operational_department"] == "ICU"
        assert patient_a.operational_decision["nurse_override"] is True
        # Hospital B's identically-ID'd patient must be completely unaffected.
        assert patient_b.operational_decision["operational_department"] == "ADMITTED_GEN"
        assert patient_b.operational_decision["nurse_override"] is False

        hr.reset_default_registry()


class TestQueueManagementAPI:
    def _client(self, monkeypatch):
        from triageguard_agent.tools import simulation_tools

        monkeypatch.setattr(simulation_tools, "_simulator_instances", {})
        import api_server
        from fastapi.testclient import TestClient

        return TestClient(api_server.app), simulation_tools.get_simulator(None)

    def test_override_endpoint_success_and_rejection(self, monkeypatch):
        client, sim = self._client(monkeypatch)
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_triaged_patient("API1", "ADMITTED_GEN"))

        r = client.post(
            "/api/simulation/queue/override",
            json={"patient_id": "API1", "department": "ICU", "reason": "test"},
        )
        assert r.status_code == 200
        assert r.json()["operational_department"] == "ICU"

        r2 = client.post(
            "/api/simulation/queue/override",
            json={"patient_id": "API1", "department": "NOT_A_REAL_DEPARTMENT"},
        )
        assert r2.status_code == 400

    def test_reorder_department_endpoint(self, monkeypatch):
        client, sim = self._client(monkeypatch)
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_triaged_patient("A1", "ICU"))
        sim.patient_flow.enqueue_patient(_triaged_patient("A2", "ICU"))

        r = client.post(
            "/api/simulation/queue/reorder-department",
            json={"patient_id": "A2", "department": "ICU", "new_index": 0},
        )
        assert r.status_code == 200
        assert r.json()["moved"] is True

        r2 = client.post(
            "/api/simulation/queue/reorder-department",
            json={"patient_id": "GHOST", "department": "ICU", "new_index": 0},
        )
        assert r2.status_code == 404


class TestExistingRoutingUnchanged:
    def test_triage_patient_still_populates_ai_snapshot_fields(self):
        """triage_patient() must still work exactly as before, now with the
        two new (additive) ai_operational_department/nurse_override fields
        present and correctly initialized."""
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        patient = sim.trigger_arrival(target_acuity=1)
        res = sim.triage_patient(patient)
        op = res["operational_decision"]

        assert op["operational_department"] == op["ai_operational_department"]
        assert op["nurse_override"] is False
        assert op["override_reason"] is None
        # Pre-existing keys untouched.
        assert "clinical_department" in op
        assert "capacity_warning" in op
