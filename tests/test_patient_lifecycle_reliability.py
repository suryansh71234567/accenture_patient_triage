"""
test_patient_lifecycle_reliability.py
--------------------------------------
Regression tests for the patient-lifecycle reliability phase:

1.  Duplicate patient id (demo + real) rejected on manual registration.
2.  Scenario switch preserves real patients, replaces only demo patients.
3.  Vitals/observation updates on an ACTIVE SimulatedPatient.
4.  One canonical triage/re-triage operation (REST + agent tool identical).
5.  Requeue after re-triage changes the recommended department.
6.  Admitted patients cannot be re-triaged.
7.  Waiting queue / admitted cohort are mutually exclusive (waiting_count
    accuracy).
8.  Dashboard vs Live Hospital "waiting" population consistency
    (untriaged_count).
9.  Historical-only (file-based) patient can still be brought into the live
    simulation as a returning patient.
10. Hospital isolation for all of the above.

Reuses existing fixtures/patterns from test_queue_management.py and
test_multi_hospital_simulation.py (HospitalStateService.reset_instance(),
the sandboxed HospitalRegistry pattern, TestClient against api_server.app).
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
from triageguard_agent.tools import simulation_tools


@pytest.fixture(autouse=True)
def reset_singleton():
    HospitalStateService.reset_instance()
    yield
    HospitalStateService.reset_instance()


def _triaged_patient(patient_id: str, operational_dept: str, clinical_dept: str = None,
                      nurse_override: bool = False, override_reason: str = None) -> SimulatedPatient:
    clinical_dept = clinical_dept or operational_dept
    return SimulatedPatient(
        patient_id=patient_id,
        age=55,
        sex="M",
        chief_complaint="test",
        vitals={"hr": 80, "rr": 16, "spo2": 98, "sbp": 120, "dbp": 80, "temp": 37.0, "pain": 0},
        acuity=3,
        arrival_time_min=0,
        expected_los_min=60,
        status=PatientStatus.TRIAGED,
        clinical_assessment={"department": clinical_dept},
        operational_decision={
            "clinical_department": clinical_dept,
            "operational_department": operational_dept,
            "ai_operational_department": operational_dept if not nurse_override else clinical_dept,
            "nurse_override": nurse_override,
            "override_reason": override_reason,
            "available_beds_in_clinical_dept": 5,
            "operating_mode": "NORMAL",
            "lambda": 0.5,
            "capacity_warning": False,
            "confirmation_required": False,
            "recommendation_summary": "test",
        },
    )


def _arrived_patient(patient_id: str, acuity: int = 3, cardiac: bool = False) -> SimulatedPatient:
    return SimulatedPatient(
        patient_id=patient_id,
        age=50,
        sex="F",
        chief_complaint="test complaint",
        vitals={"hr": 90, "rr": 18, "spo2": 97, "sbp": 130, "dbp": 85, "temp": 37.2, "pain": 2},
        acuity=acuity,
        arrival_time_min=0,
        expected_los_min=60,
        status=PatientStatus.ARRIVED,
        metadata={"cardiac_hint": cardiac},
    )


_CLINICAL_OUTPUT_TEMPLATE = {
    "acuity_tier": 3,
    "reconciled_admission_risk": 0.5,
    "reconciled_icu_risk": 0.1,
    "branches_agree": True,
    "confidence_note": "test",
    "top_diagnoses": ["test dx"],
    "red_flags": [],
}


def _clinical_output(department: str) -> dict:
    return {**_CLINICAL_OUTPUT_TEMPLATE, "department": department, "department_reasoning": f"routed to {department}"}


# ---------------------------------------------------------------------------
# 1. New patient full lifecycle
# ---------------------------------------------------------------------------

class TestNewPatientFullLifecycle:
    def test_register_triage_queue_admit(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()

        patient = _arrived_patient("NEWPT-1", acuity=2, cardiac=True)
        sim.patient_flow.enqueue_patient(patient)
        assert patient.status == PatientStatus.ARRIVED
        assert sim.patient_flow.untriaged_queue == [patient]

        res = sim.triage_patient(patient)
        assert patient.status == PatientStatus.TRIAGED
        op = res["operational_decision"]
        assert op["retriage"] is False
        assert op["nurse_override"] is False
        assert op["operational_department"] == op["ai_operational_department"]
        # Automatically in its recommended department's queue (live-computed).
        from triageguard_agent.simulation.patient_flow import department_of
        assert department_of(patient) == op["operational_department"]

        admit_res = sim.admit_patient("NEWPT-1")
        assert patient.status == PatientStatus.IN_TREATMENT
        assert admit_res["department"] == op["operational_department"]
        # No longer in the waiting/triage queue.
        assert sim.patient_flow.get_patient("NEWPT-1") is patient
        assert all(p.patient_id != "NEWPT-1" for p in sim.patient_flow.full_waiting_queue)


# ---------------------------------------------------------------------------
# 2 & 3. Duplicate patient id
# ---------------------------------------------------------------------------

class TestDuplicatePatientId:
    def _client(self, monkeypatch):
        monkeypatch.setattr(simulation_tools, "_simulator_instances", {})
        import api_server
        from fastapi.testclient import TestClient
        return TestClient(api_server.app), simulation_tools.get_simulator(None)

    def test_duplicate_against_real_patient_rejected(self, monkeypatch):
        client, sim = self._client(monkeypatch)
        sim.patient_flow.clear()

        payload = {"patient_id": "DUPTEST-1", "chief_complaint": "chest pain", "age": 40, "sex": "M", "acuity": 3}
        r1 = client.post("/api/simulation/manual-arrival", json=payload)
        assert r1.status_code == 200

        r2 = client.post("/api/simulation/manual-arrival", json=payload)
        assert r2.status_code == 409
        # Still exactly one patient with this id, not two.
        matches = [p for p in sim.patient_flow.full_waiting_queue if p.patient_id == "DUPTEST-1"]
        assert len(matches) == 1

    def test_duplicate_against_demo_patient_rejected(self, monkeypatch):
        client, sim = self._client(monkeypatch)
        sim.load_scenario("NORMAL_DAY")  # (re-)populate default-hospital demo pool
        demo_id = next(iter(sim._presimulated_patient_ids), None)
        assert demo_id is not None, "expected at least one presimulated patient for NORMAL_DAY"

        r = client.post(
            "/api/simulation/manual-arrival",
            json={"patient_id": demo_id, "chief_complaint": "chest pain", "age": 40, "sex": "M", "acuity": 3},
        )
        assert r.status_code == 409
        matches = [p for p in sim.patient_flow.full_waiting_queue if p.patient_id == demo_id]
        matches += [p for p in sim.patient_flow._admitted_cohort.values() if p.patient_id == demo_id]
        assert len(matches) == 1

    def test_historical_only_patient_allowed(self, monkeypatch):
        """A patient_id that exists only in data/patients/*.json (not active
        in this hospital's simulation) is a normal returning-patient
        registration, not a conflict."""
        client, sim = self._client(monkeypatch)
        sim.patient_flow.clear()

        r = client.post(
            "/api/simulation/manual-arrival",
            json={"patient_id": "52", "chief_complaint": "follow-up", "age": 60, "sex": "M", "acuity": 3},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["has_history"] is True
        assert sim.patient_flow.get_patient("52") is not None


# ---------------------------------------------------------------------------
# 4. Scenario switch preserves real patients, replaces demo patients
# ---------------------------------------------------------------------------

class TestScenarioSwitchPreservesRealPatients:
    def test_manual_arrived_patient_survives_scenario_switch(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        p = _arrived_patient("REAL-ARRIVED-1")
        sim.patient_flow.enqueue_patient(p)

        sim.load_scenario("BUSY_DAY")

        assert sim.patient_flow.get_patient("REAL-ARRIVED-1") is p
        assert p.status == PatientStatus.ARRIVED

    def test_manual_queued_triaged_patient_survives_scenario_switch(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        p = _triaged_patient("REAL-TRIAGED-1", "ADMITTED_GEN")
        sim.patient_flow.enqueue_patient(p)

        sim.load_scenario("SURGE_MASS_CASUALTY")

        found = sim.patient_flow.get_patient("REAL-TRIAGED-1")
        assert found is p
        assert found.status == PatientStatus.TRIAGED
        assert found.operational_decision["operational_department"] == "ADMITTED_GEN"

    def test_manual_admitted_patient_survives_scenario_switch(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        p = _arrived_patient("REAL-ADMIT-1")
        sim.patient_flow.enqueue_patient(p)
        sim.triage_patient(p)
        admit_res = sim.admit_patient("REAL-ADMIT-1")
        dept = admit_res["department"]

        sim.load_scenario("RESOURCE_CONSTRAINED")

        # The manually-admitted patient must still be admitted, in the
        # admitted cohort, with the department they were actually admitted
        # to — a scenario switch resetting the department's baseline
        # occupancy number is unrelated to (and must not silently discard)
        # this specific patient's own admission record.
        found = sim.patient_flow.get_patient("REAL-ADMIT-1")
        assert found is p
        assert found.status == PatientStatus.IN_TREATMENT
        assert "REAL-ADMIT-1" in sim.patient_flow._admitted_cohort
        assert sim.patient_flow._admitted_cohort["REAL-ADMIT-1"].department == dept

    def test_demo_patients_are_replaced_on_scenario_switch(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        normal_day_demo_ids = set(sim._presimulated_patient_ids)
        assert normal_day_demo_ids, "expected NORMAL_DAY to inject demo patients"

        sim.load_scenario("SURGE_MASS_CASUALTY")
        surge_demo_ids = set(sim._presimulated_patient_ids)
        assert surge_demo_ids, "expected SURGE_MASS_CASUALTY to inject demo patients"

        # The NORMAL_DAY-only demo patients must be gone now.
        removed_ids = normal_day_demo_ids - surge_demo_ids
        for pid in removed_ids:
            assert sim.patient_flow.get_patient(pid) is None

    def test_demo_ids_cannot_silently_overwrite_real_patient(self):
        """If a manually-registered patient happens to use the same id as a
        demo-pool entry, injecting that scenario's demo set must skip that
        id rather than overwrite the real patient."""
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim._presimulated_patient_ids = set()

        target_demo_id = "10016742"  # in NORMAL_DAY's TRIAGED_IDS_BY_SCENARIO
        real_patient = _arrived_patient(target_demo_id)
        sim.patient_flow.enqueue_patient(real_patient)

        sim.load_scenario("NORMAL_DAY")  # would normally inject 10016742 as TRIAGED

        found = sim.patient_flow.get_patient(target_demo_id)
        assert found is real_patient
        assert found.status == PatientStatus.ARRIVED  # untouched, not overwritten with demo TRIAGED data


# ---------------------------------------------------------------------------
# 5. Vitals update on a SimulatedPatient
# ---------------------------------------------------------------------------

class TestVitalsUpdate:
    def test_update_vitals_merges_in_place_no_duplicate(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("VIT-1")
        sim.patient_flow.enqueue_patient(p)

        updated = sim.update_patient_vitals("VIT-1", {"hr": 140, "spo2": 88})
        assert updated is p
        assert p.vitals["hr"] == 140
        assert p.vitals["spo2"] == 88
        # Unspecified vitals untouched.
        assert p.vitals["rr"] == 18
        assert len(sim.patient_flow.full_waiting_queue) == 1

    def test_update_vitals_rejects_out_of_range(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("VIT-2")
        sim.patient_flow.enqueue_patient(p)

        with pytest.raises(ValueError):
            sim.update_patient_vitals("VIT-2", {"spo2": 250})

    def test_update_vitals_unknown_patient_raises_keyerror(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        with pytest.raises(KeyError):
            sim.update_patient_vitals("GHOST", {"hr": 100})

    def test_update_vitals_rejects_discharged_patient(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("VIT-3")
        sim.patient_flow.enqueue_patient(p)
        sim.patient_flow.admit_patient(p, "DISCHARGE")

        with pytest.raises(ValueError):
            sim.update_patient_vitals("VIT-3", {"hr": 100})

    def test_update_vitals_endpoint(self, monkeypatch):
        monkeypatch.setattr(simulation_tools, "_simulator_instances", {})
        import api_server
        from fastapi.testclient import TestClient
        client = TestClient(api_server.app)
        sim = simulation_tools.get_simulator(None)
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_arrived_patient("VIT-API-1"))

        r = client.post("/api/simulation/patient/VIT-API-1/vitals", json={"hr": 155})
        assert r.status_code == 200
        assert r.json()["vitals"]["hr"] == 155


# ---------------------------------------------------------------------------
# 6-8. First triage, re-triage, requeue, nurse-override interaction
# ---------------------------------------------------------------------------

class TestTriageAndRetriage:
    def test_first_triage(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("TRI-1")
        sim.patient_flow.enqueue_patient(p)

        res = sim.triage_patient(p)
        op = res["operational_decision"]
        assert p.status == PatientStatus.TRIAGED
        assert op["retriage"] is False
        assert "previous_operational_department" not in op
        assert op["nurse_override"] is False

    def test_second_triage_is_explicit_retriage_not_stale_cache(self, monkeypatch):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("TRI-2")
        sim.patient_flow.enqueue_patient(p)

        outputs = iter([_clinical_output("ADMITTED_GEN"), _clinical_output("ADMITTED_GEN")])
        monkeypatch.setattr(sim, "_evaluate_clinical_truth", lambda patient: next(outputs))

        res1 = sim.triage_patient(p)
        assert res1["operational_decision"]["retriage"] is False

        res2 = sim.triage_patient(p)  # patient.status is now TRIAGED
        assert res2["operational_decision"]["retriage"] is True
        # Genuinely recomputed (both calls consumed), not a cached echo.
        with pytest.raises(StopIteration):
            next(outputs)

    def test_retriage_with_changed_vitals_changes_recommendation(self, monkeypatch):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("TRI-3")
        sim.patient_flow.enqueue_patient(p)

        outputs = iter([_clinical_output("ADMITTED_GEN"), _clinical_output("ICU")])
        monkeypatch.setattr(sim, "_evaluate_clinical_truth", lambda patient: next(outputs))

        res1 = sim.triage_patient(p)
        assert res1["clinical_assessment"]["department"] == "ADMITTED_GEN"
        assert res1["operational_decision"]["operational_department"] == "ADMITTED_GEN"

        sim.update_patient_vitals("TRI-3", {"hr": 150, "spo2": 85})
        res2 = sim.triage_patient(p)
        assert res2["clinical_assessment"]["department"] == "ICU"
        assert res2["operational_decision"]["operational_department"] == "ICU"
        assert res2["operational_decision"]["retriage"] is True

    def test_requeue_after_department_change(self, monkeypatch):
        """After re-triage changes the recommended department, the patient
        must be grouped into the NEW department's queue and not the old
        one — with no duplicate entry created."""
        from triageguard_agent.simulation.patient_flow import department_of

        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("TRI-4")
        sim.patient_flow.enqueue_patient(p)

        outputs = iter([_clinical_output("ADMITTED_GEN"), _clinical_output("ICU")])
        monkeypatch.setattr(sim, "_evaluate_clinical_truth", lambda patient: next(outputs))

        sim.triage_patient(p)
        assert department_of(p) == "ADMITTED_GEN"

        sim.triage_patient(p)
        assert department_of(p) == "ICU"

        # Exactly one entry for this patient — no duplicate enqueue.
        matches = [x for x in sim.patient_flow.full_waiting_queue if x.patient_id == "TRI-4"]
        assert len(matches) == 1

    def test_retriage_after_nurse_override_resets_and_flags_for_review(self, monkeypatch):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("TRI-5")
        sim.patient_flow.enqueue_patient(p)

        outputs = iter([_clinical_output("ADMITTED_GEN"), _clinical_output("ADMITTED_GEN")])
        monkeypatch.setattr(sim, "_evaluate_clinical_truth", lambda patient: next(outputs))

        sim.triage_patient(p)
        override_res = sim.override_department("TRI-5", "ICU", reason="family requested")
        assert override_res["nurse_override"] is True
        assert p.operational_decision["operational_department"] == "ICU"

        res2 = sim.triage_patient(p)
        op2 = res2["operational_decision"]

        # New AI recommendation replaces the old one; the stale override
        # does not silently keep governing the new assessment.
        assert op2["operational_department"] == "ADMITTED_GEN"
        assert op2["nurse_override"] is False
        assert op2["retriage"] is True
        # But history is not erased — it's visible in the audit fields.
        assert op2["previous_operational_department"] == "ICU"
        assert op2["previous_nurse_override"] is True
        assert op2["previous_override_reason"] == "family requested"
        # Flagged for explicit nurse review rather than silently resolved.
        assert op2["confirmation_required"] is True

        # An audit event was emitted recording the re-triage.
        events = sim.events.get_history(event_type=EventType.PATIENT_RETRIAGED, patient_id="TRI-5")
        assert len(events) == 1

    def test_retriage_without_prior_override_does_not_flag_review_unnecessarily(self, monkeypatch):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("TRI-6")
        sim.patient_flow.enqueue_patient(p)

        outputs = iter([_clinical_output("ADMITTED_GEN"), _clinical_output("ADMITTED_GEN")])
        monkeypatch.setattr(sim, "_evaluate_clinical_truth", lambda patient: next(outputs))

        sim.triage_patient(p)
        res2 = sim.triage_patient(p)
        op2 = res2["operational_decision"]
        assert op2["retriage"] is True
        assert op2["previous_nurse_override"] is False
        assert op2["confirmation_required"] is False


# ---------------------------------------------------------------------------
# 9. Admitted patients cannot be re-triaged
# ---------------------------------------------------------------------------

class TestAdmittedPatientCannotBeRetriaged:
    def test_simulator_rejects_triage_of_admitted_patient(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("ADM-1")
        sim.patient_flow.enqueue_patient(p)
        sim.triage_patient(p)
        sim.admit_patient("ADM-1")

        status_before = p.status
        dept_before = p.department
        occ_before = sim.state_service.get_state(dept_before)["occupied"]

        with pytest.raises(ValueError):
            sim.triage_patient(p)

        assert p.status == status_before
        assert p.department == dept_before
        assert sim.state_service.get_state(dept_before)["occupied"] == occ_before
        # Not re-inserted into the waiting/triage queue.
        assert all(x.patient_id != "ADM-1" for x in sim.patient_flow.full_waiting_queue)

    def test_rest_endpoint_returns_409_for_admitted_patient(self, monkeypatch):
        monkeypatch.setattr(simulation_tools, "_simulator_instances", {})
        import api_server
        from fastapi.testclient import TestClient
        client = TestClient(api_server.app)
        sim = simulation_tools.get_simulator(None)
        sim.patient_flow.clear()
        p = _arrived_patient("ADM-2")
        sim.patient_flow.enqueue_patient(p)
        sim.triage_patient(p)
        sim.admit_patient("ADM-2")

        r = client.post("/api/simulation/triage/ADM-2")
        assert r.status_code == 409

    def test_agent_tool_fails_for_admitted_patient(self, monkeypatch):
        monkeypatch.setattr(simulation_tools, "_simulator_instances", {})
        sim = simulation_tools.get_simulator(None)
        sim.patient_flow.clear()
        p = _arrived_patient("ADM-3")
        sim.patient_flow.enqueue_patient(p)
        sim.triage_patient(p)
        sim.admit_patient("ADM-3")

        result = simulation_tools.triage_simulated_patient("ADM-3")
        assert result.success is False
        assert result.error["code"] == "INVALID_PATIENT_STATE"


# ---------------------------------------------------------------------------
# 10. REST vs agent-tool triage consistency
# ---------------------------------------------------------------------------

class TestRestVsAgentTriageConsistency:
    def test_both_entry_points_call_the_same_canonical_operation(self, monkeypatch):
        monkeypatch.setattr(simulation_tools, "_simulator_instances", {})
        import api_server
        from fastapi.testclient import TestClient
        client = TestClient(api_server.app)
        sim = simulation_tools.get_simulator(None)
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_arrived_patient("CONSIST-REST"))
        sim.patient_flow.enqueue_patient(_arrived_patient("CONSIST-TOOL"))

        r = client.post("/api/simulation/triage/CONSIST-REST")
        assert r.status_code == 200
        rest_op = r.json()["operational_decision"]

        tool_result = simulation_tools.triage_simulated_patient("CONSIST-TOOL")
        assert tool_result.success is True
        tool_op = tool_result.data["operational_decision"]

        assert set(rest_op.keys()) == set(tool_op.keys())
        assert rest_op["retriage"] == tool_op["retriage"] == False

        # Re-triage via each path — both must recompute (not cache) and
        # both must mark retriage True.
        r2 = client.post("/api/simulation/triage/CONSIST-REST")
        assert r2.json()["operational_decision"]["retriage"] is True
        tool_result2 = simulation_tools.triage_simulated_patient("CONSIST-TOOL")
        assert tool_result2.data["operational_decision"]["retriage"] is True


# ---------------------------------------------------------------------------
# 11-13. Waiting queue / admitted state accuracy + dashboard consistency
# ---------------------------------------------------------------------------

class TestWaitingAndAdmittedStateAccuracy:
    def test_admit_removes_patient_from_waiting_queue(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("WQ-1")
        sim.patient_flow.enqueue_patient(p)
        sim.triage_patient(p)

        assert sim.patient_flow.waiting_count == 1
        sim.admit_patient("WQ-1")
        assert sim.patient_flow.waiting_count == 0
        assert sim.patient_flow.admitted_count == 1
        assert all(x.patient_id != "WQ-1" for x in sim.patient_flow.triaged_queue)

    def test_waiting_count_accurate_before_and_after_admission(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        for i in range(3):
            pt = _arrived_patient(f"WQ-COUNT-{i}")
            sim.patient_flow.enqueue_patient(pt)
            sim.triage_patient(pt)
        assert sim.patient_flow.waiting_count == 3

        sim.admit_patient("WQ-COUNT-0")
        assert sim.patient_flow.waiting_count == 2
        assert sim.patient_flow.admitted_count == 1

    def test_department_queue_excludes_admitted_patients_at_the_source(self):
        """Not relying only on frontend filtering: the backend's own
        full_waiting_queue must not contain an admitted patient at all."""
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        p = _arrived_patient("WQ-2")
        sim.patient_flow.enqueue_patient(p)
        sim.triage_patient(p)
        sim.admit_patient("WQ-2")

        dash = sim.get_live_dashboard()
        assert all(pd["patient_id"] != "WQ-2" for pd in dash["full_queue"])
        assert dash["waiting_count"] == 0
        assert dash["admitted_count"] == 1

    def test_dashboard_and_live_hospital_waiting_populations_match(self):
        """untriaged_count (the field Dashboard.tsx now reads) must equal
        an independent client-side status==ARRIVED filter over full_queue
        (what LiveHospital.tsx computes) — same population, computed two
        ways, must agree."""
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.patient_flow.enqueue_patient(_arrived_patient("DASH-1"))
        sim.patient_flow.enqueue_patient(_arrived_patient("DASH-2"))
        triaged = _arrived_patient("DASH-3")
        sim.patient_flow.enqueue_patient(triaged)
        sim.triage_patient(triaged)

        dash = sim.get_live_dashboard()
        client_side_arrived = [p for p in dash["full_queue"] if p["status"] == "ARRIVED"]
        assert dash["untriaged_count"] == len(client_side_arrived) == 2
        client_side_triaged = [p for p in dash["full_queue"] if p["status"] == "TRIAGED"]
        assert dash["triaged_count"] == len(client_side_triaged) == 1


# ---------------------------------------------------------------------------
# 12. Hospital isolation for the new behaviors
# ---------------------------------------------------------------------------

class TestHospitalIsolationForNewBehaviors:
    @pytest.fixture()
    def sandbox(self, tmp_path, monkeypatch):
        from triageguard_agent.hospital import hospital_registry as hr

        hr.reset_default_registry()
        test_registry = hr.HospitalRegistry(manifest_path=tmp_path / "hospitals" / "registry.json")
        monkeypatch.setattr(hr, "_default_registry", test_registry)
        monkeypatch.setattr(simulation_tools, "_simulator_instances", {})

        departments = {
            "ICU": {"capacity": 8, "occupied": 2, "status": "OPEN"},
            "ADMITTED_GEN": {"capacity": 20, "occupied": 5, "status": "OPEN"},
            "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
        }
        test_registry.register("iso_a", "Hospital A", config_dict={"departments": departments})
        test_registry.register("iso_b", "Hospital B", config_dict={"departments": departments})

        yield test_registry
        hr.reset_default_registry()

    def test_same_patient_id_active_in_both_hospitals_independently(self, sandbox):
        sim_a = HospitalSimulator(hospital_id="iso_a", scenario="NORMAL_DAY")
        sim_b = HospitalSimulator(hospital_id="iso_b", scenario="NORMAL_DAY")

        sim_a.patient_flow.enqueue_patient(_arrived_patient("SHARED-ID"))
        sim_b.patient_flow.enqueue_patient(_arrived_patient("SHARED-ID"))

        sim_a.update_patient_vitals("SHARED-ID", {"hr": 199})
        pa = sim_a.patient_flow.get_patient("SHARED-ID")
        pb = sim_b.patient_flow.get_patient("SHARED-ID")
        assert pa.vitals["hr"] == 199
        assert pb.vitals["hr"] != 199

    def test_retriage_in_one_hospital_does_not_affect_another(self, sandbox, monkeypatch):
        sim_a = HospitalSimulator(hospital_id="iso_a", scenario="NORMAL_DAY")
        sim_b = HospitalSimulator(hospital_id="iso_b", scenario="NORMAL_DAY")

        pa = _arrived_patient("ISO-RETRIAGE")
        pb = _arrived_patient("ISO-RETRIAGE")
        sim_a.patient_flow.enqueue_patient(pa)
        sim_b.patient_flow.enqueue_patient(pb)

        outputs_a = iter([_clinical_output("ADMITTED_GEN"), _clinical_output("ICU")])
        monkeypatch.setattr(sim_a, "_evaluate_clinical_truth", lambda patient: next(outputs_a))
        monkeypatch.setattr(sim_b, "_evaluate_clinical_truth", lambda patient: _clinical_output("ADMITTED_GEN"))

        sim_a.triage_patient(pa)
        sim_a.triage_patient(pa)  # re-triage in A only
        sim_b.triage_patient(pb)

        assert pa.operational_decision["retriage"] is True
        assert pb.operational_decision["retriage"] is False

    def test_admitted_patient_in_one_hospital_does_not_block_triage_in_another(self, sandbox):
        sim_a = HospitalSimulator(hospital_id="iso_a", scenario="NORMAL_DAY")
        sim_b = HospitalSimulator(hospital_id="iso_b", scenario="NORMAL_DAY")

        pa = _arrived_patient("ISO-ADMIT")
        sim_a.patient_flow.enqueue_patient(pa)
        sim_a.triage_patient(pa)
        sim_a.admit_patient("ISO-ADMIT")

        pb = _arrived_patient("ISO-ADMIT")  # same id, different hospital, not admitted there
        sim_b.patient_flow.enqueue_patient(pb)
        res_b = sim_b.triage_patient(pb)  # must succeed — hospital B's own patient is only ARRIVED
        assert res_b["operational_decision"]["retriage"] is False

        with pytest.raises(ValueError):
            sim_a.triage_patient(pa)  # hospital A's patient is genuinely admitted
