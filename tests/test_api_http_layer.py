"""
test_api_http_layer.py
-----------------------
Phase 6B Part 7: direct HTTP-level coverage for api_server.py's mutation
endpoints. The underlying business logic (HospitalSimulator.step/
triage_patient/admit_patient/override_department/etc., HospitalRegistry,
the LLM planning loop) is already heavily covered elsewhere by calling those
classes/functions directly — this file exists because NONE of the existing
suite exercised the actual FastAPI route wiring itself (status codes,
Pydantic request validation, response shape, error mapping) for most of
these endpoints, so a bug introduced only in api_server.py's thin wrapper
(wrong field name, wrong status code, wrong param passthrough) would not
have been caught by any existing test.

Uses the same fully-isolated sandbox pattern as test_hospital_onboarding_api.py
so nothing here touches (or is polluted by) the real repo's shared
data/hospitals or the process-wide default HospitalStateService singleton.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest
from fastapi.testclient import TestClient

from triageguard_agent.hospital import hospital_registry as hr
from triageguard_agent.hospital.hospital_state_service import HospitalStateService
from triageguard_agent.tools import simulation_tools
from triageguard_router.policy import artifacts

DEPARTMENTS = {
    "ICU": {"capacity": 8, "occupied": 2, "status": "OPEN"},
    "ADMITTED_GEN": {"capacity": 20, "occupied": 5, "status": "OPEN"},
    "ED_OBS": {"capacity": 10, "occupied": 3, "status": "OPEN"},
    "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
}


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    hr.reset_default_registry()
    HospitalStateService.reset_instance()
    monkeypatch.setattr(simulation_tools, "_simulator_instances", {})
    test_registry = hr.HospitalRegistry(manifest_path=tmp_path / "hospitals" / "registry.json")
    monkeypatch.setattr(hr, "_default_registry", test_registry)
    monkeypatch.setattr(artifacts, "_POLICY_DIR", tmp_path / "routing_policy")
    yield test_registry
    hr.reset_default_registry()
    HospitalStateService.reset_instance()


@pytest.fixture()
def client(sandbox):
    import api_server
    return TestClient(api_server.app)


@pytest.fixture()
def hospital(client):
    """A fresh, empty, isolated hospital — HTTP-registered like the real UI does."""
    r = client.post("/api/hospitals", json={
        "hospital_id": "test_h1", "hospital_name": "Test H1", "departments": DEPARTMENTS,
    })
    assert r.status_code == 200
    return "test_h1"


def _arrive(client, hospital, acuity=3):
    r = client.post("/api/simulation/arrival", params={"hospital_id": hospital, "target_acuity": acuity})
    assert r.status_code == 200
    return r.json()["patient_id"]


def _triage(client, hospital, patient_id):
    r = client.post(f"/api/simulation/triage/{patient_id}", params={"hospital_id": hospital})
    assert r.status_code == 200
    return r.json()


class TestSimulationScenario:
    def test_happy_path_returns_full_dashboard_shape(self, client, hospital):
        r = client.post("/api/simulation/scenario", json={"name": "BUSY_DAY", "hospital_id": hospital})
        assert r.status_code == 200
        body = r.json()
        for key in ("time", "scenario", "load", "departments", "full_queue", "waiting_count", "triaged_count", "admitted_count"):
            assert key in body

    def test_unknown_scenario_name_is_a_400_not_a_500(self, client, hospital):
        r = client.post("/api/simulation/scenario", json={"name": "NOT_A_REAL_SCENARIO", "hospital_id": hospital})
        assert r.status_code == 400

    def test_missing_required_name_field_is_a_422(self, client, hospital):
        r = client.post("/api/simulation/scenario", json={"hospital_id": hospital})
        assert r.status_code == 422


class TestSimulationStep:
    def test_happy_path_advances_clock_and_returns_expected_shape(self, client, hospital):
        before = client.get("/api/simulation/dashboard", params={"hospital_id": hospital}).json()["time"]
        r = client.post("/api/simulation/step", json={"minutes": 30, "auto_generate_arrivals": False, "hospital_id": hospital})
        assert r.status_code == 200
        body = r.json()
        for key in ("time", "sim_time_minutes", "discharged_count", "new_arrivals_count", "waiting_queue_count", "admitted_count", "operating_mode"):
            assert key in body
        assert body["time"] != before

    def test_zero_minutes_is_rejected_with_400(self, client, hospital):
        r = client.post("/api/simulation/step", json={"minutes": 0, "hospital_id": hospital})
        assert r.status_code == 400

    def test_negative_minutes_is_rejected_with_400(self, client, hospital):
        r = client.post("/api/simulation/step", json={"minutes": -5, "hospital_id": hospital})
        assert r.status_code == 400

    def test_defaults_apply_when_fields_omitted(self, client, hospital):
        r = client.post("/api/simulation/step", json={"hospital_id": hospital})
        assert r.status_code == 200  # minutes defaults to 15, auto_generate_arrivals to True


class TestSimulationArrival:
    def test_happy_path_creates_a_waiting_patient(self, client, hospital):
        r = client.post("/api/simulation/arrival", params={"hospital_id": hospital})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ARRIVED"
        assert "patient_id" in body

        dash = client.get("/api/simulation/dashboard", params={"hospital_id": hospital}).json()
        assert dash["waiting_count"] == 1

    def test_with_target_acuity(self, client, hospital):
        r = client.post("/api/simulation/arrival", params={"hospital_id": hospital, "target_acuity": 1})
        assert r.status_code == 200
        assert r.json()["acuity"] == 1

    def test_isolated_to_its_own_hospital(self, client, hospital):
        pid = _arrive(client, hospital)
        default_dash = client.get("/api/simulation/dashboard").json()
        # The default hospital's own demo pool is untouched by this hospital's arrival.
        default_ids = [p["patient_id"] for p in default_dash.get("full_queue", [])]
        assert pid not in default_ids


class TestManualArrival:
    def test_happy_path_new_walk_in(self, client, hospital):
        r = client.post("/api/simulation/manual-arrival", json={
            "patient_id": "WALK-1", "chief_complaint": "ankle sprain", "age": 30, "sex": "F", "acuity": 4,
            "hospital_id": hospital,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["patient_id"] == "WALK-1"
        assert body["status"] == "ARRIVED"
        assert body["has_history"] is False

    def test_missing_required_fields_is_422(self, client, hospital):
        r = client.post("/api/simulation/manual-arrival", json={"hospital_id": hospital})
        assert r.status_code == 422

    def test_temperature_out_of_range_is_400(self, client, hospital):
        r = client.post("/api/simulation/manual-arrival", json={
            "patient_id": "WALK-2", "chief_complaint": "test", "age": 40, "temperature": 90, "hospital_id": hospital,
        })
        assert r.status_code == 400

    def test_duplicate_active_patient_is_409(self, client, hospital):
        client.post("/api/simulation/manual-arrival", json={
            "patient_id": "WALK-3", "chief_complaint": "test", "age": 40, "hospital_id": hospital,
        })
        r = client.post("/api/simulation/manual-arrival", json={
            "patient_id": "WALK-3", "chief_complaint": "different complaint", "age": 50, "hospital_id": hospital,
        })
        assert r.status_code == 409

    def test_concurrent_duplicate_registration_over_real_http_is_still_safely_rejected(self, client, hospital):
        """End-to-end proof the Phase 6B lock fix holds through the real
        FastAPI/Starlette request path, not just the Python-level unit test."""
        import concurrent.futures

        def register():
            return client.post("/api/simulation/manual-arrival", json={
                "patient_id": "RACE-HTTP", "chief_complaint": "test", "age": 40, "hospital_id": hospital,
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda _: register(), range(5)))

        statuses = sorted(r.status_code for r in results)
        assert statuses == [200, 409, 409, 409, 409]
        dash = client.get("/api/simulation/dashboard", params={"hospital_id": hospital}).json()
        matches = [p for p in dash["full_queue"] if p["patient_id"] == "RACE-HTTP"]
        assert len(matches) == 1


class TestTriage:
    def test_happy_path_first_triage(self, client, hospital):
        pid = _arrive(client, hospital, acuity=2)
        r = client.post(f"/api/simulation/triage/{pid}", params={"hospital_id": hospital})
        assert r.status_code == 200
        body = r.json()
        assert "clinical_assessment" in body
        assert "operational_decision" in body
        assert body["operational_decision"]["retriage"] is False

    def test_retriage_of_an_already_triaged_patient_succeeds(self, client, hospital):
        pid = _arrive(client, hospital)
        _triage(client, hospital, pid)
        r = client.post(f"/api/simulation/triage/{pid}", params={"hospital_id": hospital})
        assert r.status_code == 200
        assert r.json()["operational_decision"]["retriage"] is True

    def test_nonexistent_patient_is_404(self, client, hospital):
        r = client.post("/api/simulation/triage/NO-SUCH-PATIENT", params={"hospital_id": hospital})
        assert r.status_code == 404

    def test_triaging_an_admitted_patient_is_409(self, client, hospital):
        pid = _arrive(client, hospital)
        decision = _triage(client, hospital, pid)
        dept = decision["operational_decision"]["operational_department"]
        sid = _new_session(client)
        client.post("/api/simulation/admit", json={"session_id": sid, "patient_id": pid, "department": dept, "hospital_id": hospital})
        # The admit is WRITE-gated: the call above only proposes it. Confirm
        # it so the patient is actually admitted before re-triaging.
        confirm = client.post("/api/tools/confirm", json={"session_id": sid, "approve": True})
        assert confirm.status_code == 200
        r = client.post(f"/api/simulation/triage/{pid}", params={"hospital_id": hospital})
        assert r.status_code == 409


class TestQueueReorder:
    def test_happy_path(self, client, hospital):
        p1 = _arrive(client, hospital)
        _arrive(client, hospital)
        r = client.post("/api/simulation/queue/reorder", json={"patient_id": p1, "new_index": 1, "hospital_id": hospital})
        assert r.status_code == 200
        assert r.json()["moved"] is True

    def test_nonexistent_patient_is_404(self, client, hospital):
        r = client.post("/api/simulation/queue/reorder", json={"patient_id": "GHOST", "new_index": 0, "hospital_id": hospital})
        assert r.status_code == 404

    def test_missing_required_fields_is_422(self, client, hospital):
        r = client.post("/api/simulation/queue/reorder", json={"hospital_id": hospital})
        assert r.status_code == 422


class TestDepartmentReorder:
    def test_happy_path(self, client, hospital):
        p1 = _arrive(client, hospital)
        d1 = _triage(client, hospital, p1)
        dept = d1["operational_decision"]["operational_department"]
        r = client.post("/api/simulation/queue/reorder-department", json={
            "patient_id": p1, "department": dept, "new_index": 0, "hospital_id": hospital,
        })
        assert r.status_code == 200

    def test_patient_not_in_that_department_is_404(self, client, hospital):
        p1 = _arrive(client, hospital)
        _triage(client, hospital, p1)
        r = client.post("/api/simulation/queue/reorder-department", json={
            "patient_id": p1, "department": "ED_OBS_WRONG_NAME", "new_index": 0, "hospital_id": hospital,
        })
        assert r.status_code == 404


class TestOverrideDepartment:
    def test_happy_path(self, client, hospital):
        pid = _arrive(client, hospital)
        _triage(client, hospital, pid)
        r = client.post("/api/simulation/queue/override", json={
            "patient_id": pid, "department": "ED_OBS", "reason": "family request", "hospital_id": hospital,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["operational_department"] == "ED_OBS"
        assert body["nurse_override"] is True

    def test_unknown_department_is_400(self, client, hospital):
        pid = _arrive(client, hospital)
        _triage(client, hospital, pid)
        r = client.post("/api/simulation/queue/override", json={
            "patient_id": pid, "department": "NOT_A_REAL_DEPT", "hospital_id": hospital,
        })
        assert r.status_code == 400

    def test_closed_department_is_400(self, client, hospital):
        pid = _arrive(client, hospital)
        _triage(client, hospital, pid)
        # Close ED_OBS via the same state path a nurse's capacity update would use.
        sim = simulation_tools.get_simulator(hospital)
        sim.state_service.apply_update("ED_OBS", {"status": "CLOSED", "occupied": 0})
        r = client.post("/api/simulation/queue/override", json={
            "patient_id": pid, "department": "ED_OBS", "hospital_id": hospital,
        })
        assert r.status_code == 400

    def test_nonexistent_patient_is_404(self, client, hospital):
        r = client.post("/api/simulation/queue/override", json={
            "patient_id": "GHOST", "department": "ED_OBS", "hospital_id": hospital,
        })
        assert r.status_code == 404


class TestVitalsUpdate:
    def test_happy_path(self, client, hospital):
        pid = _arrive(client, hospital)
        r = client.post(f"/api/simulation/patient/{pid}/vitals", json={"hr": 110, "hospital_id": hospital})
        assert r.status_code == 200
        assert r.json()["vitals"]["hr"] == 110

    def test_nonexistent_patient_is_404(self, client, hospital):
        r = client.post("/api/simulation/patient/GHOST/vitals", json={"hr": 100, "hospital_id": hospital})
        assert r.status_code == 404

    def test_out_of_range_value_is_400(self, client, hospital):
        pid = _arrive(client, hospital)
        r = client.post(f"/api/simulation/patient/{pid}/vitals", json={"hr": 9999, "hospital_id": hospital})
        assert r.status_code == 400

    def test_boundary_values_are_accepted(self, client, hospital):
        pid = _arrive(client, hospital)
        r = client.post(f"/api/simulation/patient/{pid}/vitals", json={
            "hr": 300, "rr": 0, "spo2": 100, "sbp": 0, "dbp": 200, "temp": 45, "pain": 10, "hospital_id": hospital,
        })
        assert r.status_code == 200


def _new_session(client):
    r = client.post("/api/session", json={"role": "nurse"})
    assert r.status_code == 200
    return r.json()["session_id"]


class TestSession:
    def test_create_session_happy_path(self, client):
        r = client.post("/api/session", json={"role": "nurse"})
        assert r.status_code == 200
        body = r.json()
        assert body["role"] == "nurse"
        assert body["session_id"]

    def test_get_state_for_unknown_session_is_404(self, client):
        r = client.get("/api/session/does-not-exist")
        assert r.status_code == 404

    def test_get_state_for_a_real_session_is_200(self, client):
        sid = _new_session(client)
        r = client.get(f"/api/session/{sid}")
        assert r.status_code == 200


class TestToolsExecuteAndConfirm:
    def test_execute_a_read_tool_returns_executed_immediately(self, client, hospital):
        sid = _new_session(client)
        r = client.post("/api/tools/execute", json={
            "session_id": sid, "tool_name": "get_live_simulation_dashboard", "kwargs": {"hospital_id": hospital},
        })
        assert r.status_code == 200
        assert r.json()["status"] == "executed"

    def test_execute_a_write_tool_requires_confirmation(self, client, hospital):
        pid = _arrive(client, hospital)
        decision = _triage(client, hospital, pid)
        dept = decision["operational_decision"]["operational_department"]
        sid = _new_session(client)
        r = client.post("/api/tools/execute", json={
            "session_id": sid, "tool_name": "admit_simulated_patient",
            "kwargs": {"patient_id": pid, "department": dept, "hospital_id": hospital},
        })
        assert r.status_code == 200
        assert r.json()["status"] == "awaiting_confirmation"

    def test_confirm_without_a_pending_action_is_400(self, client):
        sid = _new_session(client)
        r = client.post("/api/tools/confirm", json={"session_id": sid, "approve": True})
        assert r.status_code == 400

    def test_execute_unknown_session_is_404(self, client):
        r = client.post("/api/tools/execute", json={
            "session_id": "no-such-session", "tool_name": "get_live_simulation_dashboard", "kwargs": {},
        })
        assert r.status_code == 404

    def test_confirming_a_pending_write_actually_admits_the_patient(self, client, hospital):
        pid = _arrive(client, hospital)
        decision = _triage(client, hospital, pid)
        dept = decision["operational_decision"]["operational_department"]
        sid = _new_session(client)
        client.post("/api/tools/execute", json={
            "session_id": sid, "tool_name": "admit_simulated_patient",
            "kwargs": {"patient_id": pid, "department": dept, "hospital_id": hospital},
        })
        r = client.post("/api/tools/confirm", json={"session_id": sid, "approve": True})
        assert r.status_code == 200

        dash = client.get("/api/simulation/dashboard", params={"hospital_id": hospital}).json()
        assert dash["admitted_count"] == 1

    def test_rejecting_a_pending_write_does_not_admit_the_patient(self, client, hospital):
        pid = _arrive(client, hospital)
        decision = _triage(client, hospital, pid)
        dept = decision["operational_decision"]["operational_department"]
        sid = _new_session(client)
        client.post("/api/tools/execute", json={
            "session_id": sid, "tool_name": "admit_simulated_patient",
            "kwargs": {"patient_id": pid, "department": dept, "hospital_id": hospital},
        })
        r = client.post("/api/tools/confirm", json={"session_id": sid, "approve": False})
        assert r.status_code == 200

        dash = client.get("/api/simulation/dashboard", params={"hospital_id": hospital}).json()
        assert dash["admitted_count"] == 0


class TestAdmit:
    def test_happy_path_end_to_end(self, client, hospital):
        pid = _arrive(client, hospital)
        decision = _triage(client, hospital, pid)
        dept = decision["operational_decision"]["operational_department"]
        sid = _new_session(client)
        r = client.post("/api/simulation/admit", json={
            "session_id": sid, "patient_id": pid, "department": dept, "hospital_id": hospital,
        })
        assert r.status_code == 200
        # WRITE-tool gated — first call always awaits confirmation (never
        # auto-commits), same contract as /api/tools/execute for this tool.
        assert r.json()["status"] == "awaiting_confirmation"

    def test_confirmed_admit_is_reflected_in_the_dashboard(self, client, hospital):
        pid = _arrive(client, hospital)
        decision = _triage(client, hospital, pid)
        dept = decision["operational_decision"]["operational_department"]
        sid = _new_session(client)
        client.post("/api/simulation/admit", json={
            "session_id": sid, "patient_id": pid, "department": dept, "hospital_id": hospital,
        })
        client.post("/api/tools/confirm", json={"session_id": sid, "approve": True})
        dash = client.get("/api/simulation/dashboard", params={"hospital_id": hospital}).json()
        assert dash["admitted_count"] == 1
        assert dash["waiting_count"] == 0

    def test_concurrent_double_admit_over_real_http_only_admits_once(self, client, hospital):
        """End-to-end proof of the Phase 6B admit_patient() fix through the
        real HTTP path: two concurrent confirmed admits of the SAME patient
        must not double-consume a bed.

        Admits explicitly into "ICU" (always present in this sandbox's fixed
        DEPARTMENTS) rather than whatever department the clinical/routing
        model happens to recommend for this randomly-generated arrival —
        that recommendation can legitimately be a department outside this
        test hospital's 4-department catalog (e.g. "CICU"), which isn't a
        bug, just not what this concurrency test needs to exercise. The
        AdmitRequest.department override takes precedence over the triage
        recommendation (see admit_patient()'s target_dept resolution).
        """
        import concurrent.futures

        pid = _arrive(client, hospital)
        _triage(client, hospital, pid)
        dept = "ICU"
        before = client.get("/api/simulation/dashboard", params={"hospital_id": hospital}).json()
        icu_before = next(d for d in before["departments"] if d["name"] == dept)["occupied"]

        def admit_flow():
            sid = _new_session(client)
            client.post("/api/simulation/admit", json={
                "session_id": sid, "patient_id": pid, "department": dept, "hospital_id": hospital,
            })
            return client.post("/api/tools/confirm", json={"session_id": sid, "approve": True})

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda _: admit_flow(), range(5)))

        after = client.get("/api/simulation/dashboard", params={"hospital_id": hospital}).json()
        icu_after = next(d for d in after["departments"] if d["name"] == dept)["occupied"]
        assert icu_after - icu_before == 1  # exactly one real admission consumed exactly one bed
        assert after["admitted_count"] == 1
