"""
test_hospital_onboarding_api.py
----------------------------------
Phase 8: hospital onboarding + nurse calibration HTTP endpoints
(POST /api/hospitals, GET/POST .../calibration/...). These are thin wrappers
around already-tested backend functions (HospitalRegistry.register,
scenarios_for_hospital, fit_hospital_policy, artifacts.save_bayesian_policy)
— this file only verifies the HTTP wiring and end-to-end "the newly
calibrated policy is actually used in routing", not the underlying Bayesian
math (covered by test_live_hospital_routing.py / test_hospital_policy_calibration.py).
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
    """Fully isolated registry + artifact directory — never touches the
    real repo's data/hospitals/ or data/routing_policy/."""
    hr.reset_default_registry()
    HospitalStateService.reset_instance()
    # simulation_tools caches one HospitalSimulator per hospital_id for the
    # life of the process — reset it too, so a hospital_id reused across
    # tests (including "default") always gets a fresh simulator bound to
    # THIS test's sandboxed registry instead of a stale cached one.
    monkeypatch.setattr(simulation_tools, "_simulator_instances", {})
    test_registry = hr.HospitalRegistry(manifest_path=tmp_path / "hospitals" / "registry.json")
    # Patching the module-level singleton (not the per-module import) so
    # every caller of get_default_registry(), regardless of which module
    # imported the function reference, sees this same sandboxed registry.
    monkeypatch.setattr(hr, "_default_registry", test_registry)
    monkeypatch.setattr(artifacts, "_POLICY_DIR", tmp_path / "routing_policy")
    yield test_registry
    hr.reset_default_registry()
    HospitalStateService.reset_instance()


@pytest.fixture()
def client(sandbox):
    import api_server
    return TestClient(api_server.app)


def _register(client, hospital_id, name="Test Hospital", departments=None):
    return client.post(
        "/api/hospitals",
        json={
            "hospital_id": hospital_id,
            "hospital_name": name,
            "departments": DEPARTMENTS if departments is None else departments,
        },
    )


class TestNewHospitalStartsEmpty:
    """Quick fix: a newly registered hospital must never inherit the demo
    patient pool (or any other hospital's live patients) — exercises the
    REAL POST /api/hospitals -> GET /api/simulation/dashboard flow the
    running app actually uses, not just HospitalSimulator constructed
    directly in a unit test."""

    def test_new_hospital_starts_with_zero_patients(self, client):
        _register(client, "fresh_clinic")
        dash = client.get("/api/simulation/dashboard", params={"hospital_id": "fresh_clinic"}).json()
        assert dash["waiting_count"] == 0
        assert dash["admitted_count"] == 0
        assert dash["full_queue"] == []

    def test_new_hospital_never_contains_default_hospitals_patients(self, client):
        default_dash = client.get("/api/simulation/dashboard").json()
        default_patient_ids = {p["patient_id"] for p in default_dash["full_queue"]}
        assert default_patient_ids, "sanity check: default hospital's demo pool should be non-empty"

        _register(client, "fresh_clinic_2")
        new_dash = client.get("/api/simulation/dashboard", params={"hospital_id": "fresh_clinic_2"}).json()
        new_patient_ids = {p["patient_id"] for p in new_dash["full_queue"]}

        assert new_patient_ids == set()
        assert new_patient_ids.isdisjoint(default_patient_ids)

    def test_adding_patient_to_new_hospital_does_not_affect_default(self, client):
        default_before = client.get("/api/simulation/dashboard").json()["waiting_count"]

        _register(client, "fresh_clinic_3")
        r = client.post("/api/simulation/arrival", params={"hospital_id": "fresh_clinic_3", "target_acuity": 3})
        assert r.status_code == 200

        new_dash = client.get("/api/simulation/dashboard", params={"hospital_id": "fresh_clinic_3"}).json()
        assert new_dash["waiting_count"] == 1

        default_after = client.get("/api/simulation/dashboard").json()["waiting_count"]
        assert default_after == default_before

    def test_new_hospital_stays_empty_across_every_scenario(self, client):
        _register(client, "fresh_clinic_4")
        for scenario_name in ["NORMAL_DAY", "BUSY_DAY", "SURGE_MASS_CASUALTY", "RESOURCE_CONSTRAINED", "NIGHT_SHIFT"]:
            r = client.post("/api/simulation/scenario", json={"name": scenario_name, "hospital_id": "fresh_clinic_4"})
            assert r.status_code == 200
            assert r.json()["full_queue"] == []


class TestRegisterHospital:
    def test_register_new_hospital_appears_in_selector(self, client):
        r = _register(client, "clinic_x")
        assert r.status_code == 200
        assert r.json()["hospital_id"] == "clinic_x"

        listed = client.get("/api/hospitals").json()
        assert any(h["hospital_id"] == "clinic_x" for h in listed)

    def test_register_missing_departments_rejected(self, client):
        r = _register(client, "empty_x", departments={})
        assert r.status_code == 400

    def test_register_duplicate_hospital_id_rejected(self, client):
        _register(client, "dup_x")
        r = _register(client, "dup_x")
        assert r.status_code == 400


class TestCalibrationWorkflow:
    def test_scenarios_only_show_available_departments(self, client):
        _register(client, "clinic_y")
        r = client.get("/api/hospitals/clinic_y/calibration/scenarios")
        assert r.status_code == 200
        body = r.json()
        assert body["scenario_count"] > 0
        for scenario in body["scenarios"]:
            for dept in scenario["candidate_departments"]:
                assert dept in DEPARTMENTS  # CICU never offered — this hospital has none

    def test_submit_calibration_saves_namespaced_policy_and_status_flips(self, client):
        _register(client, "clinic_z")
        assert client.get("/api/hospitals/clinic_z/calibration/status").json()["calibrated"] is False

        scenarios = client.get("/api/hospitals/clinic_z/calibration/scenarios").json()["scenarios"]
        responses = {s["scenario_id"]: s["preferred_department"] for s in scenarios}

        r = client.post("/api/hospitals/clinic_z/calibration/submit", json={"responses": responses})
        assert r.status_code == 200
        assert r.json()["calibrated"] is True

        assert client.get("/api/hospitals/clinic_z/calibration/status").json()["calibrated"] is True

        # Namespacing: a second, uncalibrated hospital must be unaffected.
        _register(client, "clinic_untouched")
        assert client.get("/api/hospitals/clinic_untouched/calibration/status").json()["calibrated"] is False

    def test_submit_rejects_invalid_department_choice(self, client):
        _register(client, "clinic_bad")
        scenarios = client.get("/api/hospitals/clinic_bad/calibration/scenarios").json()["scenarios"]
        scenario_id = scenarios[0]["scenario_id"]
        r = client.post(
            "/api/hospitals/clinic_bad/calibration/submit",
            json={"responses": {scenario_id: "NOT_A_REAL_DEPARTMENT"}},
        )
        assert r.status_code == 400

    def test_calibrated_policy_actually_used_in_routing(self, client):
        """End-to-end: after calibration, route_with_hospital_policy for this
        hospital_id returns a real (non-None) result — the 'use the policy'
        requirement — through the same entry point live routing uses."""
        _register(client, "clinic_live")
        scenarios = client.get("/api/hospitals/clinic_live/calibration/scenarios").json()["scenarios"]
        responses = {s["scenario_id"]: s["preferred_department"] for s in scenarios}
        client.post("/api/hospitals/clinic_live/calibration/submit", json={"responses": responses})

        from triageguard_router.reconciler import reconcile
        from triageguard_router.router import route
        from triageguard_router.policy.live_routing import route_with_hospital_policy

        xgb_output = {
            "icu_risk_2h": 0.75, "icu_risk_2h_confidence": 0.85,
            "icu_risk_6h": 0.70, "icu_risk_6h_confidence": 0.85,
            "icu_risk_12h": 0.65, "icu_risk_12h_confidence": 0.85,
            "admission_risk": 0.88, "admission_risk_confidence": 0.85,
            "information_completeness": 0.95,
        }
        rag_output = {
            "structured_output": {
                "urgency": "high", "evidence_strength": 4, "escalation_concern": False,
                "top_diagnoses": [], "red_flags": [],
            }
        }
        reconciled = reconcile(xgb_output, rag_output)
        preferred = route(reconciled)["department"]

        result = route_with_hospital_policy(reconciled, xgb_output, preferred, hospital_id="clinic_live")
        assert result is not None
        assert result["routing"]["allocated_department"] is not None
