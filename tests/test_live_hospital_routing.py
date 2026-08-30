"""
test_live_hospital_routing.py
-------------------------------
Multi-hospital Step 6: connecting a hospital's calibrated policy + live
state to the live routing path (combined_pipeline.py / live_routing.py).

Covers
------
1. hospital_id reaches live routing (combined_pipeline.run kwarg).
2. Correct calibrated policy is loaded for each hospital.
3. Different hospital policies produce different routing scores/decisions
   for the IDENTICAL clinical patient.
4. Hospital state (occupancy) is respected — one hospital's full ICU
   doesn't affect another's routing.
5. Policy cannot cross hospital boundaries.
6. An infeasible (full) department is never the final allocation, even
   when it's the clinically preferred one.
7. Existing routing tests (test_routing_policy.py) still pass — run
   separately, not duplicated here.
8. End-to-end TriageGuardPipeline.run() for two hospitals.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_router.reconciler import reconcile
from triageguard_router.router import route
from triageguard_router.policy.live_routing import route_with_hospital_policy
from triageguard_router.policy.hospital_calibration import NurseResponses, fit_hospital_policy
from triageguard_router.policy import artifacts


def _dept(capacity, occupied, status="OPEN"):
    return {"capacity": capacity, "occupied": occupied, "status": status}


FACILITY_A = {
    "ICU": _dept(10, 6), "CICU": _dept(6, 3),
    "ADMITTED_GEN": _dept(50, 30), "ED_OBS": _dept(20, 10), "DISCHARGE": _dept(999, 0),
}
FACILITY_B = {  # no CICU, different capacities -> genuinely different calibration
    "ICU": _dept(8, 4), "ADMITTED_GEN": _dept(25, 15), "ED_OBS": _dept(10, 5), "DISCHARGE": _dept(999, 0),
}

# One fixed, realistic high-ICU-risk clinical patient, built through the
# REAL reconciler+router (unchanged by this step) — same patient reused
# for every hospital in these tests.
_XGB_OUTPUT = {
    "icu_risk_2h": 0.75, "icu_risk_2h_confidence": 0.85,
    "icu_risk_6h": 0.70, "icu_risk_6h_confidence": 0.85,
    "icu_risk_12h": 0.65, "icu_risk_12h_confidence": 0.85,
    "admission_risk": 0.88, "admission_risk_confidence": 0.85,
    "information_completeness": 0.95,
}
_RAG_OUTPUT = {
    "structured_output": {
        "urgency": "high", "evidence_strength": 4, "escalation_concern": False,
        "top_diagnoses": ["sepsis"], "red_flags": ["tachycardia"],
    }
}


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """
    Fully isolated hospital registry + policy artifact directory — never
    touches the real repo's data/hospitals/ or data/routing_policy/.
    """
    from triageguard_agent.hospital import hospital_registry as hr
    from triageguard_agent.hospital.hospital_state_service import HospitalStateService

    HospitalStateService.reset_instance()
    test_registry = hr.HospitalRegistry(manifest_path=tmp_path / "hospitals" / "registry.json")
    monkeypatch.setattr(hr, "get_default_registry", lambda: test_registry)
    monkeypatch.setattr(artifacts, "_POLICY_DIR", tmp_path / "routing_policy")

    test_registry.register("hosp_a", "Hospital A", config_dict={"departments": FACILITY_A})
    test_registry.register("hosp_b", "Hospital B", config_dict={"departments": FACILITY_B})

    policy_a = fit_hospital_policy("hosp_a", FACILITY_A, NurseResponses(hospital_id="hosp_a"))
    policy_b = fit_hospital_policy(
        "hosp_b", FACILITY_B,
        NurseResponses(hospital_id="hosp_b", responses={"S02_icu_full_gen_available": "ED_OBS"}),
    )
    artifacts.save_bayesian_policy(policy_a, hospital_id="hosp_a")
    artifacts.save_bayesian_policy(policy_b, hospital_id="hosp_b")

    yield test_registry
    HospitalStateService.reset_instance()


def _clinical_preferred_department() -> str:
    reconciled = reconcile(_XGB_OUTPUT, _RAG_OUTPUT)
    decision = route(reconciled)
    return decision["department"]


class TestPolicySelectionPerHospital:
    def test_clinical_preference_is_icu_for_this_patient(self, sandbox):
        assert _clinical_preferred_department() == "ICU"

    def test_two_hospitals_produce_different_routing_results(self, sandbox):
        reconciled = reconcile(_XGB_OUTPUT, _RAG_OUTPUT)
        preferred = route(reconciled)["department"]

        result_a = route_with_hospital_policy(reconciled, _XGB_OUTPUT, preferred, hospital_id="hosp_a")
        result_b = route_with_hospital_policy(reconciled, _XGB_OUTPUT, preferred, hospital_id="hosp_b")

        assert result_a is not None and result_b is not None
        # Same clinical facts on both sides — clinical assessment must be identical.
        assert result_a["clinical_assessment"] == result_b["clinical_assessment"]
        # But the hospital-specific policy scores must differ (different
        # facility, different nurse calibration -> different fitted weights).
        assert result_a["department_scores"] != result_b["department_scores"]

    def test_policy_cannot_cross_hospital_boundary(self, sandbox):
        from triageguard_router.policy.config import PolicyConfig

        policy_a = artifacts.load_bayesian_policy(PolicyConfig(), hospital_id="hosp_a")
        policy_b = artifacts.load_bayesian_policy(PolicyConfig(), hospital_id="hosp_b")
        import numpy as np
        assert not np.allclose(policy_a.w_map, policy_b.w_map)

    def test_no_policy_falls_back_to_none(self, sandbox):
        sandbox.register("hosp_c", "Hospital C (uncalibrated)", config_dict={"departments": FACILITY_A})
        reconciled = reconcile(_XGB_OUTPUT, _RAG_OUTPUT)
        preferred = route(reconciled)["department"]
        result = route_with_hospital_policy(reconciled, _XGB_OUTPUT, preferred, hospital_id="hosp_c")
        assert result is None


class TestHospitalStateRespected:
    def test_full_icu_at_hospital_a_forces_step_down_there_only(self, sandbox):
        reconciled = reconcile(_XGB_OUTPUT, _RAG_OUTPUT)
        preferred = route(reconciled)["department"]
        assert preferred == "ICU"

        ctx_a = sandbox.get("hosp_a")
        patch = ctx_a.state_service.validate_update("ICU", {"occupied": 10})  # fill ICU completely
        ctx_a.state_service.apply_update("ICU", patch)

        result_a = route_with_hospital_policy(reconciled, _XGB_OUTPUT, preferred, hospital_id="hosp_a")
        result_b = route_with_hospital_policy(reconciled, _XGB_OUTPUT, preferred, hospital_id="hosp_b")

        assert result_a["routing"]["allocated_department"] != "ICU"
        assert result_a["routing"]["resource_constraint"] is True
        # Hospital B's ICU was never touched — still allocates ICU directly.
        assert result_b["routing"]["allocated_department"] == "ICU"
        assert result_b["routing"]["resource_constraint"] is False

    def test_infeasible_department_never_allocated_even_if_preferred(self, sandbox):
        reconciled = reconcile(_XGB_OUTPUT, _RAG_OUTPUT)
        preferred = route(reconciled)["department"]

        ctx_a = sandbox.get("hosp_a")
        patch = ctx_a.state_service.validate_update("ICU", {"occupied": 0, "status": "CLOSED"})
        ctx_a.state_service.apply_update("ICU", patch)

        result = route_with_hospital_policy(reconciled, _XGB_OUTPUT, preferred, hospital_id="hosp_a")
        allocated = result["routing"]["allocated_department"]
        assert allocated != "ICU"
        if allocated is not None:
            dept_state = ctx_a.state_service.get_state(allocated)
            assert dept_state["status"] == "OPEN"
            assert dept_state["available"] > 0


class TestEndToEndPipeline:
    def test_pipeline_run_produces_different_hospital_routing_for_two_hospitals(self, sandbox):
        from triageguard_router import combined_pipeline as cp

        pipeline = object.__new__(cp.TriageGuardPipeline)

        class _FakeXgb:
            def predict(self, patient):
                return dict(_XGB_OUTPUT)

        class _FakeRag:
            def run(self, patient, hospital_id=None):
                return {"response": "", **_RAG_OUTPUT, "patient_history": [], "similar_cases": []}

        pipeline.xgb = _FakeXgb()
        pipeline.rag = _FakeRag()

        result_a = pipeline.run({"patient_id": 1, "age": 55}, hospital_id="hosp_a")
        result_b = pipeline.run({"patient_id": 1, "age": 55}, hospital_id="hosp_b")

        # Clinical preference is hospital-independent — must match.
        assert result_a["department"] == result_b["department"] == "ICU"
        # hospital_id is threaded through and echoed back correctly.
        assert result_a["hospital_id"] == "hosp_a"
        assert result_b["hospital_id"] == "hosp_b"
        # Hospital-specific routing actually differs.
        assert result_a["hospital_routing"] is not None
        assert result_b["hospital_routing"] is not None
        assert result_a["hospital_routing"]["department_scores"] != result_b["hospital_routing"]["department_scores"]

    def test_pipeline_run_without_hospital_id_does_not_crash(self, sandbox):
        """Backward compatibility: omitting hospital_id must still work."""
        from triageguard_router import combined_pipeline as cp

        pipeline = object.__new__(cp.TriageGuardPipeline)

        class _FakeXgb:
            def predict(self, patient):
                return dict(_XGB_OUTPUT)

        class _FakeRag:
            def run(self, patient, hospital_id=None):
                return {"response": "", **_RAG_OUTPUT, "patient_history": [], "similar_cases": []}

        pipeline.xgb = _FakeXgb()
        pipeline.rag = _FakeRag()

        result = pipeline.run({"patient_id": 1, "age": 55})
        assert result["department"] == "ICU"
        assert "hospital_routing" in result  # may be None or a real dict — must not KeyError/crash
