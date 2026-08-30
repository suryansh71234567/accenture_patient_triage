"""
test_multi_hospital_simulation.py
------------------------------------
Multi-hospital Step 7: simulation subsystem + RL environment hospital
awareness.

Covers Part E (isolation) and Part F (end-to-end demonstration).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import numpy as np
import pytest

from triageguard_agent.simulation.hospital_simulator import HospitalSimulator
from triageguard_agent.tools import simulation_tools
from triageguard_router.policy import artifacts
from triageguard_router.policy.hospital_calibration import (
    NurseResponses,
    fit_hospital_policy,
    train_hospital_rl_policy,
)


def _dept(capacity, occupied, status="OPEN"):
    return {"capacity": capacity, "occupied": occupied, "status": status}


FACILITY_A = {
    "ICU": _dept(10, 6), "CICU": _dept(6, 3),
    "ADMITTED_GEN": _dept(50, 30), "ED_OBS": _dept(20, 10), "DISCHARGE": _dept(999, 0),
}
FACILITY_B = {  # no CICU, smaller ICU -> genuinely different facility
    "ICU": _dept(4, 1), "ADMITTED_GEN": _dept(20, 5), "ED_OBS": _dept(8, 2), "DISCHARGE": _dept(999, 0),
}


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Fully isolated hospital registry + policy artifact directory."""
    from triageguard_agent.hospital import hospital_registry as hr
    from triageguard_agent.hospital.hospital_state_service import HospitalStateService

    HospitalStateService.reset_instance()
    test_registry = hr.HospitalRegistry(manifest_path=tmp_path / "hospitals" / "registry.json")
    monkeypatch.setattr(hr, "get_default_registry", lambda: test_registry)
    monkeypatch.setattr(artifacts, "_POLICY_DIR", tmp_path / "routing_policy")
    # simulation_tools keeps its own per-hospital simulator cache — reset
    # between tests so hospital_a/hospital_b never leak across test runs.
    monkeypatch.setattr(simulation_tools, "_simulator_instances", {})

    test_registry.register("hosp_a", "Hospital A", config_dict={"departments": FACILITY_A})
    test_registry.register("hosp_b", "Hospital B", config_dict={"departments": FACILITY_B})

    yield test_registry
    HospitalStateService.reset_instance()


class TestIndependentSimulations:
    def test_two_simulators_coexist_with_independent_state(self, sandbox):
        sim_a = HospitalSimulator(hospital_id="hosp_a", scenario="NORMAL_DAY")
        sim_b = HospitalSimulator(hospital_id="hosp_b", scenario="NORMAL_DAY")

        assert sim_a.state_service is not sim_b.state_service
        # Hospital B genuinely lacks CICU — scenario loading correctly skips
        # departments a hospital doesn't have (load_scenario's existing
        # `department_exists` guard, unchanged).
        assert "CICU" in sim_a.state_service.get_all()
        assert "CICU" not in sim_b.state_service.get_all()

    def test_occupancy_change_in_a_never_affects_b(self, sandbox):
        sim_a = HospitalSimulator(hospital_id="hosp_a")
        sim_b = HospitalSimulator(hospital_id="hosp_b")

        b_before = sim_b.state_service.get_state("ICU")["occupied"]
        patch = sim_a.state_service.validate_update("ICU", {"occupied": 10})
        sim_a.state_service.apply_update("ICU", patch)

        assert sim_a.state_service.get_state("ICU")["occupied"] == 10
        assert sim_b.state_service.get_state("ICU")["occupied"] == b_before

    def test_events_in_a_do_not_affect_b(self, sandbox):
        sim_a = HospitalSimulator(hospital_id="hosp_a")
        sim_b = HospitalSimulator(hospital_id="hosp_b")

        b_events_before = len(sim_b.events.get_history())
        sim_a.trigger_arrival(target_acuity=2)
        assert len(sim_b.events.get_history()) == b_events_before
        assert sim_a.events is not sim_b.events

    def test_patient_flow_in_a_does_not_affect_b(self, sandbox):
        sim_a = HospitalSimulator(hospital_id="hosp_a")
        sim_b = HospitalSimulator(hospital_id="hosp_b")

        assert sim_a.patient_flow is not sim_b.patient_flow
        sim_a.trigger_arrival(target_acuity=3)
        assert len(list(sim_a.patient_flow.peek_waiting(count=10))) >= 1
        assert len(list(sim_b.patient_flow.peek_waiting(count=10))) == 0

    def test_default_hospital_id_preserves_existing_behavior(self, sandbox):
        sim_default = HospitalSimulator()
        assert sim_default.hospital_id == "default"
        # Bound to the same live registry-managed default context as before.
        from triageguard_agent.hospital.hospital_registry import DEFAULT_HOSPITAL_ID
        assert sim_default.state_service is sandbox.get(DEFAULT_HOSPITAL_ID).state_service

    def test_non_default_hospital_starts_with_empty_queue(self, sandbox):
        """A newly registered (non-default) hospital must start with zero
        patients — no shared demo-pool patients — so it can be populated
        with its own real arrivals via manual intake / triggered arrivals."""
        sim = HospitalSimulator(hospital_id="hosp_a", scenario="NORMAL_DAY")
        assert sim.patient_flow.waiting_count == 0
        assert sim.patient_flow.admitted_count == 0


class TestPresimulatedPatientIsolation:
    """Phase 7 finding: build_simulated_patient() used to assign the
    module-level pool_entry's clinical_assessment/operational_decision
    dicts directly onto the new SimulatedPatient (no copy). Since demo
    patients can be built more than once from the same template (e.g. the
    default hospital reloading a scenario), that aliasing let a later
    mutation (operating_mode/lambda) silently leak into an earlier build's
    already-displayed patient data. Only the default hospital receives
    demo patients (non-default hospitals start empty — see
    HospitalSimulator._inject_presimulated_patients), so this is tested
    directly against build_simulated_patient() rather than via two
    hospitals."""

    def test_build_simulated_patient_does_not_alias_template_dicts(self):
        from triageguard_agent.simulation.presimulated_patients import (
            build_simulated_patient,
            get_patient_by_id,
        )

        entry = get_patient_by_id("10016742")
        assert entry is not None

        patient_1 = build_simulated_patient(entry, sim_time_min=0)
        patient_2 = build_simulated_patient(entry, sim_time_min=0)

        # Same template, but must be independent dict objects, not aliases
        # of the shared module-level pool entry.
        assert patient_1.operational_decision is not patient_2.operational_decision
        assert patient_1.clinical_assessment is not patient_2.clinical_assessment

        # Mutating one build's copy must never affect the other's.
        original_mode_2 = patient_2.operational_decision["operating_mode"]
        patient_1.operational_decision["operating_mode"] = "CRITICAL"
        assert patient_2.operational_decision["operating_mode"] == original_mode_2


class TestHospitalIdentityThroughSimulation:
    def test_simulated_patient_hospital_id_reaches_pipeline(self, sandbox, monkeypatch):
        captured = {}

        class _FakePipeline:
            def run(self, patient_data, hospital_id=None):
                captured["hospital_id_in_dict"] = patient_data.get("hospital_id")
                return {"department": "ICU", "hospital_routing": None, "xgb": {}, "rag_response": ""}

        from triageguard_agent.tools import assessment_tools
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: _FakePipeline())

        sim_a = HospitalSimulator(hospital_id="hosp_a")
        patient = sim_a.trigger_arrival(target_acuity=2)
        sim_a.triage_patient(patient)

        assert captured["hospital_id_in_dict"] == "hosp_a"

    def test_simulation_tools_thread_hospital_id_through(self, sandbox, monkeypatch):
        captured = {}

        class _FakePipeline:
            def run(self, patient_data, hospital_id=None):
                captured["hospital_id_in_dict"] = patient_data.get("hospital_id")
                return {"department": "ADMITTED_GEN", "hospital_routing": None, "xgb": {}, "rag_response": ""}

        from triageguard_agent.tools import assessment_tools
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: _FakePipeline())

        arrival = simulation_tools.trigger_patient_arrival(target_acuity=3, hospital_id="hosp_b")
        assert arrival.success
        patient_id = arrival.data["patient_id"]

        result = simulation_tools.triage_simulated_patient(patient_id, hospital_id="hosp_b")
        assert result.success
        assert captured["hospital_id_in_dict"] == "hosp_b"

    def test_hospital_a_and_b_simulator_caches_are_independent(self, sandbox):
        sim_a1 = simulation_tools.get_simulator("hosp_a")
        sim_a2 = simulation_tools.get_simulator("hosp_a")
        sim_b = simulation_tools.get_simulator("hosp_b")
        assert sim_a1 is sim_a2          # cached per hospital
        assert sim_a1 is not sim_b       # never shared across hospitals


class TestRLEnvironmentHospitalScoping:
    def _fast_config(self):
        from triageguard_router.policy.config import PolicyConfig, RLConfig
        return PolicyConfig(rl=RLConfig(episodes=3, steps_per_episode=3, minutes_per_step=15))

    def test_hospital_scoped_env_uses_that_hospitals_facility_not_global(self, sandbox):
        from triageguard_router.policy.simulation_env import RoutingEnv

        env_a = RoutingEnv(config=self._fast_config(), hospital_id="hosp_a")

        def dummy_policy(phi_mat, mask, candidates):
            feasible_idx = [i for i, m in enumerate(mask) if m]
            return feasible_idx[0] if feasible_idx else 0

        result = env_a.run_episode(dummy_policy, seed=1)
        assert result is not None
        # Hospital A's real, live state was never touched by training.
        ctx_a = sandbox.get("hosp_a")
        assert ctx_a.state_service.get_state("ICU")["occupied"] == 6  # unchanged from FACILITY_A

    def test_default_env_behavior_unchanged(self, sandbox):
        from triageguard_router.policy.simulation_env import RoutingEnv

        env = RoutingEnv(config=self._fast_config())  # hospital_id=None
        assert env.hospital_id is None

        def dummy_policy(phi_mat, mask, candidates):
            feasible_idx = [i for i, m in enumerate(mask) if m]
            return feasible_idx[0] if feasible_idx else 0

        result = env.run_episode(dummy_policy, seed=1)
        assert result is not None

    def test_rl_policy_artifacts_do_not_collide_between_hospitals(self, sandbox):
        cfg = self._fast_config()
        policy_a = fit_hospital_policy("hosp_a", FACILITY_A, NurseResponses(hospital_id="hosp_a"))
        policy_b = fit_hospital_policy("hosp_b", FACILITY_B, NurseResponses(hospital_id="hosp_b"))

        rl_a = train_hospital_rl_policy("hosp_a", policy_a, config=cfg)
        rl_b = train_hospital_rl_policy("hosp_b", policy_b, config=cfg)

        artifacts.save_rl_policy(rl_a, hospital_id="hosp_a")
        artifacts.save_rl_policy(rl_b, hospital_id="hosp_b")

        from triageguard_router.policy.config import PolicyConfig
        loaded_a = artifacts.load_rl_policy(PolicyConfig(), hospital_id="hosp_a")
        loaded_b = artifacts.load_rl_policy(PolicyConfig(), hospital_id="hosp_b")

        assert np.allclose(loaded_a.w.detach().numpy(), rl_a.w.detach().numpy())
        assert np.allclose(loaded_b.w.detach().numpy(), rl_b.w.detach().numpy())
        # Different facility/policy inputs -> different RL weights.
        assert not np.allclose(loaded_a.w.detach().numpy(), loaded_b.w.detach().numpy())


class TestEndToEndTwoHospitalDemonstration:
    def test_two_hospitals_isolated_and_can_route_differently(self, sandbox, monkeypatch):
        """Part F: different facility/config, different occupancy, simulated
        patient, triage, hospital-specific routing — for two hospitals."""
        # Distinct nurse calibration + saved policy per hospital (Step 5/6).
        policy_a = fit_hospital_policy("hosp_a", FACILITY_A, NurseResponses(hospital_id="hosp_a"))
        policy_b = fit_hospital_policy(
            "hosp_b", FACILITY_B,
            NurseResponses(hospital_id="hosp_b", responses={"S07_gen_available": "ED_OBS"}),
        )
        artifacts.save_bayesian_policy(policy_a, hospital_id="hosp_a")
        artifacts.save_bayesian_policy(policy_b, hospital_id="hosp_b")

        from triageguard_router import combined_pipeline as cp

        pipeline = object.__new__(cp.TriageGuardPipeline)

        xgb_output = {
            "icu_risk_2h": 0.75, "icu_risk_2h_confidence": 0.85,
            "icu_risk_6h": 0.70, "icu_risk_6h_confidence": 0.85,
            "icu_risk_12h": 0.65, "icu_risk_12h_confidence": 0.85,
            "admission_risk": 0.88, "admission_risk_confidence": 0.85,
            "information_completeness": 0.95,
        }
        rag_output = {
            "response": "",
            "structured_output": {
                "urgency": "high", "evidence_strength": 4, "escalation_concern": False,
                "top_diagnoses": ["sepsis"], "red_flags": ["tachycardia"],
            },
            "patient_history": [], "similar_cases": [],
        }

        class _FakeXgb:
            def predict(self, patient):
                return dict(xgb_output)

        class _FakeRag:
            def run(self, patient, hospital_id=None):
                return dict(rag_output)

        pipeline.xgb = _FakeXgb()
        pipeline.rag = _FakeRag()

        result_a = pipeline.run({"patient_id": 1, "age": 55}, hospital_id="hosp_a")
        result_b = pipeline.run({"patient_id": 1, "age": 55}, hospital_id="hosp_b")

        # Same clinical patient -> identical clinical preference.
        assert result_a["department"] == result_b["department"] == "ICU"
        # But hospital-specific routing genuinely differs (different
        # facility, different calibration, different occupancy).
        assert result_a["hospital_routing"] is not None
        assert result_b["hospital_routing"] is not None
        assert result_a["hospital_routing"]["department_scores"] != result_b["hospital_routing"]["department_scores"]

        # Isolation: hospital B's state was never touched by hospital A's run.
        ctx_b = sandbox.get("hosp_b")
        assert ctx_b.state_service.get_state("ICU")["occupied"] == 1  # unchanged from FACILITY_B
