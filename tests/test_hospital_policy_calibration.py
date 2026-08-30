"""
test_hospital_policy_calibration.py
-------------------------------------
Multi-hospital Step 5: nurse response collection + hospital-specific
Bayesian policy fitting + namespaced artifact storage.

Covers
------
* Hospital A and Hospital B can each independently complete calibration.
* Different nurse responses produce different fitted policies.
* Mixing responses/hospital_id across hospitals is rejected.
* Policy artifacts are stored separately per hospital_id (namespaced paths).
* Loading hospital A's policy never returns hospital B's.
* Recalibrating/re-saving A never touches B's file or in-memory content.
* Default (no hospital_id) behavior is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import numpy as np
import pytest

from triageguard_router.policy import artifacts
from triageguard_router.policy.bayesian_policy import BayesianLinearPolicy
from triageguard_router.policy.hospital_calibration import (
    NurseResponses,
    apply_responses,
    fit_hospital_policy,
)
from triageguard_router.policy.facility_calibration import scenarios_for_hospital


def _dept(capacity, occupied, status="OPEN"):
    return {"capacity": capacity, "occupied": occupied, "status": status}


FACILITY_A = {
    "ICU": _dept(10, 7),
    "CICU": _dept(6, 4),
    "ADMITTED_GEN": _dept(50, 38),
    "ED_OBS": _dept(20, 12),
    "DISCHARGE": _dept(999, 0),
}

FACILITY_B = {  # no CICU -> a genuinely different scenario set than A
    "ICU": _dept(8, 5),
    "ADMITTED_GEN": _dept(30, 20),
    "ED_OBS": _dept(10, 6),
    "DISCHARGE": _dept(999, 0),
}


class TestApplyResponses:
    def test_override_changes_preferred_department(self):
        scenarios = scenarios_for_hospital("hosp_a", FACILITY_A)["scenarios"]
        s02 = next(s for s in scenarios if s.scenario_id == "S02_icu_full_gen_available")
        assert s02.preferred_department == "ADMITTED_GEN"  # template default

        responses = NurseResponses(hospital_id="hosp_a", responses={"S02_icu_full_gen_available": "ED_OBS"})
        calibrated = apply_responses(scenarios, responses)
        new_s02 = next(s for s in calibrated if s.scenario_id == "S02_icu_full_gen_available")
        assert new_s02.preferred_department == "ED_OBS"

    def test_invalid_chosen_department_rejected(self):
        scenarios = scenarios_for_hospital("hosp_a", FACILITY_A)["scenarios"]
        responses = NurseResponses(
            hospital_id="hosp_a",
            responses={"S02_icu_full_gen_available": "NOT_A_REAL_DEPARTMENT"},
        )
        with pytest.raises(ValueError):
            apply_responses(scenarios, responses)

    def test_missing_scenario_id_falls_back_to_template_default(self):
        scenarios = scenarios_for_hospital("hosp_a", FACILITY_A)["scenarios"]
        responses = NurseResponses(hospital_id="hosp_a", responses={})  # nothing answered
        calibrated = apply_responses(scenarios, responses)
        assert [s.preferred_department for s in calibrated] == [s.preferred_department for s in scenarios]


class TestFitHospitalPolicy:
    def test_two_hospitals_produce_different_policies(self):
        responses_a = NurseResponses(hospital_id="hosp_a", responses={})
        responses_b = NurseResponses(hospital_id="hosp_b", responses={})

        policy_a = fit_hospital_policy("hosp_a", FACILITY_A, responses_a)
        policy_b = fit_hospital_policy("hosp_b", FACILITY_B, responses_b)

        assert policy_a.fitted and policy_b.fitted
        assert policy_a.training_metadata["hospital_id"] == "hosp_a"
        assert policy_b.training_metadata["hospital_id"] == "hosp_b"
        # Different facility -> different (filtered+rescaled) training set -> different weights.
        assert not np.allclose(policy_a.w_map, policy_b.w_map)

    def test_different_responses_same_facility_produce_different_weights(self):
        baseline = fit_hospital_policy("hosp_a", FACILITY_A, NurseResponses(hospital_id="hosp_a", responses={}))
        overridden = fit_hospital_policy(
            "hosp_a",
            FACILITY_A,
            NurseResponses(
                hospital_id="hosp_a",
                responses={
                    "S02_icu_full_gen_available": "ED_OBS",
                    "S05_cicu_full_icu_available": "ADMITTED_GEN",
                    "S07_gen_available": "ED_OBS",
                },
            ),
        )
        assert not np.allclose(baseline.w_map, overridden.w_map)

    def test_mismatched_hospital_id_rejected(self):
        wrong = NurseResponses(hospital_id="hosp_b", responses={})
        with pytest.raises(ValueError, match="hosp_b"):
            fit_hospital_policy("hosp_a", FACILITY_A, wrong)


class TestArtifactIsolation:
    def test_save_and_load_roundtrip_per_hospital(self, tmp_path):
        policy_a = fit_hospital_policy("hosp_a", FACILITY_A, NurseResponses(hospital_id="hosp_a"))
        artifacts.save_bayesian_policy(policy_a, hospital_id="hosp_a", base_dir=tmp_path)

        from triageguard_router.policy.config import PolicyConfig
        loaded = artifacts.load_bayesian_policy(PolicyConfig(), hospital_id="hosp_a", base_dir=tmp_path)
        assert np.allclose(loaded.w_map, policy_a.w_map)

    def test_hospital_a_and_b_artifacts_never_collide(self, tmp_path):
        from triageguard_router.policy.config import PolicyConfig

        policy_a = fit_hospital_policy("hosp_a", FACILITY_A, NurseResponses(hospital_id="hosp_a"))
        policy_b = fit_hospital_policy("hosp_b", FACILITY_B, NurseResponses(hospital_id="hosp_b"))

        path_a = artifacts.save_bayesian_policy(policy_a, hospital_id="hosp_a", base_dir=tmp_path)
        path_b = artifacts.save_bayesian_policy(policy_b, hospital_id="hosp_b", base_dir=tmp_path)
        assert path_a != path_b
        assert path_a.parent != path_b.parent

        loaded_a = artifacts.load_bayesian_policy(PolicyConfig(), hospital_id="hosp_a", base_dir=tmp_path)
        loaded_b = artifacts.load_bayesian_policy(PolicyConfig(), hospital_id="hosp_b", base_dir=tmp_path)
        assert np.allclose(loaded_a.w_map, policy_a.w_map)
        assert np.allclose(loaded_b.w_map, policy_b.w_map)
        assert not np.allclose(loaded_a.w_map, loaded_b.w_map)

    def test_recalibrating_a_does_not_modify_b(self, tmp_path):
        from triageguard_router.policy.config import PolicyConfig

        policy_a1 = fit_hospital_policy("hosp_a", FACILITY_A, NurseResponses(hospital_id="hosp_a"))
        policy_b = fit_hospital_policy("hosp_b", FACILITY_B, NurseResponses(hospital_id="hosp_b"))
        artifacts.save_bayesian_policy(policy_a1, hospital_id="hosp_a", base_dir=tmp_path)
        artifacts.save_bayesian_policy(policy_b, hospital_id="hosp_b", base_dir=tmp_path)
        b_before = artifacts.load_bayesian_policy(PolicyConfig(), hospital_id="hosp_b", base_dir=tmp_path).w_map.copy()

        # Recalibrate A with different nurse responses and re-save.
        policy_a2 = fit_hospital_policy(
            "hosp_a", FACILITY_A,
            NurseResponses(hospital_id="hosp_a", responses={"S02_icu_full_gen_available": "ED_OBS"}),
        )
        artifacts.save_bayesian_policy(policy_a2, hospital_id="hosp_a", base_dir=tmp_path)

        b_after = artifacts.load_bayesian_policy(PolicyConfig(), hospital_id="hosp_b", base_dir=tmp_path).w_map
        assert np.allclose(b_before, b_after)

    def test_default_hospital_id_behavior_unchanged(self, tmp_path):
        """hospital_id=None must resolve to the same (base_dir-relative) path
        shape as before this step existed — no hospital subdirectory."""
        from triageguard_router.policy.config import PolicyConfig

        policy = fit_hospital_policy("hosp_a", FACILITY_A, NurseResponses(hospital_id="hosp_a"))
        path = artifacts.save_bayesian_policy(policy, base_dir=tmp_path)
        assert path == tmp_path / "bayesian_policy.json"

        loaded = artifacts.load_bayesian_policy(PolicyConfig(), base_dir=tmp_path)
        assert np.allclose(loaded.w_map, policy.w_map)

    def test_artifacts_exist_is_hospital_scoped(self, tmp_path):
        assert not artifacts.artifacts_exist(hospital_id="hosp_a", base_dir=tmp_path)
        policy = fit_hospital_policy("hosp_a", FACILITY_A, NurseResponses(hospital_id="hosp_a"))
        artifacts.save_bayesian_policy(policy, hospital_id="hosp_a", base_dir=tmp_path)
        assert artifacts.artifacts_exist(hospital_id="hosp_a", base_dir=tmp_path)
        assert not artifacts.artifacts_exist(hospital_id="hosp_b", base_dir=tmp_path)
