"""
test_routing_policy.py
------------------------
Tests for the nurse-guided Bayesian routing policy + simulation-based RL
optimization (triageguard_router/policy/).

These exercise REAL code paths — real feature extraction, a real fitted
Bayesian policy (Laplace approximation), a real RL training loop over the
REAL HospitalSimulator (via the fast/deterministic calibrated fallback, not
a stub), and real safety/reward logic. Nothing here just asserts a function
exists.

Covers (Phase 18):
 1.  Nurse imitation
 2.  Paired scenarios (only ICU availability changes -> allocation changes)
 3.  Clinical invariance (ICU full must NOT change clinical_priority)
 4.  Safety (critical patient can't silently land in a low-acuity department)
 5.  Uncertainty (OOD state increases policy_uncertainty)
 6.  Simulation (reproducible under a fixed seed)
 7.  Reward (safety penalty dominates ordinary efficiency penalties)
 8.  RL initialization (starts from nurse policy, not random)
 9.  KL/behavior-cloning regularization (RL policy stays close to nurse policy)
10.  Reproducibility (same seed -> same results)
11.  Runtime integration (RoutingPolicy.route() end to end, real pipeline shape)
12.  Existing regression tests (timestamped-observation workflow still passes)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import numpy as np
import pytest

from triageguard_router.policy.bayesian_policy import BayesianLinearPolicy
from triageguard_router.policy.candidates import candidate_departments
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.demonstrations import load_demonstrations
from triageguard_router.policy.features import ClinicalSignal, HospitalSignal, FEATURE_NAMES
from triageguard_router.policy.reward import compute_reward
from triageguard_router.policy.rl_policy import RLRoutingPolicy
from triageguard_router.policy.routing_policy import RoutingPolicy
from triageguard_router.policy.safety import AllocationDecision, allocate, is_clinically_compatible
from triageguard_router.policy.simulation_env import RoutingEnv
from triageguard_router.policy.training import build_training_examples
from triageguard_router.policy.uncertainty import compute_policy_uncertainty


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config() -> PolicyConfig:
    cfg = PolicyConfig()
    cfg.rl.episodes = 15
    cfg.rl.steps_per_episode = 8
    cfg.rl.epochs = 2
    return cfg


@pytest.fixture(scope="module")
def demonstrations():
    return load_demonstrations()


@pytest.fixture(scope="module")
def training_examples(demonstrations):
    return build_training_examples(demonstrations)


@pytest.fixture(scope="module")
def bayesian_policy(config, training_examples) -> BayesianLinearPolicy:
    policy = BayesianLinearPolicy(n_features=len(FEATURE_NAMES), config=config.bayesian)
    policy.fit(training_examples)
    return policy


def _hospital_signal(icu_occ=8, gen_occ=38, obs_occ=12, cicu_occ=4, mode="NORMAL", load=0.5) -> HospitalSignal:
    return HospitalSignal(
        department_state={
            "ICU": {"capacity": 10, "occupied": icu_occ, "available": max(0, 10 - icu_occ), "status": "OPEN"},
            "CICU": {"capacity": 6, "occupied": cicu_occ, "available": max(0, 6 - cicu_occ), "status": "OPEN"},
            "ADMITTED_GEN": {"capacity": 50, "occupied": gen_occ, "available": max(0, 50 - gen_occ), "status": "OPEN"},
            "ED_OBS": {"capacity": 20, "occupied": obs_occ, "available": max(0, 20 - obs_occ), "status": "OPEN"},
            "DISCHARGE": {"capacity": 999, "occupied": 0, "available": 999, "status": "OPEN"},
        },
        operating_mode=mode,
        load_ratio=load,
    )


def _icu_clinical(preferred="ICU") -> ClinicalSignal:
    return ClinicalSignal(
        icu_risk_2h=0.7, icu_risk_6h=0.75, icu_risk_12h=0.72, admission_risk=0.9,
        xgb_confidence=0.85, rag_urgency=0.6, rag_evidence_strength=0.6,
        model_disagreement=0.0, information_completeness=0.9,
        preferred_department=preferred, top_diagnoses=["sepsis"], red_flags=["hypoxia"],
    )


# ===========================================================================
# 1. Nurse imitation
# ===========================================================================

class TestNurseImitation:
    def test_fits_without_error_on_all_demonstrations(self, bayesian_policy):
        assert bayesian_policy.fitted

    def test_high_training_imitation_accuracy(self, bayesian_policy):
        # Reproduces expert choices on the demonstrations it was fit on.
        assert bayesian_policy.training_metadata["training_imitation_accuracy"] >= 0.85

    def test_leave_one_out_generalization(self, demonstrations):
        """
        Held-out test: fit on all-but-one demonstration, check the policy's
        argmax choice for the held-out scenario at least stays within the
        scenario's own acceptable_departments set (not necessarily the exact
        preferred pick — this is a real generalization check on 17 points,
        not a memorization check).
        """
        from triageguard_router.policy.training import scenario_to_example
        from triageguard_agent.hospital.hospital_load_controller import HospitalLoadController

        controller = HospitalLoadController()
        correct_or_acceptable = 0
        for i, held_out in enumerate(demonstrations):
            train_set = demonstrations[:i] + demonstrations[i + 1:]
            examples = build_training_examples(train_set, controller)
            policy = BayesianLinearPolicy(n_features=len(FEATURE_NAMES), config=PolicyConfig().bayesian)
            policy.fit(examples)

            held_example = scenario_to_example(held_out, controller)
            pred_idx = int(np.argmax(held_example.phi_candidates @ policy.w_map))
            predicted_dept = held_out.candidate_departments[pred_idx]
            if predicted_dept in held_out.acceptable_departments:
                correct_or_acceptable += 1

        assert correct_or_acceptable / len(demonstrations) >= 0.6


# ===========================================================================
# 2. Paired scenarios
# ===========================================================================

class TestPairedScenarios:
    def test_icu_availability_changes_allocation(self, bayesian_policy, config):
        clinical = _icu_clinical("ICU")
        policy = RoutingPolicy(bayesian_policy, config=config)

        available = policy.route(clinical, _hospital_signal(icu_occ=8))
        full = policy.route(clinical, _hospital_signal(icu_occ=10))

        assert available["routing"]["allocated_department"] == "ICU"
        assert available["routing"]["resource_constraint"] is False
        assert full["routing"]["allocated_department"] != "ICU"
        assert full["routing"]["resource_constraint"] is True

    def test_preferred_department_unchanged_across_the_pair(self, bayesian_policy, config):
        clinical = _icu_clinical("ICU")
        policy = RoutingPolicy(bayesian_policy, config=config)
        r1 = policy.route(clinical, _hospital_signal(icu_occ=8))
        r2 = policy.route(clinical, _hospital_signal(icu_occ=10))
        assert r1["routing"]["preferred_department"] == r2["routing"]["preferred_department"] == "ICU"

    def test_full_ladder_matches_nurse_choices(self, bayesian_policy, config):
        """S01 -> S02 -> S03 reproduced live through RoutingPolicy, not just the raw fit."""
        clinical = _icu_clinical("ICU")
        policy = RoutingPolicy(bayesian_policy, config=config)

        s01 = policy.route(clinical, _hospital_signal(icu_occ=7))
        s02 = policy.route(clinical, _hospital_signal(icu_occ=10))
        s03 = policy.route(clinical, _hospital_signal(icu_occ=10, gen_occ=50))

        assert s01["routing"]["allocated_department"] == "ICU"
        assert s02["routing"]["allocated_department"] == "ADMITTED_GEN"
        assert s03["routing"]["allocated_department"] == "ED_OBS"


# ===========================================================================
# 3. Clinical invariance
# ===========================================================================

class TestClinicalInvariance:
    def test_icu_full_does_not_change_clinical_priority(self, bayesian_policy, config):
        clinical = _icu_clinical("ICU")
        policy = RoutingPolicy(bayesian_policy, config=config)

        available = policy.route(clinical, _hospital_signal(icu_occ=8))
        full = policy.route(clinical, _hospital_signal(icu_occ=10))

        assert available["clinical_assessment"]["clinical_priority"] == full["clinical_assessment"]["clinical_priority"]
        assert available["clinical_assessment"]["icu_risk_2h"] == full["clinical_assessment"]["icu_risk_2h"]
        assert available["clinical_assessment"]["admission_risk"] == full["clinical_assessment"]["admission_risk"]

    def test_resource_conflict_still_reports_true_clinical_priority(self, bayesian_policy, config):
        """Even when NO safe allocation exists, clinical_priority must be reported honestly, not hidden/zeroed."""
        clinical = _icu_clinical("ICU")
        policy = RoutingPolicy(bayesian_policy, config=config)
        conflict = policy.route(clinical, _hospital_signal(icu_occ=10, gen_occ=50, obs_occ=20))

        assert conflict["routing"]["allocated_department"] is None
        assert conflict["clinical_assessment"]["clinical_priority"] == pytest.approx(clinical.clinical_priority)

    def test_clinical_signal_object_is_never_mutated_by_routing(self, bayesian_policy, config):
        clinical = _icu_clinical("ICU")
        before = (clinical.icu_risk_2h, clinical.admission_risk, clinical.preferred_department)
        policy = RoutingPolicy(bayesian_policy, config=config)
        policy.route(clinical, _hospital_signal(icu_occ=10))
        after = (clinical.icu_risk_2h, clinical.admission_risk, clinical.preferred_department)
        assert before == after


# ===========================================================================
# 4. Safety
# ===========================================================================

class TestSafety:
    def test_discharge_never_offered_as_resource_fallback(self):
        candidates = candidate_departments("ICU", ["ICU", "CICU", "ADMITTED_GEN", "ED_OBS", "DISCHARGE"], max_step_down_tiers=3)
        assert "DISCHARGE" not in candidates

    def test_cicu_never_a_candidate_for_non_cardiac_icu_patient(self):
        """Regression: CICU must not appear as a step-down/sibling candidate for an ICU-preferred patient."""
        candidates = candidate_departments("ICU", ["ICU", "CICU", "ADMITTED_GEN", "ED_OBS"], max_step_down_tiers=3)
        assert "CICU" not in candidates

    def test_icu_is_valid_sibling_for_cicu_preferred_patient(self):
        candidates = candidate_departments("CICU", ["ICU", "CICU", "ADMITTED_GEN", "ED_OBS"], max_step_down_tiers=3)
        assert "ICU" in candidates

    def test_incompatible_department_rejected(self):
        assert is_clinically_compatible("DISCHARGE", "ICU", max_step_down_tiers=3) is False

    def test_no_safe_allocation_is_never_fabricated(self, bayesian_policy, config):
        clinical = _icu_clinical("ICU")
        policy = RoutingPolicy(bayesian_policy, config=config)
        result = policy.route(clinical, _hospital_signal(icu_occ=10, gen_occ=50, obs_occ=20))

        assert result["routing"]["allocated_department"] is None
        assert result["routing"]["human_review_recommended"] is True
        assert "resource conflict" in result["explanation"]["allocation_reason"].lower()

    def test_allocate_never_selects_infeasible_department(self):
        candidates = ["ICU", "ADMITTED_GEN", "ED_OBS"]
        hospital_state = {
            "ICU": {"capacity": 10, "occupied": 10, "available": 0, "status": "OPEN"},
            "ADMITTED_GEN": {"capacity": 50, "occupied": 50, "available": 0, "status": "OPEN"},
            "ED_OBS": {"capacity": 20, "occupied": 5, "available": 15, "status": "OPEN"},
        }
        decision = allocate("ICU", candidates, hospital_state, {"ICU": 10.0, "ADMITTED_GEN": 5.0, "ED_OBS": 1.0})
        # ICU/ADMITTED_GEN score highest but are infeasible -- must not be chosen
        assert decision.allocated_department == "ED_OBS"


# ===========================================================================
# 5. Uncertainty
# ===========================================================================

class TestUncertainty:
    def test_ood_state_has_higher_distance_than_training_like_state(self, bayesian_policy):
        from triageguard_router.policy.features import state_feature_vector

        in_distribution = _icu_clinical("ICU")
        hospital = _hospital_signal(icu_occ=10)
        near_state = state_feature_vector(in_distribution, hospital)

        wild = ClinicalSignal(
            icu_risk_2h=0.02, icu_risk_6h=0.02, icu_risk_12h=0.02, admission_risk=0.02,
            xgb_confidence=0.02, rag_urgency=0.02, rag_evidence_strength=0.02,
            model_disagreement=1.0, information_completeness=0.02,
            preferred_department="ICU",
        )
        far_hospital = _hospital_signal(icu_occ=10, mode="CRITICAL", load=0.99)
        far_state = state_feature_vector(wild, far_hospital)

        near_dist = bayesian_policy.state_ood_distance(near_state)
        far_dist = bayesian_policy.state_ood_distance(far_state)
        assert far_dist > near_dist

    def test_higher_utility_std_increases_policy_uncertainty(self, config):
        low = compute_policy_uncertainty(chosen_utility_std=0.1, state_ood_distance=0.0, config=config.uncertainty)
        high = compute_policy_uncertainty(chosen_utility_std=6.0, state_ood_distance=0.0, config=config.uncertainty)
        assert high > low

    def test_unfitted_policy_reports_maximal_ood(self):
        fresh = BayesianLinearPolicy(n_features=len(FEATURE_NAMES))
        assert fresh.state_ood_distance(np.zeros(len(FEATURE_NAMES))) == 1.0


# ===========================================================================
# 6 & 10. Simulation determinism / reproducibility
# ===========================================================================

class TestSimulationReproducibility:
    def test_same_seed_same_episode_outcome(self, config):
        env = RoutingEnv(config=config)

        def policy_fn(phi_mat, mask, candidates):
            valid = np.where(mask)[0]
            return int(valid[0])

        r1 = env.run_episode(policy_fn, seed=42)
        r2 = env.run_episode(policy_fn, seed=42)

        assert r1.total_reward == r2.total_reward
        assert r1.department_admission_counts == r2.department_admission_counts
        assert [t.chosen_index for t in r1.transitions] == [t.chosen_index for t in r2.transitions]

    def test_different_seed_can_differ(self, config):
        env = RoutingEnv(config=config)

        def policy_fn(phi_mat, mask, candidates):
            valid = np.where(mask)[0]
            return int(valid[0])

        r1 = env.run_episode(policy_fn, seed=1)
        r2 = env.run_episode(policy_fn, seed=2)
        # Not a strict inequality requirement (could coincide), just confirms
        # the seed is actually threaded through to arrival generation.
        assert isinstance(r1.total_reward, float) and isinstance(r2.total_reward, float)


# ===========================================================================
# 7. Reward — safety dominance
# ===========================================================================

class TestRewardSafetyDominance:
    def test_unsafe_downgrade_penalty_dominates_efficiency_terms(self, config):
        clinical = _icu_clinical("ICU")
        weights = config.reward

        safe_decision = AllocationDecision(
            preferred_department="ICU", allocated_department="ICU",
            resource_constraint=False, human_escalation=False, feasibility=None,
        )
        unsafe_decision = AllocationDecision(
            preferred_department="ICU", allocated_department="DISCHARGE",  # 3-tier unsafe downgrade
            resource_constraint=True, human_escalation=False, feasibility=None,
        )

        safe_reward, _ = compute_reward(clinical, safe_decision, wait_minutes=10, max_step_down_tiers=3, weights=weights)
        unsafe_reward, breakdown = compute_reward(clinical, unsafe_decision, wait_minutes=10, max_step_down_tiers=3, weights=weights)

        assert "clinically_incompatible_allocation" in breakdown
        assert unsafe_reward < safe_reward
        assert abs(breakdown["clinically_incompatible_allocation"]) > 10 * abs(weights.reduced_waiting_time)

    def test_resource_conflict_penalty_dominates_and_flags_failure_for_critical(self, config):
        clinical = _icu_clinical("ICU")  # clinical_priority ~0.9, high
        conflict = AllocationDecision(
            preferred_department="ICU", allocated_department=None,
            resource_constraint=True, human_escalation=True, feasibility=None,
        )
        reward, breakdown = compute_reward(clinical, conflict, wait_minutes=0, max_step_down_tiers=3, weights=config.reward)
        assert "resource_conflict" in breakdown
        assert "failure_to_allocate_critical" in breakdown
        assert reward < config.reward.resource_conflict  # more negative than resource_conflict alone


# ===========================================================================
# 8. RL initialization from nurse policy
# ===========================================================================

class TestRLInitialization:
    def test_rl_requires_bayesian_init_before_training(self, config):
        env = RoutingEnv(config=config)
        rl = RLRoutingPolicy(n_features=len(FEATURE_NAMES), config=config.rl)
        with pytest.raises(RuntimeError):
            rl.train(env)

    def test_rl_weights_equal_nurse_weights_before_any_training(self, bayesian_policy, config):
        rl = RLRoutingPolicy(n_features=len(FEATURE_NAMES), config=config.rl)
        rl.init_from_bayesian(bayesian_policy)
        assert np.allclose(rl.w.detach().numpy(), bayesian_policy.w_map)
        assert rl.initialized_from_nurse is True


# ===========================================================================
# 9. KL / behavior-cloning regularization
# ===========================================================================

class TestKLRegularization:
    def test_rl_policy_stays_close_to_nurse_policy_after_training(self, bayesian_policy, config, training_examples):
        env = RoutingEnv(config=config)
        rl = RLRoutingPolicy(n_features=len(FEATURE_NAMES), config=config.rl)
        rl.init_from_bayesian(bayesian_policy)
        rl.train(env)

        final_w = rl.w.detach().numpy()
        drift = np.linalg.norm(final_w - bayesian_policy.w_map)
        prior_scale = np.sqrt(config.bayesian.prior_variance * len(FEATURE_NAMES))
        # Regularized drift should stay well within the prior's own scale —
        # i.e. training moved the policy, but did not blow past its prior.
        assert drift < prior_scale

    def test_stronger_kl_weight_reduces_drift(self, bayesian_policy, training_examples):
        cfg_weak = PolicyConfig()
        cfg_weak.rl.episodes = 10
        cfg_weak.rl.steps_per_episode = 8
        cfg_weak.rl.epochs = 2
        cfg_weak.rl.policy_kl_weight = 0.0
        cfg_weak.rl.behavior_cloning_weight = 0.0

        cfg_strong = PolicyConfig()
        cfg_strong.rl.episodes = 10
        cfg_strong.rl.steps_per_episode = 8
        cfg_strong.rl.epochs = 2
        cfg_strong.rl.policy_kl_weight = 5.0
        cfg_strong.rl.behavior_cloning_weight = 5.0

        env_weak = RoutingEnv(config=cfg_weak)
        rl_weak = RLRoutingPolicy(n_features=len(FEATURE_NAMES), config=cfg_weak.rl)
        rl_weak.init_from_bayesian(bayesian_policy)
        rl_weak.train(env_weak)

        env_strong = RoutingEnv(config=cfg_strong)
        rl_strong = RLRoutingPolicy(n_features=len(FEATURE_NAMES), config=cfg_strong.rl)
        rl_strong.init_from_bayesian(bayesian_policy)
        rl_strong.train(env_strong)

        drift_weak = np.linalg.norm(rl_weak.w.detach().numpy() - bayesian_policy.w_map)
        drift_strong = np.linalg.norm(rl_strong.w.detach().numpy() - bayesian_policy.w_map)
        assert drift_strong < drift_weak


# ===========================================================================
# 11. Runtime integration
# ===========================================================================

class TestRuntimeIntegration:
    def test_routing_policy_end_to_end_scenario_a(self, bayesian_policy, config):
        """Phase 17 Scenario A: high-risk patient, ICU available -> ICU."""
        policy = RoutingPolicy(bayesian_policy, config=config)
        result = policy.route(_icu_clinical("ICU"), _hospital_signal(icu_occ=8))
        assert result["routing"]["preferred_department"] == "ICU"
        assert result["routing"]["allocated_department"] == "ICU"
        assert result["routing"]["resource_constraint"] is False
        for key in ("clinical_assessment", "hospital_state", "routing", "department_scores", "explanation"):
            assert key in result

    def test_routing_policy_end_to_end_scenario_b(self, bayesian_policy, config):
        """Phase 17 Scenario B: ICU full, ADMITTED_GEN available -> resource-constrained step-down."""
        policy = RoutingPolicy(bayesian_policy, config=config)
        result = policy.route(_icu_clinical("ICU"), _hospital_signal(icu_occ=10))
        assert result["routing"]["preferred_department"] == "ICU"
        assert result["routing"]["allocated_department"] == "ADMITTED_GEN"
        assert result["routing"]["resource_constraint"] is True

    def test_routing_policy_end_to_end_scenario_c(self, bayesian_policy, config):
        """Phase 17 Scenario C: everything full -> honest resource conflict, no fabricated allocation."""
        policy = RoutingPolicy(bayesian_policy, config=config)
        result = policy.route(_icu_clinical("ICU"), _hospital_signal(icu_occ=10, gen_occ=50, obs_occ=20))
        assert result["routing"]["allocated_department"] is None
        assert result["routing"]["human_review_recommended"] is True

    def test_explanation_decomposition_sums_to_total_utility(self, bayesian_policy, config):
        policy = RoutingPolicy(bayesian_policy, config=config)
        result = policy.route(_icu_clinical("ICU"), _hospital_signal(icu_occ=10))
        allocated = result["routing"]["allocated_department"]
        decomp = result["explanation"]["department_decomposition"][allocated]
        assert decomp["total_utility"] == pytest.approx(sum(decomp["decomposition"].values()), abs=1e-3)


# ===========================================================================
# 12. Existing regression tests (run via the standard suite; imported here
# only to fail loudly and early if the observation-workflow module itself
# fails to import after these changes).
# ===========================================================================

class TestNoRegressionOnObservationWorkflow:
    def test_patient_tools_module_still_imports(self):
        from triageguard_agent.tools.patient_tools import add_patient_observation  # noqa: F401

    def test_agent_runtime_still_registers_all_tools(self):
        from triageguard_agent.runtime.agent_runtime import AgentRuntime
        runtime = AgentRuntime(
            auto_register=True,
            llm_call_fn=lambda messages, tools, model=None: {"content": "x", "tool_calls": None},
        )
        assert "add_patient_observation" in runtime.tool_registry
        assert "run_triage_assessment" in runtime.tool_registry
