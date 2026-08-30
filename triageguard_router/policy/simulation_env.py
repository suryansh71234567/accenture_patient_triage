"""
simulation_env.py
------------------
RL environment for policy optimization (Phase 7/8), built ENTIRELY on top of
the existing triageguard_agent.simulation machinery
(HospitalSimulator/PatientFlowManager/EventEngine/HospitalLoadController) —
no second simulator is implemented here.

Per-episode rollout:
    reset a fresh HospitalSimulator (deterministic under a fixed seed)
        -> repeatedly: simulator.step() (existing arrivals/LOS/bed-release)
        -> for each newly arrived patient: clinical assessment via the
           existing calibrated fallback rule (fast/offline/reproducible —
           see hospital_simulator.calibrated_clinical_fallback; the live
           network-attached XGBoost+RAG pipeline is used at REAL runtime,
           see routing_policy.py, not for RL training rollouts)
        -> candidate set + feasibility mask (candidates.py / safety.py)
        -> policy selects a department among the FEASIBLE candidates
        -> simulator.admit_patient() actually occupies the bed (existing
           method — reuses admission/LOS/discharge machinery unchanged)
        -> reward.compute_reward() scores the decision

Each patient-allocation decision is one RL "step". The env exposes
transitions as (phi_candidates, action_mask, chosen_index, reward) tuples
for rl_policy.py to consume.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from triageguard_agent.hospital.hospital_load_controller import HospitalLoadController
from triageguard_agent.hospital.hospital_state_service import HospitalStateService
from triageguard_agent.simulation.hospital_simulator import HospitalSimulator, calibrated_clinical_fallback
from triageguard_router.policy.candidates import candidate_departments
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.features import ClinicalSignal, HospitalSignal, phi_matrix
from triageguard_router.policy.reward import compute_reward
from triageguard_router.policy.safety import compute_feasibility

_ESCALATION_BY_ACUITY = {1: 1.0, 2: 0.8, 3: 0.4, 4: 0.2, 5: 0.1}


def clinical_signal_from_fallback(fallback: Dict[str, Any]) -> ClinicalSignal:
    """
    Adapt HospitalSimulator.calibrated_clinical_fallback()'s output (a
    run_triage_assessment-shaped dict, but with a single risk number rather
    than per-horizon icu_risk_2h/6h/12h) into a ClinicalSignal. The three
    horizons are set equal since the rule-based fallback does not model
    time-horizon-specific risk — documented simplification, RL-training-only
    (the real runtime path uses the real per-horizon XGBoost output).
    """
    icu_risk = float(fallback["reconciled_icu_risk"])
    admission_risk = float(fallback["reconciled_admission_risk"])
    acuity = int(fallback.get("acuity_tier", 3))
    top_diagnoses = fallback.get("top_diagnoses", [])
    red_flags = fallback.get("red_flags", [])
    return ClinicalSignal(
        icu_risk_2h=icu_risk,
        icu_risk_6h=icu_risk,
        icu_risk_12h=icu_risk,
        admission_risk=admission_risk,
        xgb_confidence=0.8,           # rule-based fallback treated as moderately confident
        rag_urgency=_ESCALATION_BY_ACUITY.get(acuity, 0.4),
        rag_evidence_strength=min(1.0, (len(top_diagnoses) + len(red_flags)) / 6.0),
        model_disagreement=0.0,
        information_completeness=1.0,
        preferred_department=fallback["department"],
        top_diagnoses=top_diagnoses,
        red_flags=red_flags,
    )


@dataclass
class Transition:
    phi_candidates: np.ndarray       # (n_candidates, n_features)
    candidates: List[str]
    action_mask: np.ndarray          # bool (n_candidates,)
    chosen_index: int
    reward: float
    reward_breakdown: Dict[str, float]
    resource_constraint: bool
    human_escalation: bool


@dataclass
class EpisodeResult:
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    unsafe_allocation_count: int = 0
    resource_conflict_count: int = 0
    high_priority_wait_minutes: List[float] = field(default_factory=list)
    department_admission_counts: Dict[str, int] = field(default_factory=dict)


class RoutingEnv:
    """
    A thin RL wrapper around HospitalSimulator. `policy_fn` is any callable
    (phi_candidates, action_mask, candidate_department_names) -> chosen_index
    — this env does not know or care whether that's the Bayesian policy, the
    RL policy being trained, or the plain heuristic baseline (evaluate.py
    uses all three for the ablation study).
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.load_controller = HospitalLoadController()

    def run_episode(
        self,
        policy_fn,
        scenario_name: str = "NORMAL_DAY",
        seed: Optional[int] = None,
    ) -> EpisodeResult:
        cfg = self.config
        seed = self.config.random_seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)  # RLRoutingPolicy.select_action samples via np.random — must be seeded too

        HospitalStateService.reset_instance()
        simulator = HospitalSimulator(scenario=scenario_name)

        result = EpisodeResult()

        for _ in range(cfg.rl.steps_per_episode):
            step_info = simulator.step(minutes=cfg.rl.minutes_per_step, auto_generate_arrivals=True)

            waiting = list(simulator.patient_flow.peek_waiting(count=step_info["new_arrivals_count"] + 5))
            # Only process patients that just arrived and are still waiting —
            # avoids double-processing a patient already allocated earlier.
            to_process = [p for p in waiting if p.status.value == "ARRIVED"]

            for patient in to_process:
                # Left in _waiting_queue for now: HospitalSimulator.admit_patient()
                # looks the patient up via patient_flow.get_patient(), which
                # searches the waiting queue. Removed explicitly below, after
                # admission (or immediately on a resource conflict) — the
                # ARRIVED-status filter above already prevents re-processing
                # an admitted patient on a later step even before removal.
                fallback = calibrated_clinical_fallback(patient.acuity, patient.chief_complaint, patient.metadata)
                clinical = clinical_signal_from_fallback(fallback)

                dept_states = simulator.state_service.get_all()
                load_ratio = self.load_controller.calculate_load(dept_states)
                operating_mode = self.load_controller.calculate_operating_mode(load_ratio)
                hospital = HospitalSignal(department_state=dept_states, operating_mode=operating_mode, load_ratio=load_ratio)

                candidates = candidate_departments(
                    clinical.preferred_department, dept_states.keys(), cfg.max_step_down_tiers
                )
                if not candidates:
                    continue
                feasibility = compute_feasibility(candidates, dept_states)
                phi_mat = phi_matrix(candidates, clinical, hospital)

                if feasibility.resource_conflict:
                    chosen_index = None
                    allocated_dept = None
                else:
                    chosen_index = policy_fn(phi_mat, feasibility.action_mask, candidates)
                    allocated_dept = candidates[chosen_index]

                from triageguard_router.policy.safety import AllocationDecision
                allocation = AllocationDecision(
                    preferred_department=clinical.preferred_department,
                    allocated_department=allocated_dept,
                    resource_constraint=(allocated_dept != clinical.preferred_department),
                    human_escalation=feasibility.resource_conflict,
                    feasibility=feasibility,
                )

                wait_minutes = float(cfg.rl.minutes_per_step)
                reward, breakdown = compute_reward(
                    clinical, allocation, wait_minutes, cfg.max_step_down_tiers, cfg.reward
                )

                if allocated_dept is not None:
                    simulator.admit_patient(patient.patient_id, department=allocated_dept)
                    result.department_admission_counts[allocated_dept] = (
                        result.department_admission_counts.get(allocated_dept, 0) + 1
                    )
                else:
                    result.resource_conflict_count += 1

                if patient in simulator.patient_flow._waiting_queue:
                    simulator.patient_flow._waiting_queue.remove(patient)

                if breakdown.get("unsafe_downgrade") or breakdown.get("clinically_incompatible_allocation"):
                    result.unsafe_allocation_count += 1
                if clinical.clinical_priority >= 0.6:
                    result.high_priority_wait_minutes.append(wait_minutes)

                result.total_reward += reward
                if chosen_index is not None:
                    result.transitions.append(Transition(
                        phi_candidates=phi_mat,
                        candidates=candidates,
                        action_mask=feasibility.action_mask,
                        chosen_index=chosen_index,
                        reward=reward,
                        reward_breakdown=breakdown,
                        resource_constraint=allocation.resource_constraint,
                        human_escalation=allocation.human_escalation,
                    ))

        return result
