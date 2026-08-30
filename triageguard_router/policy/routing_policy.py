"""
routing_policy.py
------------------
Top-level orchestrator (Phase 13/14) — the ONLY entry point most callers
should need.

    reconciled clinical output (reconciler.reconcile())
            + xgb output (TriageGuardPredictor.predict())
            + router.route()'s preferred department
            + live hospital state (HospitalStateService)
        -> feature extraction (features.py)
        -> Bayesian policy scores (bayesian_policy.py) [or RL policy, if trained]
        -> uncertainty (uncertainty.py)
        -> hard safety / feasibility (safety.py)
        -> final allocation
        -> faithful explanation (explain.py)

Never recomputes or overrides clinical_priority / preferred_department —
those are read once from the existing router/reconciler output and passed
through unchanged into the final result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from triageguard_router.policy.candidates import candidate_departments
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.explain import explain_department
from triageguard_router.policy.features import ClinicalSignal, HospitalSignal, phi, state_feature_vector
from triageguard_router.policy.safety import allocate
from triageguard_router.policy.uncertainty import compute_policy_uncertainty, uncertainty_requires_review


class RoutingPolicy:
    """
    Wraps a fitted BayesianLinearPolicy (and, once trained, an RLRoutingPolicy)
    plus the safety/uncertainty layers into one callable that produces the
    Phase 14 structured result.
    """

    def __init__(
        self,
        bayesian_policy,               # BayesianLinearPolicy, fitted
        config: Optional[PolicyConfig] = None,
        rl_policy=None,                # Optional[RLRoutingPolicy], fitted
        use_rl: bool = False,
    ):
        self.bayesian_policy = bayesian_policy
        self.rl_policy = rl_policy
        self.use_rl = use_rl and rl_policy is not None
        self.config = config or PolicyConfig()

    def route(
        self,
        clinical: ClinicalSignal,
        hospital: HospitalSignal,
    ) -> Dict[str, Any]:
        preferred = clinical.preferred_department
        candidates = candidate_departments(
            preferred, hospital.department_state.keys(), self.config.max_step_down_tiers
        )

        if not candidates:
            return self._resource_conflict_result(clinical, hospital, candidates=[])

        bayesian_scores = self.bayesian_policy.predict(candidates, clinical, hospital)

        department_scores: Dict[str, float] = {}
        policy_probability: Dict[str, float] = {}
        utility_mean: Dict[str, float] = {}
        utility_std: Dict[str, float] = {}
        w = self.rl_policy.w.detach().numpy() if self.use_rl else self.bayesian_policy.w_map

        for dept in candidates:
            phi_vec = phi(dept, clinical, hospital)
            if self.use_rl:
                mean = float(phi_vec @ w)
                std = bayesian_scores[dept]["utility_std"]  # RL policy has no posterior; reuse Bayesian uncertainty
            else:
                mean = bayesian_scores[dept]["utility_mean"]
                std = bayesian_scores[dept]["utility_std"]
            utility_mean[dept] = mean
            utility_std[dept] = std
            policy_probability[dept] = bayesian_scores[dept]["probability"]
            department_scores[dept] = mean - self.config.bayesian.uncertainty_penalty_weight * std

        decision = allocate(preferred, candidates, hospital.department_state, department_scores)

        if decision.allocated_department is None:
            return self._resource_conflict_result(clinical, hospital, candidates, decision, department_scores)

        state_vec_ood = self.bayesian_policy.state_ood_distance(state_feature_vector(clinical, hospital))
        chosen_std = utility_std[decision.allocated_department]
        policy_uncertainty = compute_policy_uncertainty(chosen_std, state_vec_ood, self.config.uncertainty)
        needs_review = uncertainty_requires_review(policy_uncertainty, self.config.uncertainty)

        explanations = {}
        for dept in candidates:
            phi_vec = phi(dept, clinical, hospital)
            explanations[dept] = explain_department(
                phi_vec, w, utility_std[dept], self.config.bayesian.uncertainty_penalty_weight
            )

        allocation_reason = (
            f"{preferred} is available — direct allocation."
            if not decision.resource_constraint else
            f"{preferred} is at capacity; {decision.allocated_department} is the highest-scoring "
            f"clinically acceptable, currently available alternative."
        )
        policy_reason = (
            f"Learned routing policy scores {decision.allocated_department} highest "
            f"(utility={department_scores[decision.allocated_department]:.2f}) among feasible options "
            f"{decision.feasibility.feasible}."
        )

        return {
            "clinical_assessment": {
                "clinical_priority": round(clinical.clinical_priority, 4),
                "icu_risk_2h": clinical.icu_risk_2h,
                "icu_risk_6h": clinical.icu_risk_6h,
                "icu_risk_12h": clinical.icu_risk_12h,
                "admission_risk": clinical.admission_risk,
                "xgb_confidence": clinical.xgb_confidence,
                "rag_urgency": clinical.rag_urgency,
                "rag_evidence_strength": clinical.rag_evidence_strength,
                "model_disagreement": clinical.model_disagreement,
                "preferred_department": preferred,
            },
            "hospital_state": {
                "operating_mode": hospital.operating_mode,
                "load_ratio": hospital.load_ratio,
                **{f"{d.lower()}_available": hospital.department_state.get(d, {}).get("available", 0) for d in candidates},
            },
            "routing": {
                "preferred_department": preferred,
                "allocated_department": decision.allocated_department,
                "resource_constraint": decision.resource_constraint,
                "policy_confidence": round(policy_probability[decision.allocated_department], 4),
                "policy_uncertainty": round(policy_uncertainty, 4),
                "human_review_recommended": needs_review,
                "policy_source": "rl" if self.use_rl else "bayesian",
            },
            "department_scores": {d: round(v, 4) for d, v in department_scores.items()},
            "department_probabilities": {d: round(v, 4) for d, v in policy_probability.items()},
            "explanation": {
                "primary_reason": _clinical_reason(clinical),
                "allocation_reason": allocation_reason,
                "policy_reason": policy_reason,
                "department_decomposition": explanations,
            },
        }

    def _resource_conflict_result(
        self,
        clinical: ClinicalSignal,
        hospital: HospitalSignal,
        candidates: List[str],
        decision=None,
        department_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Phase 17 Scenario C: NEVER fabricate a safe allocation. Returns a
        structured escalation instead.
        """
        return {
            "clinical_assessment": {
                "clinical_priority": round(clinical.clinical_priority, 4),
                "icu_risk_2h": clinical.icu_risk_2h,
                "icu_risk_6h": clinical.icu_risk_6h,
                "icu_risk_12h": clinical.icu_risk_12h,
                "admission_risk": clinical.admission_risk,
                "xgb_confidence": clinical.xgb_confidence,
                "rag_urgency": clinical.rag_urgency,
                "rag_evidence_strength": clinical.rag_evidence_strength,
                "model_disagreement": clinical.model_disagreement,
                "preferred_department": clinical.preferred_department,
            },
            "hospital_state": {
                "operating_mode": hospital.operating_mode,
                "load_ratio": hospital.load_ratio,
            },
            "routing": {
                "preferred_department": clinical.preferred_department,
                "allocated_department": None,
                "resource_constraint": True,
                "policy_confidence": 0.0,
                "policy_uncertainty": 1.0,
                "human_review_recommended": True,
                "policy_source": "n/a",
            },
            "department_scores": department_scores or {},
            "department_probabilities": {},
            "explanation": {
                "primary_reason": _clinical_reason(clinical),
                "allocation_reason": (
                    "No candidate department is both clinically acceptable and currently "
                    "available — resource conflict. This is a genuine escalation, not a routing failure."
                ),
                "policy_reason": "No feasible action exists; the policy layer was not invoked for allocation.",
                "department_decomposition": {},
            },
        }


def _clinical_reason(clinical: ClinicalSignal) -> str:
    if clinical.clinical_priority >= 0.7:
        return f"High near-term risk (priority={clinical.clinical_priority:.2f}) makes {clinical.preferred_department} clinically preferred."
    if clinical.clinical_priority >= 0.35:
        return f"Moderate risk (priority={clinical.clinical_priority:.2f}) indicates admission-level care ({clinical.preferred_department})."
    return f"Low risk (priority={clinical.clinical_priority:.2f}) — {clinical.preferred_department} is clinically appropriate."
