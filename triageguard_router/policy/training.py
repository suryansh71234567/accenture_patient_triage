"""
training.py
------------
Converts NurseScenario demonstrations (schema.py) into the
(ClinicalSignal, HospitalSignal, phi-per-candidate, chosen index) form
bayesian_policy.BayesianLinearPolicy.fit() consumes.

Reuses the REAL HospitalLoadController to derive operating_mode/load_ratio
from each scenario's hospital state, rather than re-deriving that logic here.
"""

from __future__ import annotations

from typing import Dict, List

from triageguard_agent.hospital.hospital_load_controller import HospitalLoadController
from triageguard_router.policy.bayesian_policy import DemonstrationExample
from triageguard_router.policy.features import ClinicalSignal, HospitalSignal, phi_matrix, state_feature_vector
from triageguard_router.policy.schema import ClinicalState, DepartmentState, NurseScenario

_ESCALATION_SCORE = {"emergent": 1.0, "urgent": 0.6, "routine": 0.2, "unknown": 0.4}


def department_state_dict(hospital_state: Dict[str, DepartmentState]) -> Dict[str, Dict]:
    return {
        dept: {
            "capacity": st.capacity,
            "occupied": st.occupied,
            "available": st.available,
            "status": st.status,
        }
        for dept, st in hospital_state.items()
    }


def to_clinical_signal(cs: ClinicalState, preferred_department: str) -> ClinicalSignal:
    evidence = min(1.0, cs.rag_evidence_strength / 6.0)
    return ClinicalSignal(
        icu_risk_2h=cs.icu_risk_2h,
        icu_risk_6h=cs.icu_risk_6h,
        icu_risk_12h=cs.icu_risk_12h,
        admission_risk=cs.admission_risk,
        xgb_confidence=cs.xgb_confidence,
        rag_urgency=_ESCALATION_SCORE.get(cs.rag_urgency, 0.4),
        rag_evidence_strength=evidence,
        model_disagreement=0.0 if cs.branches_agree else 1.0,
        information_completeness=cs.information_completeness,
        preferred_department=preferred_department,
        top_diagnoses=list(cs.top_diagnoses),
        red_flags=list(cs.red_flags),
    )


def to_hospital_signal(
    hospital_state: Dict[str, DepartmentState],
    load_controller: HospitalLoadController,
) -> HospitalSignal:
    state_dict = department_state_dict(hospital_state)
    load_ratio = load_controller.calculate_load(state_dict)
    operating_mode = load_controller.calculate_operating_mode(load_ratio)
    return HospitalSignal(department_state=state_dict, operating_mode=operating_mode, load_ratio=load_ratio)


def scenario_to_example(
    scenario: NurseScenario,
    load_controller: HospitalLoadController,
) -> DemonstrationExample:
    # is_preferred_department/acuity_gap features reference the CLINICAL
    # preference (constant across a ladder); the training LABEL is the
    # nurse's actual scenario-specific choice (may differ under constraint).
    clinical = to_clinical_signal(scenario.clinical_state, scenario.clinical_preferred_department)
    hospital = to_hospital_signal(scenario.hospital_state, load_controller)
    phi_mat = phi_matrix(scenario.candidate_departments, clinical, hospital)
    chosen_index = scenario.candidate_departments.index(scenario.preferred_department)
    state_vec = state_feature_vector(clinical, hospital)
    return DemonstrationExample(
        phi_candidates=phi_mat,
        candidate_departments=list(scenario.candidate_departments),
        chosen_index=chosen_index,
        state_features=state_vec,
    )


def build_training_examples(
    scenarios: List[NurseScenario],
    load_controller: HospitalLoadController | None = None,
) -> List[DemonstrationExample]:
    controller = load_controller or HospitalLoadController()
    return [scenario_to_example(s, controller) for s in scenarios]
