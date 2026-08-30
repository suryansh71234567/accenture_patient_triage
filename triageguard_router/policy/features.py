"""
features.py
-----------
Compact, interpretable feature extraction (Phase 3).

Every feature is derived from EXISTING pipeline output
(triageguard_router.reconciler.reconcile() / xgb predictor output) and
EXISTING hospital state (triageguard_agent.hospital.hospital_state_store's
{capacity, occupied, available, status} dict) — nothing here recomputes or
duplicates clinical risk, and nothing here is a patient ID / encounter ID /
opaque embedding.

Two feature groups:
  * STATE features  — depend only on the clinical assessment + overall
    hospital load, not on which candidate department is being scored.
  * ACTION features — depend on a specific candidate department `a`
    (its occupancy, its acuity distance from the clinically preferred
    department, and a one-hot identity so the linear model can learn
    department-specific baseline preferences).

phi(s, a) = STATE features ++ ACTION features(a), a single flat vector, the
same for the Bayesian policy and the RL policy (both are linear/shallow
utility functions over this vector — see bayesian_policy.py / rl_policy.py).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from triageguard_router.policy.candidates import acuity_tier, _KNOWN_ACUITY_ORDER

# Escalation-level -> 0-1 score, matching reconciler._ESCALATION_SCORE exactly
# (reused as a value convention, not reimplemented — reconciler already
# returns this under "rag_escalation_score" in its own output dict, so
# callers normally just read that field rather than re-deriving it here).
_ESCALATION_SCORE = {"emergent": 1.0, "urgent": 0.6, "routine": 0.2, "unknown": 0.4}

_OPERATING_MODE_PRESSURE = {"NORMAL": 0.0, "HIGH_LOAD": 0.5, "CRITICAL": 1.0}

_ONE_HOT_DEPARTMENTS = [d for group in _KNOWN_ACUITY_ORDER for d in group]  # CICU,ICU,ADMITTED_GEN,ED_OBS,DISCHARGE

STATE_FEATURE_NAMES = [
    "bias",
    "icu_risk_2h",
    "icu_risk_6h",
    "icu_risk_12h",
    "admission_risk",
    "xgb_confidence",
    "rag_urgency",
    "rag_evidence_strength",
    "model_disagreement",
    "information_completeness",
    "operating_mode_pressure",
    "overall_load_ratio",
]

ACTION_FEATURE_NAMES = [
    "is_preferred_department",
    "acuity_gap",
    "occupancy_ratio",
    "has_availability",
    "available_beds_normalized",
] + [f"dept_is_{d}" for d in _ONE_HOT_DEPARTMENTS] + ["dept_is_other"]

FEATURE_NAMES = STATE_FEATURE_NAMES + ACTION_FEATURE_NAMES


@dataclass
class ClinicalSignal:
    """
    The clinical half of phi(s,a) — built once per patient from reconciler /
    xgb output, reused unchanged across every candidate department. This is
    also where clinical_priority (Phase 14) is defined; nothing downstream
    of this dataclass's construction can change these numbers.
    """
    icu_risk_2h: float
    icu_risk_6h: float
    icu_risk_12h: float
    admission_risk: float
    xgb_confidence: float
    rag_urgency: float             # 0-1
    rag_evidence_strength: float   # 0-1
    model_disagreement: float      # 0 or 1
    information_completeness: float
    preferred_department: str
    top_diagnoses: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)

    @property
    def clinical_priority(self) -> float:
        """
        Single-scalar clinical urgency (Phase 14's clinical_priority field):
        the worst-case near-term risk across ICU horizons and admission —
        i.e. "how clinically urgent is this patient, at minimum." Purely a
        read of existing reconciled/xgb numbers; never adjusted by resource
        availability anywhere in this module or downstream.
        """
        return max(self.icu_risk_2h, self.icu_risk_6h, self.icu_risk_12h, self.admission_risk)

    @classmethod
    def from_pipeline_output(
        cls,
        reconciled: Dict[str, Any],
        xgb_output: Dict[str, Any],
        preferred_department: str,
        rag_history_count: int = 0,
        rag_similar_count: int = 0,
    ) -> "ClinicalSignal":
        """
        Build from the REAL outputs of reconciler.reconcile() and
        TriageGuardPredictor.predict() (the same dicts combined_pipeline.py
        already produces) — no recomputation of clinical numbers happens here.
        """
        top_diagnoses = reconciled.get("top_diagnoses", [])
        red_flags = reconciled.get("red_flags", [])
        # Evidence strength: a real, bounded count of supporting signal —
        # diagnoses + red flags + retrieved history/similar-case documents —
        # normalized to [0,1]. Not fabricated: every term is a real count
        # from real RAG/XGBoost output.
        evidence_count = len(top_diagnoses) + len(red_flags) + rag_history_count + rag_similar_count
        rag_evidence_strength = min(1.0, evidence_count / 6.0)

        return cls(
            icu_risk_2h=float(xgb_output.get("icu_risk_2h", 0.0)),
            icu_risk_6h=float(xgb_output.get("icu_risk_6h", 0.0)),
            icu_risk_12h=float(xgb_output.get("icu_risk_12h", 0.0)),
            admission_risk=float(reconciled.get("reconciled_admission_risk", xgb_output.get("admission_risk", 0.0))),
            xgb_confidence=float(xgb_output.get("admission_risk_confidence", 0.5)),
            rag_urgency=float(reconciled.get("rag_escalation_score", 0.4)),
            rag_evidence_strength=rag_evidence_strength,
            model_disagreement=0.0 if reconciled.get("branches_agree", True) else 1.0,
            information_completeness=float(reconciled.get("information_completeness", 0.5)),
            preferred_department=preferred_department,
            top_diagnoses=top_diagnoses,
            red_flags=red_flags,
        )


@dataclass
class HospitalSignal:
    """The hospital-state half of phi(s,a)."""
    department_state: Dict[str, Dict[str, Any]]   # {dept: {capacity, occupied, available, status}}
    operating_mode: str
    load_ratio: float


def state_feature_vector(clinical: ClinicalSignal, hospital: HospitalSignal) -> np.ndarray:
    """The STATE half of phi(s,a) — identical across all candidate departments."""
    return np.array([
        1.0,  # bias
        clinical.icu_risk_2h,
        clinical.icu_risk_6h,
        clinical.icu_risk_12h,
        clinical.admission_risk,
        clinical.xgb_confidence,
        clinical.rag_urgency,
        clinical.rag_evidence_strength,
        clinical.model_disagreement,
        clinical.information_completeness,
        _OPERATING_MODE_PRESSURE.get(hospital.operating_mode, 0.5),
        hospital.load_ratio,
    ], dtype=np.float64)


def action_feature_vector(
    department: str,
    clinical: ClinicalSignal,
    hospital: HospitalSignal,
) -> np.ndarray:
    """The ACTION half of phi(s,a) — specific to one candidate department."""
    dept_state = hospital.department_state.get(department, {"capacity": 0, "occupied": 0, "available": 0, "status": "CLOSED"})
    capacity = max(0, int(dept_state.get("capacity", 0)))
    occupied = max(0, int(dept_state.get("occupied", 0)))
    available = max(0, int(dept_state.get("available", max(0, capacity - occupied))))
    occupancy_ratio = 0.0 if capacity <= 0 else min(1.0, occupied / capacity)
    is_open = dept_state.get("status", "OPEN") == "OPEN"
    has_availability = 1.0 if (is_open and available > 0) else 0.0
    available_norm = 0.0 if capacity <= 0 else min(1.0, available / capacity)

    is_preferred = 1.0 if department == clinical.preferred_department else 0.0
    acuity_gap = float(max(0, acuity_tier(department) - acuity_tier(clinical.preferred_department)))

    one_hot = [1.0 if department == d else 0.0 for d in _ONE_HOT_DEPARTMENTS]
    is_other = 1.0 if department not in _ONE_HOT_DEPARTMENTS else 0.0

    return np.array(
        [is_preferred, acuity_gap, occupancy_ratio, has_availability, available_norm]
        + one_hot + [is_other],
        dtype=np.float64,
    )


def phi(department: str, clinical: ClinicalSignal, hospital: HospitalSignal) -> np.ndarray:
    """Full feature vector phi(s,a) for one candidate department."""
    return np.concatenate([
        state_feature_vector(clinical, hospital),
        action_feature_vector(department, clinical, hospital),
    ])


def phi_matrix(
    departments: List[str],
    clinical: ClinicalSignal,
    hospital: HospitalSignal,
) -> np.ndarray:
    """phi(s,a) stacked for every candidate department, shape (n_candidates, n_features)."""
    return np.stack([phi(d, clinical, hospital) for d in departments])
