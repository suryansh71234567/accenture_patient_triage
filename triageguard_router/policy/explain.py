"""
explain.py
----------
Faithful per-candidate utility decomposition (Phase 15).

The routing policy's utility is a plain linear model: U(s,a) = w^T phi(s,a).
That means the total is EXACTLY a sum of per-feature contributions
(w_i * phi_i) — this module just groups those real contributions into
human-readable buckets. Nothing here is invented or asked of an LLM; if the
feature set changes, the groups below are the only thing that needs
updating, and every number displayed is traceable back to a real weight and
a real feature value.

department_scores (used by safety.allocate() to break ties among feasible
candidates) are utility_mean - uncertainty_penalty_weight * utility_std —
a risk-averse (lower-confidence-bound) score — so "uncertainty adjustment"
here is a real, load-bearing part of the score, not cosmetic.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from triageguard_router.policy.features import FEATURE_NAMES

_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

_GROUPS: Dict[str, List[str]] = {
    "clinical_suitability": [
        "bias", "icu_risk_2h", "icu_risk_6h", "icu_risk_12h", "admission_risk",
        "xgb_confidence", "rag_urgency", "rag_evidence_strength",
        "model_disagreement", "information_completeness",
        "is_preferred_department", "acuity_gap",
    ],
    "resource_availability": ["occupancy_ratio", "has_availability", "available_beds_normalized"],
    "hospital_preference": [n for n in FEATURE_NAMES if n.startswith("dept_is_")],
    "queue_pressure": ["operating_mode_pressure", "overall_load_ratio"],
}


def explain_department(
    phi_vec: np.ndarray,
    w: np.ndarray,
    utility_std: float,
    uncertainty_penalty_weight: float,
) -> Dict[str, Any]:
    contributions = phi_vec * w  # elementwise — the real per-feature terms of w^T phi
    decomposition: Dict[str, float] = {}
    for group, names in _GROUPS.items():
        idx = [_FEATURE_INDEX[n] for n in names]
        decomposition[group] = round(float(contributions[idx].sum()), 4)

    uncertainty_adjustment = round(-uncertainty_penalty_weight * utility_std, 4)
    decomposition["uncertainty_adjustment"] = uncertainty_adjustment

    total_utility = round(sum(decomposition.values()), 4)

    return {
        "decomposition": decomposition,
        "total_utility": total_utility,
    }
