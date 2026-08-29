"""
router.py
---------
Adaptive router: takes the reconciled risk signal and allocates
the patient to the appropriate department.

Departments (in descending acuity order)
-----------------------------------------
CICU          Cardiac ICU   — cardiac emergency signals
ICU           General ICU   — non-cardiac critical
ADMITTED_GEN  General ward  — needs admission, not ICU
ED_OBS        ED Observation / short stay
DISCHARGE     Safe to discharge
"""

from __future__ import annotations
from typing import Dict


# ---------------------------------------------------------------------------
# Cardiac diagnosis keywords (for CICU vs ICU split)
# ---------------------------------------------------------------------------
_CARDIAC_KEYWORDS = {
    "acs", "ami", "mi", "nstemi", "stemi",
    "aortic", "cardiac", "heart failure", "hf",
    "arrhythmia", "afib", "vt", "vf", "pe",
    "dissection", "tamponade", "cath",
}


def _is_cardiac(top_diagnoses: list) -> bool:
    joined = " ".join(d.lower() for d in top_diagnoses)
    return any(kw in joined for kw in _CARDIAC_KEYWORDS)


# ---------------------------------------------------------------------------
# Thresholds — tunable
# ---------------------------------------------------------------------------
ICU_RISK_THRESHOLD        = 0.35   # reconciled_icu_risk above this → ICU
ADMISSION_THRESHOLD       = 0.50   # reconciled_admission_risk above → admit
OBS_THRESHOLD             = 0.30   # above this but below ADMISSION → obs


def route(reconciled: Dict) -> Dict:
    """
    Parameters
    ----------
    reconciled : output of reconciler.reconcile()

    Returns
    -------
    {
        "department":           str,   one of CICU/ICU/ADMITTED_GEN/ED_OBS/DISCHARGE
        "department_reasoning": str,   short human-readable explanation
        "acuity_tier":          int,   1 (highest) – 5 (lowest)
        "escalation_level":     str,   matches RAG field
        "reconciled_admission_risk": float,
        "reconciled_icu_risk":  float,
        "red_flags":            list[str],
        "top_diagnoses":        list[str],
        "confidence_note":      str,
    }
    """
    admission_risk = reconciled["reconciled_admission_risk"]
    icu_risk       = reconciled["reconciled_icu_risk"]
    red_flags      = reconciled.get("red_flags", [])
    top_diagnoses  = reconciled.get("top_diagnoses", [])
    branches_agree = reconciled.get("branches_agree", True)
    rag_escalation = reconciled.get("rag_escalation_score", 0.4)
    completeness   = reconciled.get("information_completeness", 1.0)

    # ── Emergency escalation override ──────────────────────────────────────
    # If RAG says emergent and we have meaningful agreement, floor to ICU
    forced_icu = (rag_escalation >= 1.0 and not branches_agree)

    # ── Department decision tree ───────────────────────────────────────────
    if icu_risk >= ICU_RISK_THRESHOLD or forced_icu:
        if _is_cardiac(top_diagnoses):
            department = "CICU"
            reasoning  = (
                f"ICU-level risk (reconciled ICU risk {icu_risk:.1%}) "
                f"with cardiac diagnosis signals ({top_diagnoses[:2]})"
            )
            acuity = 1
        else:
            department = "ICU"
            reasoning  = (
                f"ICU-level risk (reconciled ICU risk {icu_risk:.1%}), "
                f"non-cardiac presentation"
            )
            acuity = 1

    elif admission_risk >= ADMISSION_THRESHOLD:
        department = "ADMITTED_GEN"
        reasoning  = (
            f"Admission indicated (reconciled admission risk {admission_risk:.1%}), "
            f"ICU threshold not met ({icu_risk:.1%} < {ICU_RISK_THRESHOLD:.1%})"
        )
        acuity = 3

    elif admission_risk >= OBS_THRESHOLD:
        department = "ED_OBS"
        reasoning  = (
            f"Borderline admission risk ({admission_risk:.1%}), "
            f"placed in ED observation for reassessment"
        )
        acuity = 3

    else:
        department = "DISCHARGE"
        reasoning  = (
            f"Low admission risk ({admission_risk:.1%}), "
            f"safe for discharge with follow-up instructions"
        )
        acuity = 5

    # Low-completeness warning — escalate one tier conservatively
    if completeness < 0.3 and department == "DISCHARGE":
        department = "ED_OBS"
        reasoning += " [UPGRADED: insufficient data to safely discharge]"
        acuity = 3

    return {
        "department":                department,
        "department_reasoning":      reasoning,
        "acuity_tier":               acuity,
        "reconciled_admission_risk": reconciled["reconciled_admission_risk"],
        "reconciled_icu_risk":       reconciled["reconciled_icu_risk"],
        "red_flags":                 red_flags,
        "top_diagnoses":             top_diagnoses,
        "confidence_note":           reconciled.get("confidence_note", ""),
        "branches_agree":            branches_agree,
    }
