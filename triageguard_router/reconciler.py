"""
reconciler.py
-------------
Trust/reconciliation layer that combines XGBoost risk scores with
RAG structured reasoning into a single weighted risk signal.

Logic
-----
- XGBoost weight scales with `information_completeness` (how much data we have).
- RAG weight = 1 - XGBoost weight (fills the gap when data is sparse).
- ICU signals from both branches are OR-combined conservatively.
- When branches disagree on admission, we take the more cautious side.
"""

from __future__ import annotations
from typing import Dict


# ---------------------------------------------------------------------------
# Escalation → numeric score mapping
# ---------------------------------------------------------------------------
_ESCALATION_SCORE = {
    "emergent":  1.0,
    "urgent":    0.6,
    "routine":   0.2,
    "unknown":   0.4,   # conservative default
}

_DISPOSITION_ADMITTED = {"admit", "admitted", "admission"}


def reconcile(xgb_output: Dict, rag_output: Dict) -> Dict:
    """
    Combine XGBoost numeric risks with RAG structured reasoning.

    Parameters
    ----------
    xgb_output : dict returned by TriageGuardPredictor.predict()
        Keys: icu_risk_2h, icu_risk_6h, icu_risk_12h, admission_risk,
              *_confidence, *_raw, information_completeness
    rag_output : dict returned by RAGPipeline.run()
        Must contain key 'structured_output' with:
            disposition, escalation_level, top_diagnoses, red_flags

    Returns
    -------
    dict with:
        reconciled_admission_risk   float [0,1]
        reconciled_icu_risk         float [0,1]   (max of time horizons)
        xgb_weight                  float [0,1]   trust weight on XGBoost
        rag_weight                  float [0,1]   trust weight on RAG
        rag_escalation_score        float [0,1]
        rag_admits                  bool
        branches_agree              bool
        red_flags                   list[str]
        top_diagnoses               list[str]
        information_completeness    float [0,1]
        confidence_note             str   human-readable trust summary
    """

    # ── XGBoost signals ────────────────────────────────────────────────────
    completeness   = float(xgb_output.get("information_completeness", 0.5))
    xgb_admission  = float(xgb_output.get("admission_risk", 0.5))
    xgb_icu_2h     = float(xgb_output.get("icu_risk_2h",   0.0))
    xgb_icu_6h     = float(xgb_output.get("icu_risk_6h",   0.0))
    xgb_icu_12h    = float(xgb_output.get("icu_risk_12h",  0.0))
    xgb_icu        = max(xgb_icu_2h, xgb_icu_6h, xgb_icu_12h)

    # ── RAG signals ────────────────────────────────────────────────────────
    so = rag_output.get("structured_output", {})
    disposition      = str(so.get("disposition", "unknown")).lower()
    escalation_level = str(so.get("escalation_level", "unknown")).lower()
    top_diagnoses    = so.get("top_diagnoses", [])
    red_flags        = so.get("red_flags", [])

    rag_escalation_score = _ESCALATION_SCORE.get(escalation_level, 0.4)
    rag_admits           = disposition in _DISPOSITION_ADMITTED

    # Derive a 0-1 risk score from the RAG branch
    rag_admission_proxy = 0.4 * rag_escalation_score + 0.6 * float(rag_admits)

    # ── Trust weights ──────────────────────────────────────────────────────
    # When data is complete, trust XGBoost more; when sparse, trust RAG more.
    xgb_weight = completeness                  # [0,1]
    rag_weight = 1.0 - completeness            # [0,1]

    # ── Reconcile ──────────────────────────────────────────────────────────
    reconciled_admission = (
        xgb_weight * xgb_admission +
        rag_weight * rag_admission_proxy
    )

    # ICU: keep XGBoost's estimate but boost if RAG says emergent
    icu_boost = 0.2 if escalation_level == "emergent" else 0.0
    reconciled_icu = min(1.0, xgb_icu + icu_boost)

    # Branch agreement: both admit OR both don't admit
    xgb_admits      = xgb_admission >= 0.5
    branches_agree  = (xgb_admits == rag_admits)

    # Safety override: if branches disagree, take the more cautious path
    if not branches_agree:
        # Bump reconciled admission toward 0.6 as a conservative floor
        reconciled_admission = max(reconciled_admission, 0.55)

    # ── Confidence note ────────────────────────────────────────────────────
    if completeness >= 0.8:
        note = f"High data completeness ({completeness:.0%}) — XGBoost dominant"
    elif completeness >= 0.4:
        note = f"Partial data ({completeness:.0%}) — blended XGBoost + RAG"
    else:
        note = f"Low data ({completeness:.0%}) — RAG dominant"

    if not branches_agree:
        note += "; branches disagree → conservative admission floor applied"

    return {
        "reconciled_admission_risk":  round(reconciled_admission, 4),
        "reconciled_icu_risk":        round(reconciled_icu, 4),
        "xgb_weight":                 round(xgb_weight, 4),
        "rag_weight":                 round(rag_weight, 4),
        "rag_escalation_score":       round(rag_escalation_score, 4),
        "rag_admits":                 rag_admits,
        "branches_agree":             branches_agree,
        "red_flags":                  red_flags,
        "top_diagnoses":              top_diagnoses,
        "information_completeness":   round(completeness, 4),
        "confidence_note":            note,
    }
