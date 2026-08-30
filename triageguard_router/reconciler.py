"""
reconciler.py
-------------
Trust/reconciliation layer that combines XGBoost risk scores with
RAG structured reasoning into a single weighted risk signal.

Logic (Master MD Part II — "Trust/reconciliation")
---------------------------------------------------
Model contributions are weighted by how CONFIDENT each branch actually is
in its own output — not by how much raw data happens to be present:

    Cx = XGBoost's own calibrated confidence for the relevant target
    Cr = RAG's self-rated evidence_strength (1-5) / 5
    BaseScore = (Cx * xgb_risk + Cr * rag_risk) / (Cx + Cr)

A branch that is itself unsure (low confidence / low evidence_strength) is
trusted less, regardless of how many raw vitals happen to be on file. This
replaces an earlier version that weighted purely by `information_completeness`,
which could dilute a highly-confident XGBoost prediction just because a few
vitals were missing, even when XGBoost itself was not actually uncertain
about the ones it had.

Escalation is intentionally one-sided: disagreement between the branches and
an explicit RAG escalation concern can only push risk up (via `min(1, x+boost)`
and `max(x, floor)`), never pull a risk estimate down.
"""

from __future__ import annotations
from typing import Dict


# ---------------------------------------------------------------------------
# Urgency → numeric score mapping (Master MD Part II policy mapping —
# not a medical probability)
# ---------------------------------------------------------------------------
_URGENCY_SCORE = {
    "low":      0.20,
    "moderate": 0.45,
    "high":     0.70,
    "critical": 0.90,
}
_URGENCY_DEFAULT = 0.45   # "unknown" -> moderate, conservative default

# Urgency levels at which RAG is treated as leaning toward admission, for
# branch-agreement comparison (there is no more explicit "disposition" field
# to read — the LLM no longer makes that call, see llm_reasoner.py).
_RAG_ADMIT_URGENCY = {"high", "critical"}

_ICU_HORIZONS = ("2h", "6h", "12h")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
            urgency, evidence_strength, escalation_concern,
            top_diagnoses, red_flags (see llm_reasoner.py)

    Returns
    -------
    dict with:
        reconciled_admission_risk   float [0,1]
        reconciled_icu_risk         float [0,1]   (max of time horizons)
        xgb_confidence              float [0,1]   Cx used for admission blend
        rag_confidence              float [0,1]   Cr used for admission blend (evidence_strength/5)
        xgb_icu_confidence          float [0,1]   Cx used for ICU blend
        rag_escalation_score        float [0,1]   kept for router.py's forced-ICU check
        rag_urgency                 str
        rag_admits                  bool
        escalation_concern          bool
        branches_agree              bool
        red_flags                   list[str]
        top_diagnoses               list[str]
        information_completeness    float [0,1]
        confidence_note             str   human-readable trust summary
    """

    # ── XGBoost signals ────────────────────────────────────────────────────
    completeness  = _safe_float(xgb_output.get("information_completeness"), 0.5)
    xgb_admission = _safe_float(xgb_output.get("admission_risk"), 0.5)
    xgb_admission_conf = _safe_float(xgb_output.get("admission_risk_confidence"), 0.0)

    xgb_icu_by_horizon = {h: _safe_float(xgb_output.get(f"icu_risk_{h}"), 0.0) for h in _ICU_HORIZONS}
    xgb_icu = max(xgb_icu_by_horizon.values())

    icu_confs = [
        _safe_float(xgb_output.get(f"icu_risk_{h}_confidence"), None)
        for h in _ICU_HORIZONS
    ]
    icu_confs = [c for c in icu_confs if c is not None]
    xgb_icu_conf = sum(icu_confs) / len(icu_confs) if icu_confs else 0.0

    # ── RAG signals ────────────────────────────────────────────────────────
    so = rag_output.get("structured_output", {}) or {}
    urgency            = str(so.get("urgency", "unknown")).lower()
    escalation_concern = bool(so.get("escalation_concern", False))
    top_diagnoses      = so.get("top_diagnoses", [])
    red_flags          = so.get("red_flags", [])

    evidence_strength = _safe_float(so.get("evidence_strength"), None)
    if evidence_strength is None:
        evidence_strength = 3.0   # mid-point default if the LLM omitted/garbled it
    evidence_strength = min(5.0, max(1.0, evidence_strength))

    rag_risk_score = _URGENCY_SCORE.get(urgency, _URGENCY_DEFAULT)
    rag_admits     = (urgency in _RAG_ADMIT_URGENCY) or escalation_concern
    rag_confidence = evidence_strength / 5.0   # Cr

    # ── Confidence-weighted blend (BaseScore) ──────────────────────────────
    def _blend(xgb_value: float, xgb_conf: float) -> float:
        denom = xgb_conf + rag_confidence
        if denom <= 1e-9:
            # Neither branch is confident — fall back to an even split
            # rather than dividing by zero.
            return 0.5 * xgb_value + 0.5 * rag_risk_score
        return (xgb_conf * xgb_value + rag_confidence * rag_risk_score) / denom

    reconciled_admission = _blend(xgb_admission, xgb_admission_conf)
    reconciled_icu = _blend(xgb_icu, xgb_icu_conf)

    # ── Asymmetric escalation (one-sided — can only raise risk) ────────────
    icu_boost = 0.2 if escalation_concern else 0.0
    reconciled_icu = min(1.0, reconciled_icu + icu_boost)

    xgb_admits     = xgb_admission >= 0.5
    branches_agree = (xgb_admits == rag_admits)
    if not branches_agree:
        reconciled_admission = max(reconciled_admission, 0.55)

    # rag_escalation_score is retained under its old name for router.py's
    # forced-ICU override, which reads this exact key.
    rag_escalation_score = 1.0 if escalation_concern else rag_risk_score

    # ── Confidence note ────────────────────────────────────────────────────
    if xgb_admission_conf >= 0.5 and rag_confidence >= 0.5:
        note = (
            f"Both branches confident (XGBoost {xgb_admission_conf:.0%}, "
            f"RAG {rag_confidence:.0%}) — balanced blend"
        )
    elif xgb_admission_conf >= rag_confidence:
        note = (
            f"XGBoost more confident ({xgb_admission_conf:.0%}) than RAG "
            f"evidence ({rag_confidence:.0%}) — XGBoost-leaning blend"
        )
    else:
        note = (
            f"RAG evidence stronger ({rag_confidence:.0%}) than XGBoost "
            f"confidence ({xgb_admission_conf:.0%}) — RAG-leaning blend"
        )
    if not branches_agree:
        note += "; branches disagree → conservative admission floor applied"

    return {
        "reconciled_admission_risk":  round(reconciled_admission, 4),
        "reconciled_icu_risk":        round(reconciled_icu, 4),
        "xgb_confidence":             round(xgb_admission_conf, 4),
        "rag_confidence":             round(rag_confidence, 4),
        "xgb_icu_confidence":         round(xgb_icu_conf, 4),
        "rag_escalation_score":       round(rag_escalation_score, 4),
        "rag_urgency":                urgency,
        "rag_admits":                 rag_admits,
        "escalation_concern":         escalation_concern,
        "branches_agree":             branches_agree,
        "red_flags":                  red_flags,
        "top_diagnoses":              top_diagnoses,
        "information_completeness":   round(completeness, 4),
        "confidence_note":            note,
    }
