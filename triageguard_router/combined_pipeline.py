"""
combined_pipeline.py
--------------------
End-to-end TriageGuard pipeline.

Patient dict (raw triage input)
    → XGBoost branch   (parallel)
    → RAG branch       (parallel, via OpenRouter)
    → Reconciler
    → Router
    → Final allocation dict

Usage
-----
from triageguard_router.combined_pipeline import TriageGuardPipeline

pipeline = TriageGuardPipeline()
result   = pipeline.run(patient_dict)
print(result["department"])          # e.g. "CICU"
print(result["structured_output"])   # RAG JSON
print(result["xgb"])                 # XGBoost risk scores
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Dict, Optional

# ── path setup ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]   # aic_hackathon/
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "triageguard_xgb"))

from triageguard_xgb.src.inference.predict import TriageGuardPredictor
from triageguard_rag.src.pipeline.rag_pipeline import RAGPipeline
from triageguard_router.reconciler import reconcile
from triageguard_router.router import route
from triageguard_router.policy.live_routing import route_with_hospital_policy


class TriageGuardPipeline:
    """
    Loads both branches once and runs them for every patient.

    Parameters
    ----------
    xgb_models_dir : path to triageguard_xgb/models (default: auto-detected)
    rag_config_path : path to triageguard_rag/config/config.yaml (default: auto)
    """

    def __init__(
        self,
        xgb_models_dir: Optional[Path] = None,
        rag_config_path: Optional[Path] = None,
    ):
        if xgb_models_dir is None:
            xgb_models_dir = _REPO / "triageguard_xgb" / "models"
        if rag_config_path is None:
            rag_config_path = _REPO / "triageguard_rag" / "config" / "config.yaml"

        print("Loading XGBoost branch...")
        self.xgb = TriageGuardPredictor(str(xgb_models_dir))

        print("Loading RAG branch...")
        self.rag = RAGPipeline(config_path=rag_config_path)

        print("TriageGuardPipeline ready.\n")

    def run(self, patient: Dict, hospital_id: Optional[str] = None) -> Dict:
        """
        Run the full pipeline for one patient.

        The patient dict must satisfy both branch schemas:
            XGBoost fields: age, sex, hr_arrival, hr_current, ... (vitals)
            RAG fields:     patient_id, chiefcomplaint, heartrate, o2sat, ...

        hospital_id : scopes RAG retrieval (never mixes another hospital's
            historical cases into this patient's reasoning) AND, since
            Step 6, selects that hospital's own calibrated routing policy
            + live bed state for "hospital_routing" below. Falls back to
            patient["hospital_id"] if not given. Does not affect XGBoost
            (shared model) or reconcile()/route() (clinical preference
            computation — unchanged, hospital-agnostic).

        Returns
        -------
        {
          "department":             str,   # CLINICAL preference — unchanged by hospital policy
          "department_reasoning":   str,
          "acuity_tier":            int,
          "reconciled_admission_risk": float,
          "reconciled_icu_risk":    float,
          "xgb":                    dict,   # full XGBoost output
          "hospital_id":            str | None,
          "hospital_routing":       dict | None,  # RoutingPolicy.route() result, or None if
                                                   # this hospital has no calibrated policy yet —
                                                   # see routing["allocated_department"] for the
                                                   # actual resource-aware final allocation.
          "rag_response":           str,    # free-text LLM narrative
          "structured_output":      dict,   # parsed RAG JSON
          "patient_history":        list,
          "similar_cases":          list,
          "red_flags":              list,
          "top_diagnoses":          list,
          "branches_agree":         bool,
          "confidence_note":        str,
        }
        """
        # ── Adapt field names between schemas ──────────────────────────────
        # XGBoost expects: age, sex, hr_arrival, hr_current, spo2_current, ...
        # RAG expects:     patient_id, chiefcomplaint, heartrate, o2sat, ...
        # We adapt the patient dict for each branch separately.

        xgb_patient = _to_xgb_schema(patient)
        xgb_out = self.xgb.predict(xgb_patient)

        # RAG inference — map field names if needed
        rag_patient = _to_rag_schema(patient)
        hospital_id = hospital_id or patient.get("hospital_id")
        rag_out = self.rag.run(rag_patient, hospital_id=hospital_id)

        # Reconcile + Route (clinical preference — unchanged, hospital-agnostic)
        reconciled = reconcile(xgb_out, rag_out)
        decision   = route(reconciled)

        # Hospital-specific, resource-aware allocation layered on top of the
        # unchanged clinical preference above (Step 6). None if this
        # hospital has no calibrated policy yet — decision["department"]
        # remains the answer in that case, exactly as before this existed.
        hospital_routing = route_with_hospital_policy(
            reconciled=reconciled,
            xgb_output=xgb_out,
            clinical_preferred_department=decision["department"],
            hospital_id=hospital_id,
            rag_history_count=len(rag_out.get("patient_history", []) or []),
            rag_similar_count=len(rag_out.get("similar_cases", []) or []),
        )

        return {
            **decision,
            "xgb":             xgb_out,
            "hospital_id":     hospital_id,
            "hospital_routing": hospital_routing,
            "rag_response":    rag_out.get("response", ""),
            "structured_output": rag_out.get("structured_output", {}),
            "patient_history": rag_out.get("patient_history", []),
            "similar_cases":   rag_out.get("similar_cases", []),
        }


# ---------------------------------------------------------------------------
# Schema adapter
# ---------------------------------------------------------------------------

def _to_rag_schema(patient: Dict) -> Dict:
    """
    Map XGBoost-style field names to RAG-style field names.
    Falls back gracefully when fields are absent.
    Passthrough if the patient dict already has RAG fields.
    """
    def _get(*keys, default=None):
        for k in keys:
            if k in patient and patient[k] is not None:
                return patient[k]
        return default

    return {
        "patient_id":    _get("patient_id", "subject_id", default=-1),
        "chiefcomplaint": _get("chiefcomplaint", "triage_complaint", default=""),
        "acuity":        _get("acuity", default=None),
        "heartrate":     _get("heartrate", "hr_current", "hr_arrival", default=None),
        "resprate":      _get("resprate", "rr_current", "rr_arrival", default=None),
        "o2sat":         _get("o2sat", "spo2_current", "spo2_arrival", default=None),
        "sbp":           _get("sbp", "sbp_current", "sbp_arrival", default=None),
        "dbp":           _get("dbp", "dbp_current", "dbp_arrival", default=None),
        "temperature":   _get("temperature", "temp_current", "temp_arrival", default=None),
        "pain":          _get("pain", default=None),
    }


def _to_xgb_schema(patient: Dict) -> Dict:
    """
    Map RAG-style field names to XGBoost-style field names.
    Provides sensible defaults for any XGBoost field that cannot be
    derived from the patient dict (unknown → 0 / None).
    """
    def _get(*keys, default=None):
        for k in keys:
            if k in patient and patient[k] is not None:
                return patient[k]
        return default

    hr  = _get("hr_current",  "hr_arrival",  "heartrate")
    rr  = _get("rr_current",  "rr_arrival",  "resprate")
    sp  = _get("spo2_current","spo2_arrival", "o2sat")
    sbp = _get("sbp_current", "sbp_arrival",  "sbp")
    dbp = _get("dbp_current", "dbp_arrival",  "dbp")
    tmp = _get("temp_current","temp_arrival",  "temperature")

    return {
        # Demographics — defaults: age 50, sex unknown (→ Female=0)
        "age":                          _get("age", default=50),
        "sex":                          _get("sex", default="F"),

        # Vitals at arrival — use current if no arrival field present
        "hr_arrival":                   hr,
        "rr_arrival":                   rr,
        "spo2_arrival":                 sp,
        "sbp_arrival":                  sbp,
        "dbp_arrival":                  dbp,
        "temp_arrival":                 tmp,

        # Vitals current
        "hr_current":                   hr,
        "rr_current":                   rr,
        "spo2_current":                 sp,
        "sbp_current":                  sbp,
        "dbp_current":                  dbp,
        "temp_current":                 tmp,

        # Clinical text
        "triage_complaint":             _get("triage_complaint", "chiefcomplaint", default=""),
        "time_elapsed_minutes":         _get("time_elapsed_minutes", default=0),

        # History flags — default 0 (unknown)
        "previous_ed_visits":           _get("previous_ed_visits", default=0),
        "previous_hospital_admissions": _get("previous_hospital_admissions", default=0),
        "previous_icu_admissions":      _get("previous_icu_admissions", default=0),
        "cardiovascular_history":       _get("cardiovascular_history", default=0),
        "respiratory_history":          _get("respiratory_history", default=0),
        "renal_history":                _get("renal_history", default=0),
        "diabetes_history":             _get("diabetes_history", default=0),
        "neurological_history":         _get("neurological_history", default=0),
        "malignancy_history":           _get("malignancy_history", default=0),
    }
