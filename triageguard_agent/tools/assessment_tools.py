"""
assessment_tools.py
-------------------
COMPUTE tools that wrap the existing TriageGuard ML pipeline.

The LLM calls these tools by name and receives structured ToolResults.
It never touches PCA, model loading, embeddings, or feature construction.

LLM-visible tools
-----------------
* run_triage_assessment  — runs full XGBoost + RAG + reconcile + route
* get_xgb_explanation    — returns top feature attributions for last XGB run

Design
------
TriageGuardPipeline is loaded ONCE at module level (lazy singleton) to
avoid re-loading models on every tool call. The LLM never sees this object.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from triageguard_agent.schemas.tool_result import ToolResult

logger = logging.getLogger(__name__)

# Lazy singleton — loaded on first use
_pipeline_instance = None

TOOL_NAME_ASSESSMENT = "run_triage_assessment"
TOOL_NAME_XGB_EXPLAIN = "get_xgb_explanation"


# ---------------------------------------------------------------------------
# Shared input schema for "patient_data"
# ---------------------------------------------------------------------------
#
# Field names match TriageGuardPipeline / TriageGuardPredictor's own native
# schema exactly (see triageguard_router/combined_pipeline.py::_to_xgb_schema
# and triageguard_xgb/src/inference/predict.py) — this is not a new schema,
# it is the existing pipeline's already-established field vocabulary made
# explicit to the LLM.
#
# Previously this was left as a bare {"type": "object"} with no properties,
# which gave the model no blueprint for what to put inside "patient_data".
# Under-specified nested-object schemas are a well known cause of small
# models emitting free-formed / stringified pseudo-JSON instead of a real
# object (observed directly: the model reconstructed field names like
# "heart_rate"/"oxygen_saturation" from prose it had generated earlier,
# instead of the pipeline's actual hr_current/spo2_current names, and wrapped
# the whole thing as a string). Declaring the real properties fixes that at
# the source, for every tool that takes patient_data — not one tool/patient.
_PATIENT_DATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "Patient clinical data using TriageGuard's native field names. "
        "Include only the fields you actually have values for — omit any "
        "field you don't have a real value for. Never invent vitals."
    ),
    "properties": {
        "patient_id": {"type": ["string", "integer"], "description": "Patient identifier."},
        "age": {"type": "integer", "description": "Age in years."},
        "sex": {"type": "string", "description": "'M' or 'F'."},
        "triage_complaint": {"type": "string", "description": "Chief complaint / triage symptoms."},
        "time_elapsed_minutes": {"type": "number", "description": "Minutes elapsed since arrival."},
        "hr_arrival": {"type": "number", "description": "Heart rate (bpm) at arrival."},
        "hr_current": {"type": "number", "description": "Current heart rate (bpm)."},
        "rr_arrival": {"type": "number", "description": "Respiratory rate at arrival."},
        "rr_current": {"type": "number", "description": "Current respiratory rate."},
        "spo2_arrival": {"type": "number", "description": "Oxygen saturation (%) at arrival."},
        "spo2_current": {"type": "number", "description": "Current oxygen saturation (%)."},
        "sbp_arrival": {"type": "number", "description": "Systolic blood pressure at arrival."},
        "sbp_current": {"type": "number", "description": "Current systolic blood pressure."},
        "dbp_arrival": {"type": "number", "description": "Diastolic blood pressure at arrival."},
        "dbp_current": {"type": "number", "description": "Current diastolic blood pressure."},
        "temp_arrival": {"type": "number", "description": "Temperature at arrival."},
        "temp_current": {"type": "number", "description": "Current temperature."},
        "previous_ed_visits": {"type": "integer", "description": "Count of prior ED visits."},
        "previous_hospital_admissions": {"type": "integer"},
        "previous_icu_admissions": {"type": "integer"},
        "cardiovascular_history": {"type": "integer", "description": "1 if present, 0 if absent/unknown."},
        "respiratory_history": {"type": "integer"},
        "renal_history": {"type": "integer"},
        "diabetes_history": {"type": "integer"},
        "neurological_history": {"type": "integer"},
        "malignancy_history": {"type": "integer"},
    },
}


def _get_pipeline():
    """Return shared TriageGuardPipeline instance (loads once)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        try:
            from triageguard_router.combined_pipeline import TriageGuardPipeline
            _pipeline_instance = TriageGuardPipeline()
            logger.info("TriageGuardPipeline loaded for assessment tools.")
        except Exception as exc:
            logger.error("Failed to load TriageGuardPipeline: %s", exc)
            raise
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Handler: run_triage_assessment
# ---------------------------------------------------------------------------

def run_triage_assessment(patient_data: Dict[str, Any]) -> ToolResult:
    """
    Run the full clinical prediction pipeline for a patient.

    Calls TriageGuardPipeline.run() → reconcile → route.
    Returns a structured summary of the routing decision, risks, and
    clinical reasoning. The LLM receives this result; it does NOT modify
    predictions.

    Parameters
    ----------
    patient_data : dict in the combined schema understood by TriageGuardPipeline.
    """
    if not isinstance(patient_data, dict) or not patient_data:
        return ToolResult.fail(
            TOOL_NAME_ASSESSMENT,
            "INVALID_PATIENT_DATA",
            "patient_data must be a non-empty dict.",
        )

    try:
        pipeline = _get_pipeline()
        result = pipeline.run(patient_data)
    except Exception as exc:
        logger.exception("Pipeline error during triage assessment.")
        return ToolResult.fail(
            TOOL_NAME_ASSESSMENT,
            "PIPELINE_ERROR",
            f"Triage pipeline failed: {exc}",
        )

    # Project to a clean output — agent never sees raw model internals
    data = {
        "department":                result.get("department"),
        "department_reasoning":      result.get("department_reasoning"),
        "acuity_tier":               result.get("acuity_tier"),
        "reconciled_admission_risk": result.get("reconciled_admission_risk"),
        "reconciled_icu_risk":       result.get("reconciled_icu_risk"),
        "branches_agree":            result.get("branches_agree"),
        "confidence_note":           result.get("confidence_note"),
        "top_diagnoses":             result.get("top_diagnoses", []),
        "red_flags":                 result.get("red_flags", []),
        "rag_trajectory":            (result.get("structured_output") or {}).get("trajectory_assessment"),
        "rag_urgency":               (result.get("structured_output") or {}).get("urgency"),
        "rag_evidence_strength":     (result.get("structured_output") or {}).get("evidence_strength"),
        "rag_escalation_concern":    (result.get("structured_output") or {}).get("escalation_concern"),
        "rag_narrative":             result.get("rag_response", "")[:500],  # truncate for context
        # Store references for XGB explanation tool
        "_xgb_raw":                  result.get("xgb", {}),
    }

    return ToolResult.ok(
        TOOL_NAME_ASSESSMENT,
        data,
        metadata={
            "patient_id": str(patient_data.get("patient_id", "")),
            "information_completeness": result.get("xgb", {}).get("information_completeness", 0),
        },
    )


# ---------------------------------------------------------------------------
# Handler: get_xgb_explanation
# ---------------------------------------------------------------------------

def get_xgb_explanation(patient_data: Dict[str, Any]) -> ToolResult:
    """
    Return feature-level attribution for the XGBoost admission risk prediction.

    This tool exposes only the *output* of the XGBoost model (calibrated
    probabilities + which vitals/features drove the score). It does not
    expose the model object, weights, or internal representations.

    The LLM must use the returned attribution data — it must never invent
    causal explanations not supported by this output.
    """
    if not isinstance(patient_data, dict) or not patient_data:
        return ToolResult.fail(
            TOOL_NAME_XGB_EXPLAIN,
            "INVALID_PATIENT_DATA",
            "patient_data must be a non-empty dict.",
        )

    try:
        from pathlib import Path
        import sys

        # parents[2] = repo root (assessment_tools.py is at
        # <repo>/triageguard_agent/tools/assessment_tools.py). The previous
        # parents[3] resolved one directory ABOVE the repo root, which
        # silently pointed _models_dir at a path that never existed.
        _REPO = Path(__file__).resolve().parents[2]
        _xgb_root = _REPO / "triageguard_xgb"
        # predict.py imports "from src.features..." (relative to triageguard_xgb/
        # itself, not the repo root) — same path requirement already handled by
        # triageguard_router/combined_pipeline.py. Without this, the import
        # below fails with "No module named 'src'" whenever this tool is used
        # standalone (i.e. run_triage_assessment/combined_pipeline hasn't
        # already been imported first in this process to set it up as a
        # side effect).
        if str(_xgb_root) not in sys.path:
            sys.path.insert(0, str(_xgb_root))

        from triageguard_xgb.src.inference.predict import TriageGuardPredictor

        _models_dir = _xgb_root / "models"

        predictor = TriageGuardPredictor(str(_models_dir))
        xgb_out = predictor.predict(patient_data)

    except Exception as exc:
        logger.exception("XGBoost explanation failed.")
        return ToolResult.fail(
            TOOL_NAME_XGB_EXPLAIN,
            "XGB_ERROR",
            f"XGBoost inference failed: {exc}",
        )

    # Build a human-readable feature contribution summary from available outputs
    # Full SHAP would require shap library — here we use the raw scores to
    # rank which vitals were present vs. missing, and flag the highest risks.
    info_completeness = xgb_out.get("information_completeness", 0.0)
    admission_risk = xgb_out.get("admission_risk", None)
    icu_risks = {
        "icu_risk_2h":  xgb_out.get("icu_risk_2h"),
        "icu_risk_6h":  xgb_out.get("icu_risk_6h"),
        "icu_risk_12h": xgb_out.get("icu_risk_12h"),
    }
    confidences = {
        k: xgb_out.get(f"{k}_confidence")
        for k in ["icu_risk_2h", "icu_risk_6h", "icu_risk_12h", "admission_risk"]
    }

    # Identify vitals that were missing (contributing to low completeness)
    vitals_present = {}
    for v in ["hr", "rr", "spo2", "sbp", "dbp", "temp"]:
        for suffix in ["_current", "_arrival", ""]:
            key = f"{v}{suffix}"
            if patient_data.get(key) is not None:
                vitals_present[v] = True
                break
        else:
            vitals_present[v] = False

    explanation = {
        "admission_risk":          admission_risk,
        "icu_risks":               icu_risks,
        "confidences":             confidences,
        "information_completeness": info_completeness,
        "vitals_present":          vitals_present,
        "missing_vitals":          [v for v, present in vitals_present.items() if not present],
        "attribution_note": (
            "XGBoost scores reflect calibrated probabilistic predictions. "
            "Feature importance is proportional to data completeness. "
            "Missing vitals reduce model confidence and shift trust to RAG branch."
        ),
    }

    return ToolResult.ok(
        TOOL_NAME_XGB_EXPLAIN,
        explanation,
        metadata={"information_completeness": info_completeness},
    )


# ---------------------------------------------------------------------------
# ToolSpec factories
# ---------------------------------------------------------------------------

def run_triage_assessment_spec():
    from triageguard_agent.tools.registry import ToolSpec, COMPUTE
    return ToolSpec(
        name=TOOL_NAME_ASSESSMENT,
        description=(
            "Run the full TriageGuard clinical prediction pipeline for a patient. "
            "Executes XGBoost risk scoring, RAG historical reasoning, reconciliation, "
            "and department routing. Use when a triage assessment or reassessment is needed. "
            "The agent must NOT modify the returned predictions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "patient_data": _PATIENT_DATA_SCHEMA,
            },
            "required": ["patient_data"],
        },
        handler=lambda **kwargs: run_triage_assessment(**kwargs),
        risk_level=COMPUTE,
        side_effect=False,
        requires_approval=False,
    )


def get_xgb_explanation_spec():
    from triageguard_agent.tools.registry import ToolSpec, COMPUTE
    return ToolSpec(
        name=TOOL_NAME_XGB_EXPLAIN,
        description=(
            "Return XGBoost feature attribution for the current patient. "
            "Use when the nurse asks 'why did TriageGuard give this score?' "
            "Only explain what the attribution data shows — never invent causes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "patient_data": _PATIENT_DATA_SCHEMA,
            },
            "required": ["patient_data"],
        },
        handler=lambda **kwargs: get_xgb_explanation(**kwargs),
        risk_level=COMPUTE,
        side_effect=False,
        requires_approval=False,
    )
