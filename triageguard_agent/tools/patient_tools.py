"""
patient_tools.py
----------------
READ tools for patient data retrieval.

These tools present a clean, high-level interface to patient data.
They do NOT expose raw database rows, PCA artefacts, or embedding details.

For the hackathon prototype, patient data is loaded from JSON files in
triageguard_agent/data/patients/. In production this would be a proper
patient data service.

LLM-visible tools
-----------------
* get_patient_summary      — current demographics + vitals snapshot
* get_patient_observations — chronological observation timeline
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from triageguard_agent.schemas.tool_result import ToolResult

# Default patient data directory (relative to this file's package root)
_AGENT_ROOT = Path(__file__).resolve().parents[1]
_PATIENTS_DIR = _AGENT_ROOT / "data" / "patients"

TOOL_NAME_SUMMARY = "get_patient_summary"
TOOL_NAME_OBSERVATIONS = "get_patient_observations"


def _load_patient_file(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a patient dict from data/patients/<patient_id>.json.
    Returns None if the file does not exist.
    """
    path = _PATIENTS_DIR / f"{patient_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Handler: get_patient_summary
# ---------------------------------------------------------------------------

def get_patient_summary(patient_id: str) -> ToolResult:
    """
    Retrieve the current demographics and vitals snapshot for a patient.

    Returns a flat dict with:
        patient_id, age, sex, chief_complaint,
        current vitals (hr, rr, spo2, sbp, dbp, temp, pain),
        acuity, time_elapsed_minutes,
        last_updated (ISO timestamp).
    """
    if not patient_id:
        return ToolResult.fail(
            TOOL_NAME_SUMMARY,
            "MISSING_PATIENT_ID",
            "patient_id is required.",
        )

    record = _load_patient_file(str(patient_id))
    if record is None:
        return ToolResult.fail(
            TOOL_NAME_SUMMARY,
            "PATIENT_NOT_FOUND",
            f"No patient record found for patient_id={patient_id!r}. "
            "Verify the ID before proceeding.",
        )

    # Project only safe, high-level fields — no internal model artefacts
    summary = {
        "patient_id":            record.get("patient_id", patient_id),
        "age":                   record.get("age"),
        "sex":                   record.get("sex"),
        "chief_complaint":       record.get("chiefcomplaint") or record.get("triage_complaint", ""),
        "acuity":                record.get("acuity"),
        "time_elapsed_minutes":  record.get("time_elapsed_minutes", 0),
        "vitals": {
            "heart_rate":        record.get("heartrate") or record.get("hr_current"),
            "resp_rate":         record.get("resprate") or record.get("rr_current"),
            "spo2":              record.get("o2sat") or record.get("spo2_current"),
            "sbp":               record.get("sbp") or record.get("sbp_current"),
            "dbp":               record.get("dbp") or record.get("dbp_current"),
            "temperature":       record.get("temperature") or record.get("temp_current"),
            "pain_score":        record.get("pain"),
        },
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    return ToolResult.ok(
        TOOL_NAME_SUMMARY,
        summary,
        metadata={"source": "patient_file", "patient_id": patient_id},
    )


# ---------------------------------------------------------------------------
# Handler: get_patient_observations
# ---------------------------------------------------------------------------

def get_patient_observations(
    patient_id: str,
    encounter_id: Optional[str] = None,
) -> ToolResult:
    """
    Retrieve the chronological observation timeline for a patient.

    If encounter_id is provided, filters to that encounter.
    Returns a list of observation dicts, ordered oldest-first.
    """
    if not patient_id:
        return ToolResult.fail(
            TOOL_NAME_OBSERVATIONS,
            "MISSING_PATIENT_ID",
            "patient_id is required.",
        )

    record = _load_patient_file(str(patient_id))
    if record is None:
        return ToolResult.fail(
            TOOL_NAME_OBSERVATIONS,
            "PATIENT_NOT_FOUND",
            f"No patient record found for patient_id={patient_id!r}.",
        )

    observations = record.get("observations", [])
    if encounter_id:
        observations = [
            o for o in observations
            if str(o.get("encounter_id", "")) == str(encounter_id)
        ]

    return ToolResult.ok(
        TOOL_NAME_OBSERVATIONS,
        {"patient_id": patient_id, "observations": observations},
        metadata={
            "encounter_id": encounter_id,
            "count": len(observations),
        },
    )


# ---------------------------------------------------------------------------
# ToolSpec factories (imported by the runtime to register these tools)
# ---------------------------------------------------------------------------

def get_patient_summary_spec() -> Dict[str, Any]:
    """Return the registration spec for get_patient_summary."""
    from triageguard_agent.tools.registry import ToolSpec, READ
    return ToolSpec(
        name=TOOL_NAME_SUMMARY,
        description=(
            "Retrieve the current demographics and vitals snapshot for a patient. "
            "Use when the nurse asks about a patient's current state. "
            "Never use RAG to answer simple factual patient questions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient identifier"},
            },
            "required": ["patient_id"],
        },
        handler=lambda **kwargs: get_patient_summary(**kwargs),
        risk_level=READ,
        side_effect=False,
        requires_approval=False,
    )


def get_patient_observations_spec() -> Dict[str, Any]:
    """Return the registration spec for get_patient_observations."""
    from triageguard_agent.tools.registry import ToolSpec, READ
    return ToolSpec(
        name=TOOL_NAME_OBSERVATIONS,
        description=(
            "Retrieve the chronological observation timeline for a patient. "
            "Use when historical context or trend analysis is needed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "patient_id":   {"type": "string"},
                "encounter_id": {"type": "string", "description": "Optional encounter filter"},
            },
            "required": ["patient_id"],
        },
        handler=lambda **kwargs: get_patient_observations(**kwargs),
        risk_level=READ,
        side_effect=False,
        requires_approval=False,
    )
