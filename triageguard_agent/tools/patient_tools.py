"""
patient_tools.py
----------------
READ and WRITE tools for patient data.

These tools present a clean, high-level interface to patient data.
They do NOT expose raw database rows, PCA artefacts, or embedding details.

For the hackathon prototype, patient data is loaded from JSON files in
triageguard_agent/data/patients/. In production this would be a proper
patient data service.

LLM-visible tools
-----------------
* get_patient_summary      — current demographics + vitals snapshot
* get_patient_observations — chronological observation timeline
* add_patient_observation  — record a new timestamped vital (WRITE, requires approval)
"""

from __future__ import annotations
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from triageguard_agent.schemas.tool_result import ToolResult

# Default patient data directory (relative to this file's package root)
_AGENT_ROOT = Path(__file__).resolve().parents[1]
_PATIENTS_DIR = _AGENT_ROOT / "data" / "patients"

TOOL_NAME_SUMMARY = "get_patient_summary"
TOOL_NAME_OBSERVATIONS = "get_patient_observations"
TOOL_NAME_ADD_OBSERVATION = "add_patient_observation"

# Serialises read-modify-write access to a single patient file across threads
# within this process (mirrors the intent of HospitalStateStore's lock —
# patient files have no such store here, so the lock lives at module level).
_WRITE_LOCK = threading.Lock()

# Observation-type vocabulary, matching triageguard_agent/skills/patient_update/SKILL.md
# exactly: field names on the per-observation dict, the corresponding "current
# state" field on the top-level patient record (naming differs between the two
# — see 52.json), and the validated clinical range.
_OBSERVATION_FIELDS: Dict[str, Dict[str, Any]] = {
    "heart_rate":  {"obs_field": "heart_rate", "current_field": "heartrate",   "min": 0,  "max": 300, "unit": "bpm"},
    "spo2":        {"obs_field": "spo2",        "current_field": "o2sat",       "min": 0,  "max": 100, "unit": "%"},
    "resp_rate":   {"obs_field": "resp_rate",   "current_field": "resprate",   "min": 0,  "max": 100, "unit": "/min"},
    "sbp":         {"obs_field": "sbp",         "current_field": "sbp",        "min": 0,  "max": 300, "unit": "mmHg"},
    "dbp":         {"obs_field": "dbp",         "current_field": "dbp",        "min": 0,  "max": 200, "unit": "mmHg"},
    "temperature": {"obs_field": "temperature", "current_field": "temperature", "min": 30, "max": 45,  "unit": "°C"},
}


def _patient_file_path(patient_id: str) -> Path:
    return _PATIENTS_DIR / f"{patient_id}.json"


def _load_patient_file(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a patient dict from data/patients/<patient_id>.json.
    Returns None if the file does not exist.
    """
    path = _patient_file_path(patient_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_patient_record(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Public, non-tool helper: return the full raw patient record dict (including
    the "observations" history list), or None if the patient does not exist.

    Used by callers that need the whole record (e.g. AgentRuntime building a
    fresh run_triage_assessment payload after a write) rather than the
    trimmed, LLM-facing view get_patient_summary returns.
    """
    return _load_patient_file(str(patient_id))


def _write_patient_file(patient_id: str, record: Dict[str, Any]) -> None:
    """
    Atomically persist `record` to data/patients/<patient_id>.json.
    Writes to a temp file in the same directory then os.replace()s it, so a
    crash mid-write can never leave a partially-written/corrupt patient file.
    """
    path = _patient_file_path(patient_id)
    tmp_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    os.replace(tmp_path, path)


def build_assessment_input(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the flat patient_data dict run_triage_assessment expects, from a raw
    patient record (as returned by get_patient_record / stored on disk).

    The raw record's own field names (heartrate, o2sat, resprate, sbp, dbp,
    temperature, chiefcomplaint, ...) are already accepted directly by
    TriageGuardPipeline's schema adapters (see
    triageguard_router/combined_pipeline.py::_to_xgb_schema/_to_rag_schema),
    so this only needs to drop the "observations" history list — the adapters
    ignore any other extra/unknown keys.
    """
    return {k: v for k, v in record.items() if k != "observations"}


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
# Handler: add_patient_observation
# ---------------------------------------------------------------------------

def add_patient_observation(
    patient_id: str,
    observation_type: str,
    value: Any,
    note: Optional[str] = None,
    **_ignored: Any,
) -> ToolResult:
    """
    Record a new observation for a patient, timestamped by the system clock.

    The timestamp is NEVER taken from the caller — it is always
    datetime.now(timezone.utc) at the moment this handler runs. Any
    "timestamp" a model passes anyway (via **_ignored) is silently discarded
    rather than trusted or erroring, so a model that ignores the declared
    schema still can't influence the clinical record's timestamp.

    Preserves history: the previous observation(s) are never modified or
    removed — a new entry is appended to record["observations"], and only
    the top-level "current" field for this vital is updated, so the caller
    can compare previous vs. new value.

    If the new value is identical to the current value for this vital, this
    is treated as a duplicate/no-op: nothing is written, and the result
    reports duplicate=True (mirrors ingest_hospital_records' duplicate-skip
    convention).
    """
    if not patient_id:
        return ToolResult.fail(
            TOOL_NAME_ADD_OBSERVATION,
            "MISSING_PATIENT_ID",
            "patient_id is required.",
        )

    field_info = _OBSERVATION_FIELDS.get(observation_type)
    if field_info is None:
        return ToolResult.fail(
            TOOL_NAME_ADD_OBSERVATION,
            "INVALID_OBSERVATION_TYPE",
            f"Unknown observation_type {observation_type!r}. "
            f"Must be one of: {sorted(_OBSERVATION_FIELDS)}.",
        )

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ToolResult.fail(
            TOOL_NAME_ADD_OBSERVATION,
            "INVALID_VALUE",
            f"value {value!r} for {observation_type!r} must be a number.",
        )
    if not (field_info["min"] <= numeric_value <= field_info["max"]):
        return ToolResult.fail(
            TOOL_NAME_ADD_OBSERVATION,
            "INVALID_VALUE",
            f"{observation_type} value {numeric_value} is outside the valid "
            f"range [{field_info['min']}, {field_info['max']}] {field_info['unit']}.",
        )
    # Preserve int-vs-float shape when the input was a whole number, so
    # "125" and "125.0" both store as a clean 125 like the rest of 52.json.
    stored_value = int(numeric_value) if numeric_value.is_integer() else numeric_value

    with _WRITE_LOCK:
        record = _load_patient_file(str(patient_id))
        if record is None:
            return ToolResult.fail(
                TOOL_NAME_ADD_OBSERVATION,
                "PATIENT_NOT_FOUND",
                f"No patient record found for patient_id={patient_id!r}.",
            )

        current_field = field_info["current_field"]
        previous_value = record.get(current_field)

        if previous_value is not None and float(previous_value) == numeric_value:
            return ToolResult.ok(
                TOOL_NAME_ADD_OBSERVATION,
                {
                    "patient_id": str(patient_id),
                    "observation_type": observation_type,
                    "previous_value": previous_value,
                    "new_value": stored_value,
                    "duplicate": True,
                    "message": (
                        f"{observation_type} is already {previous_value} — "
                        "no new observation recorded."
                    ),
                },
                metadata={"duplicate": True},
            )

        timestamp = datetime.now(timezone.utc).isoformat()
        observations = record.setdefault("observations", [])
        last_encounter_id = observations[-1].get("encounter_id") if observations else None

        new_observation = {
            "encounter_id": last_encounter_id,
            "timestamp": timestamp,
            "type": observation_type,
            field_info["obs_field"]: stored_value,
        }
        if note:
            new_observation["note"] = note
        observations.append(new_observation)
        record[current_field] = stored_value

        _write_patient_file(str(patient_id), record)

    return ToolResult.ok(
        TOOL_NAME_ADD_OBSERVATION,
        {
            "patient_id": str(patient_id),
            "observation_type": observation_type,
            "previous_value": previous_value,
            "new_value": stored_value,
            "unit": field_info["unit"],
            "timestamp": timestamp,
            "duplicate": False,
        },
        metadata={"patient_id": str(patient_id), "duplicate": False},
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


def add_patient_observation_spec() -> Dict[str, Any]:
    """Return the registration spec for add_patient_observation."""
    from triageguard_agent.tools.registry import ToolSpec, WRITE
    return ToolSpec(
        name=TOOL_NAME_ADD_OBSERVATION,
        description=(
            "Record a new clinical observation (vital sign) for a patient — e.g. the nurse "
            "says a heart rate, SpO2, respiratory rate, blood pressure, or temperature reading "
            "has changed or is now some value. Do NOT include a timestamp — one is recorded "
            "automatically from the system clock the moment this tool runs; never invent or "
            "supply a timestamp yourself. The previous value is preserved in the patient's "
            "observation history, not overwritten. WRITE tool — requires nurse confirmation "
            "before committing. After a successful confirmed write, the triage assessment is "
            "automatically rerun using the updated value — you do not need to call "
            "run_triage_assessment yourself for this."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "patient_id": {"type": ["string", "integer"], "description": "Patient identifier."},
                "observation_type": {
                    "type": "string",
                    "enum": sorted(_OBSERVATION_FIELDS),
                    "description": "Which vital is being recorded.",
                },
                "value": {
                    "type": "number",
                    "description": "The new numeric value for this vital, in its standard unit.",
                },
                "note": {
                    "type": "string",
                    "description": "Optional free-text clinical note about this observation.",
                },
            },
            "required": ["patient_id", "observation_type", "value"],
        },
        handler=lambda **kwargs: add_patient_observation(**kwargs),
        risk_level=WRITE,
        side_effect=True,
        requires_approval=True,
    )
