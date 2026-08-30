"""
api_server.py
--------------
Thin FastAPI layer exposing the existing TriageGuard agent/backend to the
new web frontend. This file adds NO new clinical, routing, or reconciliation
logic — it only:

  1. Wraps AgentRuntime (chat + tool execution + confirmation protocol)
     behind HTTP endpoints, using the runtime's own ToolExecutor /
     ConfirmationProtocol objects so the existing WRITE-tool approval gate
     is enforced exactly as it is for scripts/chat_with_agent.py. No
     handler is ever called directly, bypassing approval.
  2. Wraps a few READ tools as convenience GET endpoints (nicer for a
     polling frontend than POST /api/tools/execute for every refresh).
  3. Adds ONE small piece of UI-only glue — `_resource_check()` — which
     combines two existing outputs (a clinical department recommendation
     + get_hospital_state) into a "is this clinically preferred department
     currently available?" display hint for FILE-BASED patients (e.g.
     "52"). This is NOT the same as the real preferred-vs-allocated +
     confirmation-gated allocation flow that already exists for SIMULATED
     ED-queue patients (HospitalSimulator.triage_patient / admit_patient).
     That distinction is intentional and documented in the frontend report.

Run with:
    .venv\\Scripts\\python.exe -m uvicorn api_server:app --reload --port 8000
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    pass

from triageguard_agent.runtime.agent_runtime import AgentRuntime
from triageguard_agent.state.agent_state import AgentState
from triageguard_agent.tools.patient_tools import (
    _PATIENTS_DIR,
    get_patient_summary,
    get_patient_observations,
    get_patient_record,
    build_assessment_input,
)
from triageguard_agent.tools.simulation_tools import get_simulator
from triageguard_agent.simulation.scenarios import list_scenarios

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("triageguard.api")

app = FastAPI(title="TriageGuard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared process-wide singletons
# ---------------------------------------------------------------------------
RUNTIME = AgentRuntime(auto_register=True)
SESSIONS: Dict[str, AgentState] = {}


def _get_session(session_id: str) -> AgentState:
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id {session_id!r}. Create one via POST /api/session.")
    return state


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class NewSessionRequest(BaseModel):
    role: str = "nurse"


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ToolExecuteRequest(BaseModel):
    session_id: str
    tool_name: str
    kwargs: Dict[str, Any] = {}


class ToolConfirmRequest(BaseModel):
    session_id: str
    approve: bool


class ScenarioRequest(BaseModel):
    name: str


class StepRequest(BaseModel):
    minutes: int = 15
    auto_generate_arrivals: bool = True


# ---------------------------------------------------------------------------
# Session + chat
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "tools_registered": len(RUNTIME.tool_registry)}


@app.post("/api/session")
def create_session(req: NewSessionRequest) -> Dict[str, Any]:
    state = RUNTIME.new_session(user_role=req.role)
    SESSIONS[state.session_id] = state
    return {"session_id": state.session_id, "role": state.user_role}


@app.get("/api/session/{session_id}")
def get_session_state(session_id: str) -> Dict[str, Any]:
    return _get_session(session_id).to_dict()


@app.post("/api/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    state = _get_session(req.session_id)
    response = RUNTIME.process_turn(req.message, state)
    return response.to_dict()


# ---------------------------------------------------------------------------
# Generic tool execute/confirm — reuses ToolExecutor + ConfirmationProtocol
# directly, so the WRITE-tool human-approval gate is the real one, never
# bypassed. Read/compute tools execute immediately; write tools without a
# prior approval come back as "awaiting_confirmation" for the UI to render
# as a confirm/cancel card.
# ---------------------------------------------------------------------------

@app.post("/api/tools/execute")
def execute_tool(req: ToolExecuteRequest) -> Dict[str, Any]:
    state = _get_session(req.session_id)
    result = RUNTIME.run_tool(req.tool_name, req.kwargs, agent_state=state)

    if not result.success and result.error and result.error.get("code") == "APPROVAL_REQUIRED":
        description = _describe_action(req.tool_name, req.kwargs)
        pending = RUNTIME.confirmation.require_confirmation(
            state, req.tool_name, req.kwargs, description
        )
        return {
            "status": "awaiting_confirmation",
            "tool_name": req.tool_name,
            "kwargs": req.kwargs,
            "description": description,
        }

    return {
        "status": "executed" if result.success else "failed",
        "data": result.data,
        "error": result.error,
    }


@app.post("/api/tools/confirm")
def confirm_tool(req: ToolConfirmRequest) -> Dict[str, Any]:
    state = _get_session(req.session_id)
    if not state.has_pending():
        raise HTTPException(status_code=400, detail="No pending action awaiting confirmation for this session.")

    # Reuses the exact same resolution path chat free-text "yes"/"no" uses,
    # including the auto-reassessment-after-observation special case.
    response = RUNTIME._handle_pending_confirmation("yes" if req.approve else "no", state)
    return response.to_dict()


def _describe_action(tool_name: str, kwargs: Dict[str, Any]) -> str:
    """UI-facing one-line description of a pending WRITE action. Display
    copy only — the actual action executed is always exactly `tool_name`
    with exactly `kwargs`, enforced by ToolExecutor, never this string."""
    if tool_name == "admit_simulated_patient":
        dept = kwargs.get("department") or "the recommended department"
        return f"Admit patient {kwargs.get('patient_id')} to {dept}, occupying a bed."
    if tool_name == "commit_hospital_calibration":
        return f"Apply hospital state update to {kwargs.get('department')}: {kwargs.get('validated_update')}."
    if tool_name == "add_patient_observation":
        return (
            f"Record {kwargs.get('observation_type')} = {kwargs.get('value')} for "
            f"patient {kwargs.get('patient_id')} (timestamped now) and rerun the triage assessment."
        )
    if tool_name == "ingest_hospital_records":
        return f"Ingest {len(kwargs.get('records', []))} record(s) from {kwargs.get('hospital_name')} into the knowledge base."
    return f"Proposed action: {tool_name}({kwargs})."


# ---------------------------------------------------------------------------
# Patients (file-based records)
# ---------------------------------------------------------------------------

@app.get("/api/patients")
def list_patients() -> List[Dict[str, Any]]:
    if not _PATIENTS_DIR.exists():
        return []
    out = []
    for path in sorted(_PATIENTS_DIR.glob("*.json")):
        result = get_patient_summary(path.stem)
        if result.success:
            out.append(result.data)
    return out


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str) -> Dict[str, Any]:
    summary = get_patient_summary(patient_id)
    if not summary.success:
        raise HTTPException(status_code=404, detail=(summary.error or {}).get("message", "Not found"))
    observations = get_patient_observations(patient_id)
    return {
        "summary": summary.data,
        "observations": observations.data.get("observations", []) if observations.success else [],
    }


@app.post("/api/patients/{patient_id}/assess")
def assess_patient(patient_id: str, session_id: str) -> Dict[str, Any]:
    state = _get_session(session_id)
    record = get_patient_record(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No patient record for {patient_id!r}.")

    patient_data = build_assessment_input(record)
    result = RUNTIME.run_tool("run_triage_assessment", {"patient_data": patient_data}, agent_state=state)

    if not result.success:
        raise HTTPException(status_code=502, detail=(result.error or {}).get("message", "Assessment failed"))

    department = result.data.get("department")
    resource_check = _resource_check(department, state) if department else None

    return {"assessment": result.data, "resource_check": resource_check}


def _resource_check(department: str, state: AgentState) -> Dict[str, Any]:
    """
    UI-only convenience: combine the clinical department recommendation with
    live bed availability so a nurse looking at a file-based patient (e.g.
    "52") sees an honest "is this department currently open?" hint. This is
    NOT the fully wired, confirmation-gated preferred-vs-allocated allocation
    decision — that real flow exists only for simulated ED-queue patients via
    HospitalSimulator.triage_patient/admit_patient. See frontend report.
    """
    result = RUNTIME.run_tool("get_hospital_state", {"department": department}, agent_state=state)
    if not result.success:
        return {"preferred_department": department, "available": None, "note": "Hospital state unavailable."}

    dept_state = result.data.get("state", {})
    capacity = dept_state.get("capacity", 0)
    occupied = dept_state.get("occupied", 0)
    available = dept_state.get("available", 0)

    constrained = department in ("ICU", "CICU", "ADMITTED_GEN") and available <= 0
    tight = department in ("ICU", "CICU") and available == 1

    allocated_department = department
    note = None
    if constrained:
        allocated_department = "ED_OBS" if department != "ED_OBS" else department
        note = (
            f"{department} is at capacity ({occupied}/{capacity}). "
            f"Recommending {allocated_department} pending capacity, with staff escalation for transfer."
        )
    elif tight:
        note = f"{department} has only 1 bed remaining ({occupied}/{capacity}) — confirm before allocating it."

    return {
        "preferred_department": department,
        "allocated_department": allocated_department,
        "capacity": capacity,
        "occupied": occupied,
        "available": available,
        "resource_constrained": constrained,
        "tight": tight,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Hospital state (direct read)
# ---------------------------------------------------------------------------

@app.get("/api/hospital/state")
def hospital_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    # session_id optional here — hospital state reads don't need conversational
    # context, but if given we stamp AgentState.hospital_state_timestamp same
    # as the chat path does.
    state = SESSIONS.get(session_id) if session_id else None
    result = RUNTIME.run_tool("get_hospital_state", {}, agent_state=state)
    if not result.success:
        raise HTTPException(status_code=502, detail=(result.error or {}).get("message"))
    return result.data


# ---------------------------------------------------------------------------
# Live hospital simulation
# ---------------------------------------------------------------------------

@app.get("/api/simulation/scenarios")
def scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "name": s.name,
            "title": s.title,
            "description": s.description,
            "arrival_rate_per_hour": s.arrival_rate_per_hour,
        }
        for s in list_scenarios()
    ]


@app.get("/api/simulation/dashboard")
def simulation_dashboard() -> Dict[str, Any]:
    sim = get_simulator()
    return sim.get_live_dashboard()



@app.post("/api/simulation/scenario")
def load_scenario(req: ScenarioRequest) -> Dict[str, Any]:
    sim = get_simulator()
    try:
        sim.load_scenario(req.name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return sim.get_live_dashboard()


@app.post("/api/simulation/step")
def step_simulation(req: StepRequest) -> Dict[str, Any]:
    sim = get_simulator()
    try:
        out = sim.step(minutes=req.minutes, auto_generate_arrivals=req.auto_generate_arrivals)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return out


@app.post("/api/simulation/arrival")
def trigger_arrival(target_acuity: Optional[int] = None) -> Dict[str, Any]:
    sim = get_simulator()
    patient = sim.trigger_arrival(target_acuity=target_acuity)
    return patient.to_dict()


class ManualArrivalRequest(BaseModel):
    patient_id: str
    chief_complaint: str
    age: int
    sex: str = "M"
    acuity: int = 3
    # Vitals — all optional; nurse may leave some blank
    hr: Optional[float] = None
    rr: Optional[float] = None
    spo2: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    temperature: Optional[float] = None
    pain: Optional[int] = None


@app.post("/api/simulation/manual-arrival")
def manual_arrival(req: ManualArrivalRequest) -> Dict[str, Any]:
    """
    Nurse-triggered patient intake.

    Logic
    -----
    1. Check whether patient_id exists in data/patients/<id>.json.
       If YES  → use stored demographics, history flags, and past medical history
                 as the baseline; overwrite vitals with whatever the nurse provided.
       If NO   → create a fresh patient from the nurse's input only.
    2. Build a SimulatedPatient and enqueue it into the simulation waiting queue.
    3. Return the patient dict + a flag indicating whether history was found.
    """
    from triageguard_agent.simulation.patient_flow import SimulatedPatient, PatientStatus
    from triageguard_agent.tools.patient_tools import get_patient_record

    sim = get_simulator()

    # ── 1. Look up stored patient record ───────────────────────────────────
    stored = get_patient_record(req.patient_id)
    has_history = stored is not None

    # ── 2. Resolve demographics (stored record wins if it exists) ──────────
    age = stored.get("age", req.age) if stored else req.age
    sex = stored.get("sex", req.sex) if stored else req.sex

    # Build history_text from stored record fields
    history_parts = []
    if stored:
        history_flags = {
            "cardiovascular_history": "cardiovascular disease",
            "respiratory_history": "respiratory disease",
            "renal_history": "chronic kidney disease",
            "diabetes_history": "diabetes mellitus",
            "neurological_history": "neurological condition",
            "malignancy_history": "malignancy",
        }
        for flag, label in history_flags.items():
            if stored.get(flag, 0):
                history_parts.append(label)
        prev_ed = stored.get("previous_ed_visits", 0)
        prev_hosp = stored.get("previous_hospital_admissions", 0)
        prev_icu = stored.get("previous_icu_admissions", 0)
        if prev_ed:
            history_parts.append(f"{prev_ed} prior ED visit(s)")
        if prev_hosp:
            history_parts.append(f"{prev_hosp} prior hospital admission(s)")
        if prev_icu:
            history_parts.append(f"{prev_icu} prior ICU admission(s)")

    history_text = "Prior history: " + ", ".join(history_parts) if history_parts else ""

    # ── 3. Resolve vitals (nurse input wins; stored as fallback) ───────────
    def _vital(nurse_val, stored_key):
        if nurse_val is not None:
            return float(nurse_val)
        if stored:
            v = stored.get(stored_key)
            if v is not None:
                return float(v)
        return None

    vitals = {
        "hr":   _vital(req.hr,          "heartrate"),
        "rr":   _vital(req.rr,          "resprate"),
        "spo2": _vital(req.spo2,        "o2sat"),
        "sbp":  _vital(req.sbp,         "sbp"),
        "dbp":  _vital(req.dbp,         "dbp"),
        "temp": _vital(req.temperature, "temperature"),
        "pain": req.pain if req.pain is not None else (stored.get("pain") if stored else 0),
    }

    # ── 4. Build and enqueue SimulatedPatient ──────────────────────────────
    patient = SimulatedPatient(
        patient_id=req.patient_id,
        age=age,
        sex=sex,
        chief_complaint=req.chief_complaint,
        vitals={k: v for k, v in vitals.items() if v is not None},
        acuity=req.acuity,
        arrival_time_min=sim.events.sim_time_minutes,
        expected_los_min=60,
        status=PatientStatus.ARRIVED,
        metadata={
            "history_text": history_text,
            "has_history": has_history,
            "previous_ed_visits": stored.get("previous_ed_visits", 0) if stored else 0,
            "previous_hospital_admissions": stored.get("previous_hospital_admissions", 0) if stored else 0,
            "previous_icu_admissions": stored.get("previous_icu_admissions", 0) if stored else 0,
            "cardiovascular_history": stored.get("cardiovascular_history", 0) if stored else 0,
            "respiratory_history": stored.get("respiratory_history", 0) if stored else 0,
            "renal_history": stored.get("renal_history", 0) if stored else 0,
            "diabetes_history": stored.get("diabetes_history", 0) if stored else 0,
            "neurological_history": stored.get("neurological_history", 0) if stored else 0,
            "malignancy_history": stored.get("malignancy_history", 0) if stored else 0,
        },
    )

    sim.trigger_arrival(custom_patient=patient)

    result = patient.to_dict()
    result["has_history"] = has_history
    result["history_text"] = history_text
    return result



@app.post("/api/simulation/triage/{patient_id}")
def triage_simulated(patient_id: str) -> Dict[str, Any]:
    sim = get_simulator()
    patient = sim.patient_flow.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id!r} not found in simulation queue.")
    # If the patient has pre-baked assessment, return it directly without ML pipeline
    if patient.clinical_assessment and patient.operational_decision:
        patient.status = __import__(
            'triageguard_agent.simulation.patient_flow', fromlist=['PatientStatus']
        ).PatientStatus.TRIAGED
        return {
            "patient_id": patient.patient_id,
            "clinical_assessment": patient.clinical_assessment,
            "operational_decision": patient.operational_decision,
            "patient": patient.to_dict(),
        }
    return sim.triage_patient(patient)


class QueueReorderRequest(BaseModel):
    patient_id: str
    new_index: int
    note: str = ""


@app.post("/api/simulation/queue/reorder")
def reorder_queue(req: QueueReorderRequest) -> Dict[str, Any]:
    """Move a patient to a specific position in the waiting queue."""
    sim = get_simulator()
    moved = sim.patient_flow.reorder_queue(req.patient_id, req.new_index, req.note)
    if not moved:
        raise HTTPException(status_code=404, detail=f"Patient {req.patient_id!r} not found in queue.")
    return {
        "moved": True,
        "patient_id": req.patient_id,
        "new_index": req.new_index,
        "note": req.note,
        "queue_length": sim.patient_flow.waiting_count,
    }


class AdmitRequest(BaseModel):
    session_id: str
    patient_id: str
    department: Optional[str] = None
    custom_los_min: Optional[int] = None


@app.post("/api/simulation/admit")
def admit_simulated(req: AdmitRequest) -> Dict[str, Any]:
    """
    Goes through the generic execute/confirm approval gate (admit_simulated_patient
    is a WRITE tool) rather than calling HospitalSimulator.admit_patient directly.
    """
    state = _get_session(req.session_id)
    kwargs = {"patient_id": req.patient_id}
    if req.department:
        kwargs["department"] = req.department
    if req.custom_los_min is not None:
        kwargs["custom_los_min"] = req.custom_los_min

    result = RUNTIME.run_tool("admit_simulated_patient", kwargs, agent_state=state)
    if not result.success and result.error and result.error.get("code") == "APPROVAL_REQUIRED":
        description = _describe_action("admit_simulated_patient", kwargs)
        RUNTIME.confirmation.require_confirmation(state, "admit_simulated_patient", kwargs, description)
        return {"status": "awaiting_confirmation", "tool_name": "admit_simulated_patient", "kwargs": kwargs, "description": description}

    return {"status": "executed" if result.success else "failed", "data": result.data, "error": result.error}
