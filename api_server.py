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
import threading
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
from triageguard_agent.hospital.hospital_registry import get_default_registry
from triageguard_router.policy import artifacts
from triageguard_router.policy.facility_calibration import scenarios_for_hospital
from triageguard_router.policy.hospital_calibration import NurseResponses, fit_hospital_policy
from triageguard_router.policy.live_routing import _artifact_hospital_id

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

# FastAPI's sync `def` endpoints run on a threadpool, so two manual-arrival
# requests for the same patient_id (double-click, two tabs, a retried
# request) can genuinely execute concurrently. manual_arrival()'s own
# "reject if already active" check is check-then-act, not atomic, so without
# this lock two concurrent requests can both pass the check before either
# has inserted into the waiting queue, producing two SimulatedPatient
# objects for one id — corrupting every id-keyed lookup thereafter. One
# process-wide lock is enough: this endpoint is called rarely (one new
# registration at a time), never a hot path.
_MANUAL_ARRIVAL_LOCK = threading.Lock()


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
    hospital_id: Optional[str] = None


class ToolExecuteRequest(BaseModel):
    session_id: str
    tool_name: str
    kwargs: Dict[str, Any] = {}


class ToolConfirmRequest(BaseModel):
    session_id: str
    approve: bool


class ScenarioRequest(BaseModel):
    name: str
    hospital_id: Optional[str] = None


class StepRequest(BaseModel):
    minutes: int = 15
    auto_generate_arrivals: bool = True
    hospital_id: Optional[str] = None


class DepartmentConfig(BaseModel):
    capacity: int
    occupied: int = 0
    status: str = "OPEN"


class RegisterHospitalRequest(BaseModel):
    hospital_id: str
    hospital_name: str
    departments: Dict[str, DepartmentConfig]


class CalibrationSubmitRequest(BaseModel):
    responses: Dict[str, str]


# ---------------------------------------------------------------------------
# Session + chat
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "tools_registered": len(RUNTIME.tool_registry)}


@app.get("/api/hospitals")
def list_hospitals() -> List[Dict[str, str]]:
    """List registered hospitals (id/name/config path) via the existing HospitalRegistry."""
    return get_default_registry().list_hospitals()


@app.post("/api/hospitals")
def register_hospital(req: RegisterHospitalRequest) -> Dict[str, str]:
    """
    Register a new hospital via the existing HospitalRegistry.register() —
    no new persistence mechanism, just a thin HTTP wrapper. Reuses the same
    validation register() already performs (empty/duplicate hospital_id).
    """
    if not req.departments:
        raise HTTPException(status_code=400, detail="At least one department is required.")

    config_dict = {"departments": {name: cfg.model_dump() for name, cfg in req.departments.items()}}
    try:
        ctx = get_default_registry().register(
            req.hospital_id.strip(), req.hospital_name.strip(), config_dict=config_dict
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"hospital_id": ctx.hospital_id, "hospital_name": ctx.hospital_name}


@app.get("/api/hospitals/{hospital_id}/calibration/status")
def calibration_status(hospital_id: str) -> Dict[str, Any]:
    """Whether this hospital already has a saved calibrated policy (artifacts.artifacts_exist)."""
    if not get_default_registry().exists(hospital_id):
        raise HTTPException(status_code=404, detail=f"Hospital {hospital_id!r} is not registered.")
    return {
        "hospital_id": hospital_id,
        "calibrated": artifacts.artifacts_exist(hospital_id=_artifact_hospital_id(hospital_id)),
    }


@app.get("/api/hospitals/{hospital_id}/calibration/scenarios")
def calibration_scenarios(hospital_id: str) -> Dict[str, Any]:
    """
    The existing nurse-calibration scenarios (demonstrations.py), adapted to
    this hospital's own facility departments via facility_calibration.py's
    scenarios_for_hospital() — candidate_departments already excludes any
    department this hospital doesn't have.
    """
    try:
        ctx = get_default_registry().get(hospital_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    facility_departments = ctx.state_service.get_all()
    result = scenarios_for_hospital(hospital_id, facility_departments)
    return {
        "hospital_id": hospital_id,
        "scenario_count": result["scenario_count"],
        "scenarios": [s.to_dict() for s in result["scenarios"]],
    }


@app.post("/api/hospitals/{hospital_id}/calibration/submit")
def submit_calibration(hospital_id: str, req: CalibrationSubmitRequest) -> Dict[str, Any]:
    """
    Fit + save this hospital's calibrated Bayesian routing policy from the
    nurse's scenario answers, using the existing calibration pipeline
    unchanged (NurseResponses -> fit_hospital_policy -> save_bayesian_policy).
    Artifacts are namespaced per hospital_id (artifacts.py, unchanged) — this
    can never write into another hospital's saved policy.
    """
    try:
        ctx = get_default_registry().get(hospital_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not req.responses:
        raise HTTPException(status_code=400, detail="At least one scenario response is required.")

    facility_departments = ctx.state_service.get_all()
    responses = NurseResponses(hospital_id=hospital_id, responses=req.responses)
    try:
        policy = fit_hospital_policy(hospital_id, facility_departments, responses)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    saved_path = artifacts.save_bayesian_policy(policy, hospital_id=_artifact_hospital_id(hospital_id))

    return {
        "hospital_id": hospital_id,
        "calibrated": True,
        "trained_scenarios": policy.training_metadata.get("scenario_count"),
        "artifact_path": str(saved_path),
    }


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
    if req.hospital_id:
        # Stamped every turn (not just at session creation) so switching the
        # hospital selector mid-conversation takes effect on the very next
        # message — every tool call this turn resolves via AgentRuntime.run_tool's
        # hospital_id auto-fill instead of silently defaulting to "default".
        state.hospital_id = req.hospital_id
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
def assess_patient(patient_id: str, session_id: str, hospital_id: Optional[str] = None) -> Dict[str, Any]:
    state = _get_session(session_id)
    record = get_patient_record(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No patient record for {patient_id!r}.")

    patient_data = build_assessment_input(record)
    if hospital_id:
        # TriageGuardPipeline.run() reads hospital_id off the patient dict
        # when not passed as an explicit kwarg (see combined_pipeline.py) —
        # run_triage_assessment's own signature only takes patient_data, so
        # this is the existing, intended way to thread it through.
        patient_data["hospital_id"] = hospital_id
    result = RUNTIME.run_tool("run_triage_assessment", {"patient_data": patient_data}, agent_state=state)

    if not result.success:
        raise HTTPException(status_code=502, detail=(result.error or {}).get("message", "Assessment failed"))

    resource_check = _resource_check(result.data, state, hospital_id) if result.data.get("department") else None

    return {"assessment": result.data, "resource_check": resource_check}


def _resource_check(assessment: Dict[str, Any], state: AgentState, hospital_id: Optional[str] = None) -> Dict[str, Any]:
    """
    UI-only convenience: combine the clinical department recommendation with
    live bed availability so a nurse looking at a file-based patient (e.g.
    "52") sees an honest "is this department currently open?" hint.

    Phase 6: when this hospital has a calibrated routing policy
    (assessment["policy_applied"], set by run_triage_assessment), trust its
    already-computed operational_department/resource_constraint — the same
    Bayesian/RL-driven, resource-aware, never-fabricated decision the
    simulated-ED-queue flow uses (HospitalSimulator.triage_patient) — instead
    of recomputing a cruder ICU/CICU/ADMITTED_GEN-only threshold check here.
    Falls back to that threshold check exactly as before when no calibrated
    policy exists for this hospital.
    """
    department = assessment.get("department")
    kwargs: Dict[str, Any] = {"department": department}
    if hospital_id:
        kwargs["hospital_id"] = hospital_id
    result = RUNTIME.run_tool("get_hospital_state", kwargs, agent_state=state)
    if not result.success:
        return {"preferred_department": department, "available": None, "note": "Hospital state unavailable."}

    dept_state = result.data.get("state", {})
    capacity = dept_state.get("capacity", 0)
    occupied = dept_state.get("occupied", 0)
    available = dept_state.get("available", 0)

    if assessment.get("policy_applied") and assessment.get("operational_department"):
        allocated_department = assessment["operational_department"]
        constrained = bool(assessment.get("resource_constraint", False))
        tight = bool(assessment.get("human_review_recommended", False)) and not constrained
        note = None
        if constrained:
            note = (
                f"{department} is resource-constrained per this hospital's calibrated routing "
                f"policy. Recommending {allocated_department}."
            )
        elif tight:
            note = "This hospital's routing policy flagged this allocation for human review."
    else:
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
def hospital_state(session_id: Optional[str] = None, hospital_id: Optional[str] = None) -> Dict[str, Any]:
    # session_id optional here — hospital state reads don't need conversational
    # context, but if given we stamp AgentState.hospital_state_timestamp same
    # as the chat path does.
    state = SESSIONS.get(session_id) if session_id else None
    kwargs: Dict[str, Any] = {}
    if hospital_id:
        kwargs["hospital_id"] = hospital_id
    result = RUNTIME.run_tool("get_hospital_state", kwargs, agent_state=state)
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
def simulation_dashboard(hospital_id: Optional[str] = None) -> Dict[str, Any]:
    sim = get_simulator(hospital_id)
    return sim.get_live_dashboard()



@app.post("/api/simulation/scenario")
def load_scenario(req: ScenarioRequest) -> Dict[str, Any]:
    sim = get_simulator(req.hospital_id)
    try:
        sim.load_scenario(req.name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return sim.get_live_dashboard()


@app.post("/api/simulation/step")
def step_simulation(req: StepRequest) -> Dict[str, Any]:
    sim = get_simulator(req.hospital_id)
    try:
        out = sim.step(minutes=req.minutes, auto_generate_arrivals=req.auto_generate_arrivals)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return out


@app.post("/api/simulation/arrival")
def trigger_arrival(target_acuity: Optional[int] = None, hospital_id: Optional[str] = None) -> Dict[str, Any]:
    sim = get_simulator(hospital_id)
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
    hospital_id: Optional[str] = None


@app.post("/api/simulation/manual-arrival")
def manual_arrival(req: ManualArrivalRequest) -> Dict[str, Any]:
    """
    Nurse-triggered patient intake.

    Logic
    -----
    0. Reject if patient_id is already ACTIVE in this hospital's simulation
       (waiting queue or admitted cohort) — including a presimulated demo
       patient. Without this check, registering an id that collides with an
       already-active patient (demo or real) would silently create a
       second SimulatedPatient object sharing that id, corrupting every
       id-keyed lookup (triage/admit/override/get_patient) and the
       frontend's React key for that card. This is deliberately distinct
       from step 1 below: a patient who only exists in HISTORICAL file
       records (not currently active in this hospital's simulation) is not
       a conflict — that is the normal "returning patient" case, and is
       allowed to proceed exactly as before.
    1. Check whether patient_id exists in data/patients/<id>.json.
       If YES  → use stored demographics, history flags, and past medical history
                 as the baseline; overwrite vitals with whatever the nurse provided.
       If NO   → create a fresh patient from the nurse's input only.
    2. Build a SimulatedPatient and enqueue it into the simulation waiting queue.
    3. Return the patient dict + a flag indicating whether history was found.
    """
    from triageguard_agent.simulation.patient_flow import SimulatedPatient, PatientStatus
    from triageguard_agent.tools.patient_tools import get_patient_record

    sim = get_simulator(req.hospital_id)

    # The collision check (0) and the enqueue (4) must be atomic — see
    # _MANUAL_ARRIVAL_LOCK's comment. Everything in between is pure/read-only
    # computation from immutable stored records and is safe to run under the
    # same lock; this endpoint is never a hot path, so holding it for the
    # whole sequence costs nothing in practice.
    with _MANUAL_ARRIVAL_LOCK:
        # ── 0. Reject id collisions with an already-ACTIVE patient ─────────
        existing = sim.patient_flow.get_patient(req.patient_id)
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Patient {req.patient_id!r} is already active in this hospital "
                    f"(status={existing.status.value}). This may be one of the "
                    "hospital's demo patients — pick a different id, or work with "
                    "the existing patient (triage/admit/override) instead of "
                    "re-registering it."
                ),
            )

        # ── 1. Look up stored patient record ────────────────────────────────
        stored = get_patient_record(req.patient_id)
        has_history = stored is not None

        # ── 2. Resolve demographics (stored record wins if it exists) ──────
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

        # ── 3. Resolve vitals (nurse input wins; stored as fallback) ───────
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

        # ── 3.5 Validate temperature is a physiologically plausible °C reading ─
        # Nurse-entered temperature is Celsius everywhere else in the app (the
        # UI, update_patient_vitals, patient_tools' file-based validation) — a
        # value entered in the wrong unit must be rejected here too, not
        # silently accepted and fed into the live clinical assessment. Bounds
        # mirror HospitalSimulator._VITALS_RANGES["temp"] exactly.
        if vitals["temp"] is not None:
            temp_lo, temp_hi = sim._VITALS_RANGES["temp"]
            if not (temp_lo <= vitals["temp"] <= temp_hi):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Vital 'temp' value {vitals['temp']} is outside the valid "
                        f"range [{temp_lo}, {temp_hi}] (expected degrees Celsius)."
                    ),
                )

        # ── 4. Build and enqueue SimulatedPatient ──────────────────────────
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
def triage_simulated(patient_id: str, hospital_id: Optional[str] = None) -> Dict[str, Any]:
    """
    First triage AND re-triage both go through HospitalSimulator's one
    canonical triage_patient() — no stale-cache shortcut here (that used to
    return a cached result for any already-triaged patient without
    recomputing, and diverged from the agent-tool entry point, which always
    recomputed). Both entry points now behave identically; see
    triage_patient()'s own docstring for the first-triage-vs-re-triage and
    already-admitted-rejection rules.
    """
    sim = get_simulator(hospital_id)
    patient = sim.patient_flow.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id!r} not found in simulation queue.")
    try:
        return sim.triage_patient(patient)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class QueueReorderRequest(BaseModel):
    patient_id: str
    new_index: int
    note: str = ""
    hospital_id: Optional[str] = None


@app.post("/api/simulation/queue/reorder")
def reorder_queue(req: QueueReorderRequest) -> Dict[str, Any]:
    """Move a patient to a specific position in the waiting queue."""
    sim = get_simulator(req.hospital_id)
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


class DepartmentReorderRequest(BaseModel):
    patient_id: str
    department: str
    new_index: int
    note: str = ""
    hospital_id: Optional[str] = None


@app.post("/api/simulation/queue/reorder-department")
def reorder_department_queue(req: DepartmentReorderRequest) -> Dict[str, Any]:
    """Drag-and-drop reorder within one department's triaged queue (Phase 9)."""
    sim = get_simulator(req.hospital_id)
    moved = sim.patient_flow.reorder_within_department(req.patient_id, req.department, req.new_index, req.note)
    if not moved:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {req.patient_id!r} not found in the {req.department!r} queue.",
        )
    return {
        "moved": True,
        "patient_id": req.patient_id,
        "department": req.department,
        "new_index": req.new_index,
    }


class QueueOverrideRequest(BaseModel):
    patient_id: str
    department: str
    reason: str = ""
    hospital_id: Optional[str] = None


@app.post("/api/simulation/queue/override")
def override_department(req: QueueOverrideRequest) -> Dict[str, Any]:
    """
    Nurse cross-department override (drag a patient into a different
    department queue). Rejects infeasible moves (unknown/closed/full
    department) cleanly with a 400 rather than fabricating an allocation —
    HospitalSimulator.override_department() checks the same live hospital
    state the routing policy itself reads.
    """
    sim = get_simulator(req.hospital_id)
    try:
        result = sim.override_department(req.patient_id, req.department, req.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


class AdmitRequest(BaseModel):
    session_id: str
    patient_id: str
    department: Optional[str] = None
    custom_los_min: Optional[int] = None
    hospital_id: Optional[str] = None


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
    if req.hospital_id:
        kwargs["hospital_id"] = req.hospital_id

    result = RUNTIME.run_tool("admit_simulated_patient", kwargs, agent_state=state)
    if not result.success and result.error and result.error.get("code") == "APPROVAL_REQUIRED":
        description = _describe_action("admit_simulated_patient", kwargs)
        RUNTIME.confirmation.require_confirmation(state, "admit_simulated_patient", kwargs, description)
        return {"status": "awaiting_confirmation", "tool_name": "admit_simulated_patient", "kwargs": kwargs, "description": description}

    return {"status": "executed" if result.success else "failed", "data": result.data, "error": result.error}


class SimulatedVitalsUpdateRequest(BaseModel):
    hr: Optional[float] = None
    rr: Optional[float] = None
    spo2: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    temp: Optional[float] = None
    pain: Optional[float] = None
    hospital_id: Optional[str] = None


@app.post("/api/simulation/patient/{patient_id}/vitals")
def update_simulated_patient_vitals(patient_id: str, req: SimulatedVitalsUpdateRequest) -> Dict[str, Any]:
    """
    Record new current observations/vitals for an ACTIVE simulated patient
    (waiting queue or admitted cohort) — mirrors add_patient_observation's
    role for file-based patients, but for the hospital-scoped simulation
    queue this app actually drives (Dashboard/Live Hospital/department
    queue board). Does not itself re-triage — the caller/nurse calls
    POST /api/simulation/triage/{patient_id} afterward to re-triage with
    the updated vitals, same two-step composition add_patient_observation +
    run_triage_assessment already uses for file-based patients.
    """
    sim = get_simulator(req.hospital_id)
    vitals = {
        "hr": req.hr, "rr": req.rr, "spo2": req.spo2,
        "sbp": req.sbp, "dbp": req.dbp, "temp": req.temp, "pain": req.pain,
    }
    try:
        patient = sim.update_patient_vitals(patient_id, vitals)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return patient.to_dict()
