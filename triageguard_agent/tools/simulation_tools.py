"""
simulation_tools.py
-------------------
Tools for interacting with the live hospital simulation environment.

Exposes tools to:
- Inspect the live hospital simulation dashboard and waiting queues.
- Advance simulation time and trigger automated bed releases.
- Ingest synthetic patient arrivals for triage evaluation.
- Reconcile clinical decisions with live hospital operational constraints.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from triageguard_agent.schemas.tool_result import ToolResult
from triageguard_agent.simulation.hospital_simulator import HospitalSimulator

logger = logging.getLogger(__name__)

# One simulator per hospital_id (Step 7) — was a single shared instance;
# a bare dict keyed by hospital_id is NOT a second hospital registry (no
# identity/config lives here, only a process-local cache of which
# HospitalSimulator object is currently active for that hospital_id, same
# pattern the registry itself uses for HospitalStateService instances).
_simulator_instances: Dict[str, HospitalSimulator] = {}


def _key(hospital_id: Optional[str]) -> str:
    from triageguard_agent.hospital.hospital_registry import DEFAULT_HOSPITAL_ID
    return hospital_id or DEFAULT_HOSPITAL_ID


def get_simulator(hospital_id: Optional[str] = None) -> HospitalSimulator:
    """Return or initialize the shared HospitalSimulator instance for this hospital."""
    key = _key(hospital_id)
    if key not in _simulator_instances:
        _simulator_instances[key] = HospitalSimulator(hospital_id=key)
    return _simulator_instances[key]


def reset_simulator(
    scenario_name: Optional[str] = None,
    hospital_id: Optional[str] = None,
) -> HospitalSimulator:
    """Reset this hospital's simulator instance (test isolation / scenario switching)."""
    key = _key(hospital_id)
    _simulator_instances[key] = HospitalSimulator(scenario=scenario_name, hospital_id=key)
    return _simulator_instances[key]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def get_live_simulation_dashboard(hospital_id: Optional[str] = None) -> ToolResult:
    """Return the current Live Hospital dashboard metrics, occupancy, and event feed."""
    try:
        sim = get_simulator(hospital_id)
        dash = sim.get_live_dashboard()
        return ToolResult.ok("get_live_simulation_dashboard", dash, metadata={"hospital_id": hospital_id})
    except Exception as exc:
        logger.exception("Failed to get live simulation dashboard.")
        return ToolResult.fail("get_live_simulation_dashboard", "SIMULATION_ERROR", str(exc))


def step_simulation_time(
    minutes: int = 15,
    auto_generate_arrivals: bool = True,
    hospital_id: Optional[str] = None,
) -> ToolResult:
    """Advance simulation time, processing patient LOS discharges and bed releases."""
    try:
        sim = get_simulator(hospital_id)
        result = sim.step(minutes=minutes, auto_generate_arrivals=auto_generate_arrivals)
        return ToolResult.ok("step_simulation_time", result, metadata={"hospital_id": hospital_id})
    except Exception as exc:
        logger.exception("Failed to step simulation time.")
        return ToolResult.fail("step_simulation_time", "SIMULATION_ERROR", str(exc))


def trigger_patient_arrival(
    target_acuity: Optional[int] = None,
    hospital_id: Optional[str] = None,
) -> ToolResult:
    """Trigger a new patient arrival into the ED waiting queue."""
    try:
        sim = get_simulator(hospital_id)
        patient = sim.trigger_arrival(target_acuity=target_acuity)
        return ToolResult.ok("trigger_patient_arrival", patient.to_dict(), metadata={"hospital_id": hospital_id})
    except Exception as exc:
        logger.exception("Failed to trigger patient arrival.")
        return ToolResult.fail("trigger_patient_arrival", "SIMULATION_ERROR", str(exc))


def triage_simulated_patient(patient_id: str, hospital_id: Optional[str] = None) -> ToolResult:
    """
    Run full Clinical + Operational triage (first triage) or re-triage
    (already TRIAGED, not yet admitted) evaluation for a patient.

    Calls the exact same HospitalSimulator.triage_patient() the REST
    endpoint (POST /api/simulation/triage/{id}) calls, so the agent/chat
    entry point and the REST entry point behave identically — neither one
    special-cases an already-triaged patient into a stale cached result,
    and both reject an already-admitted patient the same way.
    """
    try:
        sim = get_simulator(hospital_id)
        patient = sim.patient_flow.get_patient(patient_id)
        if not patient:
            return ToolResult.fail(
                "triage_simulated_patient",
                "PATIENT_NOT_FOUND",
                f"Patient {patient_id!r} not found in simulation queue.",
            )
        result = sim.triage_patient(patient)
        return ToolResult.ok("triage_simulated_patient", result, metadata={"hospital_id": hospital_id})
    except ValueError as exc:
        return ToolResult.fail("triage_simulated_patient", "INVALID_PATIENT_STATE", str(exc))
    except Exception as exc:
        logger.exception("Failed to triage simulated patient.")
        return ToolResult.fail("triage_simulated_patient", "SIMULATION_ERROR", str(exc))


def update_simulated_patient_vitals(
    patient_id: str,
    hr: Optional[float] = None,
    rr: Optional[float] = None,
    spo2: Optional[float] = None,
    sbp: Optional[float] = None,
    dbp: Optional[float] = None,
    temp: Optional[float] = None,
    pain: Optional[float] = None,
    hospital_id: Optional[str] = None,
) -> ToolResult:
    """Record new current observations/vitals for an active simulated patient."""
    try:
        sim = get_simulator(hospital_id)
        vitals = {"hr": hr, "rr": rr, "spo2": spo2, "sbp": sbp, "dbp": dbp, "temp": temp, "pain": pain}
        patient = sim.update_patient_vitals(patient_id, vitals)
        return ToolResult.ok(
            "update_simulated_patient_vitals", patient.to_dict(), metadata={"hospital_id": hospital_id}
        )
    except KeyError as exc:
        return ToolResult.fail("update_simulated_patient_vitals", "PATIENT_NOT_FOUND", str(exc))
    except ValueError as exc:
        return ToolResult.fail("update_simulated_patient_vitals", "INVALID_VALUE", str(exc))
    except Exception as exc:
        logger.exception("Failed to update simulated patient vitals.")
        return ToolResult.fail("update_simulated_patient_vitals", "SIMULATION_ERROR", str(exc))


def admit_simulated_patient(
    patient_id: str,
    department: Optional[str] = None,
    custom_los_min: Optional[int] = None,
    hospital_id: Optional[str] = None,
) -> ToolResult:
    """Commit patient admission to a department, occupying a bed."""
    try:
        sim = get_simulator(hospital_id)
        result = sim.admit_patient(patient_id=patient_id, department=department, custom_los_min=custom_los_min)
        return ToolResult.ok("admit_simulated_patient", result, metadata={"hospital_id": hospital_id})
    except Exception as exc:
        logger.exception("Failed to admit simulated patient.")
        return ToolResult.fail("admit_simulated_patient", "SIMULATION_ERROR", str(exc))


# ---------------------------------------------------------------------------
# ToolSpec Factories
# ---------------------------------------------------------------------------

_HOSPITAL_ID_PROPERTY: Dict[str, Any] = {
    "type": "string",
    "description": "Which registered hospital's simulation this applies to. Omit to use the default hospital.",
}


def get_live_simulation_dashboard_spec():
    from triageguard_agent.tools.registry import ToolSpec, READ
    return ToolSpec(
        name="get_live_simulation_dashboard",
        description="Fetch live hospital operational status, department bed gauges, waiting queue, and recent event feed.",
        input_schema={"type": "object", "properties": {"hospital_id": _HOSPITAL_ID_PROPERTY}},
        handler=lambda **kwargs: get_live_simulation_dashboard(**kwargs),
        risk_level=READ,
        side_effect=False,
        requires_approval=False,
    )


def step_simulation_time_spec():
    from triageguard_agent.tools.registry import ToolSpec, COMPUTE
    return ToolSpec(
        name="step_simulation_time",
        description="Advance hospital simulation clock, automatically discharging completed patients and releasing beds.",
        input_schema={
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Minutes to advance clock (default: 15)"},
                "auto_generate_arrivals": {"type": "boolean", "description": "Whether to auto-generate arrivals based on scenario rate"},
                "hospital_id": _HOSPITAL_ID_PROPERTY,
            },
        },
        handler=lambda **kwargs: step_simulation_time(**kwargs),
        risk_level=COMPUTE,
        side_effect=True,
        requires_approval=False,
    )


def trigger_patient_arrival_spec():
    from triageguard_agent.tools.registry import ToolSpec, COMPUTE
    return ToolSpec(
        name="trigger_patient_arrival",
        description="Generate and enqueue a new simulated patient arrival into the emergency department.",
        input_schema={
            "type": "object",
            "properties": {
                "target_acuity": {"type": "integer", "description": "Optional acuity tier (1=Critical, 5=Minor)"},
                "hospital_id": _HOSPITAL_ID_PROPERTY,
            },
        },
        handler=lambda **kwargs: trigger_patient_arrival(**kwargs),
        risk_level=COMPUTE,
        side_effect=True,
        requires_approval=False,
    )


def triage_simulated_patient_spec():
    from triageguard_agent.tools.registry import ToolSpec, COMPUTE
    return ToolSpec(
        name="triage_simulated_patient",
        description="Run clinical triage assessment and evaluate live hospital operational capacity constraints.",
        input_schema={
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient ID to evaluate"},
                "hospital_id": _HOSPITAL_ID_PROPERTY,
            },
            "required": ["patient_id"],
        },
        handler=lambda **kwargs: triage_simulated_patient(**kwargs),
        risk_level=COMPUTE,
        side_effect=False,
        requires_approval=False,
    )


def update_simulated_patient_vitals_spec():
    from triageguard_agent.tools.registry import ToolSpec, COMPUTE
    return ToolSpec(
        name="update_simulated_patient_vitals",
        description=(
            "Record new current observations/vitals (heart rate, respiratory rate, "
            "SpO2, blood pressure, temperature, pain) for a patient who is active in "
            "this hospital's live simulation (waiting queue or admitted). Use this — "
            "not add_patient_observation, which only updates file-based historical "
            "records — for a patient currently in the ED queue or department board. "
            "Does not itself re-triage; call triage_simulated_patient afterward to "
            "get a new AI/routing recommendation from the updated vitals."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient ID"},
                "hr": {"type": "number", "description": "Heart rate (bpm)"},
                "rr": {"type": "number", "description": "Respiratory rate (/min)"},
                "spo2": {"type": "number", "description": "Oxygen saturation (%)"},
                "sbp": {"type": "number", "description": "Systolic blood pressure (mmHg)"},
                "dbp": {"type": "number", "description": "Diastolic blood pressure (mmHg)"},
                "temp": {"type": "number", "description": "Temperature (C)"},
                "pain": {"type": "number", "description": "Pain score (0-10)"},
                "hospital_id": _HOSPITAL_ID_PROPERTY,
            },
            "required": ["patient_id"],
        },
        handler=lambda **kwargs: update_simulated_patient_vitals(**kwargs),
        risk_level=COMPUTE,
        side_effect=True,
        requires_approval=False,
    )


def admit_simulated_patient_spec():
    from triageguard_agent.tools.registry import ToolSpec, WRITE
    return ToolSpec(
        name="admit_simulated_patient",
        description="Admit a simulated patient to a department and occupy a bed. Requires nurse confirmation.",
        input_schema={
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient ID"},
                "department": {"type": "string", "description": "Target department (e.g. ICU, CICU, ADMITTED_GEN, ED_OBS)"},
                "custom_los_min": {"type": "integer", "description": "Optional simulated length of stay in minutes"},
                "hospital_id": _HOSPITAL_ID_PROPERTY,
            },
            "required": ["patient_id"],
        },
        handler=lambda **kwargs: admit_simulated_patient(**kwargs),
        risk_level=WRITE,
        side_effect=True,
        requires_approval=True,
    )
