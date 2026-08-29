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

# Module-level shared simulator instance (singleton for tools)
_simulator_instance: Optional[HospitalSimulator] = None


def get_simulator() -> HospitalSimulator:
    """Return or initialize the shared HospitalSimulator instance."""
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = HospitalSimulator()
    return _simulator_instance


def reset_simulator(scenario_name: Optional[str] = None) -> HospitalSimulator:
    """Reset the simulator instance (useful for test isolation or scenario switching)."""
    global _simulator_instance
    _simulator_instance = HospitalSimulator(scenario=scenario_name)
    return _simulator_instance


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def get_live_simulation_dashboard() -> ToolResult:
    """Return the current Live Hospital dashboard metrics, occupancy, and event feed."""
    try:
        sim = get_simulator()
        dash = sim.get_live_dashboard()
        return ToolResult.ok("get_live_simulation_dashboard", dash)
    except Exception as exc:
        logger.exception("Failed to get live simulation dashboard.")
        return ToolResult.fail("get_live_simulation_dashboard", "SIMULATION_ERROR", str(exc))


def step_simulation_time(minutes: int = 15, auto_generate_arrivals: bool = True) -> ToolResult:
    """Advance simulation time, processing patient LOS discharges and bed releases."""
    try:
        sim = get_simulator()
        result = sim.step(minutes=minutes, auto_generate_arrivals=auto_generate_arrivals)
        return ToolResult.ok("step_simulation_time", result)
    except Exception as exc:
        logger.exception("Failed to step simulation time.")
        return ToolResult.fail("step_simulation_time", "SIMULATION_ERROR", str(exc))


def trigger_patient_arrival(target_acuity: Optional[int] = None) -> ToolResult:
    """Trigger a new patient arrival into the ED waiting queue."""
    try:
        sim = get_simulator()
        patient = sim.trigger_arrival(target_acuity=target_acuity)
        return ToolResult.ok("trigger_patient_arrival", patient.to_dict())
    except Exception as exc:
        logger.exception("Failed to trigger patient arrival.")
        return ToolResult.fail("trigger_patient_arrival", "SIMULATION_ERROR", str(exc))


def triage_simulated_patient(patient_id: str) -> ToolResult:
    """Run full Clinical + Operational triage evaluation for a waiting patient."""
    try:
        sim = get_simulator()
        patient = sim.patient_flow.get_patient(patient_id)
        if not patient:
            return ToolResult.fail(
                "triage_simulated_patient",
                "PATIENT_NOT_FOUND",
                f"Patient {patient_id!r} not found in simulation queue.",
            )
        result = sim.triage_patient(patient)
        return ToolResult.ok("triage_simulated_patient", result)
    except Exception as exc:
        logger.exception("Failed to triage simulated patient.")
        return ToolResult.fail("triage_simulated_patient", "SIMULATION_ERROR", str(exc))


def admit_simulated_patient(
    patient_id: str,
    department: Optional[str] = None,
    custom_los_min: Optional[int] = None,
) -> ToolResult:
    """Commit patient admission to a department, occupying a bed."""
    try:
        sim = get_simulator()
        result = sim.admit_patient(patient_id=patient_id, department=department, custom_los_min=custom_los_min)
        return ToolResult.ok("admit_simulated_patient", result)
    except Exception as exc:
        logger.exception("Failed to admit simulated patient.")
        return ToolResult.fail("admit_simulated_patient", "SIMULATION_ERROR", str(exc))


# ---------------------------------------------------------------------------
# ToolSpec Factories
# ---------------------------------------------------------------------------

def get_live_simulation_dashboard_spec():
    from triageguard_agent.tools.registry import ToolSpec, READ
    return ToolSpec(
        name="get_live_simulation_dashboard",
        description="Fetch live hospital operational status, department bed gauges, waiting queue, and recent event feed.",
        input_schema={"type": "object", "properties": {}},
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
            },
            "required": ["patient_id"],
        },
        handler=lambda **kwargs: triage_simulated_patient(**kwargs),
        risk_level=COMPUTE,
        side_effect=False,
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
            },
            "required": ["patient_id"],
        },
        handler=lambda **kwargs: admit_simulated_patient(**kwargs),
        risk_level=WRITE,
        side_effect=True,
        requires_approval=True,
    )
