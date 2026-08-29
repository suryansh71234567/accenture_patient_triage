"""
triageguard_agent.simulation
----------------------------
Lightweight dynamic hospital simulation package for TriageGuard.

Exports:
- HospitalSimulator: Top-level simulation orchestrator.
- EventEngine, SimEvent, EventType: Timeline and event logging.
- PatientFlowManager, SimulatedPatient, PatientStatus: Dynamic lifecycle & LOS management.
- Scenario, SCENARIOS, get_scenario, list_scenarios: Operational scenario configurations.
"""

from triageguard_agent.simulation.event_engine import (
    EventEngine,
    SimEvent,
    EventType,
)
from triageguard_agent.simulation.scenarios import (
    Scenario,
    SCENARIOS,
    NORMAL_DAY,
    BUSY_DAY,
    SURGE_MASS_CASUALTY,
    RESOURCE_CONSTRAINED,
    NIGHT_SHIFT,
    get_scenario,
    list_scenarios,
)
from triageguard_agent.simulation.patient_flow import (
    PatientFlowManager,
    SimulatedPatient,
    PatientStatus,
)
from triageguard_agent.simulation.hospital_simulator import HospitalSimulator

__all__ = [
    "HospitalSimulator",
    "EventEngine",
    "SimEvent",
    "EventType",
    "PatientFlowManager",
    "SimulatedPatient",
    "PatientStatus",
    "Scenario",
    "SCENARIOS",
    "NORMAL_DAY",
    "BUSY_DAY",
    "SURGE_MASS_CASUALTY",
    "RESOURCE_CONSTRAINED",
    "NIGHT_SHIFT",
    "get_scenario",
    "list_scenarios",
]
