"""
test_dynamic_simulation.py
--------------------------
Unit and integration tests for the dynamic hospital simulation subsystem.

Covers:
- EventEngine clock management, event emission, listener callbacks, and log filtering.
- Scenario loading, parameter initialization, and operating mode shifts.
- Patient lifecycle progression and synthetic profile generation.
- Automated bed release upon Length of Stay (LOS) expiration.
- Clinical Truth vs Operational Truth reconciliation under normal and constrained conditions.
- Simulation tools integration with AgentRuntime.
"""

from __future__ import annotations
import pytest
from typing import Dict, Any

from triageguard_agent.simulation.event_engine import EventEngine, EventType, SimEvent
from triageguard_agent.simulation.scenarios import (
    Scenario,
    get_scenario,
    list_scenarios,
    NORMAL_DAY,
    BUSY_DAY,
    SURGE_MASS_CASUALTY,
    RESOURCE_CONSTRAINED,
)
from triageguard_agent.simulation.patient_flow import (
    PatientFlowManager,
    SimulatedPatient,
    PatientStatus,
)
from triageguard_agent.simulation.hospital_simulator import HospitalSimulator
from triageguard_agent.hospital.hospital_state_service import HospitalStateService
from triageguard_agent.runtime.agent_runtime import AgentRuntime


@pytest.fixture(autouse=True)
def reset_singletons():
    """Ensure a clean HospitalStateService singleton for every test."""
    HospitalStateService.reset_instance()
    yield
    HospitalStateService.reset_instance()


# ---------------------------------------------------------------------------
# 1. EventEngine Tests
# ---------------------------------------------------------------------------

class TestEventEngine:
    def test_clock_progression(self):
        engine = EventEngine(start_hour=10, start_minute=0)
        assert engine.formatted_time == "10:00"
        assert engine.sim_time_minutes == 0

        engine.advance_time(15)
        assert engine.formatted_time == "10:15"
        assert engine.sim_time_minutes == 15

        engine.advance_time(90)
        assert engine.formatted_time == "11:45"
        assert engine.sim_time_minutes == 105

    def test_event_emission_and_formatting(self):
        engine = EventEngine(start_hour=14, start_minute=0)
        ev = engine.emit(
            EventType.PATIENT_ARRIVAL,
            "Patient PAT-1 arrived in ED",
            patient_id="PAT-1",
            data={"acuity": 2},
        )
        assert ev.event_type == EventType.PATIENT_ARRIVAL
        assert ev.formatted_time == "14:00"
        assert "PAT-1 arrived" in ev.format_log_line()
        assert len(engine.get_history()) == 1

    def test_listener_dispatch(self):
        engine = EventEngine()
        captured = []

        def on_event(event: SimEvent):
            captured.append(event)

        engine.add_listener(on_event)
        engine.emit(EventType.BED_OPENED, "Bed opened in ICU", department="ICU")

        assert len(captured) == 1
        assert captured[0].department == "ICU"

        engine.remove_listener(on_event)
        engine.emit(EventType.BED_CLOSED, "Bed closed in ICU", department="ICU")
        assert len(captured) == 1

    def test_history_filtering(self):
        engine = EventEngine()
        engine.emit(EventType.PATIENT_ARRIVAL, "P1 arrived", patient_id="P1")
        engine.emit(EventType.PATIENT_ARRIVAL, "P2 arrived", patient_id="P2")
        engine.emit(EventType.CAPACITY_WARNING, "ICU full", department="ICU")

        assert len(engine.get_history(event_type=EventType.PATIENT_ARRIVAL)) == 2
        assert len(engine.get_history(department="ICU")) == 1
        assert len(engine.get_history(patient_id="P1")) == 1
        assert len(engine.get_recent_feed(limit=2)) == 2


# ---------------------------------------------------------------------------
# 2. Scenario Tests
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_list_scenarios(self):
        scenarios = list_scenarios()
        assert len(scenarios) >= 4
        names = [s.name for s in scenarios]
        assert "NORMAL_DAY" in names
        assert "BUSY_DAY" in names
        assert "SURGE_MASS_CASUALTY" in names
        assert "RESOURCE_CONSTRAINED" in names

    def test_get_scenario_by_name(self):
        scen = get_scenario("busy_day")
        assert scen.name == "BUSY_DAY"
        assert scen.arrival_rate_per_hour == 6.0

    def test_scenario_load_updates_hospital_state(self):
        svc = HospitalStateService.instance()
        sim = HospitalSimulator(scenario="RESOURCE_CONSTRAINED", state_service=svc)

        icu = svc.get_state("ICU")
        assert icu["capacity"] == 10
        assert icu["occupied"] == 9
        assert icu["available"] == 1

        cicu = svc.get_state("CICU")
        assert cicu["occupied"] == 6
        assert cicu["available"] == 0

        gen = svc.get_state("ADMITTED_GEN")
        assert gen["occupied"] == 50
        assert gen["available"] == 0


# ---------------------------------------------------------------------------
# 3. Patient Flow & Dynamic Lifecycle Tests
# ---------------------------------------------------------------------------

class TestPatientFlow:
    def test_synthetic_patient_generation(self):
        flow = PatientFlowManager(initial_patient_counter=100)
        p1 = flow.generate_patient(sim_time_min=0, target_acuity=1)
        assert p1.patient_id == "PAT-101"
        assert p1.acuity == 1
        assert p1.status == PatientStatus.ARRIVED
        assert p1.expected_los_min >= 60
        assert "spo2" in p1.vitals

    def test_pipeline_dict_export(self):
        flow = PatientFlowManager()
        p = flow.generate_patient(sim_time_min=0, target_acuity=2)
        d = p.to_pipeline_dict()
        assert "hr_current" in d
        assert "sbp_current" in d
        assert "triage_complaint" in d
        assert d["acuity"] == 2

    def test_queue_and_pop(self):
        flow = PatientFlowManager()
        p1 = flow.generate_patient(0, target_acuity=3)
        p2 = flow.generate_patient(0, target_acuity=4)
        flow.enqueue_patient(p1)
        flow.enqueue_patient(p2)

        assert flow.waiting_count == 2
        popped = flow.pop_next_waiting()
        assert popped.patient_id == p1.patient_id
        assert flow.waiting_count == 1

    def test_los_advancement_and_auto_discharge(self):
        flow = PatientFlowManager()
        p = flow.generate_patient(sim_time_min=0)
        flow.admit_patient(p, department="ICU", custom_los_min=30)

        assert flow.admitted_count == 1
        assert p.status == PatientStatus.IN_TREATMENT

        # Advance 15m -> not yet expired
        discharged_15 = flow.advance_time(15)
        assert len(discharged_15) == 0
        assert flow.admitted_count == 1
        assert p.elapsed_los_min == 15

        # Advance another 15m -> reaches 30m -> auto discharged
        discharged_30 = flow.advance_time(15)
        assert len(discharged_30) == 1
        assert discharged_30[0].patient_id == p.patient_id
        assert discharged_30[0].status == PatientStatus.DISCHARGED
        assert flow.admitted_count == 0


# ---------------------------------------------------------------------------
# 4. Hospital Simulator & Automated Bed Release
# ---------------------------------------------------------------------------

class TestHospitalSimulator:
    def test_simulator_step_advances_time_and_releases_bed(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        svc = sim.state_service

        # Set known starting occupancy
        svc.apply_update("ICU", {"occupied": 8, "capacity": 10})
        assert svc.get_state("ICU")["occupied"] == 8
        assert svc.get_state("ICU")["available"] == 2

        # Arrive and admit a patient with 30m LOS
        p = sim.trigger_arrival(target_acuity=1)
        sim.admit_patient(p.patient_id, department="ICU", custom_los_min=30)

        # Occupancy should have increased to 9
        assert svc.get_state("ICU")["occupied"] == 9
        assert svc.get_state("ICU")["available"] == 1

        # Advance 15 min -> still occupied
        out1 = sim.step(minutes=15, auto_generate_arrivals=False)
        assert out1["discharged_count"] == 0
        assert svc.get_state("ICU")["occupied"] == 9

        # Advance another 15 min -> 30 min reached -> patient discharged and bed released!
        out2 = sim.step(minutes=15, auto_generate_arrivals=False)
        assert out2["discharged_count"] == 1
        assert p.patient_id in out2["discharged_patient_ids"]

        # Occupancy must have decreased back to 8!
        assert svc.get_state("ICU")["occupied"] == 8
        assert svc.get_state("ICU")["available"] == 2

    def test_clinical_vs_operational_truth_when_capacity_available(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        # In NORMAL_DAY, ICU has 7/10 occupied (3 beds open)
        p = sim.trigger_arrival(target_acuity=1)
        res = sim.triage_patient(p)

        clin = res["clinical_assessment"]
        op = res["operational_decision"]

        assert clin["department"] in ("ICU", "CICU")
        assert op["operational_department"] in ("ICU", "CICU")
        assert op["capacity_warning"] is False

    def test_clinical_vs_operational_truth_when_department_full(self):
        sim = HospitalSimulator(scenario="RESOURCE_CONSTRAINED")
        svc = sim.state_service

        # Fill all inpatient departments to 100% capacity
        for dept in ["ICU", "CICU", "ADMITTED_GEN"]:
            curr = svc.get_state(dept)
            if curr:
                cap = curr["capacity"]
                svc.apply_update(dept, {"occupied": cap, "capacity": cap})
                assert svc.get_state(dept)["available"] == 0

        p = sim.patient_flow.generate_patient(sim_time_min=0, target_acuity=1)
        sim.patient_flow.enqueue_patient(p)

        res = sim.triage_patient(p)
        clin = res["clinical_assessment"]
        op = res["operational_decision"]

        # Clinical Truth records true clinical need
        assert clin["department"] in ("ICU", "CICU", "ADMITTED_GEN")
        # Operational Truth sees 0 available beds -> triggers capacity warning and routes to ED_OBS
        assert op["capacity_warning"] is True
        assert op["confirmation_required"] is True
        assert op["operational_department"] == "ED_OBS"
        assert "AT CAPACITY" in op["recommendation_summary"]

    def test_critical_reservation_warning_on_last_bed(self):
        sim = HospitalSimulator(scenario="BUSY_DAY")
        svc = sim.state_service

        p = sim.patient_flow.generate_patient(sim_time_min=0, target_acuity=1)
        sim.patient_flow.enqueue_patient(p)

        clin_eval = sim._evaluate_clinical_truth(p)
        clin_dept = clin_eval["department"]
        if clin_dept not in ("ICU", "CICU"):
            clin_dept = "ICU"

        # Set target critical department to 1 bed remaining
        curr = svc.get_state(clin_dept)
        if curr:
            cap = curr["capacity"]
            svc.apply_update(clin_dept, {"occupied": cap - 1, "capacity": cap})

        res = sim.triage_patient(p)
        op = res["operational_decision"]

        if op["clinical_department"] in ("ICU", "CICU") and op["available_beds_in_clinical_dept"] == 1:
            assert op["confirmation_required"] is True
            assert "ONLY 1 BED REMAINING" in op["recommendation_summary"]

    def test_live_dashboard_structure(self):
        sim = HospitalSimulator(scenario="BUSY_DAY")
        sim.trigger_arrival()
        dash = sim.get_live_dashboard()

        assert "time" in dash
        assert "scenario" in dash
        assert "load" in dash
        assert "departments" in dash
        assert len(dash["departments"]) >= 4
        assert dash["waiting_count"] == 1
        assert len(dash["recent_events"]) >= 1


# ---------------------------------------------------------------------------
# 5. Integration with Simulation Tools and AgentRuntime
# ---------------------------------------------------------------------------

class TestSimulationToolsAndRuntime:
    def test_runtime_registers_simulation_tools(self):
        runtime = AgentRuntime(auto_register=True)
        tool_names = [t["name"] for t in runtime.get_tools_for_llm()]

        assert "get_live_simulation_dashboard" in tool_names
        assert "step_simulation_time" in tool_names
        assert "trigger_patient_arrival" in tool_names
        assert "triage_simulated_patient" in tool_names
        assert "admit_simulated_patient" in tool_names

    def test_step_simulation_tool_execution(self):
        runtime = AgentRuntime(auto_register=True)
        res = runtime.run_tool("step_simulation_time", {"minutes": 15})
        assert res.success is True
        assert "sim_time_minutes" in res.data
        assert "load_ratio" in res.data

    def test_trigger_and_triage_simulated_patient_tools(self):
        runtime = AgentRuntime(auto_register=True)
        arr_res = runtime.run_tool("trigger_patient_arrival", {"target_acuity": 2})
        assert arr_res.success is True
        pid = arr_res.data["patient_id"]

        triage_res = runtime.run_tool("triage_simulated_patient", {"patient_id": pid})
        assert triage_res.success is True
        assert "clinical_assessment" in triage_res.data
        assert "operational_decision" in triage_res.data
