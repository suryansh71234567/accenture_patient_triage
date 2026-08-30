"""
hospital_simulator.py
---------------------
Top-level Dynamic Hospital Environment for TriageGuard.

Coordinates:
1. EventEngine (clock, event timeline, listeners)
2. PatientFlowManager (patient lifecycles, LOS tracking, automated bed release)
3. HospitalStateService & Store (thread-safe operational state)
4. HospitalLoadController (dynamic λ and operating mode recalculation)
5. Clinical Truth vs. Operational Truth reconciliation
"""

from __future__ import annotations
import logging
import random
from typing import Any, Dict, List, Optional

from triageguard_agent.hospital.hospital_state_service import HospitalStateService
from triageguard_agent.hospital.hospital_load_controller import HospitalLoadController
from triageguard_agent.simulation.event_engine import EventEngine, EventType, SimEvent
from triageguard_agent.simulation.scenarios import Scenario, get_scenario, NORMAL_DAY
from triageguard_agent.simulation.patient_flow import (
    PatientFlowManager,
    SimulatedPatient,
    PatientStatus,
)
from triageguard_agent.simulation.presimulated_patients import (
    WAITING_IDS_BY_SCENARIO,
    TRIAGED_IDS_BY_SCENARIO,
    ADMITTED_IDS_BY_SCENARIO,
    build_simulated_patient,
    get_patient_by_id,
)

logger = logging.getLogger(__name__)


def calibrated_clinical_fallback(
    acuity: int,
    chief_complaint: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministic, clinically calibrated rule used when the real
    XGBoost+RAG pipeline is unavailable (or, deliberately, by RL training —
    see triageguard_router/policy/simulation_env.py) instead of the live,
    network-attached pipeline, so thousands of training episodes stay fast,
    free, and reproducible under a fixed seed. Extracted unchanged from
    HospitalSimulator._evaluate_clinical_truth's own except-branch so both
    callers share one definition instead of two.
    """
    cardiac = bool((metadata or {}).get("cardiac_hint", False))

    if acuity == 1:
        dept = "CICU" if cardiac else "ICU"
        reasoning = "Critical acuity (Acuity 1) with severe vital instability."
        admission_risk = 0.95
        icu_risk = 0.85
    elif acuity == 2:
        dept = "CICU" if cardiac else "ICU"
        reasoning = "High acuity (Acuity 2) requiring close monitoring."
        admission_risk = 0.80
        icu_risk = 0.45
    elif acuity == 3:
        dept = "ADMITTED_GEN"
        reasoning = "Moderate acute condition requiring hospital admission."
        admission_risk = 0.65
        icu_risk = 0.15
    elif acuity == 4:
        dept = "ED_OBS"
        reasoning = "Low-moderate acuity suitable for ED observation."
        admission_risk = 0.35
        icu_risk = 0.05
    else:
        dept = "DISCHARGE"
        reasoning = "Low acuity, normal vitals, safe for discharge."
        admission_risk = 0.10
        icu_risk = 0.01

    return {
        "department": dept,
        "department_reasoning": reasoning,
        "acuity_tier": acuity,
        "reconciled_admission_risk": admission_risk,
        "reconciled_icu_risk": icu_risk,
        "branches_agree": True,
        "confidence_note": "Rule-calibrated assessment.",
        "top_diagnoses": [chief_complaint[:30]],
        "red_flags": ["Severe vitals instability"] if acuity <= 2 else [],
    }


class HospitalSimulator:
    """
    Simulates a dynamic, responsive hospital environment with continuous time progression,
    stochastic/scheduled patient flow, capacity constraints, and operational-clinical reconciliation.
    """

    def __init__(
        self,
        scenario: Optional[Scenario | str] = None,
        start_hour: int = 10,
        start_minute: int = 0,
        state_service: Optional[HospitalStateService] = None,
        hospital_id: Optional[str] = None,
    ) -> None:
        """
        hospital_id : resolves this simulator's state via the Step 2/3
            HospitalRegistry (get_default_registry().get(hospital_id)) —
            "default" (or omitted) mirrors the original single-hospital
            behavior exactly (the same process-wide HospitalStateService
            singleton). Ignored if `state_service` is given explicitly.
            self.hospital_id is always set to a concrete string ("default"
            when omitted) so it can be threaded into the clinical pipeline
            without hospital identity silently disappearing (see
            _evaluate_clinical_truth).
        """
        from triageguard_agent.hospital.hospital_registry import DEFAULT_HOSPITAL_ID

        self.hospital_id = hospital_id or DEFAULT_HOSPITAL_ID
        self.events = EventEngine(start_hour=start_hour, start_minute=start_minute)
        self.patient_flow = PatientFlowManager()
        if state_service is not None:
            self.state_service = state_service
        else:
            from triageguard_agent.hospital.hospital_registry import get_default_registry
            self.state_service = get_default_registry().get(self.hospital_id).state_service
        self.load_controller = HospitalLoadController()

        # Load initial scenario
        initial_scen = scenario or NORMAL_DAY
        self._current_scenario: Scenario = (
            get_scenario(initial_scen) if isinstance(initial_scen, str) else initial_scen
        )
        self.load_scenario(self._current_scenario, reset_clock=False)

    # ------------------------------------------------------------------
    # Scenario Management
    # ------------------------------------------------------------------

    @property
    def current_scenario(self) -> Scenario:
        return self._current_scenario

    def load_scenario(self, scenario: Scenario | str, reset_clock: bool = False) -> None:
        """Apply a scenario's configuration to the hospital state and reset queues."""
        if isinstance(scenario, str):
            scenario = get_scenario(scenario)
        self._current_scenario = scenario

        if reset_clock:
            self.events.clear()
            self.patient_flow.clear()

        # Update hospital state store via state service for each department
        for dept, cfg in scenario.department_state.items():
            if not self.state_service.department_exists(dept):
                continue
            cap = int(cfg.get("capacity", 0))
            occ = int(cfg.get("occupied", 0))
            status = cfg.get("status", "OPEN")
            patch = {
                "capacity": cap,
                "occupied": min(occ, cap),
                "status": status,
            }
            # Direct store patch for scenario setup
            self.state_service._store.apply(dept, patch)

        # Recalculate load
        load_info = self.load_controller.recalculate(self.state_service.get_all())

        self.events.emit(
            EventType.SCENARIO_CHANGED,
            f"Hospital scenario switched to {scenario.title} (Mode: {load_info['operating_mode']})",
            data={
                "scenario": scenario.name,
                "operating_mode": load_info["operating_mode"],
                "lambda": load_info["lambda"],
                "load_ratio": load_info["load_ratio"],
            },
        )
        logger.info("Loaded scenario %s into HospitalSimulator.", scenario.name)

        # Inject scenario-appropriate pre-simulated patients
        self._inject_presimulated_patients(scenario.name)

    def _inject_presimulated_patients(self, scenario_name: str) -> None:
        """Pre-populate the queue and admitted cohort with scenario-appropriate patients."""
        # Clear existing queue so switching scenarios gives a fresh state
        self.patient_flow.clear()
        t = self.events.sim_time_minutes
        load_info = self.load_controller.recalculate(self.state_service.get_all())
        lam = load_info.get("lambda", 0.6)
        mode = load_info.get("operating_mode", "NORMAL")

        # ── 1. Admitted patients (already in beds, using capacity) ─────
        for pid in ADMITTED_IDS_BY_SCENARIO.get(scenario_name, []):
            entry = get_patient_by_id(pid)
            if not entry:
                continue
            patient = build_simulated_patient(entry, t)
            op = patient.operational_decision or {}
            dept = op.get("operational_department", "ADMITTED_GEN")
            patient.status = PatientStatus.IN_TREATMENT
            patient.department = dept
            patient.elapsed_los_min = patient.expected_los_min // 3
            self.patient_flow._admitted_cohort[patient.patient_id] = patient
            # Reflect in bed occupancy
            if dept != "DISCHARGE" and self.state_service.department_exists(dept):
                curr = self.state_service.get_state(dept)
                if curr:
                    new_occ = min(curr["capacity"], curr["occupied"] + 1)
                    self.state_service.apply_update(dept, {"occupied": new_occ})

        # ── 2. Triaged patients (assessment done, awaiting admission) ──
        for pid in TRIAGED_IDS_BY_SCENARIO.get(scenario_name, []):
            entry = get_patient_by_id(pid)
            if not entry:
                continue
            patient = build_simulated_patient(entry, t)
            patient.status = PatientStatus.TRIAGED
            # Update operational decision with current mode / lambda
            if patient.operational_decision:
                patient.operational_decision["operating_mode"] = mode
                patient.operational_decision["lambda"] = lam
            self.patient_flow.enqueue_patient(patient)

        # ── 3. Waiting patients (arrived, not yet triaged) ─────────────
        for pid in WAITING_IDS_BY_SCENARIO.get(scenario_name, []):
            entry = get_patient_by_id(pid)
            if not entry:
                continue
            patient = build_simulated_patient(entry, t)
            patient.status = PatientStatus.ARRIVED
            patient.clinical_assessment = None  # not yet triaged
            patient.operational_decision = None
            self.patient_flow.enqueue_patient(patient)

        logger.info(
            "Pre-populated scenario '%s': %d waiting, %d admitted",
            scenario_name,
            self.patient_flow.waiting_count,
            self.patient_flow.admitted_count,
        )

    # ------------------------------------------------------------------
    # Time Stepping & Automated Bed Release
    # ------------------------------------------------------------------

    def step(self, minutes: int = 15, auto_generate_arrivals: bool = True) -> Dict[str, Any]:
        """
        Advance simulation time by delta minutes.

        1. Advances simulated clock.
        2. Advances LOS for all currently admitted patients.
        3. Automatically discharges patients whose LOS has expired and FREES their beds.
        4. Stochastically generates new ED arrivals based on the current scenario rate.
        5. Recalculates hospital load and λ.
        """
        if minutes <= 0:
            raise ValueError("Step minutes must be positive.")

        # 1. Advance clock
        clock_str = self.events.advance_time(minutes)

        # 2 & 3. Advance LOS and release expired beds
        discharged_patients = self.patient_flow.advance_time(minutes)
        for p in discharged_patients:
            dept = p.department
            if dept and dept != "DISCHARGE" and self.state_service.department_exists(dept):
                curr_state = self.state_service.get_state(dept)
                if curr_state and curr_state.get("occupied", 0) > 0:
                    new_occ = max(0, curr_state["occupied"] - 1)
                    self.state_service.apply_update(dept, {"occupied": new_occ})
                    self.events.emit(
                        EventType.BED_OPENED,
                        f"Bed released in {dept} (Occupancy: {new_occ}/{curr_state['capacity']})",
                        department=dept,
                    )
            self.events.emit(
                EventType.PATIENT_DISCHARGED,
                f"Patient {p.patient_id} treatment complete — discharged from {dept or 'ED'}",
                department=dept,
                patient_id=p.patient_id,
            )

        # 4. Generate new arrivals if enabled
        new_arrivals: List[SimulatedPatient] = []
        if auto_generate_arrivals:
            # Expected arrivals in this time window: λ_arr = rate_per_hour * (minutes / 60)
            expected_arrivals = (self._current_scenario.arrival_rate_per_hour * (minutes / 60.0))
            # Poisson-like integer sampling
            num_arrivals = 0
            while expected_arrivals > 0:
                if random.random() < min(1.0, expected_arrivals):
                    num_arrivals += 1
                expected_arrivals -= 1.0

            for _ in range(num_arrivals):
                patient = self.trigger_arrival()
                new_arrivals.append(patient)

        # 5. Recalculate hospital load
        all_state = self.state_service.get_all()
        load_info = self.load_controller.recalculate(all_state)

        step_result = {
            "time": clock_str,
            "sim_time_minutes": self.events.sim_time_minutes,
            "discharged_count": len(discharged_patients),
            "discharged_patient_ids": [p.patient_id for p in discharged_patients],
            "new_arrivals_count": len(new_arrivals),
            "new_patient_ids": [p.patient_id for p in new_arrivals],
            "waiting_queue_count": self.patient_flow.waiting_count,
            "admitted_count": self.patient_flow.admitted_count,
            "operating_mode": load_info["operating_mode"],
            "load_ratio": load_info["load_ratio"],
            "lambda": load_info["lambda"],
        }

        self.events.emit(
            EventType.TIME_ADVANCED,
            f"Time advanced +{minutes}m → {clock_str} (Load: {load_info['load_ratio']:.1%}, Mode: {load_info['operating_mode']})",
            data=step_result,
        )

        return step_result

    # ------------------------------------------------------------------
    # Patient Ingestion & Triage
    # ------------------------------------------------------------------

    def trigger_arrival(
        self,
        target_acuity: Optional[int] = None,
        custom_patient: Optional[SimulatedPatient] = None,
    ) -> SimulatedPatient:
        """Manually trigger a patient arrival into the ED waiting queue."""
        if custom_patient:
            patient = custom_patient
        else:
            patient = self.patient_flow.generate_patient(
                sim_time_min=self.events.sim_time_minutes,
                target_acuity=target_acuity,
                acuity_weights=self._current_scenario.acuity_weights,
            )

        self.patient_flow.enqueue_patient(patient)
        self.events.emit(
            EventType.PATIENT_ARRIVAL,
            f"Patient {patient.patient_id} arrived (Acuity {patient.acuity}: {patient.chief_complaint[:40]}...)",
            patient_id=patient.patient_id,
            data={"acuity": patient.acuity, "complaint": patient.chief_complaint},
        )
        return patient

    def triage_patient(self, patient: SimulatedPatient) -> Dict[str, Any]:
        """
        Execute full Clinical Assessment and reconcile with Operational Truth.

        Separates:
        1. Clinical Truth (XGBoost + RAG): What care does the patient clinically require?
        2. Operational Truth (Hospital State): What beds and resources are available right now?
        3. Operational Recommendation & Escalation: What should staff confirm/action?
        """
        # ── 1. Clinical Truth (Pipeline or fallback) ───────────────────────
        clinical_output = self._evaluate_clinical_truth(patient)
        clinical_dept = clinical_output.get("department", "ADMITTED_GEN")
        acuity_tier = clinical_output.get("acuity_tier", patient.acuity)

        # ── 2. Operational Truth (Live Department State) ───────────────────
        all_state = self.state_service.get_all()
        dept_state = self.state_service.get_state(clinical_dept) or {"capacity": 1, "occupied": 0, "available": 1}
        available_beds = dept_state.get("available", 0)
        capacity = dept_state.get("capacity", 0)
        occupied = dept_state.get("occupied", 0)

        load_info = self.load_controller.recalculate(all_state)
        operating_mode = load_info["operating_mode"]

        # ── 3. Synthesize Operational Recommendation ───────────────────────
        confirmation_required = False
        capacity_warning = False
        operational_dept = clinical_dept
        recommendation_notes = []

        if clinical_dept in ("ICU", "CICU"):
            if available_beds <= 0:
                capacity_warning = True
                confirmation_required = True
                # Alternate operational routing when critical care is full
                operational_dept = "ED_OBS"
                recommendation_notes.append(
                    f"{clinical_dept} is AT CAPACITY (0/{capacity} available). "
                    f"Clinical risk requires ICU-level care. "
                    "ACTION REQUIRED: Request staff escalation / external transfer OR place in ED Observation with 1:1 telemetry monitoring."
                )
                self.events.emit(
                    EventType.CAPACITY_WARNING,
                    f"{clinical_dept} capacity exhausted for critical Patient {patient.patient_id}.",
                    department=clinical_dept,
                    patient_id=patient.patient_id,
                )
            elif available_beds == 1 and operating_mode in ("HIGH_LOAD", "CRITICAL"):
                confirmation_required = True
                recommendation_notes.append(
                    f"Clinical assessment recommends {clinical_dept}. "
                    f"{clinical_dept} has ONLY 1 BED REMAINING (Load: {operating_mode}). "
                    f"Confirm reserving the final {clinical_dept} bed?"
                )
            else:
                confirmation_required = False
                recommendation_notes.append(
                    f"{clinical_dept} bed available ({available_beds}/{capacity} open). Direct admission recommended."
                )

        elif clinical_dept == "ADMITTED_GEN":
            if available_beds <= 0:
                capacity_warning = True
                confirmation_required = True
                operational_dept = "ED_OBS"
                recommendation_notes.append(
                    "General Ward is AT CAPACITY. Board patient in ED Observation pending bed discharge."
                )
                self.events.emit(
                    EventType.CAPACITY_WARNING,
                    f"General Ward full. Patient {patient.patient_id} boarded in ED Observation.",
                    department="ADMITTED_GEN",
                    patient_id=patient.patient_id,
                )
            else:
                recommendation_notes.append(f"General Ward bed allocated ({available_beds}/{capacity} open).")

        elif clinical_dept == "DISCHARGE":
            recommendation_notes.append("Low risk. Safe for discharge with home care instructions.")

        else:
            recommendation_notes.append(f"Patient routed to {clinical_dept}.")

        operational_decision = {
            "clinical_department": clinical_dept,
            "operational_department": operational_dept,
            "available_beds_in_clinical_dept": available_beds,
            "operating_mode": operating_mode,
            "lambda": load_info["lambda"],
            "capacity_warning": capacity_warning,
            "confirmation_required": confirmation_required,
            "recommendation_summary": " ".join(recommendation_notes),
        }

        # Update patient
        patient.clinical_assessment = clinical_output
        patient.operational_decision = operational_decision
        patient.status = PatientStatus.TRIAGED

        self.events.emit(
            EventType.PATIENT_TRIAGED,
            f"Patient {patient.patient_id} triaged → Clinical: {clinical_dept}, Operational: {operational_dept}",
            department=operational_dept,
            patient_id=patient.patient_id,
            data=operational_decision,
        )

        return {
            "patient_id": patient.patient_id,
            "clinical_assessment": clinical_output,
            "operational_decision": operational_decision,
            "patient": patient.to_dict(),
        }

    def admit_patient(
        self,
        patient_id: str,
        department: Optional[str] = None,
        custom_los_min: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Commit patient admission, occupying a bed in the department and updating load.
        """
        patient = self.patient_flow.get_patient(patient_id)
        if not patient:
            raise KeyError(f"Patient {patient_id!r} not found in simulation.")

        # Determine target department
        target_dept = (
            department
            or (patient.operational_decision or {}).get("operational_department")
            or (patient.clinical_assessment or {}).get("department")
            or "ADMITTED_GEN"
        )

        # Occupy bed in state service if not DISCHARGE
        if target_dept != "DISCHARGE" and self.state_service.department_exists(target_dept):
            curr = self.state_service.get_state(target_dept)
            if curr:
                if curr["occupied"] >= curr["capacity"]:
                    logger.warning(
                        "Admitting patient %s to %s above capacity (%d/%d).",
                        patient_id, target_dept, curr["occupied"], curr["capacity"],
                    )
                new_occ = min(curr["capacity"], curr["occupied"] + 1)
                self.state_service.apply_update(target_dept, {"occupied": new_occ})

        # Update patient status and add to admitted cohort
        self.patient_flow.admit_patient(patient, target_dept, custom_los_min)

        # Recalculate hospital load
        load_info = self.load_controller.recalculate(self.state_service.get_all())

        self.events.emit(
            EventType.PATIENT_ADMITTED,
            f"Patient {patient.patient_id} admitted to {target_dept} (LOS: {patient.expected_los_min}m)",
            department=target_dept,
            patient_id=patient.patient_id,
            data={"expected_los_min": patient.expected_los_min, "operating_mode": load_info["operating_mode"]},
        )

        return {
            "patient_id": patient.patient_id,
            "department": target_dept,
            "status": patient.status.value,
            "expected_los_min": patient.expected_los_min,
            "load_ratio": load_info["load_ratio"],
            "operating_mode": load_info["operating_mode"],
        }

    # ------------------------------------------------------------------
    # Internal Clinical Evaluator
    # ------------------------------------------------------------------

    def _evaluate_clinical_truth(self, patient: SimulatedPatient) -> Dict[str, Any]:
        """Run ML pipeline if available, or deterministic clinically calibrated fallback."""
        try:
            from triageguard_agent.tools.assessment_tools import run_triage_assessment
            # Preserve this simulator's hospital identity through to RAG
            # retrieval + hospital-specific routing (combined_pipeline.py
            # reads patient_data["hospital_id"]) — never silently dropped.
            patient_dict = patient.to_pipeline_dict()
            patient_dict.setdefault("hospital_id", self.hospital_id)
            result = run_triage_assessment(patient_dict)
            if result.success and result.data:
                return result.data
        except Exception as exc:
            logger.debug("TriageGuardPipeline assessment fallback: %s", exc)

        return calibrated_clinical_fallback(patient.acuity, patient.chief_complaint, patient.metadata)

    # ------------------------------------------------------------------
    # Dashboard & Visualisation View
    # ------------------------------------------------------------------

    def get_live_dashboard(self) -> Dict[str, Any]:
        """Generate a complete real-time dashboard representation."""
        all_state = self.state_service.get_all()
        load_info = self.load_controller.recalculate(all_state)

        depts_summary = []
        for name, state in all_state.items():
            if name == "DISCHARGE":
                continue
            cap = state.get("capacity", 0)
            occ = state.get("occupied", 0)
            avail = state.get("available", 0)
            pct = (occ / cap * 100) if cap > 0 else 0
            depts_summary.append({
                "name": name,
                "capacity": cap,
                "occupied": occ,
                "available": avail,
                "occupancy_pct": round(pct, 1),
                "status": state.get("status", "OPEN"),
            })

        return {
            "time": self.events.formatted_time,
            "sim_time_minutes": self.events.sim_time_minutes,
            "scenario": {
                "name": self._current_scenario.name,
                "title": self._current_scenario.title,
                "description": self._current_scenario.description,
                "arrival_rate_per_hour": self._current_scenario.arrival_rate_per_hour,
            },
            "load": {
                "load_ratio": load_info["load_ratio"],
                "operating_mode": load_info["operating_mode"],
                "lambda": load_info["lambda"],
            },
            "departments": depts_summary,
            # Compact 5-patient preview used by dashboard card
            "waiting_queue": [p.to_dict() for p in self.patient_flow.peek_waiting(5)],
            # Full queue with all statuses, for Live Hospital panel
            "full_queue": [p.to_dict() for p in self.patient_flow.full_waiting_queue],
            "waiting_count": self.patient_flow.waiting_count,
            "triaged_count": len(self.patient_flow.triaged_queue),
            "untriaged_count": len(self.patient_flow.untriaged_queue),
            "admitted_count": self.patient_flow.admitted_count,
            "recent_events": self.events.get_recent_feed(limit=8),
        }
