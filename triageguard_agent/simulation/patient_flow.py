"""
patient_flow.py
---------------
Patient lifecycle management and realistic arrival generation for hospital simulation.

Models the full progression:
ARRIVED → TRIAGED → ADMITTED → IN_TREATMENT → DISCHARGED / TRANSFERRED

Tracks Length of Stay (LOS) and automatically identifies patients whose treatment
is complete so their allocated department beds are released back to the hospital.
"""

from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def department_of(patient: "SimulatedPatient") -> Optional[str]:
    """
    A patient's current department queue, mirroring the frontend's own
    getDeptForPatient() grouping logic: the operational (resource-aware)
    department if triaged, else the clinical preference, else unknown.
    Single canonical definition reused by queue reordering and overrides so
    "which department queue is this patient in" is never computed two
    different ways.
    """
    op = patient.operational_decision or {}
    ca = patient.clinical_assessment or {}
    return op.get("operational_department") or ca.get("department")


class PatientStatus(str, Enum):
    """
    Lifecycle statuses for a simulated patient.

    Canonical meaning (documented here so the "ADMITTED vs IN_TREATMENT"
    question has one answer instead of two statuses drifting apart):

        ARRIVED      -> waiting for (first) triage. Lives in the waiting
                         queue only.
        TRIAGED      -> clinically + operationally assessed, not yet
                         admitted. Still lives in the waiting queue (the
                         department queue board is this same list, grouped
                         by operational_department — see department_of()).
        IN_TREATMENT -> admitted and occupying a bed. This is the ONLY
                         status PatientFlowManager.admit_patient() assigns
                         on a non-discharge admission. It lives in the
                         admitted cohort and is removed from the waiting
                         queue at the moment of admission (see
                         admit_patient() below) so "waiting" and "admitted"
                         are always mutually exclusive collections.
        DISCHARGED   -> treatment complete (LOS expired, or admitted
                         straight to DISCHARGE). Lives in discharge history
                         only.
        TRANSFERRED  -> reserved for a future inter-facility transfer flow;
                         not currently assigned anywhere.

    ADMITTED is intentionally kept in this enum for backward compatibility
    (external code/tests may reference the name) but is a dead value — it
    is never assigned. IN_TREATMENT is the one canonical "admitted" status
    the API and frontend consume; do not start assigning ADMITTED instead,
    since that would silently split "is this patient admitted?" across two
    different enum values again.
    """
    ARRIVED = "ARRIVED"
    TRIAGED = "TRIAGED"
    ADMITTED = "ADMITTED"  # reserved / unused — see canonical-status note above
    IN_TREATMENT = "IN_TREATMENT"
    DISCHARGED = "DISCHARGED"
    TRANSFERRED = "TRANSFERRED"


@dataclass
class SimulatedPatient:
    """Represents a simulated patient progressing through the hospital lifecycle."""
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    vitals: Dict[str, Any]
    acuity: int
    arrival_time_min: int
    expected_los_min: int
    elapsed_los_min: int = 0
    status: PatientStatus = PatientStatus.ARRIVED
    department: Optional[str] = None
    clinical_assessment: Optional[Dict[str, Any]] = None
    operational_decision: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "age": self.age,
            "sex": self.sex,
            "chief_complaint": self.chief_complaint,
            "vitals": self.vitals,
            "acuity": self.acuity,
            "arrival_time_min": self.arrival_time_min,
            "expected_los_min": self.expected_los_min,
            "elapsed_los_min": self.elapsed_los_min,
            "remaining_los_min": max(0, self.expected_los_min - self.elapsed_los_min),
            "status": self.status.value if isinstance(self.status, PatientStatus) else str(self.status),
            "department": self.department,
            "clinical_assessment": self.clinical_assessment,
            "operational_decision": self.operational_decision,
            "metadata": self.metadata,
        }

    def to_pipeline_dict(self) -> Dict[str, Any]:
        """Convert into the dictionary format expected by TriageGuardPipeline and XGBoost/RAG."""
        v = self.vitals
        numeric_id_str = "".join(filter(str.isdigit, str(self.patient_id)))
        numeric_id = int(numeric_id_str) if numeric_id_str else 101

        meta = self.metadata or {}

        return {
            "patient_id": numeric_id,
            "patient_id_str": str(self.patient_id),
            "age": self.age,
            "gender": self.sex,
            "sex": self.sex,
            "chiefcomplaint": self.chief_complaint,
            "triage_complaint": self.chief_complaint,
            "acuity": self.acuity,
            "time_elapsed_minutes": self.elapsed_los_min,
            # Current vitals
            "hr_current": v.get("hr", 80),
            "rr_current": v.get("rr", 16),
            "spo2_current": v.get("spo2", 98),
            "sbp_current": v.get("sbp", 120),
            "dbp_current": v.get("dbp", 80),
            "temp_current": v.get("temp", 37.0),
            "pain_current": v.get("pain", 0),
            # Arrival vitals (baseline)
            "hr_arrival": v.get("hr", 80),
            "rr_arrival": v.get("rr", 16),
            "spo2_arrival": v.get("spo2", 98),
            "sbp_arrival": v.get("sbp", 120),
            "dbp_arrival": v.get("dbp", 80),
            "temp_arrival": v.get("temp", 37.0),
            # Standard aliases
            "heartrate": v.get("hr", 80),
            "resprate": v.get("rr", 16),
            "o2sat": v.get("spo2", 98),
            "sbp": v.get("sbp", 120),
            "dbp": v.get("dbp", 80),
            "temperature": v.get("temp", 37.0),
            "pain": v.get("pain", 0),
            # History fields from metadata (populated by manual intake)
            "history_text": meta.get("history_text", ""),
            "previous_ed_visits": meta.get("previous_ed_visits", 0),
            "previous_hospital_admissions": meta.get("previous_hospital_admissions", 0),
            "previous_icu_admissions": meta.get("previous_icu_admissions", 0),
            "cardiovascular_history": meta.get("cardiovascular_history", 0),
            "respiratory_history": meta.get("respiratory_history", 0),
            "renal_history": meta.get("renal_history", 0),
            "diabetes_history": meta.get("diabetes_history", 0),
            "neurological_history": meta.get("neurological_history", 0),
            "malignancy_history": meta.get("malignancy_history", 0),
        }


# ---------------------------------------------------------------------------
# Clinical Templates for Synthetic Patient Generation
# ---------------------------------------------------------------------------

_PATIENT_ARCHETYPES = {
    1: [
        {
            "chief_complaint": "Crushing substernal chest pain radiating to left jaw, diaphoresis, acute dyspnea",
            "age_range": (52, 78),
            "vitals": {"hr": 118, "rr": 28, "spo2": 89, "sbp": 86, "dbp": 52, "temp": 36.8, "pain": 10},
            "los_range": (80, 140),
            "cardiac": True,
        },
        {
            "chief_complaint": "Severe septic shock, altered mental status, high fever, mottled extremities",
            "age_range": (60, 85),
            "vitals": {"hr": 138, "rr": 32, "spo2": 91, "sbp": 78, "dbp": 44, "temp": 39.6, "pain": 0},
            "los_range": (90, 150),
            "cardiac": False,
        },
    ],
    2: [
        {
            "chief_complaint": "Acute worsening chest tightness with history of CAD and shortness of breath on exertion",
            "age_range": (48, 75),
            "vitals": {"hr": 102, "rr": 22, "spo2": 94, "sbp": 162, "dbp": 96, "temp": 37.1, "pain": 7},
            "los_range": (60, 100),
            "cardiac": True,
        },
        {
            "chief_complaint": "Acute respiratory distress, severe wheezing, accessory muscle use with history of COPD",
            "age_range": (55, 78),
            "vitals": {"hr": 110, "rr": 26, "spo2": 88, "sbp": 145, "dbp": 88, "temp": 37.4, "pain": 3},
            "los_range": (50, 90),
            "cardiac": False,
        },
    ],
    3: [
        {
            "chief_complaint": "Severe right lower quadrant abdominal pain, nausea, vomiting for 12 hours",
            "age_range": (20, 55),
            "vitals": {"hr": 92, "rr": 18, "spo2": 98, "sbp": 128, "dbp": 82, "temp": 38.2, "pain": 8},
            "los_range": (40, 75),
            "cardiac": False,
        },
        {
            "chief_complaint": "Productive cough with rust-colored sputum, moderate fever, pleuritic right chest pain",
            "age_range": (40, 72),
            "vitals": {"hr": 96, "rr": 20, "spo2": 94, "sbp": 122, "dbp": 76, "temp": 38.5, "pain": 5},
            "los_range": (40, 70),
            "cardiac": False,
        },
    ],
    4: [
        {
            "chief_complaint": "Right ankle inversion injury with swelling, ecchymosis, unable to bear weight",
            "age_range": (18, 50),
            "vitals": {"hr": 78, "rr": 16, "spo2": 99, "sbp": 118, "dbp": 76, "temp": 36.8, "pain": 6},
            "los_range": (20, 40),
            "cardiac": False,
        },
        {
            "chief_complaint": "Forearm laceration from kitchen knife, bleeding controlled with pressure",
            "age_range": (22, 60),
            "vitals": {"hr": 74, "rr": 15, "spo2": 99, "sbp": 120, "dbp": 78, "temp": 36.7, "pain": 4},
            "los_range": (15, 30),
            "cardiac": False,
        },
    ],
    5: [
        {
            "chief_complaint": "Routine blood pressure medication refill request, asymptomatic",
            "age_range": (45, 75),
            "vitals": {"hr": 72, "rr": 14, "spo2": 99, "sbp": 132, "dbp": 84, "temp": 36.6, "pain": 0},
            "los_range": (10, 20),
            "cardiac": False,
        },
        {
            "chief_complaint": "Minor non-pruritic rash on left forearm for 3 days, no systemic symptoms",
            "age_range": (18, 45),
            "vitals": {"hr": 68, "rr": 14, "spo2": 100, "sbp": 115, "dbp": 72, "temp": 36.6, "pain": 0},
            "los_range": (10, 15),
            "cardiac": False,
        },
    ],
}


class PatientFlowManager:
    """
    Manages active patient queues, bed occupancy cohorts, and lifecycle transitions.
    """

    def __init__(self, initial_patient_counter: int = 100) -> None:
        self._patient_counter = initial_patient_counter
        self._waiting_queue: List[SimulatedPatient] = []
        self._admitted_cohort: Dict[str, SimulatedPatient] = {}
        self._discharged_history: List[SimulatedPatient] = []

    # ------------------------------------------------------------------
    # Patient Generation
    # ------------------------------------------------------------------

    def generate_patient(
        self,
        sim_time_min: int,
        target_acuity: Optional[int] = None,
        acuity_weights: Optional[Dict[int, float]] = None,
    ) -> SimulatedPatient:
        """Create a clinically realistic simulated patient."""
        self._patient_counter += 1
        pid = f"PAT-{self._patient_counter}"

        # Determine acuity tier
        if target_acuity is not None and target_acuity in _PATIENT_ARCHETYPES:
            acuity = target_acuity
        else:
            weights = acuity_weights or {1: 0.15, 2: 0.20, 3: 0.40, 4: 0.15, 5: 0.10}
            acuities = list(weights.keys())
            probs = [weights[a] for a in acuities]
            acuity = random.choices(acuities, weights=probs, k=1)[0]

        archetype = random.choice(_PATIENT_ARCHETYPES[acuity])
        age = random.randint(*archetype["age_range"])
        sex = random.choice(["M", "F"])
        los_min = random.randint(*archetype["los_range"])

        # Add minor natural variance to vitals
        base_vitals = archetype["vitals"]
        vitals = {
            "hr": max(45, base_vitals["hr"] + random.randint(-4, 4)),
            "rr": max(10, base_vitals["rr"] + random.randint(-1, 2)),
            "spo2": min(100, max(75, base_vitals["spo2"] + random.randint(-1, 1))),
            "sbp": max(65, base_vitals["sbp"] + random.randint(-6, 6)),
            "dbp": max(40, base_vitals["dbp"] + random.randint(-4, 4)),
            "temp": round(base_vitals["temp"] + random.uniform(-0.2, 0.2), 1),
            "pain": base_vitals["pain"],
        }

        patient = SimulatedPatient(
            patient_id=pid,
            age=age,
            sex=sex,
            chief_complaint=archetype["chief_complaint"],
            vitals=vitals,
            acuity=acuity,
            arrival_time_min=sim_time_min,
            expected_los_min=los_min,
            elapsed_los_min=0,
            status=PatientStatus.ARRIVED,
            metadata={"cardiac_hint": archetype.get("cardiac", False)},
        )

        return patient

    # ------------------------------------------------------------------
    # Queue Management
    # ------------------------------------------------------------------

    def enqueue_patient(self, patient: SimulatedPatient) -> None:
        """Add newly arrived patient to ED waiting queue."""
        self._waiting_queue.append(patient)
        logger.info("Patient %s added to waiting queue (queue length=%d).", patient.patient_id, len(self._waiting_queue))

    def reorder_queue(
        self,
        patient_id: str,
        new_index: int,
        note: str = "",
    ) -> bool:
        """
        Move a patient to a specific position in the waiting queue.
        Returns True if the move was performed, False if patient not found.
        """
        idx = next(
            (i for i, p in enumerate(self._waiting_queue) if p.patient_id == patient_id),
            None,
        )
        if idx is None:
            return False
        patient = self._waiting_queue.pop(idx)
        new_index = max(0, min(new_index, len(self._waiting_queue)))
        self._waiting_queue.insert(new_index, patient)
        if note:
            patient.metadata["queue_note"] = note
        logger.info(
            "Patient %s moved from position %d to %d. Note: %s",
            patient_id, idx, new_index, note or "(none)",
        )
        return True

    def reorder_within_department(
        self,
        patient_id: str,
        department: str,
        new_index: int,
        note: str = "",
    ) -> bool:
        """
        Move a patient to a new position among only the patients currently
        routed to `department` (department_of()), preserving every other
        patient's relative position in the underlying single waiting-queue
        list. No per-department queue is duplicated — department queues are
        always a view over this one list (department_of()), same as the
        existing frontend grouping.

        Returns False if the patient isn't found or isn't currently in
        `department` (e.g. stale client-side drag target).
        """
        idx = next((i for i, p in enumerate(self._waiting_queue) if p.patient_id == patient_id), None)
        if idx is None:
            return False
        patient = self._waiting_queue[idx]
        if department_of(patient) != department:
            return False

        self._waiting_queue.pop(idx)
        dept_members = [p for p in self._waiting_queue if department_of(p) == department]
        new_index = max(0, min(new_index, len(dept_members)))
        if new_index >= len(dept_members):
            insert_at = (self._waiting_queue.index(dept_members[-1]) + 1) if dept_members else len(self._waiting_queue)
        else:
            insert_at = self._waiting_queue.index(dept_members[new_index])
        self._waiting_queue.insert(insert_at, patient)

        if note:
            patient.metadata["queue_note"] = note
        logger.info(
            "Patient %s reordered within %s queue to position %d. Note: %s",
            patient_id, department, new_index, note or "(none)",
        )
        return True

    def pop_next_waiting(self) -> Optional[SimulatedPatient]:
        """Fetch and remove the next patient awaiting triage/admission (FIFO)."""
        if self._waiting_queue:
            return self._waiting_queue.pop(0)
        return None

    def peek_waiting(self, count: int = 5) -> List[SimulatedPatient]:
        """Inspect the front of the waiting queue without removing."""
        return list(self._waiting_queue[:count])

    @property
    def full_waiting_queue(self) -> List[SimulatedPatient]:
        """Return the complete waiting queue (all patients, all statuses)."""
        return list(self._waiting_queue)

    @property
    def triaged_queue(self) -> List[SimulatedPatient]:
        """Return only patients that have been triaged but not yet admitted."""
        return [p for p in self._waiting_queue if p.status == PatientStatus.TRIAGED]

    @property
    def untriaged_queue(self) -> List[SimulatedPatient]:
        """Return only patients that have arrived but not yet been triaged."""
        return [p for p in self._waiting_queue if p.status == PatientStatus.ARRIVED]

    @property
    def waiting_count(self) -> int:
        return len(self._waiting_queue)

    @property
    def admitted_count(self) -> int:
        return len(self._admitted_cohort)

    # ------------------------------------------------------------------
    # Bed & Cohort Management
    # ------------------------------------------------------------------

    def admit_patient(
        self,
        patient: SimulatedPatient,
        department: str,
        custom_los_min: Optional[int] = None,
    ) -> SimulatedPatient:
        """
        Move a patient to the admitted cohort (or discharge history) in the
        specified department.

        Also removes the patient from the waiting/triage queue — an
        admitted or discharged patient must never remain counted as
        "waiting"/"triaged" in the same list an admitted patient now lives
        in elsewhere. Previously this method left the patient in
        `_waiting_queue` after admission too, so `waiting_count` only ever
        grew and a patient briefly existed in two collections at once;
        removing it here makes "waiting" and "admitted" mutually exclusive
        at the source, instead of relying on callers/UI to filter by status.
        """
        patient.department = department
        patient.status = PatientStatus.IN_TREATMENT if department != "DISCHARGE" else PatientStatus.DISCHARGED
        if custom_los_min is not None:
            patient.expected_los_min = custom_los_min

        self._waiting_queue = [p for p in self._waiting_queue if p.patient_id != patient.patient_id]

        if department != "DISCHARGE":
            self._admitted_cohort[patient.patient_id] = patient
        else:
            self._discharged_history.append(patient)

        logger.info("Patient %s admitted to %s (status=%s).", patient.patient_id, department, patient.status.value)
        return patient

    def remove_patient(self, patient_id: str) -> Optional[SimulatedPatient]:
        """
        Remove and return a patient (by id) from whichever *active*
        collection currently holds them — waiting queue or admitted
        cohort — or None if not found there. Discharge history is left
        alone (already inactive). Caller is responsible for releasing any
        bed this patient was occupying in HospitalStateService; this method
        only owns queue/cohort membership, mirroring get_patient()'s scope.

        Used to selectively remove previously-injected demo patients on a
        scenario switch without discarding manually-registered or
        dynamically-arrived patients that happen to share the same
        underlying storage (see HospitalSimulator._inject_presimulated_patients).
        """
        for i, p in enumerate(self._waiting_queue):
            if p.patient_id == str(patient_id):
                return self._waiting_queue.pop(i)
        return self._admitted_cohort.pop(str(patient_id), None)

    def update_vitals(self, patient_id: str, vitals: Dict[str, Any]) -> Optional[SimulatedPatient]:
        """
        Merge new current-vitals values onto an ACTIVE patient (waiting
        queue or admitted cohort) in place — never creates a second
        patient, never touches the file-based historical patient store.

        Only non-None values in `vitals` are applied, so a partial update
        (e.g. just heart rate) never clobbers other vitals with None.
        Returns the updated patient, or None if patient_id isn't active
        (discharge history is intentionally excluded — a discharged
        encounter's vitals are historical, not "current").
        """
        patient = None
        for p in self._waiting_queue:
            if p.patient_id == str(patient_id):
                patient = p
                break
        if patient is None:
            patient = self._admitted_cohort.get(str(patient_id))
        if patient is None:
            return None

        patient.vitals = {**patient.vitals, **{k: v for k, v in vitals.items() if v is not None}}
        return patient

    def advance_time(self, delta_minutes: int) -> List[SimulatedPatient]:
        """
        Advance elapsed LOS for all admitted patients.
        Automatically discharges patients whose LOS has expired and returns them.
        """
        discharged: List[SimulatedPatient] = []
        still_admitted: Dict[str, SimulatedPatient] = {}

        for pid, patient in self._admitted_cohort.items():
            patient.elapsed_los_min += delta_minutes
            if patient.elapsed_los_min >= patient.expected_los_min:
                patient.status = PatientStatus.DISCHARGED
                discharged.append(patient)
                self._discharged_history.append(patient)
                logger.info(
                    "Patient %s LOS expired (%d/%d min) — discharged from %s.",
                    pid, patient.elapsed_los_min, patient.expected_los_min, patient.department,
                )
            else:
                still_admitted[pid] = patient

        self._admitted_cohort = still_admitted
        return discharged

    def get_admitted_by_department(self, department: str) -> List[SimulatedPatient]:
        """Return all admitted patients currently in a specific department."""
        return [p for p in self._admitted_cohort.values() if p.department == department]

    def get_patient(self, patient_id: str) -> Optional[SimulatedPatient]:
        """Search for a patient across admitted cohort, waiting queue, and discharge history."""
        if patient_id in self._admitted_cohort:
            return self._admitted_cohort[patient_id]
        for p in self._waiting_queue:
            if p.patient_id == str(patient_id):
                return p
        for p in self._discharged_history:
            if p.patient_id == str(patient_id):
                return p
        return None

    def clear(self) -> None:
        """Reset all queues and cohorts."""
        self._waiting_queue.clear()
        self._admitted_cohort.clear()
        self._discharged_history.clear()
        self._patient_counter = 100
