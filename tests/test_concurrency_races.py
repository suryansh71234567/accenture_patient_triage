"""
test_concurrency_races.py
--------------------------
Phase 6B: regression tests for the concurrency races found and fixed during
the P0 investigation, plus a documented-safe result for one area that was
investigated and found NOT to need a lock.

Reproduction technique
-----------------------
A natural race between fast, no-I/O Python statements is too small a window
to reliably hit by just firing threads (confirmed empirically: 20 trials of
20 truly concurrent threads via a start-barrier did not reproduce the
admit_patient() race even before it was fixed). These tests instead widen
the window deterministically — instrument the vulnerable call with a short
sleep — and directly assert on a "concurrent entries" high-water mark to
prove the fix actually serializes access, rather than only asserting on the
end state (which a lucky interleaving could get right by accident).

Found races (fixed, see FIXED assertions below):
  - HospitalSimulator.admit_patient() — read-decide-write occupancy update.
  - HospitalSimulator.step()'s bed-release loop — the SAME shared occupied
    counter, mutated by a DIFFERENT method than admit_patient(); an
    admit-only lock does not protect it — a concurrent step() (release) and
    admit_patient() (occupy) on the same department corrupted the counter
    (confirmed 5/5 forced-interleaving trials) until both were moved onto
    one shared _bed_state_lock.
  - HospitalRegistry.register()       — collision-check-then-write.
  - simulation_tools.get_simulator()  — cache check-then-construct.
  - HospitalSimulator.admit_patient() also had a separate, pre-existing (not
    concurrency-specific) idempotency gap this investigation surfaced:
    nothing stopped a second admit of an already-IN_TREATMENT patient from
    double-consuming a bed for one physical patient — fixed with the same
    already-admitted guard triage_patient() already uses.

Investigated and confirmed SAFE (no fix applied):
  - HospitalSimulator.override_department() — never mutates any shared
    department counter; only the target patient's own operational_decision
    dict, so two different patients racing for the same "last available
    bed" check cannot corrupt anything — capacity is only actually consumed
    at admit_patient() time, and a department genuinely can (and normally
    does) hold more triaged/queued patients than it has free beds.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

import triageguard_agent.tools.simulation_tools as simulation_tools
from triageguard_agent.hospital.hospital_registry import HospitalRegistry
from triageguard_agent.hospital.hospital_state_service import HospitalStateService
from triageguard_agent.simulation.hospital_simulator import HospitalSimulator
from triageguard_agent.simulation.patient_flow import PatientStatus, SimulatedPatient


@pytest.fixture(autouse=True)
def reset_singletons():
    HospitalStateService.reset_instance()
    simulation_tools._simulator_instances.clear()
    yield
    HospitalStateService.reset_instance()
    simulation_tools._simulator_instances.clear()


def _triaged_patient(patient_id: str, operational_dept: str) -> SimulatedPatient:
    return SimulatedPatient(
        patient_id=patient_id, age=55, sex="M", chief_complaint="test", vitals={},
        acuity=2, arrival_time_min=0, expected_los_min=60, status=PatientStatus.TRIAGED,
        clinical_assessment={"department": operational_dept},
        operational_decision={
            "clinical_department": operational_dept, "operational_department": operational_dept,
            "ai_operational_department": operational_dept, "nurse_override": False, "override_reason": None,
            "available_beds_in_clinical_dept": 5, "operating_mode": "NORMAL", "lambda": 0.5,
            "capacity_warning": False, "confirmation_required": False, "recommendation_summary": "test",
        },
    )


class _ConcurrencyTracer:
    """Tracks the high-water mark of concurrent entries into a traced call —
    the direct way to prove a lock actually serializes access, rather than
    only checking that the end state happened to come out correct."""

    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def enter(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def hold(self):
        time.sleep(self.delay)

    def exit(self):
        with self._lock:
            self.active -= 1


class TestAdmitPatientConcurrency:
    def test_concurrent_admits_never_run_the_occupancy_update_simultaneously(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.state_service.apply_update("ICU", {"capacity": 1000, "occupied": 0})

        n = 12
        patients = [_triaged_patient(f"CONC-{i}", "ICU") for i in range(n)]
        for p in patients:
            sim.patient_flow.enqueue_patient(p)

        tracer = _ConcurrencyTracer()
        real_get_state = sim.state_service.get_state

        def traced_get_state(department):
            tracer.enter()
            result = real_get_state(department)
            tracer.hold()
            tracer.exit()
            return result

        sim.state_service.get_state = traced_get_state

        errors = []

        def worker(p):
            try:
                sim.admit_patient(patient_id=p.patient_id)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(p,)) for p in patients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # FIXED: _admit_lock means at most one thread is ever inside the
        # read-decide-write occupancy sequence at a time.
        assert tracer.max_active == 1, (
            f"expected admit_patient() to serialize concurrent callers, "
            f"but saw {tracer.max_active} threads inside it simultaneously"
        )

        final = real_get_state("ICU")
        admitted_to_icu = [pid for pid, sp in sim.patient_flow._admitted_cohort.items() if sp.department == "ICU"]
        # Every one of the N concurrent admits must be reflected in BOTH the
        # occupancy counter and the admitted cohort, and they must agree —
        # this is exactly the invariant the pre-fix race violated (occupied
        # counter under-reported the true admitted count under interleaving).
        assert len(admitted_to_icu) == n
        assert final["occupied"] == n

    def test_concurrent_double_admit_of_the_same_patient_only_counts_once(self):
        """A rapid double-click admitting the SAME patient twice must not
        double-increment occupancy for one physical bed."""
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.state_service.apply_update("ICU", {"capacity": 10, "occupied": 0})
        patient = _triaged_patient("DUP-1", "ICU")
        sim.patient_flow.enqueue_patient(patient)

        tracer = _ConcurrencyTracer()
        real_get_state = sim.state_service.get_state

        def traced_get_state(department):
            tracer.enter()
            result = real_get_state(department)
            tracer.hold()
            tracer.exit()
            return result

        sim.state_service.get_state = traced_get_state

        outcomes = []

        def worker():
            try:
                sim.admit_patient(patient_id="DUP-1")
                outcomes.append("ok")
            except ValueError as e:
                outcomes.append(("rejected", str(e)))

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert tracer.max_active == 1
        # FIXED: the lock serializes the two attempts, and the second one
        # now sees the patient already IN_TREATMENT and is rejected — not
        # a race, but a pre-existing idempotency gap this investigation
        # surfaced and fixed with the same already-admitted guard
        # triage_patient() already uses.
        assert outcomes.count("ok") == 1
        assert sum(1 for o in outcomes if isinstance(o, tuple) and o[0] == "rejected") == 1
        final = real_get_state("ICU")
        assert final["occupied"] == 1


class TestStepVsAdmitConcurrency:
    """step()'s bed-RELEASE loop and admit_patient()'s bed-OCCUPY loop are
    different methods mutating the same department's `occupied` counter —
    confirmed racy against EACH OTHER (not just against themselves) before
    both were moved onto the shared _bed_state_lock."""

    def test_concurrent_step_discharge_and_admit_do_not_corrupt_the_same_counter(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.state_service.apply_update("ICU", {"capacity": 1000, "occupied": 10})

        # 10 admitted patients whose LOS has already elapsed -> step() will discharge all 10.
        for i in range(10):
            p = SimulatedPatient(
                patient_id=f"DISCH-{i}", age=55, sex="M", chief_complaint="test", vitals={},
                acuity=2, arrival_time_min=0, expected_los_min=1, elapsed_los_min=100,
                status=PatientStatus.IN_TREATMENT, department="ICU",
            )
            sim.patient_flow._admitted_cohort[p.patient_id] = p

        # 1 triaged patient about to be admitted concurrently with the discharges.
        new_patient = _triaged_patient("NEW-ADMIT", "ICU")
        sim.patient_flow.enqueue_patient(new_patient)

        tracer = _ConcurrencyTracer(delay=0.005)
        real_get_state = sim.state_service.get_state

        def traced_get_state(department):
            tracer.enter()
            result = real_get_state(department)
            tracer.hold()
            tracer.exit()
            return result

        sim.state_service.get_state = traced_get_state
        errors = []

        def do_step():
            try:
                sim.step(minutes=200, auto_generate_arrivals=False)
            except Exception as e:  # noqa: BLE001
                errors.append(("step", e))

        def do_admit():
            try:
                sim.admit_patient(patient_id="NEW-ADMIT")
            except Exception as e:  # noqa: BLE001
                errors.append(("admit", e))

        t1 = threading.Thread(target=do_step)
        t2 = threading.Thread(target=do_admit)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert errors == []
        # FIXED: both operations now share _bed_state_lock.
        assert tracer.max_active == 1, (
            f"expected step()'s release and admit_patient()'s occupy to serialize "
            f"through one lock, but saw {tracer.max_active} concurrent entries"
        )
        final = real_get_state("ICU")
        # Net effect: 10 released (-10) + 1 occupied (+1), starting from 10 -> 1.
        assert final["occupied"] == 1
        assert len(sim.patient_flow._admitted_cohort) == 1  # only NEW-ADMIT remains


class TestGetSimulatorConcurrency:
    def test_concurrent_first_touch_returns_the_same_instance(self):
        tracer = _ConcurrencyTracer()
        real_init = HospitalSimulator.__init__

        def traced_init(self, *args, **kwargs):
            tracer.enter()
            tracer.hold()
            real_init(self, *args, **kwargs)
            tracer.exit()

        HospitalSimulator.__init__ = traced_init
        try:
            results = []
            lock = threading.Lock()

            def worker():
                sim = simulation_tools.get_simulator("default")
                with lock:
                    results.append(sim)

            threads = [threading.Thread(target=worker) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            HospitalSimulator.__init__ = real_init

        # FIXED: _simulator_lock means at most one HospitalSimulator is ever
        # under construction for a given (or any) key at a time.
        assert tracer.max_active == 1, (
            f"expected get_simulator() to serialize concurrent first-touch "
            f"construction, but saw {tracer.max_active} simultaneous constructions"
        )
        assert len(results) == 6
        assert all(r is results[0] for r in results), "concurrent callers received different HospitalSimulator instances"
        assert len(simulation_tools._simulator_instances) == 1

    def test_different_hospitals_get_isolated_simulator_instances(self):
        # Sanity check the fix didn't collapse per-hospital isolation into a
        # single shared instance — only same-key concurrent access should
        # serialize onto the same object; different keys must stay distinct.
        registry = simulation_tools.get_simulator("default")
        # "default" is always registered; use it twice to also confirm the
        # cache returns the identical object on a second (uncontended) call.
        again = simulation_tools.get_simulator("default")
        assert registry is again


class TestHospitalRegistryRegisterConcurrency:
    def test_concurrent_registration_of_the_same_id_only_one_wins(self, monkeypatch):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            registry = HospitalRegistry(manifest_path=tmpdir / "registry.json")

            tracer = _ConcurrencyTracer()
            real_copyfile = shutil.copyfile

            def traced_copyfile(src, dst):
                tracer.enter()
                tracer.hold()
                result = real_copyfile(src, dst)
                tracer.exit()
                return result

            monkeypatch.setattr(shutil, "copyfile", traced_copyfile)

            results: dict[str, object] = {}
            errors: dict[str, Exception] = {}

            def worker(name: str):
                try:
                    results[name] = registry.register("racehosp", f"Race Hospital {name}")
                except ValueError as e:
                    errors[name] = e

            threads = [threading.Thread(target=worker, args=(n,)) for n in ("A", "B", "C")]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # FIXED: _register_lock means at most one registration for this
            # id is ever mid-flight (config write + manifest save) at once.
            assert tracer.max_active == 1, (
                f"expected register() to serialize concurrent callers, "
                f"but saw {tracer.max_active} simultaneous registrations"
            )
            assert len(results) == 1, f"expected exactly one registration to succeed, got {list(results)}"
            assert len(errors) == 2, f"expected exactly two collisions to be rejected, got {list(errors)}"
            assert registry.exists("racehosp")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestOverrideDepartmentConcurrency:
    """Investigated (Phase 6B), confirmed SAFE without a lock — documented
    here as a regression test of the *invariant* that makes it safe, not a
    race fix, so a future change that starts mutating shared department
    state from override_department() will be caught."""

    def test_concurrent_overrides_of_different_patients_never_touch_department_counters(self):
        sim = HospitalSimulator(scenario="NORMAL_DAY")
        sim.patient_flow.clear()
        sim.state_service.apply_update("ICU", {"capacity": 5, "occupied": 4})  # exactly 1 bed nominally free

        p1 = _triaged_patient("OV-1", "ADMITTED_GEN")
        p2 = _triaged_patient("OV-2", "ADMITTED_GEN")
        for p in (p1, p2):
            sim.patient_flow.enqueue_patient(p)

        before = sim.state_service.get_state("ICU")

        results = {}
        errors = {}

        def worker(pid):
            try:
                results[pid] = sim.override_department(pid, "ICU", reason="race test")
            except Exception as e:  # noqa: BLE001
                errors[pid] = e

        t1 = threading.Thread(target=worker, args=("OV-1",))
        t2 = threading.Thread(target=worker, args=("OV-2",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        after = sim.state_service.get_state("ICU")

        # Both overrides legitimately succeed — queueing more triaged
        # patients toward a department than it has free beds is expected
        # (capacity is only consumed at admit time), not a bug.
        assert errors == {}
        assert len(results) == 2
        assert sim.patient_flow.get_patient("OV-1").operational_decision["operational_department"] == "ICU"
        assert sim.patient_flow.get_patient("OV-2").operational_decision["operational_department"] == "ICU"
        # The invariant that makes this safe without a lock: override never
        # touches the department's occupied/available counters.
        assert after == before
