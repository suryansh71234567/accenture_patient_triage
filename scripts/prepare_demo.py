"""
prepare_demo.py
----------------
One-off demo-environment prep script (not part of the application code
path — invoked manually, once, before a recording session).

Run this BEFORE starting uvicorn normally. It:
  1. Imports api_server (module-level app/runtime init — same as a real
     server start).
  2. Force-creates the "default" hospital's simulator via the same public
     get_simulator() the app itself uses (this auto-injects the existing
     pre-simulated demo pool for the default hospital, per
     HospitalSimulator._inject_presimulated_patients — that is normal,
     pre-existing app behavior, not something this script introduces).
  3. Removes exactly the pre-simulated patients that injection added,
     using the simulator's own tracked id set (_presimulated_patient_ids)
     and the exact same removal/bed-release logic
     _inject_presimulated_patients() itself uses for a scenario switch —
     just without the subsequent re-injection step. No new removal
     mechanism is invented here.
  4. Registers the curated EVAL-* demo cohort via the real
     POST /api/simulation/manual-arrival endpoint (through a real
     FastAPI TestClient dispatch — the actual route handler runs, not a
     shortcut), exactly what the "Register Patient" UI button calls.
     Each patient is left ARRIVED, in the waiting-for-triage queue —
     never triaged or admitted here.

After this script finishes, start uvicorn WITHOUT --reload in the SAME
process image (see run_demo_server.py) so the in-memory state this script
built is what the frontend actually talks to.
"""
from __future__ import annotations

CURATED_PATIENT_IDS = [
    "EVAL-001", "EVAL-002", "EVAL-003", "EVAL-005", "EVAL-006", "EVAL-007",
    "EVAL-009", "EVAL-011", "EVAL-013", "EVAL-015", "EVAL-016", "EVAL-018",
    "EVAL-020", "EVAL-023", "EVAL-026", "EVAL-036", "EVAL-038", "EVAL-040",
]


def clear_presimulated_patients(sim) -> list[str]:
    """Mirrors the removal half of HospitalSimulator._inject_presimulated_patients
    (hospital_simulator.py) without the subsequent re-injection step."""
    removed_ids = []
    for pid in list(sim._presimulated_patient_ids):
        removed = sim.patient_flow.remove_patient(pid)
        if removed:
            removed_ids.append(pid)
            if (
                removed.department
                and removed.department != "DISCHARGE"
                and sim.state_service.department_exists(removed.department)
            ):
                curr = sim.state_service.get_state(removed.department)
                if curr and curr.get("occupied", 0) > 0:
                    sim.state_service.apply_update(
                        removed.department, {"occupied": curr["occupied"] - 1}
                    )
    sim._presimulated_patient_ids = set()
    return removed_ids


def main() -> None:
    import api_server
    from fastapi.testclient import TestClient
    from triageguard_agent.tools.simulation_tools import get_simulator
    from triageguard_agent.tools.patient_tools import get_patient_record

    sim = get_simulator("default")
    removed = clear_presimulated_patients(sim)
    print(f"Cleared {len(removed)} pre-simulated demo patients: {removed}")

    # Guard: fail loudly rather than silently registering fewer than expected.
    missing = [pid for pid in CURATED_PATIENT_IDS if get_patient_record(pid) is None]
    if missing:
        raise SystemExit(f"FATAL: missing chart record(s) for {missing} — aborting, no changes committed to live state beyond the pool clear above.")

    client = TestClient(api_server.app)
    registered = []
    for pid in CURATED_PATIENT_IDS:
        record = get_patient_record(pid)
        resp = client.post("/api/simulation/manual-arrival", json={
            "patient_id": pid,
            "chief_complaint": record["chiefcomplaint"],
            "age": record["age"],
            "sex": record["sex"],
            "acuity": record["acuity"],
            "hospital_id": "default",
        })
        if resp.status_code != 200:
            raise SystemExit(f"FATAL: failed to register {pid}: {resp.status_code} {resp.text}")
        body = resp.json()
        registered.append((pid, body["status"], body["has_history"]))

    print(f"Registered {len(registered)} curated patients as ARRIVED:")
    for pid, status, has_history in registered:
        print(f"  {pid}: status={status} has_history={has_history}")

    dash = client.get("/api/simulation/dashboard", params={"hospital_id": "default"}).json()
    print(f"\nFinal default-hospital dashboard: waiting={dash['waiting_count']} "
          f"triaged={dash['triaged_count']} admitted={dash['admitted_count']} "
          f"total_in_queue={len(dash['full_queue'])}")


if __name__ == "__main__":
    main()
