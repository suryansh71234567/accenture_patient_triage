"""
live_hospital_demo.py
---------------------
Interactive Terminal Dashboard and Live Demonstration for TriageGuard Dynamic Simulation.

Demonstrates:
- Dynamic hospital capacity and λ recalculation.
- Automated bed release upon patient Length of Stay (LOS) expiration.
- Clinical Truth (XGBoost + RAG) vs. Operational Truth (Hospital capacity) separation.
- Real-time event feed and human-in-the-loop confirmation.

Usage:
  Interactive Mode:
    .venv\\Scripts\\python.exe scripts/live_hospital_demo.py

  Automated Demonstration Mode:
    .venv\\Scripts\\python.exe scripts/live_hospital_demo.py --demo-mode
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Optional

from triageguard_agent.simulation.hospital_simulator import HospitalSimulator
from triageguard_agent.simulation.scenarios import SCENARIOS, list_scenarios
from triageguard_agent.simulation.patient_flow import PatientStatus


def render_progress_bar(occupied: int, capacity: int, width: int = 12) -> str:
    """Render an ASCII gauge bar, e.g. [########....]."""
    if capacity <= 0:
        return "[" + "." * width + "]"
    ratio = min(1.0, max(0.0, occupied / capacity))
    filled_len = int(round(width * ratio))
    bar = "#" * filled_len + "." * (width - filled_len)
    return f"[{bar}]"


def print_dashboard(sim: HospitalSimulator) -> None:
    """Render the full Live Hospital ASCII dashboard."""
    dash = sim.get_live_dashboard()
    scen = dash["scenario"]
    load = dash["load"]
    depts = dash["departments"]
    time_str = dash["time"]
    mode = load["operating_mode"]
    lam = load["lambda"]
    ratio = load["load_ratio"]

    # Header Box
    print("\n" + "=" * 70)
    print(f"  [+] TRIAGEGUARD LIVE HOSPITAL SIMULATION  --  {time_str}")
    print(f"  Scenario: {scen['title']:<20} | Mode: {mode:<10} | lambda: {lam:.3f} ({ratio:.1%})")
    print("-" * 70)

    # Department Gauges
    print(f"  {'DEPARTMENT':<14} {'OCCUPANCY':<18} {'AVAILABLE':<12} {'LOAD STATUS'}")
    print("-" * 70)
    for d in depts:
        bar = render_progress_bar(d["occupied"], d["capacity"], width=10)
        occ_str = f"{bar} {d['occupied']:>2}/{d['capacity']:<2}"
        avail_str = f"{d['available']:>2} beds"
        pct = d["occupancy_pct"]
        if pct >= 95:
            status_badge = "[!] CRITICAL"
        elif pct >= 80:
            status_badge = "[*] HIGH LOAD"
        else:
            status_badge = "[+] NORMAL"
        print(f"  {d['name']:<14} {occ_str:<18} {avail_str:<12} {status_badge}")

    print("-" * 70)
    print(f"  ED Queue: {dash['waiting_count']} waiting | Admitted in Hospital: {dash['admitted_count']} | Arrivals/hr: {scen['arrival_rate_per_hour']}")

    # Waiting queue preview
    waiting = dash.get("waiting_queue", [])
    if waiting:
        print("  Waiting Patients:")
        for p in waiting[:3]:
            print(f"    * {p['patient_id']} (Acuity {p['acuity']}): {p['chief_complaint'][:48]}...")
    else:
        print("  Waiting Patients: (None - ED queue is clear)")

    # Live Event Feed
    print("-" * 70)
    print("  >> Live Event Feed (Most recent):")
    events = dash.get("recent_events", [])
    if events:
        for ev in events[-6:]:
            print(f"    {ev}")
    else:
        print("    (No events recorded yet)")
    print("=" * 70)


def run_automated_demo(sim: HospitalSimulator) -> None:
    """Run a pre-programmed showcase of dynamic hospital agent capabilities."""
    print("\n" + "=" * 70)
    print("  >> STARTING TRIAGEGUARD LIVE AGENT DEMO")
    print("=" * 70)
    time.sleep(1)

    # Step 1: Initial Normal Day State
    print("\n[Step 1] Initializing Hospital in Normal Day scenario...")
    sim.load_scenario("NORMAL_DAY")
    print_dashboard(sim)
    time.sleep(1.5)

    # Step 2: Patient 1 arrives (STEMI - Acuity 1)
    print("\n[Step 2] Patient PAT-101 arrives with Acute STEMI chest pain...")
    p1 = sim.trigger_arrival(target_acuity=1)
    print(f"  -> Generated: {p1.patient_id} ({p1.chief_complaint})")
    time.sleep(1)

    print("\n[Step 3] Running TriageGuard Agent Clinical + Operational Assessment...")
    triage_res = sim.triage_patient(p1)
    clin = triage_res["clinical_assessment"]
    op = triage_res["operational_decision"]
    print(f"  * Clinical Truth:     Needs {clin['department']} (Acuity {clin['acuity_tier']}, ICU Risk: {clin['reconciled_icu_risk']:.1%})")
    print(f"  * Operational Truth:  {op['recommendation_summary']}")
    print(f"  * Staff Confirmation: {'REQUIRED' if op['confirmation_required'] else 'Auto-approved'}")
    time.sleep(1.5)

    print("\n[Step 4] Staff Confirms Admission -> Bed Allocated in CICU")
    sim.admit_patient(p1.patient_id)
    print_dashboard(sim)
    time.sleep(2)

    # Step 3: Switch to Resource Constrained / Surge Scenario
    print("\n[Step 5] [!] SUDDEN SURGE EVENT: Hospital shifts to RESOURCE CONSTRAINED scenario...")
    sim.load_scenario("RESOURCE_CONSTRAINED")
    print_dashboard(sim)
    time.sleep(2)

    # Step 4: High-risk patient arrives when ICU has only 1 bed left
    print("\n[Step 6] High-risk patient PAT-102 arrives during high constraint...")
    p2 = sim.trigger_arrival(target_acuity=1)
    triage_res2 = sim.triage_patient(p2)
    op2 = triage_res2["operational_decision"]
    print(f"  * Operational Assessment: {op2['recommendation_summary']}")
    print("  * Agent Dialogue: 'ICU has ONLY 1 available bed. Load is CRITICAL. Reserve remaining ICU bed? [Y/N]'")
    print("  * Staff: 'YES (Confirmed)'")
    sim.admit_patient(p2.patient_id, department="ICU", custom_los_min=45)
    print_dashboard(sim)
    time.sleep(2)

    # Step 5: Another critical patient arrives when ICU is NOW 10/10 FULL
    print("\n[Step 7] Another critical patient PAT-103 arrives when ICU is 10/10 FULL...")
    p3 = sim.trigger_arrival(target_acuity=1)
    triage_res3 = sim.triage_patient(p3)
    op3 = triage_res3["operational_decision"]
    print(f"  * Capacity Warning:   {op3['capacity_warning']}")
    print(f"  * Operational Action: {op3['recommendation_summary']}")
    print("  * Agent Dialogue: 'ICU is at capacity (0/10 beds open). Recommending ED Observation with telemetry monitoring + external transfer escalation.'")
    sim.admit_patient(p3.patient_id, department="ED_OBS", custom_los_min=30)
    print_dashboard(sim)
    time.sleep(2.5)

    # Step 6: Advance Time -> Automated Bed Release (Length of Stay Expiration)
    print("\n[Step 8] Advancing Time +45 minutes (Simulating patient recovery & discharge)...")
    step_out = sim.step(minutes=45, auto_generate_arrivals=False)
    print(f"  -> Discharged {step_out['discharged_count']} patients whose LOS expired: {step_out['discharged_patient_ids']}")
    print("  -> Bed automatically released back to department!")
    print_dashboard(sim)
    time.sleep(2)

    print("\n" + "=" * 70)
    print("  [+] AUTOMATED LIVE AGENT DEMO COMPLETE")
    print("  Key Takeaways:")
    print("  1. Patients dynamically create resource pressure over simulated time.")
    print("  2. Agent separates Clinical Truth (XGB+RAG) from Operational Truth (State).")
    print("  3. Automated LOS expiration releases beds without permanent lock-in.")
    print("  4. Human-in-the-loop confirmation gates critical bed allocations.")
    print("=" * 70 + "\n")


def interactive_loop(sim: HospitalSimulator) -> None:
    """Run interactive terminal menu."""
    while True:
        print_dashboard(sim)
        print("\nCommands:")
        print("  [S] Step Time (+15 min)        [A] Arrive New Patient")
        print("  [T] Triage Next Waiting        [C] Change Scenario")
        print("  [F] Fast Forward (+60 min)     [D] Run Automated Demo")
        print("  [Q] Quit")

        try:
            choice = input("\nEnter command: ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Live Hospital.")
            break

        if choice == "Q":
            print("Exiting Live Hospital.")
            break
        elif choice == "S":
            out = sim.step(minutes=15)
            print(f"\n>> Advanced +15m. Discharged: {out['discharged_count']}, New arrivals: {out['new_arrivals_count']}")
        elif choice == "F":
            out = sim.step(minutes=60)
            print(f"\n>> Fast forwarded +60m. Discharged: {out['discharged_count']}, New arrivals: {out['new_arrivals_count']}")
        elif choice == "A":
            p = sim.trigger_arrival()
            print(f"\n>> Generated arrival: {p.patient_id} ({p.chief_complaint})")
        elif choice == "T":
            p = sim.patient_flow.pop_next_waiting()
            if not p:
                print("\n>> No patients currently waiting in ED queue. Arriving one now...")
                p = sim.trigger_arrival()
                sim.patient_flow.pop_next_waiting()  # remove it from queue since we're triaging immediately
            
            res = sim.triage_patient(p)
            clin = res["clinical_assessment"]
            op = res["operational_decision"]
            print("\n" + "─" * 60)
            print(f"  Triage Assessment for {p.patient_id} (Acuity {p.acuity}):")
            print(f"  • Chief Complaint:  {p.chief_complaint}")
            print(f"  • Vitals:           HR={p.vitals.get('hr')} BP={p.vitals.get('sbp')}/{p.vitals.get('dbp')} SpO2={p.vitals.get('spo2')}% Temp={p.vitals.get('temp')}C")
            print(f"  • Clinical Truth:   {clin['department']} (Admission risk: {clin['reconciled_admission_risk']:.1%}, ICU risk: {clin['reconciled_icu_risk']:.1%})")
            print(f"  • Operational Truth:{op['recommendation_summary']}")
            print("─" * 60)

            confirm = input(f"Confirm admission to {op['operational_department']}? [Y/n]: ").strip().lower()
            if confirm in ("", "y", "yes"):
                sim.admit_patient(p.patient_id, department=op["operational_department"])
                print(f">> Patient {p.patient_id} admitted to {op['operational_department']}.")
            else:
                alt = input("Enter alternate department (ICU/CICU/ADMITTED_GEN/ED_OBS/DISCHARGE): ").strip().upper()
                if alt in ("ICU", "CICU", "ADMITTED_GEN", "ED_OBS", "DISCHARGE"):
                    sim.admit_patient(p.patient_id, department=alt)
                    print(f">> Patient {p.patient_id} manually routed to {alt}.")
                else:
                    print(">> Admission cancelled.")
        elif choice == "C":
            print("\nAvailable Scenarios:")
            scen_list = list_scenarios()
            for idx, sc in enumerate(scen_list, 1):
                print(f"  [{idx}] {sc.title:<22} - {sc.description}")
            try:
                s_choice = int(input("\nSelect scenario number: ").strip())
                selected_scen = scen_list[s_choice - 1]
                sim.load_scenario(selected_scen)
                print(f"\n>> Loaded {selected_scen.title}")
            except (ValueError, IndexError):
                print("\n>> Invalid selection.")
        elif choice == "D":
            run_automated_demo(sim)
        else:
            print(f"\n>> Unrecognized command: {choice}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TriageGuard Live Hospital Simulator")
    parser.add_argument("--demo-mode", action="store_true", help="Run automated demonstration showcase")
    parser.add_argument("--scenario", type=str, default="NORMAL_DAY", help="Initial scenario name")
    args = parser.parse_args()

    sim = HospitalSimulator(scenario=args.scenario)

    if args.demo_mode:
        run_automated_demo(sim)
    else:
        interactive_loop(sim)


if __name__ == "__main__":
    main()
