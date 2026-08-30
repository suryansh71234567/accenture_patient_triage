---
name: hospital_status
description: Answer questions about the current operational state of the hospital, its queues, and which patients need attention, using live tool data.
---

# Skill: Hospital Status

## Purpose
Answer hospital-wide operational questions — not just bed capacity, but queues,
which patients are waiting/triaged/awaiting admission, which department is
most constrained, and which patients are flagged for review. This is the
nurse's general operations copilot, not a single-patient lookup.

## Procedure

1. **Pick the right tool for the question.**
   - Pure capacity/occupancy ("how many ICU beds free?"): `get_hospital_state()`,
     or `get_hospital_state(department=<name>)` for one department.
   - Anything about the QUEUE — who's waiting, who's triaged, who's awaiting
     admission, which department is busiest/most constrained, which patients
     need attention or are flagged for review: `get_live_simulation_dashboard()`.
     Its `full_queue` lists every simulated patient with `status`,
     `clinical_assessment`, and `operational_decision` (which itself carries
     `clinical_department`, `operational_department`, `ai_operational_department`,
     `nurse_override`, `confirmation_required`, `capacity_warning`); its
     `departments` list carries `occupancy_pct`/`available` per department.
   - Always fetch live data — never use remembered occupancy or a queue
     snapshot from an earlier turn.

2. **Answering "which patients need attention first / are flagged for review".**
   - From `full_queue`, prioritize patients whose `operational_decision` has
     `confirmation_required=True`, `capacity_warning=True`, or
     `nurse_override=True` (an existing manual override worth a second look),
     then by acuity/clinical priority.
   - Say what "flagged" means for each one (e.g. "ICU is at capacity" vs
     "nurse overrode the AI recommendation") — don't just list IDs.

3. **Answering "which department is most constrained/busiest".**
   - Compare `occupancy_pct` (and `available`) across `departments` — report
     the actual numbers, not just a name.

4. **Check the returned timestamp (get_hospital_state only).**
   - Report when the state was last updated.
   - If `is_stale=True` for any department, flag this to the nurse.
   - Ask the nurse if the occupancy has changed since the last update.

5. **Report occupancy, capacity, and available resources.**
   - State clearly: capacity, occupied, available, status for each department.
   - Report the current operating mode and λ if available.

6. **Detect stale state proactively.**
   - If a routing or resource-allocation decision depends on hospital state, always refresh before deciding.
   - If state is stale (> 30 minutes old), ask staff to confirm current occupancy before using the data.

7. **Trigger calibration if needed.**
   - If the nurse reports that occupancy has changed, activate the `hospital_calibration` workflow.

## Rules

- **Never put current occupancy into memory between turns.** Always fetch fresh.
- **Never report stale data as current** without flagging it to the nurse.
- Hospital state is dynamic — the operational picture can change every few minutes during a busy shift.
- Do not guess occupancy. Ask the nurse if uncertain.
- When summarizing "who needs attention" or "what's flagged", you are surfacing what the system
  already computed — never phrase it as an instruction the nurse must follow. The nurse decides
  what to do with the information; a `nurse_override` already on a patient is a legitimate final
  decision, not something to flag as an error.
