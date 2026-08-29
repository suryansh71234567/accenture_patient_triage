---
name: hospital_status
description: Answer questions about the current operational state of the hospital using live tool data.
---

# Skill: Hospital Status

## Purpose
Answer questions about the current operational state of the hospital.

## Procedure

1. **Query the hospital state tool.**
   - Call `get_hospital_state()` for full hospital state.
   - Call `get_hospital_state(department=<name>)` for a specific department.
   - Always fetch live data — never use remembered occupancy.

2. **Check the returned timestamp.**
   - Report when the state was last updated.
   - If `is_stale=True` for any department, flag this to the nurse.
   - Ask the nurse if the occupancy has changed since the last update.

3. **Report occupancy, capacity, and available resources.**
   - State clearly: capacity, occupied, available, status for each department.
   - Report the current operating mode and λ if available.

4. **Detect stale state proactively.**
   - If a routing or resource-allocation decision depends on hospital state, always refresh before deciding.
   - If state is stale (> 30 minutes old), ask staff to confirm current occupancy before using the data.

5. **Trigger calibration if needed.**
   - If the nurse reports that occupancy has changed, activate the `hospital_calibration` workflow.

## Rules

- **Never put current occupancy into memory between turns.** Always fetch fresh.
- **Never report stale data as current** without flagging it to the nurse.
- Hospital state is dynamic — the operational picture can change every few minutes during a busy shift.
- Do not guess occupancy. Ask the nurse if uncertain.
