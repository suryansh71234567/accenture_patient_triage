---
name: routing
description: Handle routing and escalation requests with human approval for operational writes.
---

# Skill: Routing

## Purpose
Handle routing and escalation requests.

## Procedure

1. **Identify patient.**
   - Confirm `active_patient_id` from agent state.

2. **Determine request type.**
   - Is the nurse asking for a routing *recommendation* or an operational *action* (actually moving the patient)?
   - A recommendation is informational — no write required.
   - An operational action requires approval.

3. **Retrieve current patient assessment.**
   - Use the latest `run_triage_assessment` result if available (check working memory).
   - If stale or absent, re-run the assessment.

4. **Retrieve current hospital state.**
   - Call `get_hospital_state(department=<target>)`.
   - Check if the target department has available beds.
   - If `is_stale=True`, ask the nurse to confirm current occupancy before proceeding.

5. **Check destination availability.**
   - If available beds = 0 for the target, report this and do NOT route there.
   - Suggest the next appropriate department instead.

6. **For operational writes, create a proposed action.**
   - Call `propose_hospital_calibration` if a bed state change is needed.
   - Build a structured routing proposal for nurse confirmation.

7. **Require human approval.**
   - Use the confirmation protocol: show the proposal and ask the nurse to confirm.
   - Do NOT execute without explicit "yes" / "confirm" from the nurse.

8. **Execute only an approved action.**
   - After confirmation, call `commit_hospital_calibration` for state changes.

9. **Verify execution.**
   - Confirm the tool result is `success=True`.
   - Report any failure to the nurse immediately.

10. **Update hospital state.**
    - After a successful move, the committed tool call automatically recalculates λ and operating mode.

## Rules

- **Never allow an LLM-generated string to directly change patient location.** All writes go through the validated tool chain.
- A routing recommendation does NOT require approval. A routing action does.
- If the patient's ICU risk has changed significantly since the last assessment, re-run before routing.
