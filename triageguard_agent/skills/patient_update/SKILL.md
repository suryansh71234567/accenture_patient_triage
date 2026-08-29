---
name: patient_update
description: Record a new timestamped patient observation through the validated write tool.
---

# Skill: Patient Update

## Purpose
Record a new timestamped patient observation.

## Procedure

1. **Identify patient and encounter.**
   - Use `active_patient_id` and `active_encounter_id` from agent state.
   - If either is missing, ask the nurse before continuing.

2. **Obtain the current patient state.**
   - Call `get_patient_summary` to confirm the current baseline before recording a change.

3. **Parse only values explicitly provided by the nurse.**
   - Accept: vital signs, pain score, chief complaint updates, acuity changes.
   - Do NOT infer or guess values the nurse did not state.
   - If an observation is ambiguous (e.g. "her BP is high") ask for the specific numeric value.

4. **Validate field names, units, and value ranges.**
   - Heart rate: 0–300 bpm
   - SpO₂: 0–100%
   - Respiratory rate: 0–100/min
   - Systolic BP: 0–300 mmHg
   - Temperature: 30–45°C or 86–113°F

5. **Record the observation through the validated write tool.**
   - Use the patient observation write tool with the parsed, validated values.
   - Require nurse confirmation before committing.

6. **Update the patient's current state.**
   - After successful write, mark the previous assessment as potentially stale.

7. **Trigger reassessment if requested.**
   - If the nurse asks for a reassessment after the update, activate the `triage_assessment` skill.

8. **Report exactly what was recorded.**
   - Confirm the values that were written, not what was expected.

## Rules

- Never silently change a numerical observation.
- Never fabricate missing values.
- **Never construct the XGBoost feature vector manually in the LLM.** Feature construction belongs to the existing backend pipeline — call `run_triage_assessment` instead.
- A write operation requires nurse confirmation before execution.
