---
name: patient_lookup
description: Retrieve factual information about a patient without inventing missing information.
---

# Skill: Patient Lookup

## Purpose
Retrieve factual information about a patient without inventing missing information.

## Procedure

1. **Identify the patient.**
   - Use the `active_patient_id` from agent state.
   - If patient identity is ambiguous or absent, ask the nurse for clarification before proceeding.

2. **Retrieve current patient state.**
   - Call `get_patient_summary(patient_id=<id>)`.
   - Do NOT proceed until a successful ToolResult is returned.

3. **Retrieve timeline only when requested.**
   - Call `get_patient_observations(patient_id=<id>)` only if the nurse explicitly asks for history or trend information.

4. **Distinguish current vs. historical observations.**
   - Clearly label which values are current and which are from a prior visit.

5. **Report only what is in the data.**
   - If a value is missing or null, state "not recorded" — do not estimate or infer.
   - If a field is unavailable, say so explicitly.

## Rules

- Patient data returned by the tool is authoritative. Do not override it.
- Do NOT use the RAG branch to answer simple factual patient-data questions unless historical context is explicitly requested.
- Never expose unrelated patient information unnecessarily.
- Never fabricate clinical values that are not present in the tool result.
