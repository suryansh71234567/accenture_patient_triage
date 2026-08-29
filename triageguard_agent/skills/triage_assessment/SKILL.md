---
name: triage_assessment
description: Run the existing XGBoost and RAG clinical prediction branches for the current patient.
---

# Skill: Triage Assessment

## Purpose
Run the existing clinical prediction and reasoning branches for the current patient.

## Procedure

1. **Obtain current patient state.**
   - Call `get_patient_summary(patient_id=<id>)`.
   - Verify the result is successful before proceeding.

2. **Verify that the state is valid for inference.**
   - At minimum, a chief complaint must be present.
   - Warn the nurse if critical vitals (HR, SpO₂, BP) are missing — assessment will proceed with reduced XGBoost confidence.

3. **Run the TriageGuard pipeline.**
   - Call `run_triage_assessment(patient_data=<patient_dict>)`.
   - Do NOT modify the returned predictions.

4. **Preserve outputs of both branches separately.**
   - XGBoost: admission risk, ICU risk, confidence, information completeness.
   - RAG: disposition, escalation level, top diagnoses, red flags.

5. **Report structured results to the nurse.**
   - Present the reconciled routing decision.
   - Report the confidence note (which branch is dominant).
   - List red flags and top diagnoses.
   - If branches disagree, flag this explicitly and activate `human_review` skill.

6. **Report missing information and uncertainty.**
   - If information_completeness is low, explain that XGBoost confidence is reduced and RAG is dominant.

## Rules

- **XGBoost remains the quantitative prediction branch.** The agent does not replace it.
- **RAG remains the historical/contextual reasoning branch.** The agent does not replace it.
- The agent MUST NOT modify, override, or second-guess the prediction outputs.
- If both branches strongly disagree (`branches_agree = False`), escalate to human review.
