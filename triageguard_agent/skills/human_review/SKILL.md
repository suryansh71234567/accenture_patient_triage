---
name: human_review
description: Escalate situations that require human judgment and cannot be safely resolved by the agent alone.
---

# Skill: Human Review

## Purpose
Escalate situations requiring human judgment.

## When to Activate

Request human review when any of the following are true:

1. **Model branches strongly disagree** — `branches_agree = False` from the triage assessment.
2. **Required clinical information is missing** — critical vitals are absent and the nurse cannot provide them.
3. **Patient identity is ambiguous** — multiple patients could match the description given by the nurse.
4. **An operational write requires approval** — any WRITE tool result is pending confirmation.
5. **A tool fails during a critical workflow** — any tool returns `success=False` during triage or routing.
6. **The system cannot confidently determine the requested action** — the agent is uncertain and proceeding would be unsafe.
7. **RAG disposition is unknown** — the LLM reasoning branch could not determine a disposition.
8. **High ICU risk with low completeness** — XGBoost ICU risk is high but data completeness is very low (< 30%).

## Procedure

1. **Stop the current workflow.**
2. **State clearly why human review is needed.** Be specific — do not say "I need help." Say which condition triggered the escalation.
3. **Present the available evidence.**
   - What does XGBoost say (if available)?
   - What does RAG say (if available)?
   - What information is missing?
4. **Propose a conservative interim action** if appropriate (e.g. "Pending review, I recommend treating as high acuity.").
5. **Wait for human input** before taking any further action.

## Rules

- **The human reviewer is the final authority** for actions requiring approval.
- Do not attempt to resolve a human_review situation by inferring the "likely" correct action.
- Document the escalation reason clearly so the reviewer understands the situation quickly.
- After human review resolves the situation, return to the appropriate workflow skill.
