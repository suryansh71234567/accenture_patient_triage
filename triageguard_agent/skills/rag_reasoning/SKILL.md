---
name: rag_reasoning
description: Explain the historical reasoning produced by the RAG branch using returned evidence only.
---

# Skill: RAG Reasoning

## Purpose
Explain the historical reasoning produced by the RAG branch.

## Procedure

1. **Retrieve the structured RAG result.**
   - Use the `rag_narrative` and `rag_disposition`/`rag_escalation` fields from the last `run_triage_assessment` result.
   - If no recent result is available, call `run_triage_assessment` first.

2. **Identify historical evidence used.**
   - The RAG branch retrieves similar historical patient cases.
   - Report what patterns from those cases influenced the current reasoning.

3. **Distinguish current patient facts from historical cases.**
   - Clearly label: "Based on this patient's current presentation…" vs. "Historical cases with similar presentations showed…"

4. **Present the trajectory assessment.**
   - What does the historical evidence suggest about likely progression?
   - What escalation level did the RAG branch recommend and why?

5. **Present uncertainty and limitations.**
   - How many historical cases were retrieved?
   - Does the current patient match the retrieved cases well?
   - Is there conflicting historical evidence?

## Rules

- **Never claim that historical outcomes ARE the current patient's outcome.** Historical cases inform — they do not determine.
- **Never invent evidence.** Only present what the tool returned.
- The RAG branch should expose an auditable reasoning summary — not speculation.
- If `rag_disposition` is "unknown", report that the RAG branch could not determine a disposition and escalate to human review.
