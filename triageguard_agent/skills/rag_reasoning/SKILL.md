---
name: rag_reasoning
description: Explain the historical reasoning produced by the RAG branch using returned evidence only.
---

# Skill: RAG Reasoning

## Purpose
Explain the historical reasoning produced by the RAG branch.

## Procedure

1. **Retrieve the structured RAG result.**
   - Use the `rag_narrative`, `rag_urgency`, `rag_evidence_strength`, and `rag_escalation_concern` fields from the last `run_triage_assessment` result.
   - If no recent result is available, call `run_triage_assessment` first.

2. **Identify historical evidence used.**
   - The RAG branch retrieves similar historical patient cases.
   - Report what patterns from those cases influenced the current reasoning.

3. **Distinguish current patient facts from historical cases.**
   - Clearly label: "Based on this patient's current presentation…" vs. "Historical cases with similar presentations showed…"

4. **Present the trajectory assessment.**
   - What does the historical evidence suggest about likely progression?
   - What urgency level (`rag_urgency`) did the RAG branch assign, and did it raise an `rag_escalation_concern`?

5. **Present uncertainty and limitations.**
   - Report `rag_evidence_strength` (1-5) as how strongly the retrieved history actually supports the assessment — this is not a probability.
   - Does the current patient match the retrieved cases well?
   - Is there conflicting historical evidence?

## Rules

- **Never claim that historical outcomes ARE the current patient's outcome.** Historical cases inform — they do not determine.
- **Never invent evidence.** Only present what the tool returned.
- The RAG branch should expose an auditable reasoning summary — not speculation.
- The RAG branch never decides admit/discharge/observation — that decision was intentionally removed from its scope. If `rag_urgency` is "unknown", report that the RAG branch could not assess urgency and escalate to human review.
