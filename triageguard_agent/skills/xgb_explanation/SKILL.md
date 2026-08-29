---
name: xgb_explanation
description: Explain an XGBoost prediction using deterministic model attribution data only.
---

# Skill: XGBoost Explanation

## Purpose
Explain an XGBoost prediction using deterministic model attribution.

## Procedure

1. **Obtain the relevant prediction.**
   - If a recent `run_triage_assessment` result is available in working memory, use it.
   - Otherwise, call `run_triage_assessment` first to obtain fresh predictions.

2. **Obtain feature attribution.**
   - Call `get_xgb_explanation(patient_data=<patient_dict>)`.
   - The tool returns: admission risk, ICU risks, confidences, information completeness, vitals present/missing.

3. **Identify the strongest contributing factors.**
   - Highlight which vitals were present vs. missing.
   - Explain how information completeness affects model confidence.
   - Report which time horizon (2h/6h/12h) shows the highest ICU risk.

4. **Preserve contribution direction.**
   - State whether vitals are within or outside normal ranges.
   - Do NOT invent a causal pathway not shown by the attribution data.

5. **Present the explanation to the nurse.**
   - Use plain language. Avoid jargon like "feature vector" or "calibrated probability."
   - Example: "The system flagged an elevated heart rate and low SpO₂ as the primary drivers of the high admission risk."

## Rules

- **Do NOT describe XGBoost as having a hidden chain-of-thought.** It has predictions and feature contributions only.
- **Never invent a causal explanation** that is not supported by the attribution data returned by the tool.
- If `information_completeness` is low, explain that the score is driven more by RAG than XGBoost.
- Do not expose raw model weights, PCA components, or embedding dimensions.
