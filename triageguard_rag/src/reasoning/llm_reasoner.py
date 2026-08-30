"""
llm_reasoner.py
---------------
Sends a structured clinical prompt to an LLM via the OpenRouter API
and returns the model's reasoning text.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_doc(doc: Dict, label: str) -> str:
    meta = doc.get("metadata", {})
    # Show hospital name when present (hospital-provided records);
    # fall back to the 'source' field for existing MIMIC docs.
    hospital = meta.get("hospital_name") or meta.get("source", "N/A")
    return (
        f"[{label}] Source: {hospital} | "
        f"Stay {meta.get('stay_id', 'N/A')} | "
        f"Acuity {meta.get('acuity', 'N/A')} | "
        f"Disposition: {meta.get('disposition', 'N/A')}\n"
        f"{doc['document_text']}"
    )


def build_prompt(
    current_state: Dict,
    patient_history: List[Dict],
    similar_cases: List[Dict],
) -> str:
    """
    Construct the system + user prompt for clinical trajectory reasoning.

    Parameters
    ----------
    current_state : dict with keys:
        patient_id, chiefcomplaint, acuity, heartrate, resprate,
        o2sat, sbp, dbp, temperature, pain   (all may be None/NaN)
    patient_history  : list of retrieved docs from this patient's past.
    similar_cases    : list of retrieved docs from similar patients.
    """
    # ── current state text ─────────────────────────────────────────────────
    def fmt(v, unit=""):
        return f"{v}{unit}" if v not in (None, "None", "", float("nan")) else "N/A"

    # ── history section ────────────────────────────────────────────────────
    if patient_history:
        history_text = "\n\n".join(
            _format_doc(d, f"Past Visit {i+1}") for i, d in enumerate(patient_history)
        )
    else:
        history_text = "No prior ED visits found for this patient."

    # ── similar cases section ──────────────────────────────────────────────
    if similar_cases:
        similar_text = "\n\n".join(
            _format_doc(d, f"Similar Case {i+1}") for i, d in enumerate(similar_cases)
        )
    else:
        similar_text = "No similar cases retrieved."

    # Split current_text (built above) into identity vs. clinical-state
    # lines so the prompt can present them as two distinct sections, per
    # the Master MD's 4-section prompt structure (CURRENT PATIENT /
    # CURRENT CLINICAL STATE / PATIENT'S RELEVANT HISTORY / SIMILAR
    # HISTORICAL CASES).
    identity_text = (
        f"Patient ID      : {current_state.get('patient_id', 'N/A')}\n"
        f"Chief complaint : {fmt(current_state.get('chiefcomplaint'))}\n"
        f"Acuity level    : {fmt(current_state.get('acuity'))}\n"
    )
    clinical_state_text = (
        f"Heart rate      : {fmt(current_state.get('heartrate'), ' bpm')}\n"
        f"Resp rate       : {fmt(current_state.get('resprate'), ' /min')}\n"
        f"SpO2            : {fmt(current_state.get('o2sat'), '%')}\n"
        f"BP              : {fmt(current_state.get('sbp'))}/{fmt(current_state.get('dbp'))} mmHg\n"
        f"Temperature     : {fmt(current_state.get('temperature'), '°F')}\n"
        f"Pain score      : {fmt(current_state.get('pain'), '/10')}\n"
    )

    # ── final prompt ───────────────────────────────────────────────────────
    prompt = f"""You are a senior emergency physician assistant helping with real-time patient triage.

=== CURRENT PATIENT ===
{identity_text}

=== CURRENT CLINICAL STATE ===
{clinical_state_text}

=== PATIENT'S RELEVANT HISTORY (this patient's own prior visits) ===
{history_text}

=== SIMILAR HISTORICAL CASES (other patients — how they were actually treated and what happened) ===
{similar_text}

=== TASK ===
You are doing case-based clinical reasoning: look at how similar past
patients actually presented, how they were treated, and what happened to
them, and use that precedent the way a physician recalls prior cases to
reason about THIS patient's likely trajectory. Use the retrieved outcomes
as evidence — do not withhold or ignore an outcome that is already present
in the evidence above.

Based on the current presentation and the evidence above, reason about:
1. What is the likely clinical trajectory for this patient over the next few hours?
2. What would a meaningful change (improvement or deterioration) look like for this patient?
3. Are there any red flags in the vitals or history that warrant urgent attention?
4. Is escalation (e.g. ICU-level concern) warranted based on the trajectory and evidence?
5. How strongly does the retrieved evidence above actually support this assessment, versus general clinical judgment?

Be concise, structured, and clinically accurate. Cite specific vital values or historical findings where relevant.
You are NOT deciding the final disposition or department placement — a separate
deterministic system does that. Do not output an admit/discharge/observation decision.

After your narrative, output a JSON block (and nothing after it) in exactly this format:
```json
{{
  "trajectory_assessment": "<one or two sentence summary of the likely clinical course>",
  "possible_next_state": "<one sentence describing what deterioration or improvement would look like>",
  "urgency": "low" | "moderate" | "high" | "critical",
  "evidence_strength": <integer 1-5, how strongly the retrieved history actually supports this assessment>,
  "escalation_concern": true | false,
  "reasoning": ["<reason 1>", "<reason 2>"],
  "supporting_history": ["<specific evidence from this patient's own history, or empty list>"],
  "similar_case_summary": ["<what happened in similar retrieved cases, or empty list>"],
  "limitations": ["<what is missing or uncertain in this assessment>"],
  "top_diagnoses": ["<dx1>", "<dx2>", "<dx3>"],
  "red_flags": ["<flag1>", "<flag2>"]
}}
```"""

    return prompt


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_openrouter(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    timeout: float = 60.0,
) -> str:
    """
    Call OpenRouter chat completions and return the assistant's text.

    Raises
    ------
    httpx.HTTPStatusError on non-2xx responses.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/TriageGuard",
        "X-Title":       "TriageGuard",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role":    "system",
                "content": "You are a clinical decision support assistant specializing in emergency triage.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    logger.info("Calling OpenRouter model: %s", model)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    logger.info("Received response (%d chars).", len(content))
    return content


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def _parse_structured(response: str) -> dict:
    """
    Extract the trailing ```json ... ``` block from the LLM response.
    Returns a dict with trajectory_assessment, possible_next_state, urgency,
    evidence_strength, escalation_concern, reasoning, supporting_history,
    similar_case_summary, limitations, top_diagnoses, red_flags.
    Falls back to safe defaults if parsing fails.
    """
    import re
    defaults = {
        "trajectory_assessment": "unknown",
        "possible_next_state": "unknown",
        "urgency": "unknown",
        "evidence_strength": 3,
        "escalation_concern": False,
        "reasoning": [],
        "supporting_history": [],
        "similar_case_summary": [],
        "limitations": [],
        "top_diagnoses": [],
        "red_flags": [],
    }
    try:
        # Find the last ```json ... ``` block
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if not match:
            # Also try bare JSON object at end of response
            match = re.search(r"(\{[^{}]*\"urgency\"[^{}]*\})", response, re.DOTALL)
        if match:
            import json as _json
            parsed = _json.loads(match.group(1))
            defaults.update(parsed)
    except Exception:
        pass
    return defaults


def reason(
    current_state: Dict,
    patient_history: List[Dict],
    similar_cases: List[Dict],
    api_key: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 1000,
) -> Dict:
    """
    Build prompt, call the LLM, and return structured result.

    Returns
    -------
    {
      "prompt":           <str>,   # the full prompt sent to the LLM
      "response":         <str>,   # the full LLM answer (narrative + JSON block)
      "structured_output": {
          "trajectory_assessment": <str>,
          "possible_next_state":   <str>,
          "urgency":               "low" | "moderate" | "high" | "critical" | "unknown",
          "evidence_strength":     <int 1-5>,   # LLM-rated, not a probability
          "escalation_concern":    <bool>,
          "reasoning":             [<str>, ...],
          "supporting_history":    [<str>, ...],
          "similar_case_summary":  [<str>, ...],
          "limitations":           [<str>, ...],
          "top_diagnoses":         [<str>, ...],
          "red_flags":             [<str>, ...],
      }
    }
    """
    prompt   = build_prompt(current_state, patient_history, similar_cases)
    response = call_openrouter(
        prompt=prompt,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    structured = _parse_structured(response)
    return {"prompt": prompt, "response": response, "structured_output": structured}
