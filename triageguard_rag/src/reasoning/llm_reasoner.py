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

    current_text = (
        f"Patient ID      : {current_state.get('patient_id', 'N/A')}\n"
        f"Chief complaint : {fmt(current_state.get('chiefcomplaint'))}\n"
        f"Acuity level    : {fmt(current_state.get('acuity'))}\n"
        f"Heart rate      : {fmt(current_state.get('heartrate'), ' bpm')}\n"
        f"Resp rate       : {fmt(current_state.get('resprate'), ' /min')}\n"
        f"SpO2            : {fmt(current_state.get('o2sat'), '%')}\n"
        f"BP              : {fmt(current_state.get('sbp'))}/{fmt(current_state.get('dbp'))} mmHg\n"
        f"Temperature     : {fmt(current_state.get('temperature'), '°F')}\n"
        f"Pain score      : {fmt(current_state.get('pain'), '/10')}\n"
    )

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

    # ── final prompt ───────────────────────────────────────────────────────
    prompt = f"""You are a senior emergency physician assistant helping with real-time patient triage.

=== CURRENT PATIENT STATE ===
{current_text}

=== PATIENT'S OWN PRIOR ED VISITS ===
{history_text}

=== CLINICALLY SIMILAR PATIENTS (other patients) ===
{similar_text}

=== TASK ===
Based on the current presentation and the evidence above, reason about:
1. What are the most likely diagnoses or clinical trajectories for this patient?
2. What immediate clinical actions or escalations should be considered?
3. What is your estimated disposition (admit / discharge / observation)?
4. Are there any red flags in the vitals or history that warrant urgent attention?

Be concise, structured, and clinically accurate. Cite specific vital values or historical findings where relevant.

After your narrative, output a JSON block (and nothing after it) in exactly this format:
```json
{{
  "disposition": "admit" | "discharge" | "observation",
  "escalation_level": "routine" | "urgent" | "emergent",
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
    Returns a dict with disposition, escalation_level, top_diagnoses, red_flags.
    Falls back to safe defaults if parsing fails.
    """
    import re
    defaults = {
        "disposition": "unknown",
        "escalation_level": "unknown",
        "top_diagnoses": [],
        "red_flags": [],
    }
    try:
        # Find the last ```json ... ``` block
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if not match:
            # Also try bare JSON object at end of response
            match = re.search(r"(\{[^{}]*\"disposition\"[^{}]*\})", response, re.DOTALL)
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
          "disposition":       "admit" | "discharge" | "observation" | "unknown",
          "escalation_level": "routine" | "urgent" | "emergent" | "unknown",
          "top_diagnoses":    [<str>, ...],
          "red_flags":        [<str>, ...],
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
