"""
presimulated_patients.py
------------------------
A pool of demo patients with pre-computed clinical assessments.
These bypass the ML pipeline so the simulation always has interesting
queue content that is reliable for demo purposes.

Scenario-based selection:
  NORMAL   → 3-4 waiting + 2 admitted
  HIGH_LOAD → 7 waiting + 4 admitted
  CRITICAL  → 11 waiting + 6 admitted
"""

from __future__ import annotations
from typing import Any, Dict, List
from triageguard_agent.simulation.patient_flow import SimulatedPatient, PatientStatus


# ---------------------------------------------------------------------------
# Pre-baked patient pool
# Each patient has fully resolved clinical_assessment + operational_decision
# so no ML pipeline call is needed. The assessment mirrors what the real
# pipeline would produce for that acuity / complaint combination.
# ---------------------------------------------------------------------------

def _make_assessment(
    dept: str,
    admission_risk: float,
    icu_risk: float,
    reasoning: str,
    red_flags: List[str],
    top_dx: List[str],
    urgency: str = "moderate",
    escalation: bool = False,
    has_history: bool = False,
) -> Dict[str, Any]:
    evidence_strength = 4 if has_history else 2
    return {
        "department": dept,
        "department_reasoning": reasoning,
        "acuity_tier": 2 if icu_risk > 0.4 else (3 if admission_risk > 0.5 else 4),
        "reconciled_admission_risk": admission_risk,
        "reconciled_icu_risk": icu_risk,
        "branches_agree": True,
        "confidence_note": (
            "History-augmented assessment — RAG retrieved prior visit records."
            if has_history
            else "Assessment based on current vitals only. No prior history available — evidence strength reduced."
        ),
        "top_diagnoses": top_dx,
        "red_flags": red_flags,
        "urgency": urgency,
        "escalation_concern": escalation,
        "evidence_strength": evidence_strength,
        "rag_disposition": "Admit" if admission_risk > 0.5 else "Observe",
        "rag_escalation": "ICU" if icu_risk > 0.4 else "General",
        "rag_narrative": (
            f"{'Prior history retrieved. ' if has_history else 'No prior history found. '}"
            f"{reasoning}"
        ),
    }


def _make_op(
    clinical_dept: str,
    operational_dept: str,
    available_beds: int,
    mode: str,
    lam: float,
    cap_warning: bool = False,
    confirm: bool = False,
    summary: str = "",
) -> Dict[str, Any]:
    return {
        "clinical_department": clinical_dept,
        "operational_department": operational_dept,
        "available_beds_in_clinical_dept": available_beds,
        "operating_mode": mode,
        "lambda": lam,
        "capacity_warning": cap_warning,
        "confirmation_required": confirm,
        "recommendation_summary": summary or f"Routed to {operational_dept}.",
    }


# ---------------------------------------------------------------------------
# The pool — 20 patients, mix of MIMIC-linked (with history) and walk-ins
# ---------------------------------------------------------------------------

PATIENT_POOL: List[Dict[str, Any]] = [

    # ── Acuity 1 / Critical ─────────────────────────────────────────────

    {
        "patient_id": "PAT-CRIT-01",
        "age": 67, "sex": "M",
        "chief_complaint": "ST-elevation on arrival ECG, crushing substernal chest pain, diaphoresis",
        "vitals": {"hr": 118, "rr": 24, "spo2": 91, "sbp": 88, "dbp": 54, "temp": 36.8, "pain": 10},
        "acuity": 1, "expected_los_min": 180,
        "metadata": {"cardiac_hint": True, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "CICU", 0.97, 0.92,
            "STEMI presentation. HR 118, SpO2 91%, BP 88/54 — cardiogenic shock territory. Immediate cath lab activation required.",
            ["STEMI pattern on ECG", "Cardiogenic shock (SBP 88)", "SpO2 91% — oxygen-dependent"],
            ["STEMI", "Cardiogenic shock", "Acute MI"],
            urgency="critical", escalation=True,
        ),
        "operational_decision": _make_op("CICU", "CICU", 2, "NORMAL", 0.62, summary="CICU bed allocated. Cath lab team paged."),
    },

    {
        "patient_id": "10037928",  # MIMIC — 81F COPD+CHF, SpO2 88%
        "age": 81, "sex": "F",
        "chief_complaint": "Influenza-like illness, respiratory distress, SpO2 88% on room air",
        "vitals": {"hr": 97, "rr": 32, "spo2": 88, "sbp": 176, "dbp": 86, "temp": 99.8, "pain": 0},
        "acuity": 2, "expected_los_min": 150,
        "metadata": {"cardiac_hint": False, "has_history": True,
                     "history_text": "Prior history: cardiovascular disease, respiratory disease, 1 prior ICU admission",
                     "previous_ed_visits": 4, "previous_hospital_admissions": 3, "previous_icu_admissions": 1,
                     "cardiovascular_history": 1, "respiratory_history": 1},
        "clinical_assessment": _make_assessment(
            "ICU", 0.91, 0.72,
            "81F with COPD and CHF presenting with RR 32, SpO2 88%. Prior ICU admission for similar presentation. High risk of respiratory failure. Prior history retrieved from 4 ED visits.",
            ["SpO2 88% on RA", "RR 32 — severe tachypnoea", "Age 81 with COPD+CHF"],
            ["Acute exacerbation COPD", "Decompensated CHF", "Community-acquired pneumonia"],
            urgency="critical", escalation=True, has_history=True,
        ),
        "operational_decision": _make_op("ICU", "ICU", 3, "NORMAL", 0.62, summary="ICU bed allocated. Prior ICU history noted — escalation appropriate."),
    },

    # ── Acuity 2 / High ──────────────────────────────────────────────────

    {
        "patient_id": "10039708",  # MIMIC — 72M Hypotension, 2 prior ICU
        "age": 72, "sex": "M",
        "chief_complaint": "Hypotension (BP 98/68), altered mental status, brought by ambulance",
        "vitals": {"hr": 89, "rr": 16, "spo2": 98, "sbp": 98, "dbp": 68, "temp": 98.1, "pain": 0},
        "acuity": 2, "expected_los_min": 120,
        "metadata": {"cardiac_hint": False, "has_history": True,
                     "history_text": "Prior history: cardiovascular disease, chronic kidney disease, 2 prior ICU admissions",
                     "previous_ed_visits": 5, "previous_hospital_admissions": 4, "previous_icu_admissions": 2,
                     "cardiovascular_history": 1, "renal_history": 1},
        "clinical_assessment": _make_assessment(
            "ICU", 0.88, 0.65,
            "72M with CAD+CKD and 2 prior ICU admissions. BP 98/68 meets SIRS criteria. Altered mental status raises concern for sepsis or cardiogenic event. 12 prior visit documents retrieved.",
            ["SBP 98 — borderline shock", "Altered mental status", "2 prior ICU admissions (pattern recognition)"],
            ["Septic shock", "Cardiogenic event", "AKI on CKD"],
            urgency="critical", escalation=True, has_history=True,
        ),
        "operational_decision": _make_op("ICU", "ICU", 3, "NORMAL", 0.62, summary="ICU bed allocated. Two prior ICU admissions — high escalation confidence."),
    },

    {
        "patient_id": "10016742",  # MIMIC — 71M CAD+DM, chest pain
        "age": 71, "sex": "M",
        "chief_complaint": "Chest pain and shortness of breath, onset 2 hours ago",
        "vitals": {"hr": 112, "rr": 22, "spo2": 94, "sbp": 148, "dbp": 90, "temp": 98.9, "pain": 7},
        "acuity": 2, "expected_los_min": 120,
        "metadata": {"cardiac_hint": True, "has_history": True,
                     "history_text": "Prior history: cardiovascular disease, diabetes mellitus, 1 prior ICU admission",
                     "previous_ed_visits": 3, "previous_hospital_admissions": 2, "previous_icu_admissions": 1,
                     "cardiovascular_history": 1, "diabetes_history": 1},
        "clinical_assessment": _make_assessment(
            "CICU", 0.87, 0.61,
            "71M with CAD and DM presenting with chest pain, HR 112, SpO2 94%. Prior NSTEMI with CICU admission. Evidence strength high — 4 prior visit documents retrieved.",
            ["HR 112 — tachycardia", "SpO2 94%", "CAD with prior NSTEMI"],
            ["NSTEMI / ACS", "Unstable angina", "Pulmonary oedema"],
            urgency="high", escalation=True, has_history=True,
        ),
        "operational_decision": _make_op("CICU", "CICU", 1, "NORMAL", 0.62, confirm=True,
                                          summary="CICU has 1 bed remaining. Given prior NSTEMI history, confirm reservation."),
    },

    {
        "patient_id": "PAT-HI-02",
        "age": 55, "sex": "F",
        "chief_complaint": "Sudden onset severe headache '10/10 worst of my life', photophobia, neck stiffness",
        "vitals": {"hr": 94, "rr": 18, "spo2": 99, "sbp": 172, "dbp": 98, "temp": 37.9, "pain": 10},
        "acuity": 2, "expected_los_min": 100,
        "metadata": {"cardiac_hint": False, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "ICU", 0.84, 0.55,
            "55F presenting with thunderclap headache, meningismus, and BP 172/98. High suspicion for subarachnoid haemorrhage. No prior history — uncertainty increased.",
            ["Thunderclap headache — SAH until proven otherwise", "Meningismus", "BP 172/98"],
            ["Subarachnoid haemorrhage", "Bacterial meningitis", "Hypertensive emergency"],
            urgency="critical", escalation=True,
        ),
        "operational_decision": _make_op("ICU", "ICU", 3, "NORMAL", 0.62, summary="ICU allocated. CT head + LP required urgently."),
    },

    # ── Acuity 3 / Moderate ──────────────────────────────────────────────

    {
        "patient_id": "10012853",  # MIMIC — 64M Hypertension, CAD+DM+neuro
        "age": 64, "sex": "M",
        "chief_complaint": "Hypertensive urgency — BP 218/72, persistent headache for 2 hours",
        "vitals": {"hr": 65, "rr": 18, "spo2": 96, "sbp": 218, "dbp": 72, "temp": 97.5, "pain": 0},
        "acuity": 3, "expected_los_min": 90,
        "metadata": {"cardiac_hint": False, "has_history": True,
                     "history_text": "Prior history: cardiovascular disease, diabetes mellitus, neurological condition",
                     "previous_ed_visits": 3, "previous_hospital_admissions": 2,
                     "cardiovascular_history": 1, "diabetes_history": 1, "neurological_history": 1},
        "clinical_assessment": _make_assessment(
            "ADMITTED_GEN", 0.68, 0.14,
            "64M with CAD, DM, and prior neurological event. BP 218/72 — hypertensive urgency. No end-organ damage signs currently. Prior DVT visit retrieved. 9 prior documents available.",
            ["SBP 218 — hypertensive urgency", "Prior neurological history — stroke risk elevated"],
            ["Hypertensive urgency", "Secondary headache", "Poorly controlled hypertension"],
            urgency="high", escalation=False, has_history=True,
        ),
        "operational_decision": _make_op("ADMITTED_GEN", "ADMITTED_GEN", 5, "NORMAL", 0.62,
                                          summary="General ward bed available. BP monitoring and IV antihypertensive initiated."),
    },

    {
        "patient_id": "10005866",  # MIMIC — 45F Abdominal pain
        "age": 45, "sex": "F",
        "chief_complaint": "Severe right-sided abdominal pain 8/10, nausea and vomiting, onset 6 hours ago",
        "vitals": {"hr": 66, "rr": 18, "spo2": 99, "sbp": 111, "dbp": 68, "temp": 97.4, "pain": 8},
        "acuity": 2, "expected_los_min": 80,
        "metadata": {"cardiac_hint": False, "has_history": True,
                     "history_text": "Prior history: 3 prior ED visits for similar abdominal pain",
                     "previous_ed_visits": 3, "previous_hospital_admissions": 1},
        "clinical_assessment": _make_assessment(
            "ADMITTED_GEN", 0.74, 0.18,
            "45F with 3 prior ED visits for abdominal pain. Current presentation: RLQ pain 8/10, fever 97.4. Appendicitis vs ovarian pathology. Surgical consult indicated.",
            ["Pain 8/10 — significant", "Recurrent presentations (3 prior visits)", "Surgical abdomen possible"],
            ["Acute appendicitis", "Ovarian torsion / cyst", "Small bowel obstruction"],
            urgency="high", escalation=False, has_history=True,
        ),
        "operational_decision": _make_op("ADMITTED_GEN", "ADMITTED_GEN", 5, "NORMAL", 0.62,
                                          summary="General ward bed allocated. CT abdomen/pelvis ordered."),
    },

    {
        "patient_id": "PAT-MOD-03",
        "age": 48, "sex": "M",
        "chief_complaint": "Community-acquired pneumonia — productive cough, fever 38.8°C, pleuritic chest pain",
        "vitals": {"hr": 96, "rr": 22, "spo2": 93, "sbp": 118, "dbp": 74, "temp": 38.8, "pain": 5},
        "acuity": 3, "expected_los_min": 90,
        "metadata": {"cardiac_hint": False, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "ADMITTED_GEN", 0.69, 0.12,
            "48M walk-in with CAP presentation. SpO2 93%, RR 22, fever 38.8. PSI/PORT score indicates moderate risk. No prior history — evidence based on vitals and similar cases only.",
            ["SpO2 93%", "RR 22", "Fever 38.8°C"],
            ["Community-acquired pneumonia", "Pulmonary embolism", "Pleuritis"],
            urgency="moderate", escalation=False,
        ),
        "operational_decision": _make_op("ADMITTED_GEN", "ADMITTED_GEN", 5, "NORMAL", 0.62,
                                          summary="General ward admission. IV antibiotics and O2 supplementation initiated."),
    },

    {
        "patient_id": "10014354",  # MIMIC — 68M Leg pain/weakness, CKD+DM+neuro
        "age": 68, "sex": "M",
        "chief_complaint": "Left leg pain and acute weakness, sudden onset 1 hour ago",
        "vitals": {"hr": 72, "rr": 18, "spo2": 100, "sbp": 125, "dbp": 62, "temp": 98.6, "pain": 0},
        "acuity": 3, "expected_los_min": 90,
        "metadata": {"cardiac_hint": False, "has_history": True,
                     "history_text": "Prior history: chronic kidney disease, diabetes mellitus, neurological condition",
                     "previous_ed_visits": 6, "previous_hospital_admissions": 3,
                     "renal_history": 1, "diabetes_history": 1, "neurological_history": 1},
        "clinical_assessment": _make_assessment(
            "ADMITTED_GEN", 0.70, 0.16,
            "68M with CKD, DM, and neuro history. Sudden leg weakness — peripheral vascular event or cord compression. 23 prior visit documents retrieved. High evidence strength.",
            ["Sudden onset weakness — vascular event possible", "CKD + DM — peripheral vascular disease risk"],
            ["Peripheral arterial occlusion", "DVT", "Lumbar disc herniation with radiculopathy"],
            urgency="high", escalation=False, has_history=True,
        ),
        "operational_decision": _make_op("ADMITTED_GEN", "ADMITTED_GEN", 5, "NORMAL", 0.62,
                                          summary="General ward bed. Vascular surgery and neurology consult requested."),
    },

    {
        "patient_id": "PAT-MOD-04",
        "age": 34, "sex": "F",
        "chief_complaint": "Severe migraine, photophobia, vomiting, not responding to home analgesia",
        "vitals": {"hr": 82, "rr": 16, "spo2": 99, "sbp": 122, "dbp": 78, "temp": 37.1, "pain": 9},
        "acuity": 3, "expected_los_min": 60,
        "metadata": {"cardiac_hint": False, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "ADMITTED_GEN", 0.55, 0.07,
            "34F presenting with migraine. No prior history on file — cannot exclude secondary causes. Pain 9/10, not responding to OTC. CT head recommended to exclude organic pathology.",
            ["Pain 9/10", "No history on file — uncertain if recurrent pattern"],
            ["Migraine (primary headache)", "Secondary headache", "Intracranial hypertension"],
            urgency="moderate", escalation=False,
        ),
        "operational_decision": _make_op("ADMITTED_GEN", "ED_OBS", 5, "NORMAL", 0.62,
                                          summary="ED Observation for IV antiemetics and analgesia. CT head ordered."),
    },

    # ── Acuity 4 / Low-Moderate ──────────────────────────────────────────

    {
        "patient_id": "10002930",  # MIMIC — 58M Hypoglycemia+DM
        "age": 58, "sex": "M",
        "chief_complaint": "Hypoglycaemia, confusion, blood glucose 42 mg/dL on arrival",
        "vitals": {"hr": 89, "rr": 16, "spo2": 100, "sbp": 106, "dbp": 61, "temp": 97.3, "pain": 0},
        "acuity": 2, "expected_los_min": 60,
        "metadata": {"cardiac_hint": False, "has_history": True,
                     "history_text": "Prior history: diabetes mellitus, 4 prior ED visits for hypoglycaemia",
                     "previous_ed_visits": 4, "previous_hospital_admissions": 2, "diabetes_history": 1},
        "clinical_assessment": _make_assessment(
            "ED_OBS", 0.44, 0.06,
            "58M diabetic with 4 prior hypoglycaemic ED visits. Glucose 42 — moderate hypoglycaemia. Pattern of recurrence — diabetes management review recommended before discharge.",
            ["Recurrent hypoglycaemia (4th visit)", "Diabetes poorly controlled"],
            ["Hypoglycaemia", "Medication non-adherence", "Insulin overdose"],
            urgency="moderate", escalation=False, has_history=True,
        ),
        "operational_decision": _make_op("ED_OBS", "ED_OBS", 4, "NORMAL", 0.62,
                                          summary="ED Obs for glucose monitoring after D50W. Diabetes nurse educator to review."),
    },

    {
        "patient_id": "PAT-LOW-01",
        "age": 28, "sex": "F",
        "chief_complaint": "Right ankle sprain after fall, swelling, unable to bear weight",
        "vitals": {"hr": 76, "rr": 15, "spo2": 99, "sbp": 118, "dbp": 74, "temp": 36.8, "pain": 6},
        "acuity": 4, "expected_los_min": 40,
        "metadata": {"cardiac_hint": False, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "ED_OBS", 0.28, 0.02,
            "28F walk-in with ankle sprain. Vitals normal. Ottawa ankle rules apply — X-ray indicated. Low admission risk, no history on file.",
            ["Unable to bear weight — Ottawa positive"],
            ["Ankle fracture", "Lateral ligament sprain", "Peroneal tendon injury"],
            urgency="low", escalation=False,
        ),
        "operational_decision": _make_op("ED_OBS", "ED_OBS", 4, "NORMAL", 0.62,
                                          summary="ED observation for X-ray. Expected discharge with boot and outpatient physio."),
    },

    {
        "patient_id": "PAT-LOW-02",
        "age": 22, "sex": "M",
        "chief_complaint": "Forearm laceration from kitchen accident, bleeding controlled",
        "vitals": {"hr": 74, "rr": 15, "spo2": 100, "sbp": 120, "dbp": 76, "temp": 36.7, "pain": 4},
        "acuity": 4, "expected_los_min": 30,
        "metadata": {"cardiac_hint": False, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "ED_OBS", 0.15, 0.01,
            "22M with minor forearm laceration. Bleeding controlled. Vitals normal. Requires suturing only. Safe for rapid discharge after wound closure.",
            [],
            ["Forearm laceration", "Superficial wound"],
            urgency="low", escalation=False,
        ),
        "operational_decision": _make_op("ED_OBS", "ED_OBS", 4, "NORMAL", 0.62,
                                          summary="ED treatment for wound closure. Discharge after sutures."),
    },

    {
        "patient_id": "10015860",  # MIMIC — 55F Foot infection
        "age": 55, "sex": "F",
        "chief_complaint": "Right foot diabetic ulcer infection, swelling, fever 38.3°C",
        "vitals": {"hr": 108, "rr": 18, "spo2": 99, "sbp": 107, "dbp": 73, "temp": 100.0, "pain": 7},
        "acuity": 3, "expected_los_min": 75,
        "metadata": {"cardiac_hint": False, "has_history": True,
                     "history_text": "Prior history: diabetes mellitus, 4 prior ED visits including prior foot infection",
                     "previous_ed_visits": 4, "previous_hospital_admissions": 2, "diabetes_history": 1},
        "clinical_assessment": _make_assessment(
            "ADMITTED_GEN", 0.72, 0.13,
            "55F diabetic with recurrent foot infections — 15 prior documents retrieved. HR 108, Temp 100°F, signs of cellulitis. Sepsis risk given DM. IV antibiotics indicated.",
            ["HR 108 tachycardia", "Temp 100°F with infection", "Recurrent diabetic foot infection (pattern)"],
            ["Diabetic foot infection / cellulitis", "Sepsis", "Osteomyelitis"],
            urgency="high", escalation=False, has_history=True,
        ),
        "operational_decision": _make_op("ADMITTED_GEN", "ADMITTED_GEN", 5, "NORMAL", 0.62,
                                          summary="General ward. IV antibiotics, podiatry and endocrine consult."),
    },

    # ── Walk-in / new patients (no history) ─────────────────────────────

    {
        "patient_id": "PAT-NEW-01",
        "age": 19, "sex": "M",
        "chief_complaint": "First-ever seizure, post-ictal, witnessed tonic-clonic episode",
        "vitals": {"hr": 102, "rr": 20, "spo2": 97, "sbp": 134, "dbp": 82, "temp": 37.4, "pain": 2},
        "acuity": 2, "expected_los_min": 90,
        "metadata": {"cardiac_hint": False, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "ADMITTED_GEN", 0.77, 0.22,
            "19M first seizure. No history on file — unknown epilepsy vs provoked seizure. Post-ictal, HR 102. Neurology consult and MRI required. Evidence based on similar cases only — strength low.",
            ["First-ever seizure — full workup required", "Unknown underlying cause", "Post-ictal confusional state"],
            ["First unprovoked seizure", "Provoked seizure (metabolic)", "Intracranial mass"],
            urgency="high", escalation=False,
        ),
        "operational_decision": _make_op("ADMITTED_GEN", "ADMITTED_GEN", 5, "NORMAL", 0.62,
                                          summary="General ward. Neuro consult, MRI brain, metabolic panel."),
    },

    {
        "patient_id": "PAT-NEW-02",
        "age": 73, "sex": "F",
        "chief_complaint": "Fall at home, right hip pain, unable to stand, found on floor by family",
        "vitals": {"hr": 88, "rr": 18, "spo2": 97, "sbp": 138, "dbp": 84, "temp": 37.0, "pain": 8},
        "acuity": 2, "expected_los_min": 120,
        "metadata": {"cardiac_hint": False, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "ADMITTED_GEN", 0.82, 0.21,
            "73F found on floor — hip fracture highly likely. No prior history on file. Pain 8/10. Age 73 with unknown comorbidities — conservative approach warranted. Orthopaedic consult urgent.",
            ["Pain 8/10", "Unable to weight-bear", "Age 73 — fracture risk high", "No history — uncertainty elevated"],
            ["Femoral neck fracture", "Intertrochanteric fracture", "Pubic rami fracture"],
            urgency="high", escalation=False,
        ),
        "operational_decision": _make_op("ADMITTED_GEN", "ADMITTED_GEN", 5, "NORMAL", 0.62,
                                          summary="General ward. Ortho consult, X-ray pelvis/hip, pain management."),
    },

    {
        "patient_id": "PAT-NEW-03",
        "age": 42, "sex": "M",
        "chief_complaint": "Acute low back pain after lifting, unable to walk straight",
        "vitals": {"hr": 78, "rr": 16, "spo2": 99, "sbp": 128, "dbp": 82, "temp": 37.0, "pain": 7},
        "acuity": 4, "expected_los_min": 45,
        "metadata": {"cardiac_hint": False, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "ED_OBS", 0.22, 0.02,
            "42M acute musculoskeletal back pain after lifting. No red flags (neurological symptoms, cauda equina signs absent). No history on file. Analgesia and physio expected.",
            [],
            ["Acute lumbar muscle strain", "Lumbar disc prolapse", "Facet joint injury"],
            urgency="low", escalation=False,
        ),
        "operational_decision": _make_op("ED_OBS", "ED_OBS", 4, "NORMAL", 0.62,
                                          summary="ED observation. NSAIDs, muscle relaxants, physio review."),
    },

    {
        "patient_id": "PAT-NEW-04",
        "age": 61, "sex": "F",
        "chief_complaint": "New onset atrial fibrillation, palpitations, mildly symptomatic",
        "vitals": {"hr": 138, "rr": 18, "spo2": 98, "sbp": 142, "dbp": 88, "temp": 37.0, "pain": 2},
        "acuity": 2, "expected_los_min": 100,
        "metadata": {"cardiac_hint": True, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "CICU", 0.80, 0.38,
            "61F with new-onset AF, HR 138, BP 142/88. Haemodynamically stable but symptomatic. No prior history — cannot assess whether paroxysmal vs persistent. Cardiology review required.",
            ["HR 138 — rapid ventricular response", "New AF — aetiology unknown", "No prior cardiac history on file"],
            ["New-onset atrial fibrillation", "Thyrotoxicosis-induced AF", "Sepsis-induced AF"],
            urgency="high", escalation=False,
        ),
        "operational_decision": _make_op("CICU", "CICU", 1, "NORMAL", 0.62, confirm=True,
                                          summary="CICU has 1 bed remaining. Rate control and anticoagulation. Confirm CICU reservation."),
    },

    {
        "patient_id": "PAT-NEW-05",
        "age": 88, "sex": "M",
        "chief_complaint": "Acute confusion, new onset agitation, fever 38.5°C",
        "vitals": {"hr": 104, "rr": 22, "spo2": 96, "sbp": 116, "dbp": 68, "temp": 38.5, "pain": 0},
        "acuity": 2, "expected_los_min": 110,
        "metadata": {"cardiac_hint": False, "has_history": False, "history_text": ""},
        "clinical_assessment": _make_assessment(
            "ICU", 0.89, 0.52,
            "88M with acute delirium and fever. No history available — high uncertainty. Age 88 has severely reduced physiological reserve. Sepsis workup mandatory. High escalation risk if source not identified early.",
            ["Age 88 — minimal physiological reserve", "Acute delirium + fever — sepsis pattern", "No history — workup guided by vitals only"],
            ["Sepsis / UTI", "Aspiration pneumonia", "Urinary retention with sepsis"],
            urgency="critical", escalation=True,
        ),
        "operational_decision": _make_op("ICU", "ICU", 3, "NORMAL", 0.62,
                                          summary="ICU allocated for close monitoring. Sepsis bundle initiated."),
    },
]


# ---------------------------------------------------------------------------
# Scenario-based selection helpers
# ---------------------------------------------------------------------------

# Patients that should always be in the "WAITING for triage" queue
# (i.e., nurse needs to assess them)
WAITING_IDS_BY_SCENARIO = {
    "NORMAL_DAY":         ["PAT-MOD-03", "PAT-LOW-01", "PAT-LOW-02", "PAT-NEW-03"],
    "BUSY_DAY":           ["PAT-CRIT-01", "PAT-HI-02", "PAT-MOD-03", "PAT-MOD-04",
                           "PAT-LOW-01", "PAT-LOW-02", "PAT-NEW-01", "PAT-NEW-02"],
    "SURGE_MASS_CASUALTY":["PAT-CRIT-01", "PAT-HI-02", "PAT-MOD-03", "PAT-MOD-04",
                           "PAT-LOW-01", "PAT-LOW-02", "PAT-NEW-01", "PAT-NEW-02",
                           "PAT-NEW-03", "PAT-NEW-04", "PAT-NEW-05"],
    "NIGHT_SHIFT":        ["PAT-NEW-03", "PAT-LOW-02"],
    "RESOURCE_CONSTRAINED":["PAT-CRIT-01", "PAT-HI-02", "PAT-MOD-03", "PAT-MOD-04",
                            "PAT-LOW-01", "PAT-NEW-01", "PAT-NEW-04", "PAT-NEW-05"],
}

# Patients that should already be TRIAGED (assessment done, awaiting admission confirm)
TRIAGED_IDS_BY_SCENARIO = {
    "NORMAL_DAY":         ["10016742", "10002930"],
    "BUSY_DAY":           ["10037928", "10039708", "10016742", "10012853", "10002930"],
    "SURGE_MASS_CASUALTY":["10037928", "10039708", "10016742", "10005866", "10012853",
                           "10014354", "10002930"],
    "NIGHT_SHIFT":        ["10016742", "10012853"],
    "RESOURCE_CONSTRAINED":["10037928", "10039708", "10016742", "10012853", "10015860"],
}

# Patients that should already be ADMITTED (in beds, counting toward occupancy)
ADMITTED_IDS_BY_SCENARIO = {
    "NORMAL_DAY":         ["10014354", "PAT-NEW-02"],
    "BUSY_DAY":           ["10014354", "10015860", "PAT-NEW-02", "PAT-NEW-05"],
    "SURGE_MASS_CASUALTY":["PAT-NEW-02", "PAT-NEW-05", "PAT-NEW-04"],
    "NIGHT_SHIFT":        ["10039708", "PAT-NEW-02"],
    "RESOURCE_CONSTRAINED":["10014354", "10015860", "PAT-NEW-02", "PAT-NEW-04", "PAT-NEW-05"],
}


def get_patient_by_id(patient_id: str) -> Dict[str, Any]:
    """Return the pool entry for a given patient_id, or None."""
    for p in PATIENT_POOL:
        if p["patient_id"] == patient_id:
            return p
    return None


def build_simulated_patient(pool_entry: Dict[str, Any], sim_time_min: int) -> SimulatedPatient:
    """Construct a SimulatedPatient from a pool entry."""
    p = SimulatedPatient(
        patient_id=pool_entry["patient_id"],
        age=pool_entry["age"],
        sex=pool_entry["sex"],
        chief_complaint=pool_entry["chief_complaint"],
        vitals=dict(pool_entry["vitals"]),
        acuity=pool_entry["acuity"],
        arrival_time_min=sim_time_min,
        expected_los_min=pool_entry["expected_los_min"],
        elapsed_los_min=0,
        status=PatientStatus.ARRIVED,
        metadata=dict(pool_entry.get("metadata", {})),
    )
    # Pre-bake the assessment — this is the key part, no ML call needed
    if "clinical_assessment" in pool_entry:
        p.clinical_assessment = pool_entry["clinical_assessment"]
    if "operational_decision" in pool_entry:
        p.operational_decision = pool_entry["operational_decision"]
    return p
