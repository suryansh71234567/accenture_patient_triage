"""
batch_evaluate_50_patients.py
------------------------------
Generates 50 clinically realistic synthetic patients, runs ALL of them
through the complete TriageGuard pipeline:

    XGBoost inference
        +
    RAG retrieval + LLM reasoning
        ↓
    Reconciler (trust / lambda)
        ↓
    Router (clinical preference)
        ↓
    Live hospital routing policy (operational allocation)

Then evaluates each result for demo quality using clinical common sense
and prints a ranked shortlist of the best 15-20 to use for submission demo.

Run:
    .venv\\Scripts\\python.exe scripts\\batch_evaluate_50_patients.py
"""

import sys, os, json, time
from pathlib import Path

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "triageguard_xgb"))

from triageguard_router.combined_pipeline import TriageGuardPipeline

# ── 50 diverse patient definitions ─────────────────────────────────────────
# Each covers a different acuity tier, age group, sex, and clinical story.
# Mix of: with history flags (tests XGB feature importance) and clean walks-in
# (tests RAG retrieval + LLM evidence weighting).

PATIENTS = [
    # ── ACUITY 1 — Life threatening ────────────────────────────────────────
    {
        "patient_id": "EVAL-001", "age": 62, "sex": "M",
        "chiefcomplaint": "Sudden crushing chest pain, radiation to left arm, diaphoresis",
        "acuity": 1,
        "heartrate": 118, "resprate": 28, "o2sat": 88, "sbp": 78, "dbp": 50,
        "temperature": 36.9, "pain": 10,
        "previous_ed_visits": 2, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Known CAD and DM. Prior NSTEMI 3 years ago.",
    },
    {
        "patient_id": "EVAL-002", "age": 74, "sex": "F",
        "chiefcomplaint": "Unresponsive, found at home, GCS 5",
        "acuity": 1,
        "heartrate": 42, "resprate": 6, "o2sat": 82, "sbp": 72, "dbp": 44,
        "temperature": 35.2, "pain": 0,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 1, "malignancy_history": 0,
        "history_text": "History of prior stroke. Found unresponsive by family.",
    },
    {
        "patient_id": "EVAL-003", "age": 45, "sex": "M",
        "chiefcomplaint": "Massive haematemesis, BP falling, pallor and diaphoresis",
        "acuity": 1,
        "heartrate": 132, "resprate": 22, "o2sat": 94, "sbp": 82, "dbp": 52,
        "temperature": 36.4, "pain": 7,
        "previous_ed_visits": 3, "previous_hospital_admissions": 2, "previous_icu_admissions": 1,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Heavy alcohol use. Prior GI bleed with ICU admission.",
    },
    {
        "patient_id": "EVAL-004", "age": 29, "sex": "F",
        "chiefcomplaint": "Anaphylaxis after insect sting, throat swelling, stridor",
        "acuity": 1,
        "heartrate": 140, "resprate": 30, "o2sat": 90, "sbp": 84, "dbp": 50,
        "temperature": 37.1, "pain": 5,
        "previous_ed_visits": 1, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Known bee allergy. Previous mild reaction.",
    },
    {
        "patient_id": "EVAL-005", "age": 58, "sex": "M",
        "chiefcomplaint": "Acute aortic dissection — tearing back pain radiating to abdomen",
        "acuity": 1,
        "heartrate": 108, "resprate": 22, "o2sat": 96, "sbp": 190, "dbp": 110,
        "temperature": 37.0, "pain": 10,
        "previous_ed_visits": 0, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Hypertension poorly controlled. Marfan syndrome suspected.",
    },
    # ── ACUITY 2 — Emergent ────────────────────────────────────────────────
    {
        "patient_id": "EVAL-006", "age": 81, "sex": "F",
        "chiefcomplaint": "Acute respiratory failure, RR 32, SpO2 88%, COPD history",
        "acuity": 2,
        "heartrate": 104, "resprate": 32, "o2sat": 88, "sbp": 162, "dbp": 94,
        "temperature": 38.2, "pain": 2,
        "previous_ed_visits": 5, "previous_hospital_admissions": 4, "previous_icu_admissions": 2,
        "cardiovascular_history": 1, "respiratory_history": 1, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "COPD Gold IV and CHF. 2 prior ICU admissions for respiratory failure.",
    },
    {
        "patient_id": "EVAL-007", "age": 67, "sex": "M",
        "chiefcomplaint": "New onset AF with rapid ventricular response HR 148, palpitations",
        "acuity": 2,
        "heartrate": 148, "resprate": 18, "o2sat": 97, "sbp": 138, "dbp": 86,
        "temperature": 37.0, "pain": 3,
        "previous_ed_visits": 2, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Known paroxysmal AF. On anticoagulation. DM and HTN.",
    },
    {
        "patient_id": "EVAL-008", "age": 55, "sex": "F",
        "chiefcomplaint": "Stroke — right-sided facial droop, arm weakness, slurred speech, onset 1h ago",
        "acuity": 2,
        "heartrate": 88, "resprate": 16, "o2sat": 98, "sbp": 168, "dbp": 98,
        "temperature": 37.2, "pain": 0,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Hypertension, hypercholesterolaemia. No prior neurological events.",
    },
    {
        "patient_id": "EVAL-009", "age": 72, "sex": "M",
        "chiefcomplaint": "Hypotension and altered consciousness — BP 90/60, sepsis query",
        "acuity": 2,
        "heartrate": 112, "resprate": 22, "o2sat": 96, "sbp": 90, "dbp": 60,
        "temperature": 38.9, "pain": 2,
        "previous_ed_visits": 3, "previous_hospital_admissions": 3, "previous_icu_admissions": 1,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 1,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "CKD stage 3, DM. Recurrent UTIs. Prior sepsis ICU admission.",
    },
    {
        "patient_id": "EVAL-010", "age": 34, "sex": "F",
        "chiefcomplaint": "Status epilepticus — ongoing generalised tonic-clonic seizure 8 min",
        "acuity": 2,
        "heartrate": 124, "resprate": 20, "o2sat": 94, "sbp": 142, "dbp": 86,
        "temperature": 37.8, "pain": 0,
        "previous_ed_visits": 2, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 1, "malignancy_history": 0,
        "history_text": "Known epilepsy. Took last dose of levetiracetam 3 days ago.",
    },
    {
        "patient_id": "EVAL-011", "age": 48, "sex": "M",
        "chiefcomplaint": "Inferior STEMI — chest tightening, nausea, ST elevation V2-V6",
        "acuity": 1,
        "heartrate": 96, "resprate": 20, "o2sat": 96, "sbp": 118, "dbp": 72,
        "temperature": 37.0, "pain": 9,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Heavy smoker. No prior cardiac history. Father had MI at 50.",
    },
    {
        "patient_id": "EVAL-012", "age": 88, "sex": "M",
        "chiefcomplaint": "Acute delirium, fever 38.6°C, new urinary incontinence",
        "acuity": 2,
        "heartrate": 102, "resprate": 22, "o2sat": 95, "sbp": 118, "dbp": 68,
        "temperature": 38.6, "pain": 0,
        "previous_ed_visits": 4, "previous_hospital_admissions": 3, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 1,
        "diabetes_history": 1, "neurological_history": 1, "malignancy_history": 0,
        "history_text": "Dementia, CKD, DM. Recurrent UTI-related delirium. Very frail.",
    },
    # ── ACUITY 3 — Urgent ──────────────────────────────────────────────────
    {
        "patient_id": "EVAL-013", "age": 53, "sex": "F",
        "chiefcomplaint": "Right lower quadrant pain 8/10 with rebound, fever 38.4°C",
        "acuity": 3,
        "heartrate": 94, "resprate": 18, "o2sat": 98, "sbp": 122, "dbp": 76,
        "temperature": 38.4, "pain": 8,
        "previous_ed_visits": 1, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "No prior abdominal history.",
    },
    {
        "patient_id": "EVAL-014", "age": 64, "sex": "M",
        "chiefcomplaint": "Hypertensive urgency BP 228/118, headache, blurred vision",
        "acuity": 3,
        "heartrate": 72, "resprate": 16, "o2sat": 97, "sbp": 228, "dbp": 118,
        "temperature": 37.0, "pain": 4,
        "previous_ed_visits": 3, "previous_hospital_admissions": 2, "previous_icu_admissions": 0,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 1,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Poorly controlled HTN, CKD, DM. Prior hypertensive emergency.",
    },
    {
        "patient_id": "EVAL-015", "age": 39, "sex": "M",
        "chiefcomplaint": "Severe community-acquired pneumonia, SpO2 92%, productive cough 5 days",
        "acuity": 3,
        "heartrate": 102, "resprate": 24, "o2sat": 92, "sbp": 116, "dbp": 72,
        "temperature": 39.1, "pain": 4,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Smoker, no prior hospital history.",
    },
    {
        "patient_id": "EVAL-016", "age": 77, "sex": "F",
        "chiefcomplaint": "Fall at home, right hip pain, unable to weight-bear, pain 9/10",
        "acuity": 3,
        "heartrate": 84, "resprate": 18, "o2sat": 96, "sbp": 148, "dbp": 82,
        "temperature": 37.0, "pain": 9,
        "previous_ed_visits": 2, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Osteoporosis, hypertension. Prior wrist fracture 2 years ago.",
    },
    {
        "patient_id": "EVAL-017", "age": 60, "sex": "M",
        "chiefcomplaint": "DVT left leg confirmed on ultrasound — calf swelling 3 days",
        "acuity": 3,
        "heartrate": 82, "resprate": 16, "o2sat": 97, "sbp": 132, "dbp": 80,
        "temperature": 37.2, "pain": 5,
        "previous_ed_visits": 0, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 1,
        "history_text": "Colorectal cancer on chemotherapy. Prior DVT 1 year ago.",
    },
    {
        "patient_id": "EVAL-018", "age": 44, "sex": "F",
        "chiefcomplaint": "Diabetic ketoacidosis — blood glucose 520, vomiting, Kussmaul breathing",
        "acuity": 2,
        "heartrate": 118, "resprate": 26, "o2sat": 98, "sbp": 108, "dbp": 66,
        "temperature": 37.6, "pain": 3,
        "previous_ed_visits": 4, "previous_hospital_admissions": 3, "previous_icu_admissions": 1,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "T1DM poorly compliant. 3 prior DKA admissions, one ICU.",
    },
    {
        "patient_id": "EVAL-019", "age": 31, "sex": "M",
        "chiefcomplaint": "Acute pancreatitis — severe epigastric pain 9/10, vomiting",
        "acuity": 3,
        "heartrate": 106, "resprate": 20, "o2sat": 97, "sbp": 126, "dbp": 78,
        "temperature": 38.0, "pain": 9,
        "previous_ed_visits": 2, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Alcohol-related pancreatitis. Prior admission 1 year ago.",
    },
    {
        "patient_id": "EVAL-020", "age": 69, "sex": "M",
        "chiefcomplaint": "NSTEMI — troponin elevated, chest discomfort, diaphoresis",
        "acuity": 2,
        "heartrate": 94, "resprate": 18, "o2sat": 95, "sbp": 144, "dbp": 88,
        "temperature": 37.1, "pain": 7,
        "previous_ed_visits": 2, "previous_hospital_admissions": 2, "previous_icu_admissions": 1,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "CAD, prior CABG, DM, HTN. Prior NSTEMI with CICU stay.",
    },
    # ── ACUITY 3 continued ─────────────────────────────────────────────────
    {
        "patient_id": "EVAL-021", "age": 52, "sex": "F",
        "chiefcomplaint": "Pulmonary embolism — acute dyspnoea, pleuritic chest pain, D-dimer elevated",
        "acuity": 2,
        "heartrate": 114, "resprate": 26, "o2sat": 93, "sbp": 122, "dbp": 74,
        "temperature": 37.3, "pain": 6,
        "previous_ed_visits": 0, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 1,
        "history_text": "Breast cancer post-mastectomy on tamoxifen. Prior PE 3 years ago.",
    },
    {
        "patient_id": "EVAL-022", "age": 27, "sex": "M",
        "chiefcomplaint": "Acute asthma exacerbation — wheeze, accessory muscle use, PEFR 35%",
        "acuity": 2,
        "heartrate": 108, "resprate": 28, "o2sat": 92, "sbp": 128, "dbp": 76,
        "temperature": 37.1, "pain": 2,
        "previous_ed_visits": 4, "previous_hospital_admissions": 2, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 1, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Severe asthma. 2 prior hospital admissions. On step 4 therapy.",
    },
    {
        "patient_id": "EVAL-023", "age": 66, "sex": "F",
        "chiefcomplaint": "Cellulitis right leg — redness spreading above knee, fever 38.7°C",
        "acuity": 3,
        "heartrate": 98, "resprate": 18, "o2sat": 97, "sbp": 136, "dbp": 80,
        "temperature": 38.7, "pain": 6,
        "previous_ed_visits": 2, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "DM with peripheral vascular disease. Prior cellulitis admission.",
    },
    {
        "patient_id": "EVAL-024", "age": 41, "sex": "M",
        "chiefcomplaint": "Renal colic — sudden severe left flank pain 10/10, haematuria",
        "acuity": 3,
        "heartrate": 96, "resprate": 18, "o2sat": 99, "sbp": 142, "dbp": 88,
        "temperature": 37.2, "pain": 10,
        "previous_ed_visits": 3, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 1,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Recurrent nephrolithiasis. 3 prior ED visits for kidney stones.",
    },
    {
        "patient_id": "EVAL-025", "age": 78, "sex": "M",
        "chiefcomplaint": "Decompensated heart failure — bilateral leg oedema, orthopnoea, pink frothy sputum",
        "acuity": 2,
        "heartrate": 106, "resprate": 28, "o2sat": 90, "sbp": 168, "dbp": 96,
        "temperature": 37.0, "pain": 3,
        "previous_ed_visits": 6, "previous_hospital_admissions": 5, "previous_icu_admissions": 2,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 1,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Severe systolic HF EF 25%, CKD, prior 2 ICU admissions.",
    },
    # ── ACUITY 4 — Less urgent ─────────────────────────────────────────────
    {
        "patient_id": "EVAL-026", "age": 35, "sex": "F",
        "chiefcomplaint": "Migraine with aura — severe unilateral headache, photophobia, vomiting",
        "acuity": 4,
        "heartrate": 78, "resprate": 16, "o2sat": 99, "sbp": 118, "dbp": 74,
        "temperature": 37.0, "pain": 8,
        "previous_ed_visits": 3, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Known migraine sufferer. No organic pathology found on MRI.",
    },
    {
        "patient_id": "EVAL-027", "age": 58, "sex": "M",
        "chiefcomplaint": "Hypoglycaemia — blood glucose 38, confused, known diabetic",
        "acuity": 3,
        "heartrate": 90, "resprate": 16, "o2sat": 99, "sbp": 108, "dbp": 62,
        "temperature": 97.4, "pain": 0,
        "previous_ed_visits": 5, "previous_hospital_admissions": 2, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "T2DM on insulin. 5 prior ED hypoglycaemia visits. Non-compliant.",
    },
    {
        "patient_id": "EVAL-028", "age": 22, "sex": "F",
        "chiefcomplaint": "Acute appendicitis — McBurney point tenderness, Rovsing sign positive",
        "acuity": 3,
        "heartrate": 96, "resprate": 18, "o2sat": 98, "sbp": 122, "dbp": 76,
        "temperature": 38.3, "pain": 8,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "No prior history.",
    },
    {
        "patient_id": "EVAL-029", "age": 50, "sex": "M",
        "chiefcomplaint": "Alcohol intoxication — unsteady gait, slurred speech, GCS 13",
        "acuity": 4,
        "heartrate": 88, "resprate": 16, "o2sat": 98, "sbp": 126, "dbp": 78,
        "temperature": 36.8, "pain": 0,
        "previous_ed_visits": 8, "previous_hospital_admissions": 2, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Chronic alcohol use disorder. Multiple ED attendances.",
    },
    {
        "patient_id": "EVAL-030", "age": 19, "sex": "M",
        "chiefcomplaint": "First-ever seizure — post-ictal, witnessed tonic-clonic 3 min",
        "acuity": 2,
        "heartrate": 98, "resprate": 18, "o2sat": 98, "sbp": 128, "dbp": 80,
        "temperature": 37.4, "pain": 1,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "No prior history. University student.",
    },
    {
        "patient_id": "EVAL-031", "age": 46, "sex": "F",
        "chiefcomplaint": "Ectopic pregnancy — LIF pain, vaginal bleeding, positive pregnancy test",
        "acuity": 2,
        "heartrate": 118, "resprate": 20, "o2sat": 98, "sbp": 102, "dbp": 66,
        "temperature": 37.1, "pain": 9,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Known risk factor for ectopic — prior PID.",
    },
    {
        "patient_id": "EVAL-032", "age": 73, "sex": "F",
        "chiefcomplaint": "Lower GI bleed — bright red rectal bleeding, 3 episodes today",
        "acuity": 3,
        "heartrate": 98, "resprate": 18, "o2sat": 97, "sbp": 114, "dbp": 68,
        "temperature": 37.2, "pain": 4,
        "previous_ed_visits": 2, "previous_hospital_admissions": 2, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 1,
        "history_text": "Colorectal carcinoma. On anticoagulation for AF.",
    },
    {
        "patient_id": "EVAL-033", "age": 83, "sex": "M",
        "chiefcomplaint": "Acute urinary retention — suprapubic pain 8/10, unable to void 12h",
        "acuity": 3,
        "heartrate": 88, "resprate": 16, "o2sat": 97, "sbp": 148, "dbp": 84,
        "temperature": 37.4, "pain": 8,
        "previous_ed_visits": 3, "previous_hospital_admissions": 2, "previous_icu_admissions": 0,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 1,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 1,
        "history_text": "BPH, prostate cancer, CKD, DM. 2 prior catheterisations.",
    },
    {
        "patient_id": "EVAL-034", "age": 37, "sex": "F",
        "chiefcomplaint": "Pregnancy-induced hypertension — BP 158/104, proteinuria, headache",
        "acuity": 2,
        "heartrate": 92, "resprate": 18, "o2sat": 99, "sbp": 158, "dbp": 104,
        "temperature": 37.0, "pain": 5,
        "previous_ed_visits": 1, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "32 weeks pregnant. Prior pre-eclampsia in first pregnancy.",
    },
    {
        "patient_id": "EVAL-035", "age": 25, "sex": "M",
        "chiefcomplaint": "Traumatic head injury — MVC, GCS 13, laceration, confusion",
        "acuity": 2,
        "heartrate": 92, "resprate": 18, "o2sat": 98, "sbp": 138, "dbp": 82,
        "temperature": 37.1, "pain": 6,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "No prior history. Not wearing helmet.",
    },
    # ── ACUITY 4-5 — Minor ─────────────────────────────────────────────────
    {
        "patient_id": "EVAL-036", "age": 28, "sex": "F",
        "chiefcomplaint": "Right ankle sprain after fall — unable to weight-bear, swollen",
        "acuity": 4,
        "heartrate": 76, "resprate": 16, "o2sat": 99, "sbp": 118, "dbp": 74,
        "temperature": 36.8, "pain": 6,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "No prior history.",
    },
    {
        "patient_id": "EVAL-037", "age": 62, "sex": "M",
        "chiefcomplaint": "Chest tightness and exertional dyspnoea — new symptoms 2 weeks",
        "acuity": 3,
        "heartrate": 86, "resprate": 18, "o2sat": 96, "sbp": 148, "dbp": 90,
        "temperature": 37.0, "pain": 4,
        "previous_ed_visits": 1, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "HTN, DM, prior angina. Stopped medications 1 month ago.",
    },
    {
        "patient_id": "EVAL-038", "age": 45, "sex": "F",
        "chiefcomplaint": "Panic attack — palpitations, chest tightness, hyperventilation",
        "acuity": 4,
        "heartrate": 102, "resprate": 22, "o2sat": 99, "sbp": 124, "dbp": 78,
        "temperature": 37.0, "pain": 3,
        "previous_ed_visits": 4, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Anxiety disorder. 4 prior ED visits with same presentation.",
    },
    {
        "patient_id": "EVAL-039", "age": 71, "sex": "F",
        "chiefcomplaint": "Syncope — brief loss of consciousness, now alert, no trauma",
        "acuity": 3,
        "heartrate": 52, "resprate": 16, "o2sat": 98, "sbp": 108, "dbp": 64,
        "temperature": 36.9, "pain": 0,
        "previous_ed_visits": 1, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "HTN, complete heart block pacemaker. Prior vasovagal documented.",
    },
    {
        "patient_id": "EVAL-040", "age": 32, "sex": "M",
        "chiefcomplaint": "Minor forearm laceration — 5cm kitchen knife wound, bleeding controlled",
        "acuity": 4,
        "heartrate": 74, "resprate": 14, "o2sat": 99, "sbp": 122, "dbp": 76,
        "temperature": 36.7, "pain": 4,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "No prior history.",
    },
    {
        "patient_id": "EVAL-041", "age": 54, "sex": "M",
        "chiefcomplaint": "Acute liver failure — jaundice, encephalopathy, INR 4.2",
        "acuity": 1,
        "heartrate": 108, "resprate": 22, "o2sat": 96, "sbp": 96, "dbp": 62,
        "temperature": 37.8, "pain": 3,
        "previous_ed_visits": 2, "previous_hospital_admissions": 2, "previous_icu_admissions": 1,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 1,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Alcoholic liver disease, CKD. Prior hepatic decompensation with ICU.",
    },
    {
        "patient_id": "EVAL-042", "age": 65, "sex": "F",
        "chiefcomplaint": "Hypercalcaemia crisis — confusion, vomiting, polyuria, Ca 14.8",
        "acuity": 2,
        "heartrate": 96, "resprate": 18, "o2sat": 97, "sbp": 112, "dbp": 68,
        "temperature": 37.2, "pain": 2,
        "previous_ed_visits": 1, "previous_hospital_admissions": 1, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 1,
        "history_text": "Metastatic lung cancer on checkpoint inhibitor therapy.",
    },
    {
        "patient_id": "EVAL-043", "age": 76, "sex": "M",
        "chiefcomplaint": "Ischaemic foot — cold, pulseless, mottled right foot, sudden onset",
        "acuity": 1,
        "heartrate": 88, "resprate": 18, "o2sat": 97, "sbp": 148, "dbp": 86,
        "temperature": 37.0, "pain": 9,
        "previous_ed_visits": 2, "previous_hospital_admissions": 3, "previous_icu_admissions": 1,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 1, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "PAD, AF, DM. Prior vascular surgery. On warfarin.",
    },
    {
        "patient_id": "EVAL-044", "age": 48, "sex": "F",
        "chiefcomplaint": "New rash — generalised urticaria and angioedema, no airway involvement",
        "acuity": 3,
        "heartrate": 88, "resprate": 16, "o2sat": 99, "sbp": 128, "dbp": 78,
        "temperature": 37.0, "pain": 2,
        "previous_ed_visits": 1, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "New ACE inhibitor started 5 days ago.",
    },
    {
        "patient_id": "EVAL-045", "age": 85, "sex": "F",
        "chiefcomplaint": "Aspiration pneumonia — gurgling breathing, fever, O2 drop after meal",
        "acuity": 2,
        "heartrate": 102, "resprate": 26, "o2sat": 91, "sbp": 108, "dbp": 64,
        "temperature": 38.8, "pain": 1,
        "previous_ed_visits": 3, "previous_hospital_admissions": 3, "previous_icu_admissions": 1,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 1, "malignancy_history": 0,
        "history_text": "Advanced dementia, prior stroke, recurrent aspirations. Very frail.",
    },
    {
        "patient_id": "EVAL-046", "age": 56, "sex": "M",
        "chiefcomplaint": "Chronic low back pain flare — unable to get out of bed today",
        "acuity": 5,
        "heartrate": 76, "resprate": 14, "o2sat": 99, "sbp": 128, "dbp": 80,
        "temperature": 36.8, "pain": 7,
        "previous_ed_visits": 6, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Chronic mechanical low back pain. Frequent ED attender.",
    },
    {
        "patient_id": "EVAL-047", "age": 70, "sex": "F",
        "chiefcomplaint": "Thyroid storm — HR 148, hyperthermia 40.2°C, agitation, AF",
        "acuity": 1,
        "heartrate": 148, "resprate": 24, "o2sat": 95, "sbp": 168, "dbp": 96,
        "temperature": 40.2, "pain": 4,
        "previous_ed_visits": 1, "previous_hospital_admissions": 2, "previous_icu_admissions": 1,
        "cardiovascular_history": 1, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "Graves disease, AF. Stopped antithyroid medication abruptly.",
    },
    {
        "patient_id": "EVAL-048", "age": 42, "sex": "M",
        "chiefcomplaint": "Opiate overdose — pinpoint pupils, RR 6, GCS 8, naloxone given",
        "acuity": 1,
        "heartrate": 58, "resprate": 6, "o2sat": 86, "sbp": 88, "dbp": 54,
        "temperature": 35.8, "pain": 0,
        "previous_ed_visits": 5, "previous_hospital_admissions": 3, "previous_icu_admissions": 2,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "IV heroin use disorder. Two prior overdose ICU admissions.",
    },
    {
        "patient_id": "EVAL-049", "age": 61, "sex": "F",
        "chiefcomplaint": "Breast cancer — fever 38.8°C, neutropenia, recent chemotherapy",
        "acuity": 2,
        "heartrate": 108, "resprate": 20, "o2sat": 97, "sbp": 118, "dbp": 72,
        "temperature": 38.8, "pain": 2,
        "previous_ed_visits": 2, "previous_hospital_admissions": 3, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 1,
        "history_text": "Breast cancer cycle 3 docetaxel. ANC 0.2. Neutropenic fever protocol.",
    },
    {
        "patient_id": "EVAL-050", "age": 38, "sex": "M",
        "chiefcomplaint": "Testicular torsion — sudden severe scrotal pain, absent cremasteric reflex",
        "acuity": 2,
        "heartrate": 102, "resprate": 18, "o2sat": 99, "sbp": 132, "dbp": 82,
        "temperature": 37.2, "pain": 10,
        "previous_ed_visits": 0, "previous_hospital_admissions": 0, "previous_icu_admissions": 0,
        "cardiovascular_history": 0, "respiratory_history": 0, "renal_history": 0,
        "diabetes_history": 0, "neurological_history": 0, "malignancy_history": 0,
        "history_text": "No prior history. Onset 2 hours ago.",
    },
]

# ── Scoring logic ───────────────────────────────────────────────────────────

def clinical_ground_truth(p: dict) -> str:
    """Simple clinical rule — what should a good system decide?"""
    a = p["acuity"]
    hr = p["heartrate"]; spo2 = p["o2sat"]; sbp = p["sbp"]
    is_cardiac = p.get("cardiovascular_history", 0) and ("chest" in p["chiefcomplaint"].lower() or "cardiac" in p["chiefcomplaint"].lower() or "STEMI" in p["chiefcomplaint"] or "AF" in p["chiefcomplaint"] or "heart" in p["chiefcomplaint"].lower())
    if a == 1 or (spo2 < 90 and sbp < 90) or (hr > 130 and sbp < 90):
        return "ICU" if not is_cardiac else "CICU"
    if a == 2 or spo2 < 93 or sbp < 100:
        return "CICU" if is_cardiac else "ICU"
    if a == 3:
        return "ADMITTED_GEN"
    return "ED_OBS"


def score_result(p: dict, r: dict) -> dict:
    """
    Score a pipeline result for demo quality on 0-100 scale.
    Considers: clinical correctness, branch agreement, confidence, output richness.
    """
    score = 0
    reasons = []

    dept      = r.get("department", "")
    adm_risk  = r.get("reconciled_admission_risk", 0) or 0
    icu_risk  = r.get("reconciled_icu_risk", 0) or 0
    agree     = r.get("branches_agree", False)
    red_flags = r.get("red_flags", []) or []
    top_dx    = r.get("top_diagnoses", []) or []
    conf_note = r.get("confidence_note", "") or ""
    struct    = r.get("structured_output", {}) or {}
    urgency   = (struct.get("urgency") or "").lower()

    expected = clinical_ground_truth(p)

    # ── 1. Clinical correctness (40 pts) ───────────────────────────────────
    if dept == expected:
        score += 40
        reasons.append(f"✅ Correct dept ({dept})")
    elif (dept in ("ICU","CICU") and expected in ("ICU","CICU")):
        score += 30  # close enough — both critical
        reasons.append(f"≈ Near-correct ({dept} vs {expected})")
    elif dept == "ADMITTED_GEN" and expected == "ADMITTED_GEN":
        score += 40
        reasons.append(f"✅ Correct dept ({dept})")
    else:
        reasons.append(f"❌ Wrong dept ({dept} vs expected {expected})")

    # ── 2. Risk scores plausible given acuity (20 pts) ───────────────────
    acuity = p["acuity"]
    if acuity <= 2 and adm_risk >= 0.65 and icu_risk >= 0.35:
        score += 20; reasons.append("✅ Risk scores match high acuity")
    elif acuity == 3 and 0.45 <= adm_risk <= 0.85 and icu_risk < 0.4:
        score += 20; reasons.append("✅ Risk scores match moderate acuity")
    elif acuity >= 4 and adm_risk < 0.5:
        score += 20; reasons.append("✅ Risk scores match low acuity")
    elif acuity <= 2 and adm_risk >= 0.5:
        score += 10; reasons.append("⚠ Risk plausible but not confident")
    else:
        reasons.append("❌ Risk scores inconsistent with acuity")

    # ── 3. Branch agreement (15 pts) ─────────────────────────────────────
    if agree:
        score += 15; reasons.append("✅ XGB and RAG agree")
    else:
        score += 5; reasons.append("⚠ Branches disagree (interesting for demo)")

    # ── 4. RAG output richness (15 pts) ───────────────────────────────────
    if red_flags:
        score += 7; reasons.append(f"✅ {len(red_flags)} red flags identified")
    if top_dx and len(top_dx) >= 2:
        score += 8; reasons.append(f"✅ {len(top_dx)} differential diagnoses")

    # ── 5. Urgency label present and correct (10 pts) ─────────────────────
    critical_terms = {"critical", "immediate", "emergent"}
    high_terms     = {"high", "urgent"}
    if acuity <= 2 and any(t in urgency for t in critical_terms | high_terms):
        score += 10; reasons.append("✅ Urgency label correct for severity")
    elif acuity == 3 and any(t in urgency for t in high_terms | {"moderate"}):
        score += 10; reasons.append("✅ Urgency label correct for moderate")
    elif acuity >= 4 and any(t in urgency for t in {"low", "moderate"}):
        score += 10; reasons.append("✅ Urgency label correct for low acuity")
    else:
        reasons.append(f"⚠ Urgency '{urgency}' unclear")

    return {"score": score, "reasons": reasons, "expected_dept": expected}


# ── Main runner ─────────────────────────────────────────────────────────────

def main():
    # Force UTF-8 output on Windows to avoid cp1252 crash on arrows/checkmarks
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("  TriageGuard -- Batch Evaluation of 50 Synthetic Patients")
    print("=" * 70)
    print("Loading pipeline (XGBoost + RAG + LLM + Reconciler + Router)...")

    pipeline = TriageGuardPipeline()

    results = []
    failed  = []

    for i, p in enumerate(PATIENTS):
        pid = p["patient_id"]
        print(f"\n[{i+1:02d}/50] Running {pid} — {p['chiefcomplaint'][:55]}...")
        t0 = time.time()
        try:
            r = pipeline.run(p)
            elapsed = time.time() - t0
            scored = score_result(p, r)
            row = {
                "rank": None,
                "patient_id": pid,
                "age": p["age"],
                "sex": p["sex"],
                "acuity": p["acuity"],
                "chief_complaint": p["chiefcomplaint"],
                "has_history": bool(p.get("history_text")),
                # Pipeline outputs
                "department":            r.get("department", "—"),
                "expected_dept":         scored["expected_dept"],
                "admission_risk":        round(r.get("reconciled_admission_risk", 0) or 0, 3),
                "icu_risk":              round(r.get("reconciled_icu_risk", 0) or 0, 3),
                "branches_agree":        r.get("branches_agree", False),
                "confidence_note":       r.get("confidence_note", ""),
                "red_flags":             r.get("red_flags", []),
                "top_diagnoses":         r.get("top_diagnoses", []),
                "urgency":               (r.get("structured_output") or {}).get("urgency", ""),
                "rag_narrative":         (r.get("structured_output") or {}).get("narrative", ""),
                "xgb_admission_prob":    round(r.get("xgb", {}).get("admission_probability", 0) or 0, 3),
                "xgb_icu_2h_prob":       round(r.get("xgb", {}).get("icu_probability_2h", 0) or 0, 3),
                "hospital_routing_dept": (r.get("hospital_routing") or {}).get("allocated_department", ""),
                # Scoring
                "demo_score":   scored["score"],
                "score_reasons": scored["reasons"],
                "elapsed_s":    round(elapsed, 1),
            }
            results.append(row)
            print(f"    -> dept={row['department']} | adm={row['admission_risk']:.2f} | icu={row['icu_risk']:.2f} | score={row['demo_score']}/100 | {elapsed:.1f}s")
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append({"patient_id": pid, "error": str(e)})

    # ── Rank and print shortlist ────────────────────────────────────────────
    results.sort(key=lambda x: x["demo_score"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    out_path = ROOT / "scripts" / "eval_results_50.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "failed": failed}, f, indent=2)
    print(f"\n\nFull results saved → {out_path}")

    # ── Console report ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TOP 20 PATIENTS FOR DEMO (ranked by pipeline quality score)")
    print("=" * 70)
    for r in results[:20]:
        agree_str = "OK AGREE" if r["branches_agree"] else "! SPLIT"
        correct   = "OK" if r['department'] == r['expected_dept'] else (
                    "~" if r['department'] in ("ICU","CICU") and r['expected_dept'] in ("ICU","CICU") else "XX")
        print(
            f"\n#{r['rank']:2d}  {r['patient_id']}  Score={r['demo_score']}/100"
            f"  {correct} Dept={r['department']} (exp={r['expected_dept']})"
        )
        print(f"     {r['age']}y {r['sex']} | Acuity {r['acuity']} | {r['chief_complaint'][:60]}")
        print(f"     Adm={r['admission_risk']:.2f} ICU={r['icu_risk']:.2f} | {agree_str} | Urgency={r['urgency']}")
        if r["red_flags"]:
            print(f"     Red flags: {', '.join(str(f) for f in r['red_flags'][:3])}")
        if r["top_diagnoses"]:
            print(f"     Top Dx: {', '.join(str(d) for d in r['top_diagnoses'][:3])}")
        print(f"     Scoring: {' | '.join(r['score_reasons'][:4])}")

    print("\n" + "=" * 70)
    print(f"  SUMMARY: {len(results)}/{len(PATIENTS)} patients completed | {len(failed)} failed")
    if failed:
        print(f"  FAILED: {[f['patient_id'] for f in failed]}")
    print("=" * 70)

    # ── My clinical assessment of the system ──────────────────────────────
    correct = sum(1 for r in results if r["department"] == r["expected_dept"])
    near    = sum(1 for r in results if r["department"] in ("ICU","CICU") and r["expected_dept"] in ("ICU","CICU"))
    agree   = sum(1 for r in results if r["branches_agree"])
    avg_sc  = sum(r["demo_score"] for r in results) / len(results) if results else 0
    print(f"\n  MY CLINICAL ASSESSMENT OF THE SYSTEM:")
    print(f"  Exact dept match:   {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
    print(f"  Near match (ICU/CICU counted same): {correct+near}/{len(results)}")
    print(f"  Both branches agree: {agree}/{len(results)} ({agree/len(results)*100:.0f}%)")
    print(f"  Average demo score:  {avg_sc:.1f}/100")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
