"""
eval_audit.py — TriageGuard Evaluation Script
Runs XGBoost inference on 8 synthetic test cases and missing-vitals ablation.
This is a READ-ONLY audit script. It does not modify any model or artifact.
"""
import os, sys, json, math
import numpy as np
import pandas as pd

# ── project paths ──────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
XGB_DIR = os.path.join(REPO, "triageguard_xgb")
sys.path.insert(0, XGB_DIR)

from src.inference.predict import TriageGuardPredictor

predictor = TriageGuardPredictor(os.path.join(XGB_DIR, "models"))

# ── helper ─────────────────────────────────────────────────────────────────
def run(label, patient):
    out = predictor.predict(patient)
    missing = [v for v in ["hr","rr","spo2","sbp","dbp","temp"]
               if patient.get(f"{v}_current") is None]
    missing_str = ",".join(missing) if missing else "none"
    return {
        "label":          label,
        "icu_2h":         out["icu_risk_2h"],
        "icu_6h":         out["icu_risk_6h"],
        "icu_12h":        out["icu_risk_12h"],
        "admission":      out["admission_risk"],
        "conf_admission": out["admission_confidence"],
        "conf_icu_2h":    out["icu_risk_2h".replace("risk","confidence")],
        "missing":        missing_str,
    }

def fmt(x): return f"{x:.3f}"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — 8 SYNTHETIC TEST CASES
# ═══════════════════════════════════════════════════════════════════════════

BASE = dict(
    previous_ed_visits=0, previous_hospital_admissions=0, previous_icu_admissions=0,
    cardiovascular_history=0, respiratory_history=0, renal_history=0,
    diabetes_history=0, neurological_history=0, malignancy_history=0,
    time_elapsed_minutes=15,
)

cases = [
    # 1. Stable — low acuity, normal vitals
    dict(BASE, label="[SYNTH] 1-Stable", age=42, sex="F",
         hr_arrival=78, hr_current=76, rr_arrival=16, rr_current=16,
         spo2_arrival=99, spo2_current=99, sbp_arrival=118, sbp_current=120,
         dbp_arrival=72, dbp_current=74, temp_arrival=98.6, temp_current=98.5,
         triage_complaint="mild ankle sprain from sports"),

    # 2. Concerning — elevated HR, low SpO2, chest pain
    dict(BASE, label="[SYNTH] 2-Concerning", age=58, sex="M",
         cardiovascular_history=1, diabetes_history=1,
         hr_arrival=110, hr_current=118, rr_arrival=22, rr_current=24,
         spo2_arrival=95, spo2_current=92, sbp_arrival=145, sbp_current=130,
         dbp_arrival=88, dbp_current=80, temp_arrival=98.9, temp_current=99.1,
         triage_complaint="chest pain and shortness of breath",
         previous_ed_visits=2, previous_hospital_admissions=1),

    # 3. High-risk — near-critical, multi-system compromise
    dict(BASE, label="[SYNTH] 3-High-Risk", age=74, sex="M",
         cardiovascular_history=1, respiratory_history=1, renal_history=1,
         diabetes_history=1, previous_icu_admissions=1, previous_hospital_admissions=3,
         hr_arrival=128, hr_current=135, rr_arrival=28, rr_current=30,
         spo2_arrival=88, spo2_current=85, sbp_arrival=90, sbp_current=82,
         dbp_arrival=55, dbp_current=48, temp_arrival=101.2, temp_current=102.1,
         triage_complaint="acute respiratory failure, altered mental status",
         time_elapsed_minutes=5),

    # 4. Missing vitals — only basic demographics
    dict(BASE, label="[SYNTH] 4-Missing-Vitals", age=55, sex="F",
         hr_arrival=None, hr_current=None, rr_arrival=None, rr_current=None,
         spo2_arrival=None, spo2_current=None, sbp_arrival=None, sbp_current=None,
         dbp_arrival=None, dbp_current=None, temp_arrival=None, temp_current=None,
         triage_complaint="general malaise", time_elapsed_minutes=0),

    # 5. Low SpO2 / respiratory
    dict(BASE, label="[SYNTH] 5-Low-SpO2", age=66, sex="M",
         respiratory_history=1,
         hr_arrival=102, hr_current=108, rr_arrival=26, rr_current=28,
         spo2_arrival=90, spo2_current=87, sbp_arrival=135, sbp_current=128,
         dbp_arrival=82, dbp_current=78, temp_arrival=98.6, temp_current=99.8,
         triage_complaint="severe shortness of breath, wheezing"),

    # 6. Cardiovascular — hypertensive emergency
    dict(BASE, label="[SYNTH] 6-Cardio-HTN", age=61, sex="M",
         cardiovascular_history=1, diabetes_history=1,
         hr_arrival=96, hr_current=100, rr_arrival=18, rr_current=20,
         spo2_arrival=97, spo2_current=96, sbp_arrival=210, sbp_current=215,
         dbp_arrival=115, dbp_current=118, temp_arrival=98.2, temp_current=98.4,
         triage_complaint="severe headache and visual changes"),

    # 7. Ambiguous / intermediate
    dict(BASE, label="[SYNTH] 7-Ambiguous", age=49, sex="F",
         hr_arrival=92, hr_current=88, rr_arrival=18, rr_current=18,
         spo2_arrival=97, spo2_current=96, sbp_arrival=130, sbp_current=128,
         dbp_arrival=84, dbp_current=82, temp_arrival=99.0, temp_current=99.2,
         triage_complaint="intermittent chest tightness, fatigue",
         previous_ed_visits=1),

    # 8. Real patient from sample_patient.json (MIMIC-IV patient 10016742)
    dict(BASE, label="[REAL] 8-MIMIC-10016742", age=65, sex="M",
         cardiovascular_history=1,
         hr_arrival=108, hr_current=112, rr_arrival=20, rr_current=22,
         spo2_arrival=96, spo2_current=94, sbp_arrival=142, sbp_current=148,
         dbp_arrival=86, dbp_current=90, temp_arrival=98.4, temp_current=98.9,
         triage_complaint="chest pain and shortness of breath",
         previous_ed_visits=1, time_elapsed_minutes=20),
]

print("\n" + "="*100)
print("  STEP 5 — XGBoost Inference on 8 Test Cases")
print("="*100)
print(f"{'Case':<35} {'ICU-2h':>7} {'ICU-6h':>7} {'ICU-12h':>7} {'Admit':>7} {'Conf-Adm':>9} {'Conf-2h':>8} {'Missing'}")
print("-"*100)

rows = []
for c in cases:
    label = c.pop("label")
    r = run(label, c)
    rows.append(r)
    print(f"{r['label']:<35} {fmt(r['icu_2h']):>7} {fmt(r['icu_6h']):>7} "
          f"{fmt(r['icu_12h']):>7} {fmt(r['admission']):>7} "
          f"{fmt(r['conf_admission']):>9} {fmt(r['conf_icu_2h']):>8}   {r['missing']}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("  STEP 6 — Sanity Checks")
print("="*100)

stable   = rows[0]
concern  = rows[1]
highrisk = rows[2]

checks = [
    ("Admit(high-risk) > Admit(stable)",     highrisk["admission"]  > stable["admission"]),
    ("ICU-2h(high-risk) > ICU-2h(stable)",   highrisk["icu_2h"]     > stable["icu_2h"]),
    ("Admit(concerning) > Admit(stable)",    concern["admission"]   > stable["admission"]),
    ("All probs in [0,1]",                   all(0 <= rows[i][k] <= 1
                                                  for i in range(len(rows))
                                                  for k in ["icu_2h","icu_6h","icu_12h","admission"])),
    ("All confs in [0,1]",                   all(0 <= rows[i][k] <= 1
                                                  for i in range(len(rows))
                                                  for k in ["conf_admission","conf_icu_2h"])),
    ("ICU-12h >= ICU-2h (stable)",           stable["icu_12h"]      >= stable["icu_2h"]),
    ("ICU-12h >= ICU-6h (stable)",           stable["icu_12h"]      >= stable["icu_6h"]),
    ("ICU-12h >= ICU-2h (high-risk)",        highrisk["icu_12h"]    >= highrisk["icu_2h"]),
    ("No NaN in any output",                 all(not math.isnan(rows[i][k])
                                                  for i in range(len(rows))
                                                  for k in ["icu_2h","icu_6h","icu_12h","admission"])),
]
for name, result in checks:
    status = "PASS" if result else "FAIL"
    print(f"  [{status}] {name}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — MISSINGNESS ABLATION (cases 2-Concerning & 5-Low-SpO2)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("  STEP 7 — Missingness Ablation")
print("="*100)

base_concern = dict(
    age=58, sex="M", cardiovascular_history=1, diabetes_history=1,
    previous_ed_visits=2, previous_hospital_admissions=1,
    time_elapsed_minutes=30,
    hr_arrival=110, hr_current=118, rr_arrival=22, rr_current=24,
    spo2_arrival=95, spo2_current=92, sbp_arrival=145, sbp_current=130,
    dbp_arrival=88, dbp_current=80, temp_arrival=98.9, temp_current=99.1,
    triage_complaint="chest pain and shortness of breath",
    previous_icu_admissions=0, respiratory_history=0, renal_history=0,
    neurological_history=0, malignancy_history=0,
)

ablation_cases = [
    ("Complete", base_concern.copy()),
    ("Missing temp",        {**base_concern, "temp_current": None}),
    ("Missing temp+rr",     {**base_concern, "temp_current": None, "rr_current": None}),
    ("Half missing (temp+rr+spo2)",
                            {**base_concern, "temp_current": None, "rr_current": None, "spo2_current": None}),
    ("Only HR+SBP available",
                            {**base_concern, "temp_current": None, "rr_current": None,
                             "spo2_current": None, "dbp_current": None, "sbp_current": None,
                             "hr_current": 118}),   # keep HR
    ("All vitals missing",  {**base_concern,
                             "hr_current": None, "rr_current": None, "spo2_current": None,
                             "sbp_current": None, "dbp_current": None, "temp_current": None}),
]

print(f"\n  Subject: Concerning chest-pain patient (case 2)\n")
print(f"{'Completeness':<35} {'ICU-2h':>7} {'ICU-6h':>7} {'ICU-12h':>7} {'Admit':>7} {'Conf-Adm':>9}")
print("-"*80)
for lbl, pat in ablation_cases:
    out = predictor.predict(pat)
    print(f"  {lbl:<33} {out['icu_risk_2h']:>7.3f} {out['icu_risk_6h']:>7.3f} "
          f"{out['icu_risk_12h']:>7.3f} {out['admission_risk']:>7.3f} "
          f"{out['admission_confidence']:>9.3f}")

print("\nDone. All outputs above are raw, unmodified model results.")
