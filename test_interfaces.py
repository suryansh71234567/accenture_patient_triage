"""
Quick smoke test for both branches in one shot.
Run from repo root: .venv\Scripts\python.exe test_interfaces.py
"""
import sys, json
sys.path.insert(0, "triageguard_xgb")
sys.path.insert(0, ".")

# ── XGBoost branch ──────────────────────────────────────────────
print("=" * 60)
print("XGBoost branch")
print("=" * 60)
from triageguard_xgb.src.inference.predict import TriageGuardPredictor

xgb_predictor = TriageGuardPredictor("triageguard_xgb/models")

sample = {
    "age": 65, "sex": "M", "cardiovascular_history": 1,
    "hr_arrival": 108, "hr_current": 112,
    "rr_arrival": 20,  "rr_current": 22,
    "spo2_arrival": 96, "spo2_current": 94,
    "sbp_arrival": 142, "sbp_current": 148,
    "dbp_arrival": 86,  "dbp_current": 90,
    "temp_arrival": 98.4, "temp_current": 98.9,
    "time_elapsed_minutes": 20,
    "triage_complaint": "chest pain and shortness of breath",
    "previous_ed_visits": 1, "previous_hospital_admissions": 0,
    "previous_icu_admissions": 0, "respiratory_history": 0,
    "renal_history": 0, "diabetes_history": 0,
    "neurological_history": 0, "malignancy_history": 0,
}

xgb_out = xgb_predictor.predict(sample)
print(json.dumps(xgb_out, indent=2))

print("\nExpected keys:", sorted([
    "icu_risk_2h", "icu_risk_2h_confidence", "icu_risk_2h_raw",
    "icu_risk_6h", "icu_risk_6h_confidence", "icu_risk_6h_raw",
    "icu_risk_12h","icu_risk_12h_confidence","icu_risk_12h_raw",
    "admission_risk","admission_risk_confidence","admission_risk_raw",
    "information_completeness"
]))
print("Actual keys:  ", sorted(xgb_out.keys()))
missing = set(["icu_risk_2h","icu_risk_2h_confidence","icu_risk_2h_raw",
               "icu_risk_6h","icu_risk_6h_confidence","icu_risk_6h_raw",
               "icu_risk_12h","icu_risk_12h_confidence","icu_risk_12h_raw",
               "admission_risk","admission_risk_confidence","admission_risk_raw",
               "information_completeness"]) - set(xgb_out.keys())
print("MISSING KEYS:", missing if missing else "none — all present")

# ── RAG branch ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RAG branch — structured_output field")
print("=" * 60)
from triageguard_rag.src.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
with open("triageguard_rag/data/sample_patient.json") as f:
    patient = json.load(f)

rag_out = pipeline.run(patient)

print("structured_output:")
print(json.dumps(rag_out["structured_output"], indent=2))

so = rag_out["structured_output"]
print("\nDisposition parsed:", so.get("disposition"))
print("Escalation level:", so.get("escalation_level"))
print("Top diagnoses:", so.get("top_diagnoses"))
print("Red flags:", so.get("red_flags"))
parsed_ok = so.get("disposition") not in (None, "unknown")
print("JSON parse success:", parsed_ok)
