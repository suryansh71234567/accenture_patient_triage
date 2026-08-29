"""
scripts/run_triage.py  (fixed path bootstrap)
"""
import argparse, json, sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]   # aic_hackathon/
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "triageguard_xgb"))

from triageguard_router.combined_pipeline import TriageGuardPipeline


def main():
    p = argparse.ArgumentParser(description="TriageGuard — full triage pipeline")
    p.add_argument("--input", required=True, help="Patient JSON file path")
    p.add_argument("--json",  action="store_true", help="Dump full JSON output")
    args = p.parse_args()

    with open(args.input) as f:
        patient = json.load(f)

    pipeline = TriageGuardPipeline()
    result   = pipeline.run(patient)

    if args.json:
        out = {k: v for k, v in result.items() if k not in ("rag_response", "patient_history", "similar_cases")}
        print(json.dumps(out, indent=2))
        return

    print("\n" + "=" * 70)
    print("  TriageGuard -- Triage Decision")
    print("=" * 70)
    print(f"  Department      : {result['department']}")
    print(f"  Acuity tier     : {result['acuity_tier']} / 5")
    print(f"  Admission risk  : {result['reconciled_admission_risk']:.1%}")
    print(f"  ICU risk        : {result['reconciled_icu_risk']:.1%}")
    print(f"  Branches agree  : {result['branches_agree']}")
    print(f"  Confidence note : {result['confidence_note']}")
    print(f"\n  Reasoning       : {result['department_reasoning']}")
    print(f"\n  Top diagnoses   : {result['top_diagnoses']}")
    print(f"  Red flags       : {result['red_flags']}")
    so = result.get("structured_output", {})
    print(f"\n  RAG disposition : {so.get('disposition', 'N/A')}")
    print(f"  RAG escalation  : {so.get('escalation_level', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
