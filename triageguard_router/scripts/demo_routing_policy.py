"""
scripts/demo_routing_policy.py
--------------------------------
Real, end-to-end runtime demonstration (Phase 13/17):

    real patient
        -> TriageGuardPipeline (REAL XGBoost + REAL RAG/LLM, unmodified)
        -> reconciler.reconcile() (REAL, re-derived from the pipeline's own
           already-computed xgb/rag output — no clinical logic reimplemented)
        -> ClinicalSignal (features.py)
        -> live HospitalStateService state
        -> RoutingPolicy (trained Bayesian nurse policy)
        -> final allocation + faithful explanation

Runs the three required scenarios (Phase 17), holding the SAME real clinical
assessment fixed and varying ONLY hospital occupancy, to show clinical
priority never moves while allocation does:

    A. ICU available                              -> preferred == allocated
    B. ICU full, general ward available            -> resource-constrained step-down
    C. ICU, general ward, AND ED observation full   -> honest resource conflict

Requires OPENROUTER_API_KEY (see .env) for the real RAG call, same as
scripts/chat_with_agent.py.

Usage
-----
    .venv\\Scripts\\python.exe triageguard_router/scripts/demo_routing_policy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv(repo_root / ".env", override=True)
except ImportError:
    pass

from triageguard_router.combined_pipeline import TriageGuardPipeline
from triageguard_router.reconciler import reconcile
from triageguard_router.policy import artifacts
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.features import ClinicalSignal, HospitalSignal
from triageguard_router.policy.routing_policy import RoutingPolicy

PATIENT = {
    "patient_id": "52",
    "age": 62,
    "sex": "M",
    "chiefcomplaint": "Crushing substernal chest pain, diaphoresis, acute severe dyspnea",
    "acuity": 1,
    "heartrate": 138,
    "resprate": 30,
    "o2sat": 84,
    "sbp": 88,
    "dbp": 54,
    "temperature": 99.4,
    "pain": 10,
    "time_elapsed_minutes": 10,
    "previous_ed_visits": 2,
    "previous_hospital_admissions": 1,
    "previous_icu_admissions": 1,
    "cardiovascular_history": 1,
    "respiratory_history": 0,
    "renal_history": 0,
    "diabetes_history": 0,
    "neurological_history": 0,
    "malignancy_history": 0,
}


def _hospital_signal(icu_occ: int, gen_occ: int, obs_occ: int) -> HospitalSignal:
    return HospitalSignal(
        department_state={
            "ICU": {"capacity": 10, "occupied": icu_occ, "available": max(0, 10 - icu_occ), "status": "OPEN"},
            "CICU": {"capacity": 6, "occupied": 4, "available": 2, "status": "OPEN"},
            "ADMITTED_GEN": {"capacity": 50, "occupied": gen_occ, "available": max(0, 50 - gen_occ), "status": "OPEN"},
            "ED_OBS": {"capacity": 20, "occupied": obs_occ, "available": max(0, 20 - obs_occ), "status": "OPEN"},
            "DISCHARGE": {"capacity": 999, "occupied": 0, "available": 999, "status": "OPEN"},
        },
        operating_mode="NORMAL", load_ratio=0.5,
    )


def main() -> None:
    if not artifacts.artifacts_exist():
        print("No calibrated Bayesian policy found — run calibrate_policy.py first.")
        sys.exit(1)

    cfg = PolicyConfig.load()
    bayesian_policy = artifacts.load_bayesian_policy(cfg)
    routing_policy = RoutingPolicy(bayesian_policy, config=cfg)

    print("=" * 70)
    print("  Running REAL TriageGuard pipeline (XGBoost + RAG) for patient 52")
    print("=" * 70)
    pipeline = TriageGuardPipeline()
    result = pipeline.run(PATIENT)

    reconciled_raw = reconcile(result["xgb"], {"structured_output": result.get("structured_output", {})})
    clinical = ClinicalSignal.from_pipeline_output(
        reconciled=reconciled_raw,
        xgb_output=result["xgb"],
        preferred_department=result["department"],
        rag_history_count=len(result.get("patient_history", [])),
        rag_similar_count=len(result.get("similar_cases", [])),
    )

    print(f"\nReal clinical result: preferred_department={clinical.preferred_department}, "
          f"clinical_priority={clinical.clinical_priority:.3f}, "
          f"branches_agree={result.get('branches_agree')}")

    scenarios = [
        ("A: ICU available", _hospital_signal(icu_occ=8, gen_occ=38, obs_occ=12)),
        ("B: ICU full, general ward available", _hospital_signal(icu_occ=10, gen_occ=38, obs_occ=12)),
        ("C: ICU, general ward, AND ED observation all full", _hospital_signal(icu_occ=10, gen_occ=50, obs_occ=20)),
    ]

    for label, hospital in scenarios:
        print(f"\n{'-' * 70}\nSCENARIO {label}\n{'-' * 70}")
        routing_result = routing_policy.route(clinical, hospital)
        r = routing_result["routing"]
        print(f"  preferred_department : {r['preferred_department']}")
        print(f"  allocated_department : {r['allocated_department']}")
        print(f"  resource_constraint  : {r['resource_constraint']}")
        print(f"  policy_confidence    : {r['policy_confidence']}")
        print(f"  policy_uncertainty   : {r['policy_uncertainty']}")
        print(f"  human_review         : {r['human_review_recommended']}")
        print(f"  clinical_priority    : {routing_result['clinical_assessment']['clinical_priority']} (must match across all 3 scenarios)")
        print(f"  primary_reason       : {routing_result['explanation']['primary_reason']}")
        print(f"  allocation_reason    : {routing_result['explanation']['allocation_reason']}")


if __name__ == "__main__":
    main()
