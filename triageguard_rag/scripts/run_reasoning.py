"""
scripts/run_reasoning.py
------------------------
CLI entry point: given a JSON file describing a patient's current triage
state, run the full RAG + LLM reasoning pipeline and print the result.

Usage
-----
    python scripts/run_reasoning.py --input data/sample_patient.json
    python scripts/run_reasoning.py --input data/sample_patient.json --verbose

The input JSON must contain (at minimum):
    patient_id, chiefcomplaint

Optional vitals (set to null if unknown):
    acuity, heartrate, resprate, o2sat, sbp, dbp, temperature, pain
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the workspace root importable
repo_root = Path(__file__).resolve().parents[2]   # aic_hackathon/
sys.path.insert(0, str(repo_root))

from triageguard_rag.src.pipeline.rag_pipeline import RAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="TriageGuard RAG clinical reasoning pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a JSON file containing the current patient triage state.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Also print the retrieved documents and the full prompt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        patient_state = json.load(f)

    print("\n" + "=" * 70)
    print("  TriageGuard - RAG Clinical Reasoning Pipeline")
    print("=" * 70)
    print(f"  Patient ID      : {patient_state.get('patient_id')}")
    print(f"  Chief complaint : {patient_state.get('chiefcomplaint')}")
    print("=" * 70 + "\n")

    pipeline = RAGPipeline()
    result   = pipeline.run(patient_state)

    if args.verbose:
        print("\n---- RETRIEVED PATIENT HISTORY " + "-" * 39)
        if result["patient_history"]:
            for i, doc in enumerate(result["patient_history"], 1):
                print(f"\n[Past Visit {i}]\n{doc['document_text'][:400]}...")
        else:
            print("  (none)")

        print("\n---- RETRIEVED SIMILAR CASES " + "-" * 41)
        if result["similar_cases"]:
            for i, doc in enumerate(result["similar_cases"], 1):
                print(f"\n[Similar Case {i}]\n{doc['document_text'][:400]}...")
        else:
            print("  (none)")

        print("\n---- FULL PROMPT " + "-" * 53)
        print(result["prompt"])

    print("\n---- LLM REASONING " + "-" * 51)
    print(result["response"])
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
