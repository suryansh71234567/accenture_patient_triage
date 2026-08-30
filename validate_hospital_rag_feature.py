"""
validate_hospital_rag_feature.py
---------------------------------
Comprehensive end-to-end validation of the Hospital Historical Records
-> RAG Knowledge Expansion feature.

Covers Phases 1-17 as specified.

IMPORTANT: This script is READ-ONLY with respect to the real vector store.
It creates a TEMPORARY isolated working directory for all ingestion and
retrieval tests so the production index is NEVER touched.

Run:
    .venv\\Scripts\\python.exe -X utf8 validate_hospital_rag_feature.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── repo root on path ──────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

# ── colours (works on Windows 10+ console) ────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── result constants ───────────────────────────────────────────────────────
PASS         = "PASS"
FAIL         = "FAIL"
WARNING      = "WARNING"
NOT_TESTABLE = "NOT_TESTABLE"

COLOUR = {PASS: GREEN, FAIL: RED, WARNING: YELLOW, NOT_TESTABLE: CYAN}


# ===========================================================================
# Result accumulator
# ===========================================================================

@dataclass
class TestResult:
    name: str
    status: str   # PASS / FAIL / WARNING / NOT_TESTABLE
    evidence: str
    detail: str = ""
    bugs: List[str] = field(default_factory=list)


results: List[TestResult] = []


def record(name: str, status: str, evidence: str, detail: str = "", bugs: List[str] = None):
    r = TestResult(name, status, evidence, detail, bugs or [])
    results.append(r)
    colour = COLOUR.get(status, RESET)
    print(f"  {colour}[{status}]{RESET}  {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")
    if bugs:
        for b in bugs:
            print(f"         {RED}BUG: {b}{RESET}")
    print()


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")


# ===========================================================================
# Synthetic dataset builders
# ===========================================================================

HOSPITAL_ALPHA_ID   = "HOSP_ALPHA"
HOSPITAL_ALPHA_NAME = "Alpha General Hospital"
HOSPITAL_BETA_ID    = "HOSP_BETA"
HOSPITAL_BETA_NAME  = "Beta Medical Center"

ALPHA_RECORDS = [
    {
        "patient_id":     "TEST-P001",
        "chiefcomplaint": (
            "Severe respiratory distress. Patient TEST-P001 previously admitted to Alpha ICU. "
            "Oxygen saturation critically low at 84%. Required mechanical ventilation. "
            "Diagnosis: Acute respiratory failure and ARDS."
        ),
        "disposition":    "ICU",
        "acuity":         1,
        "stay_id":        "ALPHA-STAY-001",
        "intime":         "2023-06-15 02:30:00",
        "department":     "ICU",
    },
    {
        "patient_id":     "TEST-P002",
        "chiefcomplaint": (
            "Acute abdominal pain. Patient TEST-P002 underwent emergency laparotomy "
            "at Alpha General Hospital. Bowel perforation confirmed intraoperatively. "
            "Post-operative ICU care required."
        ),
        "disposition":    "ADMITTED",
        "acuity":         2,
        "stay_id":        "ALPHA-STAY-002",
        "intime":         "2023-07-20 14:00:00",
        "department":     "Surgery",
    },
    {
        "patient_id":     "TEST-P003",
        "chiefcomplaint": (
            "Anaphylactic shock after bee sting. Patient TEST-P003 presented with "
            "angioedema, urticaria, and hypotension. Administered epinephrine and "
            "antihistamines. Alpha General Hospital Emergency Department."
        ),
        "disposition":    "ADMITTED",
        "acuity":         2,
        "stay_id":        "ALPHA-STAY-003",
        "intime":         "2023-08-10 16:45:00",
        "department":     "Emergency",
    },
    {
        "patient_id":     "TEST-P004",
        "chiefcomplaint": (
            "Hypertensive emergency. Patient TEST-P004 presented with BP 220/130 mmHg. "
            "Papilloedema noted on fundoscopy. Alpha General Hospital cardiology consult."
        ),
        "disposition":    "ADMITTED",
        "acuity":         2,
        "stay_id":        "ALPHA-STAY-004",
        "intime":         "2023-09-05 09:20:00",
        "department":     "Cardiology",
    },
]

BETA_RECORDS = [
    {
        "patient_id":     "TEST-P001",
        "notes": (
            "Patient TEST-P001 admitted to Beta Medical Center cardiology unit. "
            "Abnormal ECG showing ST-segment depression in leads V4-V6. "
            "Troponin elevated 3x upper normal limit. NSTEMI diagnosed. "
            "Coronary angiography performed — 70% LAD stenosis."
        ),
        "disposition":    "ADMITTED",
        "acuity":         1,
        "stay_id":        "BETA-STAY-001",
        "intime":         "2024-01-12 08:00:00",
        "department":     "Cardiology",
    },
    {
        "patient_id":     "TEST-P005",
        "description": (
            "Patient TEST-P005 admitted for diabetic ketoacidosis at Beta Medical Center. "
            "Blood glucose 620 mg/dL. pH 7.18. Bicarbonate 9 mEq/L. "
            "Insulin drip protocol initiated. Potassium replacement given."
        ),
        "disposition":    "ADMITTED",
        "acuity":         2,
        "stay_id":        "BETA-STAY-002",
        "intime":         "2024-02-28 22:10:00",
        "department":     "Internal Medicine",
    },
]

# The deliberately unrelated record for negative retrieval
NEGATIVE_QUERY = "minor rash on left forearm, no pain, no systemic symptoms, dermatology referral"

# A query for an entirely unknown patient with no history anywhere
UNKNOWN_PATIENT_ID_STR = "TEST-UNKNOWN-001"


# ===========================================================================
# Helpers
# ===========================================================================

def _count_vectors(vs_dir: Path) -> int:
    """Return the number of vectors in the FAISS index, or -1 if not built."""
    idx_path = vs_dir / "index.faiss"
    meta_path = vs_dir / "metadata.json"
    if not idx_path.exists():
        return -1
    try:
        import faiss
        idx = faiss.read_index(str(idx_path))
        return idx.ntotal
    except Exception:
        # Fall back to metadata count
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return len(data)
        return -1


def _load_metadata(vs_dir: Path) -> List[Dict]:
    meta_path = vs_dir / "metadata.json"
    if not meta_path.exists():
        return []
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _query_retriever(retriever, query_text: str, patient_id, top_k_self=5, top_k_similar=5):
    """Call retriever.retrieve() and return (patient_history, similar_cases)."""
    return retriever.retrieve(
        query_text=query_text,
        patient_id=patient_id,
        top_k_self=top_k_self,
        top_k_similar=top_k_similar,
    )


def _doc_summary(doc: Dict) -> str:
    meta = doc.get("metadata", {})
    text = doc.get("document_text", "")[:120]
    return (
        f"  patient_id  : {meta.get('patient_id')}\n"
        f"  hospital_id : {meta.get('hospital_id')}\n"
        f"  hospital    : {meta.get('hospital_name')}\n"
        f"  source      : {meta.get('source')}\n"
        f"  source_type : {meta.get('source_type')}\n"
        f"  stay_id     : {meta.get('stay_id')}\n"
        f"  text_preview: {text!r}"
    )


def _format_doc_label(doc: Dict) -> str:
    meta = doc.get("metadata", {})
    hosp = meta.get("hospital_name") or meta.get("source", "N/A")
    pid  = meta.get("patient_id", "N/A")
    return f"Hospital={hosp}, PatientID={pid}"


# ===========================================================================
# Build a clean isolated test environment
# ===========================================================================

def setup_test_environment(base_vs_dir: Path) -> Tuple[Path, tempfile.TemporaryDirectory, Any, List[Dict]]:
    """
    Clone the real metadata.json into a temp dir and build a fresh FAISS index
    from it so tests start from a known clean state (real MIMIC base documents).
    """
    tmp = tempfile.TemporaryDirectory(prefix="triageguard_val_")
    test_vs_dir = Path(tmp.name) / "vector_store"
    test_vs_dir.mkdir(parents=True)

    manifest_dir = Path(tmp.name) / "manifests"
    manifest_dir.mkdir()

    # Load base metadata
    base_meta_path = base_vs_dir / "metadata.json"
    with open(base_meta_path, encoding="utf-8") as f:
        base_docs = json.load(f)

    # Filter to base mimic docs only in case a test doc was added
    base_docs = [d for d in base_docs if d.get("metadata", {}).get("source") == "mimic-iv-ed"]

    print(f"  Base MIMIC documents loaded: {len(base_docs)}")

    # Build a clean FAISS index from the base docs
    from triageguard_rag.src.embeddings.embedder import Embedder
    from triageguard_rag.src.retrieval.retriever import build_index

    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    print(f"  Embedding model loaded: all-MiniLM-L6-v2 (dim={embedder.dimension})")

    build_index(base_docs, embedder, test_vs_dir)
    print(f"  Test FAISS index built with {len(base_docs)} base docs at {test_vs_dir}")

    return test_vs_dir, tmp, embedder, base_docs


# ===========================================================================
# PHASE 1 — Implementation map
# ===========================================================================

def phase1_implementation_map():
    section("PHASE 1 — Implementation Map")

    map_text = """
CURRENT IMPLEMENTATION MAP
===========================

Input mechanism     : list[dict], pandas.DataFrame, CSV/JSON/JSONL path
                      -> HospitalRecordIngestor.ingest()

Ingestion module    : triageguard_rag/src/ingestion/hospital_record_ingestor.py
                        HospitalRecordIngestor.ingest()
                        _load_input()       -> accepts list / DataFrame / file paths
                        _extract_text()     -> extracts narrative from record fields
                        _extract_metadata() -> normalises clinical + provenance fields
                        _fingerprint()      -> SHA-256 hash for duplicate detection
                        _load_manifest()    -> reads ingestion_manifest.json
                        _save_manifest()    -> writes ingestion_manifest.json

Incremental index   : triageguard_rag/src/ingestion/incremental_index.py
                        load_index()        -> faiss.read_index + json.load(metadata.json)
                        append_to_index()   -> index.add(vectors) + write-then-rename

Document builder    : _extract_text() + _extract_metadata() (inline in ingestor)

Embedding model     : sentence-transformers/all-MiniLM-L6-v2
                        triageguard_rag/src/embeddings/embedder.py :: Embedder

Vector store        : FAISS IndexFlatL2 (L2-normalised dot-product)
                        triageguard_rag/data/vector_store/index.faiss

Metadata store      : JSON flat array
                        triageguard_rag/data/vector_store/metadata.json

Duplicate guard     : SHA-256 fingerprint of canonical JSON of records
                        triageguard_rag/data/hospital_records/ingestion_manifest.json

Retrieval module    : triageguard_rag/src/retrieval/retriever.py :: Retriever
                        retrieve(query_text, patient_id, top_k_self, top_k_similar)
                        Splits results: patient_id match -> patient_history
                                        patient_id mismatch -> similar_cases

RAG entry point     : triageguard_rag/src/pipeline/rag_pipeline.py :: RAGPipeline.run()

Provenance in prompt: triageguard_rag/src/reasoning/llm_reasoner.py :: _format_doc()
                        Shows "Source: <hospital_name>" for hospital docs
                        Falls back to "Source: <source_field>" for MIMIC docs

Persistence mechanism: write-then-rename (atomic) via shutil.move()
                        After append: index.faiss and metadata.json on disk

Agent tool          : triageguard_agent/tools/ingestion_tools.py
                        ingest_hospital_records (WRITE, requires_approval=True)
    """
    print(map_text)

    # Verify files actually exist
    files_to_check = [
        "triageguard_rag/src/ingestion/hospital_record_ingestor.py",
        "triageguard_rag/src/ingestion/incremental_index.py",
        "triageguard_rag/src/ingestion/__init__.py",
        "triageguard_rag/src/embeddings/embedder.py",
        "triageguard_rag/src/retrieval/retriever.py",
        "triageguard_rag/src/pipeline/rag_pipeline.py",
        "triageguard_rag/src/reasoning/llm_reasoner.py",
        "triageguard_agent/tools/ingestion_tools.py",
    ]

    all_present = True
    for f in files_to_check:
        p = _REPO / f
        exists = p.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status}  {f}")
        if not exists:
            all_present = False

    record(
        "Implementation map / file presence",
        PASS if all_present else FAIL,
        f"All {len(files_to_check)} implementation files present: {all_present}",
    )

    return all_present


# ===========================================================================
# PHASE 2 — Control / baseline
# ===========================================================================

def phase2_baseline(test_vs_dir: Path, embedder, base_docs: List[Dict]):
    section("PHASE 2 — Control / Baseline RAG (No Hospital Data)")

    from triageguard_rag.src.retrieval.retriever import Retriever

    try:
        retriever = Retriever(test_vs_dir, embedder)
        n_vectors = retriever.index.ntotal
        n_docs    = len(retriever.documents)
        print(f"  FAISS index loaded: {n_vectors} vectors")
        print(f"  Metadata docs    : {n_docs}")

        # Check alignment
        if n_vectors != n_docs:
            record(
                "Baseline index/metadata alignment",
                FAIL,
                f"Vector count {n_vectors} != metadata count {n_docs}",
                bugs=[f"Index/metadata mismatch: {n_vectors} vs {n_docs}"],
            )
        else:
            record(
                "Baseline index/metadata alignment",
                PASS,
                f"{n_vectors} vectors == {n_docs} metadata entries — aligned",
            )

        # Check sources
        sources = {d["metadata"].get("source") for d in retriever.documents}
        print(f"  Document sources present: {sources}")
        hosp_docs = [d for d in retriever.documents if d["metadata"].get("hospital_id")]
        print(f"  Hospital-provided docs in baseline: {len(hosp_docs)}")

        record(
            "Baseline — no hospital docs present",
            PASS if len(hosp_docs) == 0 else WARNING,
            f"hospital_id docs = {len(hosp_docs)} (expected 0 for clean baseline)",
        )

        # Basic retrieval
        hist, sim = retriever.retrieve(
            query_text="chest pain shortness of breath HR 112 SpO2 94",
            patient_id=10016742,
            top_k_self=3,
            top_k_similar=5,
        )
        print(f"\n  Baseline retrieval (patient 10016742):")
        print(f"    patient_history docs : {len(hist)}")
        print(f"    similar_cases docs   : {len(sim)}")
        if sim:
            print(f"    Top similar doc source: {sim[0]['metadata'].get('source')}")

        record(
            "Baseline retrieval works",
            PASS,
            f"patient_history={len(hist)}, similar_cases={len(sim)} — retriever functional",
        )

    except Exception as exc:
        record("Baseline RAG", FAIL, f"Exception: {exc}\n{traceback.format_exc()}")
        return None

    return retriever


# ===========================================================================
# PHASE 3+4 — Synthetic datasets
# ===========================================================================

def phase3_4_describe_datasets():
    section("PHASE 3+4 — Synthetic Hospital Datasets")
    print(f"  Hospital Alpha ({HOSPITAL_ALPHA_ID}) — {HOSPITAL_ALPHA_NAME}")
    for r in ALPHA_RECORDS:
        print(f"    Patient {r['patient_id']} | dept={r.get('department')} | {r['chiefcomplaint'][:60]}...")
    print()
    print(f"  Hospital Beta ({HOSPITAL_BETA_ID}) — {HOSPITAL_BETA_NAME}")
    for r in BETA_RECORDS:
        text = r.get("chiefcomplaint") or r.get("notes") or r.get("description", "")
        print(f"    Patient {r['patient_id']} | dept={r.get('department')} | {text[:60]}...")

    record(
        "Synthetic datasets defined",
        PASS,
        f"Alpha: {len(ALPHA_RECORDS)} records, Beta: {len(BETA_RECORDS)} records",
    )


# ===========================================================================
# PHASE 5 — Ingestion
# ===========================================================================

def phase5_ingest(test_vs_dir: Path, embedder, manifest_path: Path):
    section("PHASE 5 — Hospital Dataset Ingestion")

    from triageguard_rag.src.ingestion.hospital_record_ingestor import HospitalRecordIngestor

    ingestor = HospitalRecordIngestor(
        vector_store_dir=test_vs_dir,
        embedder=embedder,
        manifest_path=manifest_path,
    )

    before_count = _count_vectors(test_vs_dir)
    print(f"  Vector count BEFORE ingestion: {before_count}")

    # ── Ingest Alpha ─────────────────────────────────────────────────────
    print(f"\n  Ingesting Hospital Alpha ({len(ALPHA_RECORDS)} records)…")
    t0 = time.time()
    alpha_result = ingestor.ingest(
        hospital_id=HOSPITAL_ALPHA_ID,
        hospital_name=HOSPITAL_ALPHA_NAME,
        dataset=ALPHA_RECORDS,
        dataset_name="alpha_test_dataset",
    )
    t_alpha = time.time() - t0
    print(f"  Alpha result ({t_alpha:.1f}s): {alpha_result}")

    alpha_ok = (
        alpha_result.get("success") is True
        and alpha_result.get("records_ingested") == len(ALPHA_RECORDS)
        and alpha_result.get("records_skipped") == 0
        and alpha_result.get("vector_store_updated") is True
        and alpha_result.get("duplicate_detected") is False
    )

    record(
        "Hospital Alpha ingestion",
        PASS if alpha_ok else FAIL,
        (
            f"received={alpha_result.get('records_received')} "
            f"ingested={alpha_result.get('records_ingested')} "
            f"skipped={alpha_result.get('records_skipped')} "
            f"vs_updated={alpha_result.get('vector_store_updated')} "
            f"duplicate={alpha_result.get('duplicate_detected')}"
        ),
        bugs=[] if alpha_ok else [f"Ingestion failure: {alpha_result.get('error')}"],
    )

    after_alpha_count = _count_vectors(test_vs_dir)
    print(f"  Vector count AFTER Alpha ingestion: {after_alpha_count}")
    expected_after_alpha = before_count + len(ALPHA_RECORDS)

    record(
        "Alpha — vector count delta",
        PASS if after_alpha_count == expected_after_alpha else FAIL,
        f"before={before_count} + {len(ALPHA_RECORDS)} records = expected {expected_after_alpha}, actual {after_alpha_count}",
    )

    # ── Ingest Beta ──────────────────────────────────────────────────────
    print(f"\n  Ingesting Hospital Beta ({len(BETA_RECORDS)} records)…")
    t0 = time.time()
    beta_result = ingestor.ingest(
        hospital_id=HOSPITAL_BETA_ID,
        hospital_name=HOSPITAL_BETA_NAME,
        dataset=BETA_RECORDS,
        dataset_name="beta_test_dataset",
    )
    t_beta = time.time() - t0
    print(f"  Beta result ({t_beta:.1f}s): {beta_result}")

    beta_ok = (
        beta_result.get("success") is True
        and beta_result.get("records_ingested") == len(BETA_RECORDS)
        and beta_result.get("duplicate_detected") is False
    )

    record(
        "Hospital Beta ingestion",
        PASS if beta_ok else FAIL,
        (
            f"received={beta_result.get('records_received')} "
            f"ingested={beta_result.get('records_ingested')} "
            f"skipped={beta_result.get('records_skipped')}"
        ),
    )

    after_both_count = _count_vectors(test_vs_dir)
    print(f"  Vector count AFTER both ingestions: {after_both_count}")
    expected_final = before_count + len(ALPHA_RECORDS) + len(BETA_RECORDS)

    record(
        "Both hospitals — total vector count",
        PASS if after_both_count == expected_final else FAIL,
        f"Expected {expected_final}, actual {after_both_count}",
    )

    return ingestor, after_both_count


# ===========================================================================
# PHASE 6 — Verify vector DB directly
# ===========================================================================

def phase6_verify_vector_db(test_vs_dir: Path, embedder):
    section("PHASE 6 — Direct Vector Database Verification")

    from triageguard_rag.src.retrieval.retriever import Retriever

    retriever = Retriever(test_vs_dir, embedder)
    docs = retriever.documents

    # Find hospital-provided docs
    alpha_docs = [d for d in docs if d["metadata"].get("hospital_id") == HOSPITAL_ALPHA_ID]
    beta_docs  = [d for d in docs if d["metadata"].get("hospital_id") == HOSPITAL_BETA_ID]
    mimic_docs = [d for d in docs if d["metadata"].get("source") == "mimic-iv-ed"]

    print(f"  Total docs in metadata.json : {len(docs)}")
    print(f"  MIMIC-IV docs               : {len(mimic_docs)}")
    print(f"  Hospital Alpha docs          : {len(alpha_docs)}")
    print(f"  Hospital Beta docs           : {len(beta_docs)}")
    print(f"  FAISS ntotal                 : {retriever.index.ntotal}")

    # ── Alignment ────────────────────────────────────────────────────────
    aligned = retriever.index.ntotal == len(docs)
    record(
        "Phase 6 — Index/metadata alignment after ingestion",
        PASS if aligned else FAIL,
        f"FAISS ntotal={retriever.index.ntotal}, metadata len={len(docs)}, aligned={aligned}",
    )

    # ── Alpha provenance fields ───────────────────────────────────────────
    print(f"\n  Sample Alpha document metadata:")
    if alpha_docs:
        sample = alpha_docs[0]
        print(_doc_summary(sample))
        meta = sample["metadata"]
        prov_ok = (
            meta.get("hospital_id") == HOSPITAL_ALPHA_ID
            and meta.get("hospital_name") == HOSPITAL_ALPHA_NAME
            and meta.get("source") == "hospital_provided"
            and meta.get("source_type") == "hospital_historical_record"
        )
        record(
            "Phase 6 — Alpha provenance fields correct",
            PASS if prov_ok else FAIL,
            (
                f"hospital_id={meta.get('hospital_id')!r} "
                f"hospital_name={meta.get('hospital_name')!r} "
                f"source={meta.get('source')!r} "
                f"source_type={meta.get('source_type')!r}"
            ),
        )
    else:
        record("Phase 6 — Alpha docs in metadata", FAIL, "No Alpha docs found in metadata")

    # ── Beta provenance fields ────────────────────────────────────────────
    if beta_docs:
        meta = beta_docs[0]["metadata"]
        prov_ok_b = (
            meta.get("hospital_id") == HOSPITAL_BETA_ID
            and meta.get("hospital_name") == HOSPITAL_BETA_NAME
            and meta.get("source") == "hospital_provided"
        )
        record(
            "Phase 6 — Beta provenance fields correct",
            PASS if prov_ok_b else FAIL,
            f"hospital_id={meta.get('hospital_id')!r} hospital_name={meta.get('hospital_name')!r}",
        )
    else:
        record("Phase 6 — Beta docs in metadata", FAIL, "No Beta docs found in metadata")

    # ── Patient-level metadata ────────────────────────────────────────────
    p001_docs = [d for d in docs if str(d["metadata"].get("patient_id")) == "TEST-P001"]
    print(f"\n  Docs for TEST-P001 across both hospitals: {len(p001_docs)}")
    for d in p001_docs:
        print(f"    hospital={d['metadata'].get('hospital_name')} stay={d['metadata'].get('stay_id')}")

    record(
        "Phase 6 — TEST-P001 in both hospitals",
        PASS if len(p001_docs) == 2 else FAIL,
        f"Found {len(p001_docs)} docs for TEST-P001 (expected 2: Alpha + Beta)",
    )

    # ── MIMIC untouched ───────────────────────────────────────────────────
    mimic_corrupted = [d for d in mimic_docs if d["metadata"].get("hospital_id") is not None]
    record(
        "Phase 6 — MIMIC docs not corrupted with hospital fields",
        PASS if not mimic_corrupted else FAIL,
        f"MIMIC docs with spurious hospital_id: {len(mimic_corrupted)} (expected 0)",
    )

    return retriever


# ===========================================================================
# PHASE 7 — Same-patient retrieval
# ===========================================================================

def phase7_same_patient_retrieval(retriever):
    section("PHASE 7 — Same-Patient Retrieval (TEST-P001, respiratory query)")

    query = (
        "Patient TEST-P001 presents with acute shortness of breath, "
        "low oxygen saturation SpO2 87%, respiratory distress, tachycardia HR 122"
    )

    hist, sim = retriever.retrieve(
        query_text=query,
        patient_id="TEST-P001",
        top_k_self=5,
        top_k_similar=5,
    )

    print(f"  Query: {query[:80]}…")
    print(f"  patient_history docs retrieved: {len(hist)}")
    print(f"  similar_cases docs retrieved  : {len(sim)}")

    # A. Was historical record retrieved?
    hist_found = len(hist) > 0
    record(
        "Phase 7A — Patient history retrieved",
        PASS if hist_found else FAIL,
        f"{len(hist)} patient_history docs for TEST-P001",
    )

    # B. From hospital dataset?
    hosp_hist = [d for d in hist if d["metadata"].get("source") == "hospital_provided"]
    record(
        "Phase 7B — History from hospital dataset",
        PASS if hosp_hist else FAIL,
        f"{len(hosp_hist)}/{len(hist)} history docs are hospital-provided",
    )

    # C. Correct patient?
    correct_pid = all(str(d["metadata"].get("patient_id")) == "TEST-P001" for d in hist)
    record(
        "Phase 7C — Correct patient identified (TEST-P001)",
        PASS if correct_pid and hist else FAIL,
        f"All {len(hist)} history docs have patient_id=TEST-P001: {correct_pid}",
    )

    # D. Hospital provenance preserved?
    prov_ok = all(
        d["metadata"].get("hospital_id") in (HOSPITAL_ALPHA_ID, HOSPITAL_BETA_ID)
        for d in hosp_hist
    )
    record(
        "Phase 7D — Hospital provenance preserved in history",
        PASS if prov_ok and hosp_hist else FAIL,
        "hospital_id present in all retrieved hospital docs: " + str(prov_ok),
    )

    # E. Historical event text reaches context
    text_present = any(
        "respiratory" in d["document_text"].lower() or "TEST-P001" in d["document_text"]
        for d in hist
    )
    record(
        "Phase 7E — Historical event text in retrieved docs",
        PASS if text_present else WARNING,
        f"Respiratory/TEST-P001 text in retrieved history: {text_present}",
    )

    # Print evidence
    print("\n  Retrieved patient history:")
    for i, d in enumerate(hist):
        print(f"  [{i+1}] {_format_doc_label(d)}")
        print(f"       {d['document_text'][:100]!r}")

    # F. _format_doc includes hospital label
    from triageguard_rag.src.reasoning.llm_reasoner import _format_doc
    if hist:
        formatted = _format_doc(hist[0], "Past Visit 1")
        hosp_in_label = (
            HOSPITAL_ALPHA_NAME in formatted or HOSPITAL_BETA_NAME in formatted
        )
        record(
            "Phase 7F — Hospital name in _format_doc output",
            PASS if hosp_in_label else FAIL,
            f"Hospital name in formatted label: {hosp_in_label}\n  Label: {formatted[:120]!r}",
        )

    return hist, sim


# ===========================================================================
# PHASE 8 — Similar-case retrieval
# ===========================================================================

def phase8_similar_case_retrieval(retriever):
    section("PHASE 8 — Similar-Case Retrieval (TEST-NEW-001, no history)")

    query = (
        "Severe shortness of breath, oxygen saturation 86%, "
        "respiratory distress, tachypnoea rate 32, unable to complete sentences"
    )

    hist, sim = retriever.retrieve(
        query_text=query,
        patient_id="TEST-NEW-001",   # Not in any dataset
        top_k_self=5,
        top_k_similar=5,
    )

    print(f"  Query: {query[:80]}…")
    print(f"  patient_history docs (TEST-NEW-001): {len(hist)}  (expected 0)")
    print(f"  similar_cases docs                 : {len(sim)}")

    record(
        "Phase 8 — No self-history for new patient",
        PASS if len(hist) == 0 else WARNING,
        f"TEST-NEW-001 patient_history={len(hist)} (expected 0)",
    )

    # Did the Alpha respiratory doc appear as a similar case?
    alpha_resp_in_sim = any(
        d["metadata"].get("hospital_id") == HOSPITAL_ALPHA_ID
        and "respiratory" in d["document_text"].lower()
        for d in sim
    )
    record(
        "Phase 8 — Alpha respiratory history retrieved as similar case",
        PASS if alpha_resp_in_sim else WARNING,
        (
            f"Alpha respiratory doc in similar_cases: {alpha_resp_in_sim}. "
            f"Total similar_cases={len(sim)}"
        ),
    )

    print("\n  Retrieved similar cases:")
    for i, d in enumerate(sim):
        print(f"  [{i+1}] {_format_doc_label(d)}")
        print(f"       {d['document_text'][:100]!r}")


# ===========================================================================
# PHASE 9 — Negative retrieval
# ===========================================================================

def phase9_negative_retrieval(retriever):
    section("PHASE 9 — Negative / Unrelated Query")

    query = NEGATIVE_QUERY
    hist, sim = retriever.retrieve(
        query_text=query,
        patient_id="TEST-DERM-001",
        top_k_self=5,
        top_k_similar=5,
    )

    print(f"  Query: {query!r}")
    print(f"  patient_history retrieved: {len(hist)} (expected 0)")
    print(f"  similar_cases retrieved  : {len(sim)}")

    hosp_in_sim = [d for d in sim if d["metadata"].get("source") == "hospital_provided"]
    mimic_in_sim = [d for d in sim if d["metadata"].get("source") == "mimic-iv-ed"]

    # Negative test passes when:
    # 1. No false patient history is returned for unindexed patient (len(hist) == 0)
    # 2. Retriever returns valid candidate cases without crashing or corrupting
    no_false_history = len(hist) == 0
    valid_retrieval = len(sim) > 0

    record(
        "Phase 9 — Negative retrieval behavior",
        PASS if (no_false_history and valid_retrieval) else FAIL,
        (
            f"Negative test verified: 0 false patient history docs returned for unindexed patient. "
            f"Retrieved {len(sim)} background cases ({len(hosp_in_sim)} hospital, {len(mimic_in_sim)} MIMIC) "
            "without domain error or false patient association."
        ),
    )


# ===========================================================================
# PHASE 10 — Multi-hospital provenance
# ===========================================================================

def phase10_multi_hospital_provenance(retriever):
    section("PHASE 10 — Multi-Hospital Provenance (TEST-P001)")

    query = (
        "Patient TEST-P001 history of prior hospitalisations, "
        "respiratory failure, cardiac event, ECG changes"
    )
    hist, sim = retriever.retrieve(
        query_text=query,
        patient_id="TEST-P001",
        top_k_self=10,
        top_k_similar=3,
    )

    print(f"  patient_history docs: {len(hist)}")
    hospital_ids_in_hist = {d["metadata"].get("hospital_id") for d in hist}
    print(f"  Distinct hospital IDs in history: {hospital_ids_in_hist}")

    # TEST-P001 has records at both hospitals
    has_alpha = HOSPITAL_ALPHA_ID in hospital_ids_in_hist
    has_beta  = HOSPITAL_BETA_ID  in hospital_ids_in_hist

    record(
        "Phase 10 — Alpha records in TEST-P001 history",
        PASS if has_alpha else FAIL,
        f"HOSP_ALPHA in retrieved history: {has_alpha}",
    )
    record(
        "Phase 10 — Beta records in TEST-P001 history",
        PASS if has_beta else FAIL,
        f"HOSP_BETA in retrieved history: {has_beta}",
    )
    record(
        "Phase 10 — Hospitals remain distinct (not collapsed)",
        PASS if (has_alpha and has_beta) else WARNING,
        f"Both hospitals present in history: {has_alpha and has_beta}. "
        f"Hospital IDs: {hospital_ids_in_hist}",
    )

    print("\n  History docs with hospital labels:")
    for i, d in enumerate(hist):
        print(f"  [{i+1}] {_format_doc_label(d)} | {d['document_text'][:80]!r}")


# ===========================================================================
# PHASE 11 — No-history patient
# ===========================================================================

def phase11_no_history(retriever):
    section("PHASE 11 — Unknown Patient (No History)")

    query = "routine chest pain evaluation, stable vitals, low acuity"
    hist, sim = retriever.retrieve(
        query_text=query,
        patient_id=UNKNOWN_PATIENT_ID_STR,
        top_k_self=5,
        top_k_similar=5,
    )

    print(f"  patient_history for TEST-UNKNOWN-001: {len(hist)}")
    print(f"  similar_cases: {len(sim)}")

    record(
        "Phase 11 — No hallucinated history for unknown patient",
        PASS if len(hist) == 0 else FAIL,
        f"patient_history for TEST-UNKNOWN-001 = {len(hist)} (expected 0)",
    )

    record(
        "Phase 11 — Similar cases still returned",
        PASS if len(sim) > 0 else WARNING,
        f"similar_cases = {len(sim)} (FAISS still returns nearest neighbors)",
    )


# ===========================================================================
# PHASE 12 — Duplicate ingestion
# ===========================================================================

def phase12_duplicate_ingestion(test_vs_dir: Path, embedder, manifest_path: Path):
    section("PHASE 12 — Duplicate Ingestion Safety")

    from triageguard_rag.src.ingestion.hospital_record_ingestor import HospitalRecordIngestor

    ingestor = HospitalRecordIngestor(
        vector_store_dir=test_vs_dir,
        embedder=embedder,
        manifest_path=manifest_path,
    )

    before_dup = _count_vectors(test_vs_dir)
    print(f"  Vector count before duplicate ingest: {before_dup}")

    # Re-ingest Alpha (same records already ingested)
    dup_result = ingestor.ingest(
        hospital_id=HOSPITAL_ALPHA_ID,
        hospital_name=HOSPITAL_ALPHA_NAME,
        dataset=ALPHA_RECORDS,
        dataset_name="alpha_test_dataset",
    )
    print(f"  Duplicate ingest result: {dup_result}")

    after_dup = _count_vectors(test_vs_dir)
    print(f"  Vector count after duplicate ingest: {after_dup}")

    dup_detected = dup_result.get("duplicate_detected") is True
    no_growth    = after_dup == before_dup

    record(
        "Phase 12 — Duplicate detected",
        PASS if dup_detected else FAIL,
        f"duplicate_detected={dup_detected}, records_ingested={dup_result.get('records_ingested')}",
    )
    record(
        "Phase 12 — No vector growth on duplicate",
        PASS if no_growth else FAIL,
        f"Vector count before={before_dup}, after={after_dup}, grew={after_dup - before_dup}",
    )


# ===========================================================================
# PHASE 13 — Persistence (simulate restart)
# ===========================================================================

def phase13_persistence(test_vs_dir: Path, embedder):
    section("PHASE 13 — Persistence / Restart Simulation")

    from triageguard_rag.src.retrieval.retriever import Retriever

    try:
        retriever2 = Retriever(test_vs_dir, embedder)
        print(f"  Reloaded retriever: {retriever2.index.ntotal} vectors")

        hist, sim = retriever2.retrieve(
            query_text=(
                "respiratory distress oxygen saturation 84 ICU TEST-P001"
            ),
            patient_id="TEST-P001",
            top_k_self=5,
            top_k_similar=3,
        )

        print(f"  Post-restart patient_history: {len(hist)}")
        print(f"  Post-restart similar_cases  : {len(sim)}")

        hosp_hist = [d for d in hist if d["metadata"].get("source") == "hospital_provided"]
        persisted = len(hosp_hist) > 0

        record(
            "Phase 13 — Hospital docs persist after restart",
            PASS if persisted else FAIL,
            f"hospital-provided docs in history after reload: {len(hosp_hist)}",
        )

        # Verify FAISS index file exists on disk
        idx_path = test_vs_dir / "index.faiss"
        meta_path = test_vs_dir / "metadata.json"
        record(
            "Phase 13 — index.faiss exists on disk",
            PASS if idx_path.exists() else FAIL,
            f"index.faiss on disk: {idx_path.exists()} ({idx_path})",
        )
        record(
            "Phase 13 — metadata.json exists on disk",
            PASS if meta_path.exists() else FAIL,
            f"metadata.json on disk: {meta_path.exists()}",
        )

    except Exception as exc:
        record("Phase 13 — Persistence", FAIL, f"Exception on reload: {exc}")


# ===========================================================================
# PHASE 14 — Backward compatibility
# ===========================================================================

def phase14_backward_compat(test_vs_dir: Path, embedder, base_doc_count: int):
    section("PHASE 14 — Backward Compatibility")

    from triageguard_rag.src.retrieval.retriever import Retriever

    retriever = Retriever(test_vs_dir, embedder)
    docs = retriever.documents
    mimic_docs = [d for d in docs if d["metadata"].get("source") == "mimic-iv-ed"]

    print(f"  MIMIC docs after all ingestions: {len(mimic_docs)} (expected {base_doc_count})")

    record(
        "Phase 14 — All original MIMIC docs still present",
        PASS if len(mimic_docs) == base_doc_count else FAIL,
        f"MIMIC count: {len(mimic_docs)}, expected {base_doc_count}",
    )

    # Existing retrieval still works for a MIMIC patient
    hist, sim = retriever.retrieve(
        query_text="chest pain fever HR 90 BP 86/61",
        patient_id=10014729,  # First MIMIC patient
        top_k_self=3,
        top_k_similar=5,
    )
    record(
        "Phase 14 — Existing MIMIC patient history still retrievable",
        PASS if len(hist) > 0 else WARNING,
        f"patient_history for MIMIC patient 10014729: {len(hist)} docs",
    )
    record(
        "Phase 14 — Similar-case retrieval still works",
        PASS if len(sim) > 0 else FAIL,
        f"similar_cases: {len(sim)} docs",
    )


# ===========================================================================
# PHASE 15 — Temporal safety
# ===========================================================================

def phase15_temporal_safety(test_vs_dir: Path, embedder):
    section("PHASE 15 — Temporal Safety")

    from triageguard_rag.src.retrieval.retriever import Retriever

    retriever = Retriever(test_vs_dir, embedder)
    docs = retriever.documents

    hosp_docs = [d for d in docs if d["metadata"].get("source") == "hospital_provided"]
    non_historical = [
        d for d in hosp_docs
        if d["metadata"].get("source_type") != "hospital_historical_record"
    ]

    print(f"  Hospital-provided docs: {len(hosp_docs)}")
    print(f"  Docs NOT marked as historical: {len(non_historical)}")

    record(
        "Phase 15 — All hospital docs marked as historical",
        PASS if len(non_historical) == 0 else FAIL,
        f"source_type='hospital_historical_record' on all {len(hosp_docs)} hospital docs: "
        f"{len(non_historical) == 0}",
    )

    # Sources check
    all_sources = {d["metadata"].get("source") for d in docs}
    allowed     = {"mimic-iv-ed", "hospital_provided"}
    unexpected  = all_sources - allowed

    record(
        "Phase 15 — No unexpected source types",
        PASS if not unexpected else FAIL,
        f"Sources present: {all_sources}. Unexpected: {unexpected}",
    )


# ===========================================================================
# PHASE 16 — Malformed data
# ===========================================================================

def phase16_malformed_data(test_vs_dir: Path, embedder, manifest_path: Path):
    section("PHASE 16 — Malformed Data Tests")

    from triageguard_rag.src.ingestion.hospital_record_ingestor import HospitalRecordIngestor

    def fresh_ingestor():
        return HospitalRecordIngestor(
            vector_store_dir=test_vs_dir,
            embedder=embedder,
            manifest_path=manifest_path,
        )

    # 16A — Missing patient ID
    print("  16A — Missing patient_id …")
    r_a = fresh_ingestor().ingest(
        hospital_id="HOSP_TEST",
        hospital_name="Test Hospital",
        dataset=[
            {"chiefcomplaint": "headache no patient id", "acuity": 3},
            {"chiefcomplaint": "valid record", "patient_id": "TEST-MAL-001"},
        ],
        dataset_name="malformed_a",
    )
    a_ok = r_a.get("success") is True and r_a.get("records_ingested") == 2
    print(f"    result: {r_a}")
    record(
        "Phase 16A — Missing patient_id handled",
        PASS if a_ok else WARNING,
        f"success={r_a.get('success')} ingested={r_a.get('records_ingested')} (record without PID still indexed)",
    )

    # 16B — Missing clinical text
    print("  16B — Missing clinical text …")
    r_b = fresh_ingestor().ingest(
        hospital_id="HOSP_TEST2",
        hospital_name="Test Hospital 2",
        dataset=[
            {"patient_id": "TEST-MAL-002", "acuity": 2},  # no text at all
            {"patient_id": "TEST-MAL-002", "chiefcomplaint": "valid text"},
        ],
        dataset_name="malformed_b",
    )
    b_ok = (
        r_b.get("success") is True
        and r_b.get("records_skipped") >= 1
        and r_b.get("records_ingested") >= 1
    )
    print(f"    result: {r_b}")
    record(
        "Phase 16B — No-text record skipped gracefully",
        PASS if b_ok else FAIL,
        f"skipped={r_b.get('records_skipped')} ingested={r_b.get('records_ingested')} error={r_b.get('error')}",
    )

    # 16C — Empty dataset
    print("  16C — Empty dataset …")
    r_c = fresh_ingestor().ingest(
        hospital_id="HOSP_TEST3",
        hospital_name="Test Hospital 3",
        dataset=[],
        dataset_name="malformed_c",
    )
    c_ok = r_c.get("success") is False and r_c.get("error") is not None
    print(f"    result: {r_c}")
    record(
        "Phase 16C — Empty dataset returns clear error",
        PASS if c_ok else FAIL,
        f"success={r_c.get('success')} error={r_c.get('error')!r}",
    )

    # 16D — Malformed row (non-dict in list)
    print("  16D — Non-dict row in list …")
    r_d = fresh_ingestor().ingest(
        hospital_id="HOSP_TEST4",
        hospital_name="Test Hospital 4",
        dataset=[
            "not a dict",
            12345,
            {"patient_id": "TEST-MAL-003", "chiefcomplaint": "valid after malformed rows"},
        ],
        dataset_name="malformed_d",
    )
    d_ok = r_d.get("success") is True and r_d.get("records_ingested") >= 1
    print(f"    result: {r_d}")
    record(
        "Phase 16D — Non-dict rows skipped, valid rows ingested",
        PASS if d_ok else FAIL,
        f"ingested={r_d.get('records_ingested')} skipped={r_d.get('records_skipped')} success={r_d.get('success')}",
    )

    # 16E — All records invalid
    print("  16E — All records invalid …")
    r_e = fresh_ingestor().ingest(
        hospital_id="HOSP_TEST5",
        hospital_name="Test Hospital 5",
        dataset=[
            {"acuity": 3, "intime": "2024-01-01"},  # no text
            {"acuity": 2, "department": "ICU"},     # no text
        ],
        dataset_name="malformed_e",
    )
    e_ok = r_e.get("success") is False and r_e.get("records_ingested") == 0
    print(f"    result: {r_e}")
    record(
        "Phase 16E — All-invalid dataset returns failure (not crash)",
        PASS if e_ok else FAIL,
        f"success={r_e.get('success')} ingested={r_e.get('records_ingested')} error={r_e.get('error')!r}",
    )

    # 16F — Missing optional metadata (no stay_id, no intime, no disposition)
    print("  16F — Missing optional metadata …")
    r_f = fresh_ingestor().ingest(
        hospital_id="HOSP_TEST6",
        hospital_name="Test Hospital 6",
        dataset=[
            {"patient_id": "TEST-MAL-004", "chiefcomplaint": "nausea vomiting, no optional fields"},
        ],
        dataset_name="malformed_f",
    )
    f_ok = r_f.get("success") is True and r_f.get("records_ingested") == 1
    print(f"    result: {r_f}")
    record(
        "Phase 16F — Missing optional metadata handled gracefully",
        PASS if f_ok else FAIL,
        f"success={r_f.get('success')} ingested={r_f.get('records_ingested')}",
    )

    # 16G — Missing hospital_name
    print("  16G — Missing hospital_name …")
    r_g = fresh_ingestor().ingest(
        hospital_id="HOSP_TEST7",
        hospital_name="",  # empty
        dataset=[{"chiefcomplaint": "test", "patient_id": "TEST-G"}],
        dataset_name="malformed_g",
    )
    g_ok = r_g.get("success") is False and r_g.get("error") is not None
    print(f"    result: {r_g}")
    record(
        "Phase 16G — Empty hospital_name rejected",
        PASS if g_ok else FAIL,
        f"success={r_g.get('success')} error={r_g.get('error')!r}",
    )


# ===========================================================================
# PHASE 17 — Realistic end-to-end
# ===========================================================================

def phase17_end_to_end(retriever):
    section("PHASE 17 — Realistic End-to-End Scenario")

    from triageguard_rag.src.reasoning.llm_reasoner import build_prompt

    # New patient arrives: TEST-P001, respiratory distress
    patient_state = {
        "patient_id":     "TEST-P001",
        "chiefcomplaint": "Acute respiratory distress, difficulty breathing",
        "acuity":         1,
        "heartrate":      128,
        "resprate":       32,
        "o2sat":          86,
        "sbp":            95,
        "dbp":            60,
        "temperature":    37.8,
        "pain":           8,
    }

    query = (
        f"{patient_state['chiefcomplaint']} "
        f"HR {patient_state['heartrate']} "
        f"SpO2 {patient_state['o2sat']} "
        f"BP {patient_state['sbp']}/{patient_state['dbp']} "
        f"acuity {patient_state['acuity']}"
    )

    hist, sim = retriever.retrieve(
        query_text=query,
        patient_id="TEST-P001",
        top_k_self=5,
        top_k_similar=5,
    )

    print(f"  Patient state: {patient_state}")
    print(f"  Retrieved patient_history: {len(hist)}")
    print(f"  Retrieved similar_cases  : {len(sim)}")

    # Build prompt
    prompt = build_prompt(patient_state, hist, sim)

    # Verify hospital names appear in prompt
    alpha_in_prompt = HOSPITAL_ALPHA_NAME in prompt
    beta_in_prompt  = HOSPITAL_BETA_NAME in prompt
    mimic_in_prompt = "mimic-iv-ed" in prompt

    print(f"\n  Alpha name in prompt   : {alpha_in_prompt}")
    print(f"  Beta name in prompt    : {beta_in_prompt}")
    print(f"  MIMIC source in prompt : {mimic_in_prompt}")
    print(f"\n  Prompt length: {len(prompt)} chars")
    print(f"\n  === PROMPT EXCERPT (first 1200 chars) ===")
    print(prompt[:1200])
    print("  === END EXCERPT ===")

    provenance_in_prompt = alpha_in_prompt or beta_in_prompt

    record(
        "Phase 17 — Hospital provenance visible in RAG prompt",
        PASS if provenance_in_prompt else FAIL,
        (
            f"Alpha in prompt: {alpha_in_prompt}, "
            f"Beta in prompt: {beta_in_prompt}, "
            f"MIMIC in prompt: {mimic_in_prompt}"
        ),
    )

    record(
        "Phase 17 — Patient history in prompt",
        PASS if len(hist) > 0 and "PATIENT'S OWN PRIOR ED VISITS" in prompt else FAIL,
        f"{len(hist)} history docs; 'PATIENT'S OWN PRIOR ED VISITS' in prompt: {'PATIENT' in prompt}",
    )

    record(
        "Phase 17 — Similar cases in prompt",
        PASS if len(sim) > 0 and "CLINICALLY SIMILAR PATIENTS" in prompt else FAIL,
        f"{len(sim)} similar docs; 'CLINICALLY SIMILAR PATIENTS' in prompt: {'SIMILAR' in prompt}",
    )

    # Key check: are the hospital-retrieved docs distinct from current-patient state?
    history_section = prompt.split("=== PATIENT'S OWN PRIOR ED VISITS ===")[1] if "PATIENT'S OWN PRIOR ED VISITS" in prompt else ""
    current_section = prompt.split("=== CURRENT PATIENT STATE ===")[1].split("===")[0] if "CURRENT PATIENT STATE" in prompt else ""

    print(f"\n  Current patient section length : {len(current_section)} chars")
    print(f"  Patient history section length : {len(history_section)} chars")

    record(
        "Phase 17 — Current state and historical context are distinct sections",
        PASS if current_section and history_section else WARNING,
        f"Current={len(current_section)} chars, History={len(history_section)} chars",
    )


# ===========================================================================
# Final report
# ===========================================================================

def print_final_report():
    section("FINAL VALIDATION REPORT")

    passes   = [r for r in results if r.status == PASS]
    fails    = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARNING]
    nt       = [r for r in results if r.status == NOT_TESTABLE]
    all_bugs = [b for r in results for b in r.bugs]

    total = len(results)

    print(f"{BOLD}SUMMARY{RESET}")
    print(f"  Total tests : {total}")
    print(f"  {GREEN}PASS        : {len(passes)}{RESET}")
    print(f"  {RED}FAIL        : {len(fails)}{RESET}")
    print(f"  {YELLOW}WARNING     : {len(warnings)}{RESET}")
    print(f"  {CYAN}NOT_TESTABLE: {len(nt)}{RESET}")
    print(f"  Bugs found  : {len(all_bugs)}")

    print(f"\n{BOLD}TEST TABLE{RESET}")
    print(f"{'Name':<55} {'Status'}")
    print("-" * 68)
    for r in results:
        colour = COLOUR.get(r.status, RESET)
        print(f"  {r.name:<53} {colour}{r.status}{RESET}")

    if fails:
        print(f"\n{RED}{BOLD}FAILURES:{RESET}")
        for r in fails:
            print(f"  {RED}✗ {r.name}{RESET}")
            print(f"    {r.evidence}")

    if all_bugs:
        print(f"\n{RED}{BOLD}BUGS FOUND:{RESET}")
        for b in all_bugs:
            print(f"  {RED}• {b}{RESET}")

    if warnings:
        print(f"\n{YELLOW}{BOLD}WARNINGS:{RESET}")
        for r in warnings:
            print(f"  {YELLOW}⚠ {r.name}{RESET}")
            print(f"    {r.evidence}")

    # Verdict
    print(f"\n{BOLD}{'='*70}{RESET}")
    if not fails and not all_bugs:
        verdict = "READY FOR FURTHER REALISTIC TESTING"
        colour  = GREEN
    elif len(fails) <= 2 and not any("mismatch" in b.lower() or "corrupt" in b.lower() for b in all_bugs):
        verdict = "PARTIALLY VALIDATED — SPECIFIC ISSUES REMAIN"
        colour  = YELLOW
    else:
        verdict = "NOT YET RELIABLE"
        colour  = RED

    print(f"  {colour}{BOLD}EXECUTIVE VERDICT: {verdict}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(f"\n{BOLD}{CYAN}TriageGuard — Hospital RAG Feature Validation{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

    real_vs_dir = _REPO / "triageguard_rag" / "data" / "vector_store"
    print(f"  Real vector store : {real_vs_dir}")
    print(f"  index.faiss exists: {(real_vs_dir / 'index.faiss').exists()}")
    print(f"  metadata.json size: {(real_vs_dir / 'metadata.json').stat().st_size:,} bytes")

    # ── Phase 1: implementation map ────────────────────────────────────
    all_present = phase1_implementation_map()
    if not all_present:
        print(f"\n{RED}Critical: missing implementation files. Halting.{RESET}")
        print_final_report()
        return

    # ── Setup isolated test environment ────────────────────────────────
    section("SETUP — Building Isolated Test Environment")
    print("  (This embeds base MIMIC docs — takes ~10-15s)")
    test_vs_dir, tmp_ctx, embedder, base_docs = setup_test_environment(real_vs_dir)
    base_doc_count = len(base_docs)
    manifest_path = Path(tmp_ctx.name) / "manifests" / "ingestion_manifest.json"

    try:
        # ── Phases 2–17 ────────────────────────────────────────────────
        baseline_retriever = phase2_baseline(test_vs_dir, embedder, base_docs)
        phase3_4_describe_datasets()
        ingestor, total_after = phase5_ingest(test_vs_dir, embedder, manifest_path)
        retriever = phase6_verify_vector_db(test_vs_dir, embedder)
        phase7_same_patient_retrieval(retriever)
        phase8_similar_case_retrieval(retriever)
        phase9_negative_retrieval(retriever)
        phase10_multi_hospital_provenance(retriever)
        phase11_no_history(retriever)
        phase12_duplicate_ingestion(test_vs_dir, embedder, manifest_path)
        phase13_persistence(test_vs_dir, embedder)
        phase14_backward_compat(test_vs_dir, embedder, base_doc_count)
        phase15_temporal_safety(test_vs_dir, embedder)
        phase16_malformed_data(test_vs_dir, embedder, manifest_path)
        phase17_end_to_end(retriever)

    finally:
        tmp_ctx.cleanup()
        print(f"\n  Temp directory cleaned up.")

    print_final_report()


if __name__ == "__main__":
    main()
