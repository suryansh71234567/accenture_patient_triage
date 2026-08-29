"""
hospital_record_ingestor.py
---------------------------
Validates, normalises, and ingests hospital-provided historical records
into the existing TriageGuard RAG FAISS vector store.

Supported input formats
-----------------------
  * list[dict]              — Python dicts, at least one text-bearing field
  * pandas.DataFrame        — converted to list of dicts
  * str / Path → .csv       — loaded via pandas
  * str / Path → .json      — list-of-objects JSON
  * str / Path → .jsonl     — one dict per line

Minimum required per record
---------------------------
At least one of: chiefcomplaint, notes, description, document_text,
clinical_text, summary, diagnosis, chief_complaint, discharge_summary.
(Any additional text fields are concatenated.)

patient_id is strongly recommended but optional; records without one
are still indexed under patient_id=None (they participate only in
similar-case retrieval, not same-patient retrieval).

Provenance metadata attached to every ingested document
-------------------------------------------------------
  hospital_id   : str
  hospital_name : str
  source_type   : "hospital_historical_record"
  dataset_name  : str
  source        : "hospital_provided"

Duplicate / re-ingestion safety
--------------------------------
The ingestion manifest lives at:
  triageguard_rag/data/hospital_records/ingestion_manifest.json

A SHA-256 fingerprint of the canonical dataset content is stored.
Re-submitting the exact same dataset returns a result with
  duplicate_detected=True  and  records_ingested=0
without touching the vector store.

Return value
------------
{
  "success":              bool,
  "hospital_id":          str,
  "hospital_name":        str,
  "dataset_name":         str,
  "records_received":     int,
  "records_ingested":     int,
  "records_skipped":      int,
  "duplicates_skipped":   int,
  "vector_store_updated": bool,
  "duplicate_detected":   bool,   # True if entire dataset was already ingested
  "error":                str | None,
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fields inspected (in priority order) for clinical narrative text
_TEXT_FIELDS = (
    "document_text",
    "clinical_text",
    "chief_complaint",
    "chiefcomplaint",
    "notes",
    "description",
    "discharge_summary",
    "summary",
    "diagnosis",
    "procedure",
    "lab_result",
    "medication",
    "observation",
)

# Manifest path relative to the triageguard_rag root
_MANIFEST_RELATIVE = Path("data") / "hospital_records" / "ingestion_manifest.json"

# Path to default vector store
_VECTOR_STORE_RELATIVE = Path("data") / "vector_store"


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text(record: Dict[str, Any]) -> str:
    """
    Build a clinical narrative string from a single record dict.
    Concatenates all recognisable text fields.
    """
    parts: List[str] = []

    for field in _TEXT_FIELDS:
        val = record.get(field)
        if val and isinstance(val, str) and val.strip():
            parts.append(val.strip())

    # Also absorb any remaining string fields not in the list above
    for k, v in record.items():
        if k in _TEXT_FIELDS:
            continue
        if k.lower() in {"patient_id", "stay_id", "subject_id", "encounter_id",
                         "hospital_id", "hospital_name", "source", "source_type",
                         "dataset_name", "intime", "outtime", "admission_time",
                         "discharge_time", "acuity", "disposition", "department",
                         "age", "sex", "gender"}:
            continue
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(f"{k}: {v}")

    return " | ".join(parts) if parts else ""


def _extract_patient_id(record: Dict[str, Any]) -> Optional[Union[int, str]]:
    """Try common patient-ID field names."""
    for field in ("patient_id", "subject_id", "patientid", "pat_id", "pid"):
        val = record.get(field)
        if val is not None and str(val).strip() != "":
            try:
                return int(val)
            except (TypeError, ValueError):
                return str(val).strip()
    return None


def _extract_metadata(
    record: Dict[str, Any],
    hospital_id: str,
    hospital_name: str,
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Build the metadata dict for one RAG document.
    All fields are optional; missing ones default to None / empty.
    """

    def _get(*keys, default=None):
        for k in keys:
            v = record.get(k)
            if v is not None and v != "":
                return v
        return default

    return {
        # Patient / encounter identity
        "patient_id":       _extract_patient_id(record),
        "stay_id":          _get("stay_id", "encounter_id", "visit_id"),
        "intime":           str(_get("intime", "admission_time", "event_time", default="")),

        # Clinical metadata
        "disposition":      str(_get("disposition", default="")),
        "acuity":           _to_float(_get("acuity", "triage_level")),
        "department":       str(_get("department", default="")),

        # Provenance — always set
        "source":           "hospital_provided",
        "source_type":      "hospital_historical_record",
        "hospital_id":      hospital_id,
        "hospital_name":    hospital_name,
        "dataset_name":     dataset_name,
    }


def _to_float(val: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_input(
    dataset: Union[List[Dict], "pd.DataFrame", str, Path],
) -> List[Dict]:
    """
    Accept multiple input formats and return a flat list of dicts.

    Raises
    ------
    ValueError if format is unsupported or file is missing.
    """
    # ── pandas DataFrame ────────────────────────────────────────────────────
    try:
        import pandas as pd
        if isinstance(dataset, pd.DataFrame):
            logger.info("Input is a DataFrame with %d rows.", len(dataset))
            return dataset.where(dataset.notna(), None).to_dict(orient="records")
    except ImportError:
        pass

    # ── list / tuple ────────────────────────────────────────────────────────
    if isinstance(dataset, (list, tuple)):
        logger.info("Input is a list/tuple with %d items.", len(dataset))
        return list(dataset)

    # ── file path ───────────────────────────────────────────────────────────
    path = Path(dataset) if not isinstance(dataset, Path) else dataset

    if not path.exists():
        raise ValueError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            import pandas as pd
            df = pd.read_csv(path)
            logger.info("Loaded CSV with %d rows from %s", len(df), path)
            return df.where(df.notna(), None).to_dict(orient="records")
        except ImportError:
            raise ValueError("pandas is required to load CSV files.")

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info("Loaded JSON list with %d items from %s", len(data), path)
            return data
        if isinstance(data, dict):
            logger.info("Loaded single-object JSON from %s", path)
            return [data]
        raise ValueError(f"JSON file must contain a list or object, got {type(data).__name__}.")

    if suffix == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line %d: %s", lineno, exc)
        logger.info("Loaded %d records from JSONL %s", len(records), path)
        return records

    raise ValueError(
        f"Unsupported file format: '{suffix}'. "
        "Supported: .csv, .json, .jsonl (or pass a list/DataFrame)."
    )


# ---------------------------------------------------------------------------
# Fingerprinting / manifest
# ---------------------------------------------------------------------------

def _fingerprint(records: List[Dict]) -> str:
    """
    Compute a SHA-256 fingerprint over the canonical JSON representation
    of the records list.
    """
    canonical = json.dumps(records, sort_keys=True, ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_manifest(manifest_path: Path) -> Dict:
    """Load the ingestion manifest, creating it if absent."""
    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load manifest (%s) — starting fresh.", exc)
    return {"ingested_datasets": []}


def _save_manifest(manifest: Dict, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _is_duplicate(manifest: Dict, fingerprint: str) -> bool:
    for entry in manifest.get("ingested_datasets", []):
        if entry.get("dataset_hash") == fingerprint:
            return True
    return False


# ---------------------------------------------------------------------------
# Main ingestor class
# ---------------------------------------------------------------------------

class HospitalRecordIngestor:
    """
    Validates, normalises, and ingests hospital-provided historical records
    into the existing TriageGuard RAG FAISS vector store.

    Parameters
    ----------
    vector_store_dir : Path to the directory containing index.faiss and
                       metadata.json. Defaults to the canonical RAG location.
    embedder         : An existing Embedder instance. If None, a new one is
                       created with the default model.
    manifest_path    : Path to the ingestion manifest JSON file. Defaults to
                       the canonical location inside triageguard_rag/data/.
    """

    def __init__(
        self,
        vector_store_dir: Optional[Path] = None,
        embedder=None,
        manifest_path: Optional[Path] = None,
    ) -> None:
        # Resolve paths relative to the rag root
        # This file is triageguard_rag/src/ingestion/hospital_record_ingestor.py,
        # so parents[2] is triageguard_rag/ (parents[3] would be one directory
        # above the repo root, which does not exist).
        _rag_root = Path(__file__).resolve().parents[2]  # triageguard_rag/

        if vector_store_dir is None:
            vector_store_dir = _rag_root / _VECTOR_STORE_RELATIVE
        self.vector_store_dir = vector_store_dir

        if manifest_path is None:
            manifest_path = _rag_root / _MANIFEST_RELATIVE
        self.manifest_path = manifest_path

        if embedder is None:
            from triageguard_rag.src.embeddings.embedder import Embedder
            self.embedder = Embedder()
        else:
            self.embedder = embedder

    # ------------------------------------------------------------------

    def ingest(
        self,
        hospital_id: str,
        hospital_name: str,
        dataset: Union[List[Dict], "pd.DataFrame", str, Path],
        dataset_name: str = "",
    ) -> Dict[str, Any]:
        """
        Ingest a hospital-provided dataset into the RAG vector store.

        Parameters
        ----------
        hospital_id   : Unique string identifier for the hospital.
        hospital_name : Human-readable hospital name.
        dataset       : Records in any supported format (see module docstring).
        dataset_name  : Optional label for this dataset.

        Returns
        -------
        Ingestion result dict — see module docstring for schema.
        """
        hospital_id   = str(hospital_id).strip()
        hospital_name = str(hospital_name).strip()
        dataset_name  = str(dataset_name).strip() if dataset_name else "unnamed_dataset"

        result_base = {
            "success":              False,
            "hospital_id":          hospital_id,
            "hospital_name":        hospital_name,
            "dataset_name":         dataset_name,
            "records_received":     0,
            "records_ingested":     0,
            "records_skipped":      0,
            "duplicates_skipped":   0,
            "vector_store_updated": False,
            "duplicate_detected":   False,
            "error":                None,
        }

        if not hospital_id:
            result_base["error"] = "hospital_id is required and must not be empty."
            return result_base

        if not hospital_name:
            result_base["error"] = "hospital_name is required and must not be empty."
            return result_base

        # ── 1. Load raw records ─────────────────────────────────────────────
        try:
            raw_records = _load_input(dataset)
        except (ValueError, OSError) as exc:
            result_base["error"] = f"Failed to load dataset: {exc}"
            return result_base

        result_base["records_received"] = len(raw_records)

        if not raw_records:
            result_base["error"] = "Dataset is empty — no records to ingest."
            return result_base

        # ── 2. Duplicate / re-ingestion check ──────────────────────────────
        manifest = _load_manifest(self.manifest_path)
        fingerprint = _fingerprint(raw_records)

        if _is_duplicate(manifest, fingerprint):
            logger.info(
                "Dataset '%s' from hospital '%s' was already ingested (fingerprint match). "
                "Skipping to avoid duplicates.",
                dataset_name, hospital_id,
            )
            result_base.update({
                "success":            True,
                "duplicate_detected": True,
                "records_ingested":   0,
            })
            return result_base

        # ── 3. Validate + normalise ─────────────────────────────────────────
        valid_docs: List[Dict] = []
        skipped = 0
        skipped_reasons: Dict[str, int] = {}

        for i, record in enumerate(raw_records):
            if not isinstance(record, dict):
                skipped += 1
                skipped_reasons["not_a_dict"] = skipped_reasons.get("not_a_dict", 0) + 1
                logger.debug("Record %d skipped: not a dict (type=%s).", i, type(record).__name__)
                continue

            text = _extract_text(record)
            if not text.strip():
                skipped += 1
                skipped_reasons["no_text"] = skipped_reasons.get("no_text", 0) + 1
                logger.debug("Record %d skipped: no extractable clinical text.", i)
                continue

            meta = _extract_metadata(record, hospital_id, hospital_name, dataset_name)

            valid_docs.append({
                "document_text": text,
                "metadata": meta,
            })

        result_base["records_skipped"] = skipped

        if not valid_docs:
            result_base["error"] = (
                f"All {len(raw_records)} records were invalid. "
                f"Skip reasons: {skipped_reasons}. "
                "Ensure each record has at least one text-bearing field."
            )
            return result_base

        # ── 4. Embed + append to FAISS ─────────────────────────────────────
        from triageguard_rag.src.ingestion.incremental_index import append_to_index

        try:
            n_added = append_to_index(valid_docs, self.embedder, self.vector_store_dir)
        except Exception as exc:
            logger.exception("Incremental index append failed.")
            result_base["error"] = f"Vector store update failed: {exc}"
            return result_base

        # ── 5. Update manifest ─────────────────────────────────────────────
        manifest["ingested_datasets"].append({
            "dataset_hash":     fingerprint,
            "hospital_id":      hospital_id,
            "hospital_name":    hospital_name,
            "dataset_name":     dataset_name,
            "ingested_at":      datetime.now(timezone.utc).isoformat(),
            "records_received": len(raw_records),
            "records_ingested": n_added,
            "records_skipped":  skipped,
        })
        try:
            _save_manifest(manifest, self.manifest_path)
        except OSError as exc:
            # Non-fatal — index was written; just log
            logger.warning("Could not update ingestion manifest: %s", exc)

        logger.info(
            "Ingestion complete: hospital=%s dataset=%s ingested=%d skipped=%d",
            hospital_id, dataset_name, n_added, skipped,
        )

        result_base.update({
            "success":              True,
            "records_ingested":     n_added,
            "vector_store_updated": True,
        })
        return result_base
