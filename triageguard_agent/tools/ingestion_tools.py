"""
ingestion_tools.py
------------------
Agent-facing WRITE tool for ingesting hospital-provided historical records
into the TriageGuard RAG knowledge base.

LLM-visible tool
----------------
* ingest_hospital_records — validate, normalise, embed, and append hospital
  records to the existing FAISS vector store.

Design
------
This is a WRITE tool (side_effect=True, requires_approval=True).
The nurse/operator must confirm before hospital records are committed to
the knowledge base. The runtime enforces this approval gate.

The tool wraps HospitalRecordIngestor and returns a structured ToolResult
containing the ingestion summary. It does NOT expose internal FAISS or
embedding details to the LLM.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from triageguard_agent.schemas.tool_result import ToolResult

logger = logging.getLogger(__name__)

TOOL_NAME_INGEST = "ingest_hospital_records"

# Lazy singleton ingestor (shared across calls to avoid re-loading the embedder)
_ingestor_instance = None


def _get_ingestor():
    """Return shared HospitalRecordIngestor instance (loads once)."""
    global _ingestor_instance
    if _ingestor_instance is None:
        try:
            from triageguard_rag.src.ingestion.hospital_record_ingestor import (
                HospitalRecordIngestor,
            )
            _ingestor_instance = HospitalRecordIngestor()
            logger.info("HospitalRecordIngestor loaded for ingestion tool.")
        except Exception as exc:
            logger.error("Failed to load HospitalRecordIngestor: %s", exc)
            raise
    return _ingestor_instance


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def ingest_hospital_records(
    hospital_id: str,
    hospital_name: str,
    records: List[Dict[str, Any]],
    dataset_name: str = "",
) -> ToolResult:
    """
    Ingest a list of hospital historical records into the RAG knowledge base.

    Parameters
    ----------
    hospital_id   : Unique string identifier for the hospital (e.g. "hosp_a").
    hospital_name : Human-readable name (e.g. "City General Hospital").
    records       : List of patient/clinical record dicts. Each record should
                    contain at least one text-bearing field (e.g. chiefcomplaint,
                    notes, description, document_text, clinical_text).
    dataset_name  : Optional label for this batch (e.g. "admissions_2024").

    Returns
    -------
    ToolResult with data = ingestion summary dict.
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not isinstance(hospital_id, str) or not hospital_id.strip():
        return ToolResult.fail(
            TOOL_NAME_INGEST,
            "INVALID_HOSPITAL_ID",
            "hospital_id must be a non-empty string.",
        )
    if not isinstance(hospital_name, str) or not hospital_name.strip():
        return ToolResult.fail(
            TOOL_NAME_INGEST,
            "INVALID_HOSPITAL_NAME",
            "hospital_name must be a non-empty string.",
        )
    if not isinstance(records, list):
        return ToolResult.fail(
            TOOL_NAME_INGEST,
            "INVALID_RECORDS",
            "records must be a list of dicts.",
        )
    if len(records) == 0:
        return ToolResult.fail(
            TOOL_NAME_INGEST,
            "EMPTY_RECORDS",
            "records list is empty — nothing to ingest.",
        )

    # ── Ingestion ────────────────────────────────────────────────────────
    try:
        ingestor = _get_ingestor()
        ingestion_result = ingestor.ingest(
            hospital_id=hospital_id,
            hospital_name=hospital_name,
            dataset=records,
            dataset_name=dataset_name,
        )
    except Exception as exc:
        logger.exception("ingest_hospital_records: unexpected error.")
        return ToolResult.fail(
            TOOL_NAME_INGEST,
            "INGESTOR_ERROR",
            f"Ingestion failed unexpectedly: {exc}",
        )

    # ── Map result ───────────────────────────────────────────────────────
    if not ingestion_result.get("success"):
        return ToolResult.fail(
            TOOL_NAME_INGEST,
            "INGESTION_FAILED",
            ingestion_result.get("error", "Unknown ingestion error."),
            metadata={
                "hospital_id":      hospital_id,
                "records_received": ingestion_result.get("records_received", 0),
            },
        )

    return ToolResult.ok(
        TOOL_NAME_INGEST,
        {
            "hospital_id":          ingestion_result["hospital_id"],
            "hospital_name":        ingestion_result["hospital_name"],
            "dataset_name":         ingestion_result["dataset_name"],
            "records_received":     ingestion_result["records_received"],
            "records_ingested":     ingestion_result["records_ingested"],
            "records_skipped":      ingestion_result["records_skipped"],
            "duplicates_skipped":   ingestion_result["duplicates_skipped"],
            "vector_store_updated": ingestion_result["vector_store_updated"],
            "duplicate_detected":   ingestion_result["duplicate_detected"],
        },
        metadata={
            "hospital_id": hospital_id,
            "side_effect": "vector_store_appended",
        },
    )


# ---------------------------------------------------------------------------
# ToolSpec factory
# ---------------------------------------------------------------------------

def ingest_hospital_records_spec():
    """Return the ToolSpec for the ingest_hospital_records WRITE tool."""
    from triageguard_agent.tools.registry import ToolSpec, WRITE

    return ToolSpec(
        name=TOOL_NAME_INGEST,
        description=(
            "Ingest a hospital's historical patient records into the TriageGuard RAG "
            "knowledge base. After ingestion, future triage queries can retrieve relevant "
            "history and similar cases from these records, with full hospital provenance "
            "(hospital name / ID) preserved in retrieved results. "
            "WRITE tool — requires nurse/operator confirmation before committing. "
            "Provide hospital_id (unique string), hospital_name (human-readable), "
            "records (list of patient/clinical record dicts), and an optional dataset_name."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "hospital_id": {
                    "type": "string",
                    "description": "Unique identifier for the hospital (e.g. 'hosp_001').",
                },
                "hospital_name": {
                    "type": "string",
                    "description": "Human-readable hospital name (e.g. 'City General Hospital').",
                },
                "records": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "List of patient/clinical record dicts. Each should have at least "
                        "one text field: chiefcomplaint, notes, description, document_text, "
                        "clinical_text, diagnosis, or similar."
                    ),
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Optional label for this dataset batch (e.g. 'admissions_2024').",
                },
            },
            "required": ["hospital_id", "hospital_name", "records"],
        },
        handler=lambda **kwargs: ingest_hospital_records(**kwargs),
        risk_level=WRITE,
        side_effect=True,
        requires_approval=True,
    )
