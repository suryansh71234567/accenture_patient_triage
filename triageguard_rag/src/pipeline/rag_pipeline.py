"""
rag_pipeline.py
---------------
Top-level orchestrator: load config → embed → retrieve → reason.

Typical usage
-------------
from triageguard_rag.src.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()          # loads config and model once
result   = pipeline.run(patient_state)
print(result["response"])
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from dotenv import load_dotenv

from triageguard_rag.src.embeddings.embedder import Embedder
from triageguard_rag.src.retrieval.retriever import Retriever
from triageguard_rag.src.reasoning.llm_reasoner import reason

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Single-object interface to the full RAG + LLM reasoning pipeline.

    Parameters
    ----------
    config_path : path to config.yaml. Defaults to <repo_root>/triageguard_rag/config/config.yaml.
    vector_store_dir : where the FAISS index lives. Defaults to config location.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        vector_store_dir: Optional[Path] = None,
    ):
        # ── locate repo root ───────────────────────────────────────────────
        self.repo_root = Path(__file__).resolve().parents[4]   # goes up: pipeline → src → triageguard_rag → aic_hackathon → repo_root... wait, let's compute properly
        # Actual hierarchy: rag_pipeline.py is at
        #   <repo>/triageguard_rag/src/pipeline/rag_pipeline.py
        # parents[0] = pipeline/, [1] = src/, [2] = triageguard_rag/, [3] = aic_hackathon/
        self.rag_root = Path(__file__).resolve().parents[2]   # triageguard_rag/

        # ── load env (.env in repo root) ───────────────────────────────────
        load_dotenv(self.rag_root.parent / ".env")
        load_dotenv(self.rag_root / ".env")

        # ── load config ────────────────────────────────────────────────────
        if config_path is None:
            config_path = self.rag_root / "config" / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # ── API key: env var takes precedence over yaml ────────────────────
        self.api_key = (
            os.getenv("OPENROUTER_API_KEY")
            or self.cfg.get("openrouter", {}).get("api_key", "")
        )
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY in .env or config/config.yaml."
            )

        self.model       = self.cfg["openrouter"]["model"]
        self.temperature = self.cfg.get("llm", {}).get("temperature", 0.1)
        self.max_tokens  = self.cfg.get("llm", {}).get("max_tokens", 1000)
        self.top_k_self  = self.cfg.get("retrieval", {}).get("top_k_patient_history", 3)
        self.top_k_sim   = self.cfg.get("retrieval", {}).get("top_k_similar_cases", 5)

        # ── embedder ───────────────────────────────────────────────────────
        embed_model = self.cfg.get("embedding", {}).get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedder = Embedder(embed_model)

        # ── retriever ──────────────────────────────────────────────────────
        if vector_store_dir is None:
            vector_store_dir = self.rag_root / "data" / "vector_store"
        self.retriever = Retriever(vector_store_dir, self.embedder)

        logger.info("RAGPipeline ready.")

    # -----------------------------------------------------------------------

    def run(self, patient_state: Dict) -> Dict:
        """
        Run the full pipeline for one patient.

        Parameters
        ----------
        patient_state : dict with keys:
            patient_id (int), chiefcomplaint (str),
            acuity, heartrate, resprate, o2sat, sbp, dbp, temperature, pain

        Returns
        -------
        {
          "patient_id":      <int>,
          "prompt":          <str>,   # prompt sent to LLM
          "response":        <str>,   # LLM reasoning text
          "patient_history": [...],   # retrieved self docs
          "similar_cases":   [...],   # retrieved similar docs
        }
        """
        raw_pid = patient_state.get("patient_id", -1)
        try:
            patient_id = int(raw_pid)
        except (TypeError, ValueError):
            patient_id = str(raw_pid).strip()

        # Build a query string from current complaint + vitals
        query = (
            f"{patient_state.get('chiefcomplaint', '')} "
            f"HR {patient_state.get('heartrate')} "
            f"SpO2 {patient_state.get('o2sat')} "
            f"BP {patient_state.get('sbp')}/{patient_state.get('dbp')} "
            f"acuity {patient_state.get('acuity')}"
        )

        patient_history, similar_cases = self.retriever.retrieve(
            query_text=query,
            patient_id=patient_id,
            top_k_self=self.top_k_self,
            top_k_similar=self.top_k_sim,
        )

        result = reason(
            current_state=patient_state,
            patient_history=patient_history,
            similar_cases=similar_cases,
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return {
            "patient_id":       patient_id,
            "prompt":           result["prompt"],
            "response":         result["response"],
            "structured_output": result["structured_output"],
            "patient_history":  patient_history,
            "similar_cases":    similar_cases,
        }
