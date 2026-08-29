"""
retriever.py
------------
Builds and queries a FAISS flat index over clinical event documents.

Index files on disk:
  <vector_store_dir>/index.faiss   – FAISS IndexFlatL2
  <vector_store_dir>/metadata.json – list of document dicts in index order
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from triageguard_rag.src.embeddings.embedder import Embedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(
    documents: List[Dict],
    embedder: Embedder,
    vector_store_dir: Path,
) -> None:
    """
    Embed all documents and save a flat FAISS index + metadata to disk.

    Parameters
    ----------
    documents      : list of dicts with keys 'document_text' and 'metadata'.
    embedder       : Embedder instance.
    vector_store_dir : directory where index.faiss and metadata.json are saved.
    """
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    texts = [d["document_text"] for d in documents]
    logger.info("Embedding %d documents …", len(texts))
    vectors = embedder.embed(texts)  # (N, dim) float32

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    index_path = vector_store_dir / "index.faiss"
    meta_path  = vector_store_dir / "metadata.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    logger.info(
        "Saved FAISS index (%d vectors, dim=%d) to %s",
        index.ntotal, dim, index_path,
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Load a pre-built FAISS index and retrieve documents for a query.

    Usage
    -----
    retriever = Retriever(vector_store_dir, embedder)
    self_docs, similar_docs = retriever.retrieve(
        query_text, patient_id, top_k_self=3, top_k_similar=5
    )
    """

    def __init__(self, vector_store_dir: Path, embedder: Embedder):
        index_path = vector_store_dir / "index.faiss"
        meta_path  = vector_store_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run scripts/build_index.py first."
            )

        self.index    = faiss.read_index(str(index_path))
        self.embedder = embedder

        with open(meta_path, encoding="utf-8") as f:
            self.documents: List[Dict] = json.load(f)

        logger.info(
            "Loaded FAISS index with %d vectors from %s",
            self.index.ntotal, index_path,
        )

    def retrieve(
        self,
        query_text: str,
        patient_id: int,
        top_k_self: int = 3,
        top_k_similar: int = 5,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Retrieve the most relevant documents.

        Returns
        -------
        (patient_history, similar_cases)
          patient_history : up to `top_k_self` docs belonging to `patient_id`,
                            ranked by similarity to `query_text`.
          similar_cases   : up to `top_k_similar` docs from *other* patients,
                            ranked by similarity to `query_text`.
        """
        query_vec = self.embedder.embed([query_text])  # (1, dim)

        # Retrieve enough candidates to fill both buckets
        k_candidates = min(
            (top_k_self + top_k_similar) * 10,
            self.index.ntotal,
        )
        _, indices = self.index.search(query_vec, k_candidates)

        patient_history: List[Dict] = []
        similar_cases:   List[Dict] = []

        for idx in indices[0]:
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            pid = doc["metadata"].get("patient_id")

            if pid == patient_id:
                if len(patient_history) < top_k_self:
                    patient_history.append(doc)
            else:
                if len(similar_cases) < top_k_similar:
                    similar_cases.append(doc)

            if (
                len(patient_history) >= top_k_self
                and len(similar_cases) >= top_k_similar
            ):
                break

        logger.info(
            "Retrieved %d patient-history docs and %d similar-case docs",
            len(patient_history), len(similar_cases),
        )
        return patient_history, similar_cases
