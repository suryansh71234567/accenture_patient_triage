"""
incremental_index.py
--------------------
Helpers to append new documents to an existing FAISS IndexFlatL2
and its companion metadata.json without rebuilding from scratch.

Atomicity
---------
Updates are written to temporary files and then renamed over the
originals so a crash mid-write cannot leave the index in a
partially-written / inconsistent state.

Usage
-----
from triageguard_rag.src.ingestion.incremental_index import append_to_index

n_added = append_to_index(new_documents, embedder, vector_store_dir)
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from triageguard_rag.src.embeddings.embedder import Embedder

logger = logging.getLogger(__name__)

# Sentinel source dimension check value (unused vector slot)
_SENTINEL = -1


def load_index(vector_store_dir: Path):
    """
    Load the existing FAISS index and metadata list.

    Returns
    -------
    (faiss.Index, List[Dict])
    """
    index_path = vector_store_dir / "index.faiss"
    meta_path = vector_store_dir / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/build_index.py first to create the base index."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {meta_path}."
        )

    index = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        documents: List[Dict] = json.load(f)

    if index.ntotal != len(documents):
        raise ValueError(
            f"Index/metadata mismatch: FAISS has {index.ntotal} vectors "
            f"but metadata.json has {len(documents)} entries. "
            "Run build_index.py to rebuild a clean index."
        )

    logger.info(
        "Loaded existing index with %d vectors from %s",
        index.ntotal, index_path,
    )
    return index, documents


def append_to_index(
    new_documents: List[Dict],
    embedder: Embedder,
    vector_store_dir: Path,
) -> int:
    """
    Embed *new_documents* and append them to the existing FAISS index
    and metadata.json on disk.

    Parameters
    ----------
    new_documents    : list of dicts with keys 'document_text' and 'metadata'.
    embedder         : Existing Embedder instance (not re-loaded).
    vector_store_dir : Directory containing index.faiss + metadata.json.

    Returns
    -------
    Number of vectors added.
    """
    if not new_documents:
        logger.warning("append_to_index called with empty document list — nothing to do.")
        return 0

    index, existing_docs = load_index(vector_store_dir)

    texts = [d["document_text"] for d in new_documents]
    logger.info("Embedding %d new documents for incremental append …", len(texts))
    vectors = embedder.embed(texts)  # (N, dim) float32

    # Dimension consistency check
    if vectors.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: new vectors have dim {vectors.shape[1]} "
            f"but existing index expects dim {index.d}."
        )

    # Append vectors to in-memory index
    index.add(vectors)

    # Merge document lists
    merged_docs = existing_docs + new_documents

    # ── Atomic write ────────────────────────────────────────────────────────
    index_path = vector_store_dir / "index.faiss"
    meta_path = vector_store_dir / "metadata.json"

    tmp_index = index_path.with_suffix(".faiss.tmp")
    tmp_meta = meta_path.with_suffix(".json.tmp")

    try:
        faiss.write_index(index, str(tmp_index))
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(merged_docs, f, ensure_ascii=False, indent=2)

        # Atomic rename
        shutil.move(str(tmp_index), str(index_path))
        shutil.move(str(tmp_meta), str(meta_path))

    except Exception:
        # Clean up temp files on failure
        for p in (tmp_index, tmp_meta):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        raise

    logger.info(
        "Incremental append complete: %d → %d total vectors in %s",
        len(existing_docs), index.ntotal, index_path,
    )
    return len(new_documents)
