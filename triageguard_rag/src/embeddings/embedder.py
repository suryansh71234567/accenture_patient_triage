"""
embedder.py
-----------
Thin wrapper around sentence-transformers to produce L2-normalised
dense embeddings suitable for FAISS IndexFlatL2.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a SentenceTransformer model for batch embedding."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dimension)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray of shape (N, dim), dtype float32, L2-normalised.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,   # L2 normalise → cosine ≈ dot product
            convert_to_numpy=True,
        ).astype(np.float32)

        return vecs
