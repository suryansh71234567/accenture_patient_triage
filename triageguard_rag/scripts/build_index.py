"""
scripts/build_index.py
----------------------
CLI entry point: reads data/processed/events.jsonl,
embeds all documents, and saves a flat FAISS index to
data/vector_store/.
"""

import json
import logging
import sys
from pathlib import Path

# Make the workspace root importable (so triageguard_rag is a package)
repo_root = Path(__file__).resolve().parents[2]   # aic_hackathon/
sys.path.insert(0, str(repo_root))

from triageguard_rag.src.embeddings.embedder import Embedder
from triageguard_rag.src.retrieval.retriever import build_index

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    rag_root         = repo_root / "triageguard_rag"
    events_path      = rag_root / "data" / "processed" / "events.jsonl"
    vector_store_dir = rag_root / "data" / "vector_store"

    if not events_path.exists():
        logger.error(
            "events.jsonl not found at %s.\n"
            "Run scripts/prepare_documents.py first.",
            events_path,
        )
        sys.exit(1)

    logger.info("Loading documents from %s …", events_path)
    documents = []
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))

    logger.info("Loaded %d documents.", len(documents))

    # Load embedding model (uses config default)
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")

    build_index(documents, embedder, vector_store_dir)

    print(f"\nOK FAISS index built with {len(documents)} documents.")
    print(f"  index.faiss   -> {vector_store_dir / 'index.faiss'}")
    print(f"  metadata.json -> {vector_store_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
