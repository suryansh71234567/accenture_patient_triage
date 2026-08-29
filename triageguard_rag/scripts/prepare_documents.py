"""
scripts/prepare_documents.py
----------------------------
CLI entry point: reads MIMIC-IV ED tables → writes data/processed/events.jsonl
"""

import logging
import sys
from pathlib import Path

# Insert aic_hackathon/ (parent of triageguard_rag/) so the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triageguard_rag.src.data.prepare_documents import prepare_documents

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

if __name__ == "__main__":
    out = prepare_documents()
    print(f"\nOK Documents written to: {out}")
