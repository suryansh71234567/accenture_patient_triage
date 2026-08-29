"""
prepare_documents.py
--------------------
Reads MIMIC-IV ED demo tables (triage, vitalsign, edstays) and converts
each ED stay into a plain-text RAG document saved as JSONL.

Output format (one JSON object per line):
{
  "document_text": "<narrative string>",
  "metadata": {
    "patient_id":   <int>,
    "stay_id":      <int>,
    "intime":       "YYYY-MM-DD HH:MM:SS",
    "disposition":  "<str>",           # e.g. ADMITTED / HOME
    "acuity":       <float|null>,
    "source":       "mimic-iv-ed"
  }
}
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, unit=""):
    """Return '<val><unit>' or 'N/A' when val is missing."""
    if pd.isna(val):
        return "N/A"
    if isinstance(val, float) and val == int(val):
        return f"{int(val)}{unit}"
    return f"{val}{unit}"


def _build_text(stay_row: pd.Series, vitals_rows: pd.DataFrame) -> str:
    """Compose a human-readable narrative for one ED stay."""
    lines = []

    # ── triage snapshot ────────────────────────────────────────────────────
    lines.append(
        f"Patient {int(stay_row['subject_id'])} arrived at the ED on "
        f"{stay_row['intime']} via {stay_row['arrival_transport']}."
    )
    lines.append(
        f"Chief complaint: {stay_row['chiefcomplaint'] if pd.notna(stay_row.get('chiefcomplaint')) else 'not recorded'}."
    )
    lines.append(
        f"Triage acuity level: {_fmt(stay_row.get('acuity'))}."
    )
    lines.append(
        f"Triage vitals — "
        f"HR {_fmt(stay_row.get('heartrate'), ' bpm')}, "
        f"RR {_fmt(stay_row.get('resprate'), ' /min')}, "
        f"SpO2 {_fmt(stay_row.get('o2sat'), '%')}, "
        f"BP {_fmt(stay_row.get('sbp'))}/{_fmt(stay_row.get('dbp'))} mmHg, "
        f"Temp {_fmt(stay_row.get('temperature'), '°F')}, "
        f"Pain {_fmt(stay_row.get('pain'), '/10')}."
    )

    # ── subsequent vitals (if any) ─────────────────────────────────────────
    if not vitals_rows.empty:
        lines.append("Subsequent vitals recorded during stay:")
        for _, vr in vitals_rows.iterrows():
            lines.append(
                f"  [{vr['charttime']}] "
                f"HR {_fmt(vr['heartrate'], ' bpm')}, "
                f"RR {_fmt(vr['resprate'], ' /min')}, "
                f"SpO2 {_fmt(vr['o2sat'], '%')}, "
                f"BP {_fmt(vr['sbp'])}/{_fmt(vr['dbp'])} mmHg, "
                f"Temp {_fmt(vr['temperature'], '°F')}, "
                f"Pain {_fmt(vr['pain'], '/10')}, "
                f"Rhythm: {vr['rhythm'] if pd.notna(vr['rhythm']) else 'N/A'}."
            )

    # ── outcome ────────────────────────────────────────────────────────────
    lines.append(f"Disposition: {stay_row['disposition']}.")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_documents(
    dataset_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Parameters
    ----------
    dataset_dir : root of the mimic-iv-ed-demo-2.2/ed/ folder.
    output_path : where to write events.jsonl.

    Returns
    -------
    Path to the written JSONL file.
    """
    repo_root = Path(__file__).resolve().parents[3]

    if dataset_dir is None:
        dataset_dir = (
            repo_root / "dataset" / "mimic-iv-ed-demo-2.2" / "ed"
        )

    if output_path is None:
        output_path = repo_root / "triageguard_rag" / "data" / "processed" / "events.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load tables ────────────────────────────────────────────────────────
    logger.info("Loading MIMIC-IV ED tables from %s", dataset_dir)
    edstays  = pd.read_csv(dataset_dir / "edstays.csv.gz")
    triage   = pd.read_csv(dataset_dir / "triage.csv.gz")
    vitals   = pd.read_csv(dataset_dir / "vitalsign.csv.gz")

    # merge triage into edstays on stay_id
    merged = edstays.merge(triage, on=["subject_id", "stay_id"], how="left")

    # ── build documents ────────────────────────────────────────────────────
    docs_written = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for _, stay in merged.iterrows():
            stay_vitals = vitals[vitals["stay_id"] == stay["stay_id"]].copy()
            if "charttime" in stay_vitals.columns:
                stay_vitals = stay_vitals.sort_values("charttime")

            text = _build_text(stay, stay_vitals)

            doc = {
                "document_text": text,
                "metadata": {
                    "patient_id":  int(stay["subject_id"]),
                    "stay_id":     int(stay["stay_id"]),
                    "intime":      str(stay["intime"]),
                    "disposition": str(stay["disposition"]),
                    "acuity":      None if pd.isna(stay.get("acuity")) else float(stay["acuity"]),
                    "source":      "mimic-iv-ed",
                },
            }
            fout.write(json.dumps(doc) + "\n")
            docs_written += 1

    logger.info("Wrote %d documents to %s", docs_written, output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    prepare_documents()
