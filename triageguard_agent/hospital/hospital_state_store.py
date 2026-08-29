"""
hospital_state_store.py
-----------------------
In-memory hospital state store, optionally backed by a JSON file.

This is the single source of truth for all hospital operational data
within the agent process. It is NOT a database — it is a lightweight
dict with a timestamp per resource, suitable for the hackathon.

The store is initialized from hospital_config.json on first access.
All reads/writes go through HospitalStateService (never directly).
"""

from __future__ import annotations
import json
import logging
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "data" / "hospital_config.json"

# Supported resource statuses
OPEN = "OPEN"
CLOSED = "CLOSED"
RESTRICTED = "RESTRICTED"
VALID_STATUSES = frozenset({OPEN, CLOSED, RESTRICTED})


class HospitalStateStore:
    """
    Thread-safe in-memory store for hospital department state.

    Each department entry contains:
        capacity  : int   — total beds/slots
        occupied  : int   — currently occupied
        available : int   — capacity - occupied (always derived)
        status    : str   — OPEN | CLOSED | RESTRICTED
        last_updated : ISO timestamp string (UTC)

    The store also holds global config (λ thresholds, stale threshold).
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self._departments: Dict[str, Dict[str, Any]] = {}
        self._lambda_thresholds: Dict[str, float] = {
            "normal_max": 0.70,
            "high_load_max": 0.90,
        }
        self._stale_threshold_minutes: int = 30
        self._load_config(config_path or _DEFAULT_CONFIG)

    def _load_config(self, path: Path) -> None:
        """Seed state from config JSON."""
        if not path.exists():
            logger.warning("Hospital config not found at %s — using empty state.", path)
            return

        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)

        now = datetime.now(timezone.utc).isoformat()
        for dept, info in cfg.get("departments", {}).items():
            cap = int(info.get("capacity", 0))
            occ = int(info.get("occupied", 0))
            self._departments[dept] = {
                "capacity":     cap,
                "occupied":     occ,
                "available":    cap - occ,
                "status":       info.get("status", OPEN),
                "last_updated": now,
            }

        thresholds = cfg.get("lambda_thresholds", {})
        self._lambda_thresholds["normal_max"] = float(
            thresholds.get("normal_max", 0.70)
        )
        self._lambda_thresholds["high_load_max"] = float(
            thresholds.get("high_load_max", 0.90)
        )
        self._stale_threshold_minutes = int(
            cfg.get("stale_threshold_minutes", 30)
        )
        logger.info("Hospital state loaded from %s (%d departments).", path, len(self._departments))

    # ------------------------------------------------------------------
    # Public read interface
    # ------------------------------------------------------------------

    def get(self, department: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._departments.get(department)
            return deepcopy(entry) if entry else None

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return deepcopy(self._departments)

    def exists(self, department: str) -> bool:
        with self._lock:
            return department in self._departments

    def is_stale(self, department: str, threshold_minutes: Optional[int] = None) -> bool:
        """Return True if the department state hasn't been updated recently."""
        entry = self.get(department)
        if entry is None:
            return True
        threshold = threshold_minutes or self._stale_threshold_minutes
        last = entry.get("last_updated")
        if not last:
            return True
        try:
            updated_at = datetime.fromisoformat(last)
            elapsed = (datetime.now(timezone.utc) - updated_at).total_seconds() / 60
            return elapsed > threshold
        except Exception:
            return True

    @property
    def lambda_thresholds(self) -> Dict[str, float]:
        return deepcopy(self._lambda_thresholds)

    # ------------------------------------------------------------------
    # Public write interface (called only by HospitalStateService)
    # ------------------------------------------------------------------

    def apply(self, department: str, patch: Dict[str, Any]) -> None:
        """
        Apply a pre-validated patch to a department.
        Always recalculates `available` and stamps `last_updated`.
        """
        with self._lock:
            if department not in self._departments:
                raise KeyError(f"Department {department!r} not found in store.")
            entry = self._departments[department]
            entry.update(patch)
            # Recalculate available from current cap/occ
            entry["available"] = entry["capacity"] - entry["occupied"]
            # Enforce: CLOSED → available = 0
            if entry.get("status") == CLOSED:
                entry["available"] = 0
            entry["last_updated"] = datetime.now(timezone.utc).isoformat()

    def add_department(self, department: str, state: Dict[str, Any]) -> None:
        """Add a new department (admin use only)."""
        with self._lock:
            cap = int(state.get("capacity", 0))
            occ = int(state.get("occupied", 0))
            self._departments[department] = {
                "capacity":     cap,
                "occupied":     occ,
                "available":    cap - occ,
                "status":       state.get("status", OPEN),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
