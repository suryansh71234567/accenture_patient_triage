"""
hospital_state_service.py
--------------------------
Validation and persistence layer for hospital state changes.

Responsibilities
----------------
* Validate every proposed update before it touches the store.
* Enforce invariants: occupied ≤ capacity, available = capacity - occupied,
  CLOSED resources have 0 available, no negatives.
* Reject contradictory or ambiguous updates.
* Provide a process-wide singleton so all tools share the same state.

The LLM never calls this directly. It goes through hospital_tools.py.
"""

from __future__ import annotations
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from triageguard_agent.hospital.hospital_state_store import (
    HospitalStateStore,
    OPEN, CLOSED, RESTRICTED, VALID_STATUSES,
)

logger = logging.getLogger(__name__)

_MAX_REASONABLE_CAPACITY = 10_000   # sanity cap


class HospitalStateService:
    """
    Validates and applies hospital state updates.

    Use HospitalStateService.instance() to get the process-wide singleton.
    """

    _instance: Optional["HospitalStateService"] = None

    def __init__(self, store: Optional[HospitalStateStore] = None) -> None:
        self._store = store or HospitalStateStore()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls, config_path: Optional[Path] = None) -> "HospitalStateService":
        """Return (or create) the process-wide singleton."""
        if cls._instance is None:
            store = HospitalStateStore(config_path) if config_path else HospitalStateStore()
            cls._instance = cls(store)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing only)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Read interface
    # ------------------------------------------------------------------

    def get_state(self, department: str) -> Optional[Dict[str, Any]]:
        return self._store.get(department)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        return self._store.get_all()

    def is_stale(self, department: str, threshold_minutes: Optional[int] = None) -> bool:
        return self._store.is_stale(department, threshold_minutes)

    def department_exists(self, department: str) -> bool:
        return self._store.exists(department)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_update(
        self,
        department: str,
        update: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate a proposed update and return a normalised patch dict.

        Raises ValueError with a descriptive message if the update is invalid.

        Checks
        ------
        1. Department must exist.
        2. No unrecognised keys (only capacity, occupied, status allowed).
        3. No negative values.
        4. occupied ≤ capacity (using current values for any field not provided).
        5. available is always derived (never accepted from input).
        6. CLOSED status → occupied must be 0 (or set occupied=0 automatically
           only if explicitly closed with no occupancy given).
        7. Capacity cannot be reduced below current occupancy.
        """
        if not self._store.exists(department):
            raise ValueError(
                f"Department {department!r} is not in the hospital configuration. "
                f"Available departments: {list(self._store.get_all().keys())}"
            )

        allowed_keys = {"capacity", "occupied", "status"}
        bad_keys = set(update.keys()) - allowed_keys
        if bad_keys:
            raise ValueError(
                f"Unrecognised update fields: {bad_keys}. "
                f"Only {allowed_keys} are allowed."
            )

        current = self._store.get(department)
        patch: Dict[str, Any] = {}

        # --- status ---
        new_status = update.get("status", current["status"])
        if new_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {new_status!r}. Must be one of {VALID_STATUSES}."
            )
        patch["status"] = new_status

        # --- capacity ---
        if "capacity" in update:
            new_cap = _parse_int(update["capacity"], "capacity")
            if new_cap < 0:
                raise ValueError("capacity cannot be negative.")
            if new_cap > _MAX_REASONABLE_CAPACITY:
                raise ValueError(
                    f"capacity {new_cap} exceeds the sanity limit of {_MAX_REASONABLE_CAPACITY}."
                )
            patch["capacity"] = new_cap
        else:
            patch["capacity"] = current["capacity"]

        # --- occupied ---
        if "occupied" in update:
            new_occ = _parse_int(update["occupied"], "occupied")
            if new_occ < 0:
                raise ValueError("occupied cannot be negative.")
            patch["occupied"] = new_occ
        else:
            patch["occupied"] = current["occupied"]

        # --- consistency: occupied ≤ capacity ---
        if patch["occupied"] > patch["capacity"]:
            raise ValueError(
                f"occupied ({patch['occupied']}) cannot exceed "
                f"capacity ({patch['capacity']}) for {department}."
            )

        # --- consistency: capacity cannot drop below current occupancy ---
        if "capacity" in update and patch["capacity"] < current["occupied"]:
            if "occupied" not in update:
                raise ValueError(
                    f"Cannot reduce capacity to {patch['capacity']} — "
                    f"current occupancy is {current['occupied']}. "
                    "Provide an updated occupied value as well."
                )

        # --- CLOSED department must have 0 available ---
        if patch["status"] == CLOSED and patch["occupied"] > 0:
            raise ValueError(
                f"Cannot set {department} to CLOSED while occupied={patch['occupied']}. "
                "Either set occupied=0 or keep the department open."
            )

        # available is always derived — never accepted from input
        patch["available"] = patch["capacity"] - patch["occupied"]
        if patch["status"] == CLOSED:
            patch["available"] = 0

        return patch

    # ------------------------------------------------------------------
    # Write interface
    # ------------------------------------------------------------------

    def apply_update(self, department: str, validated_update: Dict[str, Any]) -> None:
        """Apply a pre-validated update to the store."""
        logger.info(
            "Applying hospital state update: department=%s update=%s",
            department, validated_update,
        )
        self._store.apply(department, validated_update)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def lambda_thresholds(self) -> Dict[str, float]:
        return self._store.lambda_thresholds


def _parse_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Field {field!r} must be an integer, got {value!r}.")
