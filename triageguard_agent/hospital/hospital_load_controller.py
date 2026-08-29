"""
hospital_load_controller.py
----------------------------
Deterministic λ (lambda) and operating mode calculator.

Rules
-----
* The agent NEVER decides λ.
* λ is a pure function of the hospital's overall occupancy ratio.
* Thresholds are configurable in hospital_config.json — not hardcoded here.
* The formula uses a stepped approach:
    load_ratio < normal_max   → NORMAL    → λ = load_ratio
    load_ratio < high_max     → HIGH_LOAD → λ = 0.5 + (load_ratio - normal_max) / (high_max - normal_max) * 0.3
    load_ratio >= high_max    → CRITICAL  → λ = 0.8 + (load_ratio - high_max) / (1 - high_max) * 0.2
* λ ∈ [0, 1] where higher λ = more conservative triage (prefer admission/ICU).

Operating modes
---------------
NORMAL   : < 70% overall occupancy (configurable)
HIGH_LOAD: 70–90%
CRITICAL : > 90%
"""

from __future__ import annotations
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Departments excluded from load calculation (e.g. DISCHARGE is always "free")
_EXCLUDED_DEPARTMENTS = frozenset({"DISCHARGE"})


class HospitalLoadController:
    """
    Recalculates operating mode and λ from the current hospital state.

    Parameters
    ----------
    normal_max    : Occupancy ratio below which hospital is NORMAL (default 0.70).
    high_load_max : Occupancy ratio below which hospital is HIGH_LOAD (default 0.90).
    """

    def __init__(
        self,
        normal_max: float = 0.70,
        high_load_max: float = 0.90,
    ) -> None:
        if not (0 < normal_max < high_load_max <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < normal_max < high_load_max ≤ 1. "
                f"Got normal_max={normal_max}, high_load_max={high_load_max}."
            )
        self.normal_max = normal_max
        self.high_load_max = high_load_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_load(self, hospital_state: Dict[str, Any]) -> float:
        """
        Compute the overall hospital occupancy ratio.

        Excludes DISCHARGE and CLOSED departments from the ratio.
        Returns 0.0 if no capacity is available (safe default).
        """
        total_capacity = 0
        total_occupied = 0

        for dept, state in hospital_state.items():
            if dept in _EXCLUDED_DEPARTMENTS:
                continue
            if state.get("status") == "CLOSED":
                continue
            cap = int(state.get("capacity", 0))
            occ = int(state.get("occupied", 0))
            total_capacity += cap
            total_occupied += occ

        if total_capacity == 0:
            logger.warning("No active capacity found — defaulting load_ratio to 0.0.")
            return 0.0

        ratio = total_occupied / total_capacity
        return round(min(ratio, 1.0), 4)

    def calculate_operating_mode(self, load_ratio: float) -> str:
        """Map load ratio to an operating mode string."""
        if load_ratio < self.normal_max:
            return "NORMAL"
        elif load_ratio < self.high_load_max:
            return "HIGH_LOAD"
        else:
            return "CRITICAL"

    def calculate_lambda(self, load_ratio: float) -> float:
        """
        Compute λ from the load ratio using a stepped linear formula.

        NORMAL    : λ scales linearly from 0 → normal_max
        HIGH_LOAD : λ scales linearly from 0.50 → 0.80
        CRITICAL  : λ scales linearly from 0.80 → 1.00
        """
        r = max(0.0, min(load_ratio, 1.0))

        if r < self.normal_max:
            # NORMAL: λ ∈ [0, 0.50]
            lam = (r / self.normal_max) * 0.50
        elif r < self.high_load_max:
            # HIGH_LOAD: λ ∈ [0.50, 0.80]
            span = self.high_load_max - self.normal_max
            progress = (r - self.normal_max) / span if span > 0 else 1.0
            lam = 0.50 + progress * 0.30
        else:
            # CRITICAL: λ ∈ [0.80, 1.00]
            span = 1.0 - self.high_load_max
            progress = (r - self.high_load_max) / span if span > 0 else 1.0
            lam = 0.80 + progress * 0.20

        return round(lam, 4)

    def recalculate(self, hospital_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full recalculation from current hospital state.

        Returns
        -------
        {
            "load_ratio":      float,
            "operating_mode":  str,   # NORMAL | HIGH_LOAD | CRITICAL
            "lambda":          float, # λ ∈ [0, 1]
            "normal_max":      float,
            "high_load_max":   float,
        }
        """
        load_ratio = self.calculate_load(hospital_state)
        operating_mode = self.calculate_operating_mode(load_ratio)
        lam = self.calculate_lambda(load_ratio)

        logger.info(
            "Load recalculated: ratio=%.3f mode=%s λ=%.4f",
            load_ratio, operating_mode, lam,
        )

        return {
            "load_ratio":    load_ratio,
            "operating_mode": operating_mode,
            "lambda":        lam,
            "normal_max":    self.normal_max,
            "high_load_max": self.high_load_max,
        }

    @classmethod
    def from_config(cls, thresholds: Dict[str, float]) -> "HospitalLoadController":
        """Construct from a config dict (e.g. from HospitalStateStore.lambda_thresholds)."""
        return cls(
            normal_max=thresholds.get("normal_max", 0.70),
            high_load_max=thresholds.get("high_load_max", 0.90),
        )
