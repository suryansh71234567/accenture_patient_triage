"""
scenarios.py
------------
Pre-configured hospital operational scenarios for TriageGuard simulation.

Scenarios define initial department occupancies, arrival rates, acuity profiles,
and default lengths of stay. They allow demonstrating that the SAME clinical
patient receives different operational recommendations based on hospital state.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Scenario:
    """Configuration definition for a hospital simulation scenario."""
    name: str
    title: str
    description: str
    arrival_rate_per_hour: float
    department_state: Dict[str, Dict[str, Any]]
    acuity_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 0.10, 2: 0.20, 3: 0.40, 4: 0.20, 5: 0.10}
    )
    mean_los_minutes: Dict[str, int] = field(
        default_factory=lambda: {
            "ICU": 90,
            "CICU": 90,
            "ADMITTED_GEN": 75,
            "ED_OBS": 45,
            "DISCHARGE": 0,
        }
    )
    surge_multiplier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "arrival_rate_per_hour": self.arrival_rate_per_hour,
            "department_state": self.department_state,
            "acuity_weights": self.acuity_weights,
            "mean_los_minutes": self.mean_los_minutes,
            "surge_multiplier": self.surge_multiplier,
        }


# ---------------------------------------------------------------------------
# Pre-built Scenarios
# ---------------------------------------------------------------------------

NORMAL_DAY = Scenario(
    name="NORMAL_DAY",
    title="Normal Day",
    description="Standard baseline hospital operating conditions with moderate volume and comfortable capacity headroom.",
    arrival_rate_per_hour=3.0,
    department_state={
        "ICU": {"capacity": 10, "occupied": 7, "status": "OPEN"},
        "CICU": {"capacity": 6, "occupied": 4, "status": "OPEN"},
        "ADMITTED_GEN": {"capacity": 50, "occupied": 36, "status": "OPEN"},
        "ED_OBS": {"capacity": 20, "occupied": 12, "status": "OPEN"},
        "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
    },
    acuity_weights={1: 0.10, 2: 0.15, 3: 0.45, 4: 0.20, 5: 0.10},
    mean_los_minutes={"ICU": 90, "CICU": 90, "ADMITTED_GEN": 60, "ED_OBS": 30, "DISCHARGE": 0},
)

BUSY_DAY = Scenario(
    name="BUSY_DAY",
    title="Busy Day",
    description="Elevated ED arrivals with high occupancy across critical and general wards (High Load operating mode).",
    arrival_rate_per_hour=6.0,
    department_state={
        "ICU": {"capacity": 10, "occupied": 8, "status": "OPEN"},
        "CICU": {"capacity": 6, "occupied": 5, "status": "OPEN"},
        "ADMITTED_GEN": {"capacity": 50, "occupied": 43, "status": "OPEN"},
        "ED_OBS": {"capacity": 20, "occupied": 16, "status": "OPEN"},
        "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
    },
    acuity_weights={1: 0.15, 2: 0.25, 3: 0.40, 4: 0.15, 5: 0.05},
    mean_los_minutes={"ICU": 105, "CICU": 105, "ADMITTED_GEN": 75, "ED_OBS": 45, "DISCHARGE": 0},
)

SURGE_MASS_CASUALTY = Scenario(
    name="SURGE_MASS_CASUALTY",
    title="Mass-Casualty / Surge",
    description="Severe surge of high-acuity patients pushing ICU, CICU, and ED Observation to near or total saturation.",
    arrival_rate_per_hour=15.0,
    department_state={
        "ICU": {"capacity": 10, "occupied": 9, "status": "OPEN"},
        "CICU": {"capacity": 6, "occupied": 6, "status": "OPEN"},
        "ADMITTED_GEN": {"capacity": 50, "occupied": 48, "status": "OPEN"},
        "ED_OBS": {"capacity": 20, "occupied": 19, "status": "OPEN"},
        "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
    },
    acuity_weights={1: 0.35, 2: 0.35, 3: 0.20, 4: 0.08, 5: 0.02},
    mean_los_minutes={"ICU": 120, "CICU": 120, "ADMITTED_GEN": 90, "ED_OBS": 60, "DISCHARGE": 0},
    surge_multiplier=2.5,
)

RESOURCE_CONSTRAINED = Scenario(
    name="RESOURCE_CONSTRAINED",
    title="Resource Constrained",
    description="Bottlenecked hospital where General ward and CICU are completely full and ICU has only 1 bed remaining.",
    arrival_rate_per_hour=4.0,
    department_state={
        "ICU": {"capacity": 10, "occupied": 9, "status": "OPEN"},
        "CICU": {"capacity": 6, "occupied": 6, "status": "OPEN"},
        "ADMITTED_GEN": {"capacity": 50, "occupied": 50, "status": "OPEN"},
        "ED_OBS": {"capacity": 20, "occupied": 18, "status": "OPEN"},
        "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
    },
    acuity_weights={1: 0.20, 2: 0.25, 3: 0.35, 4: 0.15, 5: 0.05},
    mean_los_minutes={"ICU": 90, "CICU": 90, "ADMITTED_GEN": 60, "ED_OBS": 30, "DISCHARGE": 0},
)

NIGHT_SHIFT = Scenario(
    name="NIGHT_SHIFT",
    title="Night Shift",
    description="Low ED arrival rate during overnight hours with gradual bed turnover.",
    arrival_rate_per_hour=1.5,
    department_state={
        "ICU": {"capacity": 10, "occupied": 6, "status": "OPEN"},
        "CICU": {"capacity": 6, "occupied": 3, "status": "OPEN"},
        "ADMITTED_GEN": {"capacity": 50, "occupied": 30, "status": "OPEN"},
        "ED_OBS": {"capacity": 20, "occupied": 6, "status": "OPEN"},
        "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
    },
    acuity_weights={1: 0.10, 2: 0.15, 3: 0.40, 4: 0.25, 5: 0.10},
    mean_los_minutes={"ICU": 120, "CICU": 120, "ADMITTED_GEN": 90, "ED_OBS": 45, "DISCHARGE": 0},
)


SCENARIOS: Dict[str, Scenario] = {
    "NORMAL_DAY": NORMAL_DAY,
    "BUSY_DAY": BUSY_DAY,
    "SURGE_MASS_CASUALTY": SURGE_MASS_CASUALTY,
    "RESOURCE_CONSTRAINED": RESOURCE_CONSTRAINED,
    "NIGHT_SHIFT": NIGHT_SHIFT,
}


def get_scenario(name: str) -> Scenario:
    """Fetch scenario by key (case-insensitive) or return NORMAL_DAY as fallback."""
    key = name.strip().upper()
    if key in SCENARIOS:
        return SCENARIOS[key]
    # Allow matching by title substring
    for s in SCENARIOS.values():
        if key in s.name or key in s.title.upper():
            return s
    raise KeyError(f"Unknown scenario {name!r}. Available: {list(SCENARIOS.keys())}")


def list_scenarios() -> List[Scenario]:
    """Return all available predefined scenarios."""
    return list(SCENARIOS.values())
