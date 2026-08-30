"""
facility_calibration.py
------------------------
Multi-hospital Step 4: ties a hospital_id to the subset of the existing 18
nurse-demonstration scenarios (demonstrations.py) applicable to that
hospital's actual facility configuration.

Scope note
----------
This module only SELECTS/ADAPTS scenarios and labels them with a
hospital_id. It does NOT collect nurse responses, persist anything, or
train/activate a policy — that is Step 5 (routing policy calibration).
"""

from __future__ import annotations

from typing import Any, Dict, List

from triageguard_router.policy.demonstrations import load_demonstrations
from triageguard_router.policy.schema import NurseScenario


def scenarios_for_hospital(
    hospital_id: str,
    facility_departments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    hospital_id : the hospital these scenarios are being generated for.
    facility_departments : that hospital's departments dict — e.g.
        HospitalContext.state_service.get_all() (Step 2/3 registry) or
        hospital_config.json's "departments" dict directly.

    Returns
    -------
    {
        "hospital_id": str,
        "scenario_count": int,
        "scenarios": List[NurseScenario],
    }
    """
    if not hospital_id:
        raise ValueError("hospital_id must not be empty.")

    scenarios: List[NurseScenario] = load_demonstrations(facility_departments)
    return {
        "hospital_id": hospital_id,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }
