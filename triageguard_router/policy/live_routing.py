"""
live_routing.py
----------------
Multi-hospital Step 6: connects a hospital's calibrated Bayesian policy
(Step 5 artifacts) + its live operational state (Step 2/3 registry) to an
already-computed clinical result, producing a resource-aware department
allocation.

Does NOT recompute clinical risk, reconciliation, or the clinically
preferred department — those still come from reconciler.reconcile() /
router.route() unchanged (see combined_pipeline.py). This module only
decides WHERE, given what was already decided clinically, using that
hospital's own calibrated preferences and its own real-time bed state.

If a hospital has no calibrated policy yet (no saved bayesian_policy.json),
route_with_hospital_policy() returns None so the caller falls back to the
plain clinical preference — today's pre-Step-6 behavior, unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from triageguard_router.policy import artifacts
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.features import ClinicalSignal, HospitalSignal
from triageguard_router.policy.routing_policy import RoutingPolicy


def _artifact_hospital_id(hospital_id: Optional[str]) -> Optional[str]:
    """
    "default" resolves to hospital_id=None for artifacts.py: that is the
    pre-existing fixed global path where the original single-hospital
    system's already-calibrated policy actually lives (mirrors
    hospital_registry.py's own special-casing of "default" at the state
    layer, for the same reason).
    """
    from triageguard_agent.hospital.hospital_registry import DEFAULT_HOSPITAL_ID

    if hospital_id is None or hospital_id == DEFAULT_HOSPITAL_ID:
        return None
    return hospital_id


def route_with_hospital_policy(
    reconciled: Dict[str, Any],
    xgb_output: Dict[str, Any],
    clinical_preferred_department: str,
    hospital_id: Optional[str] = None,
    rag_history_count: int = 0,
    rag_similar_count: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Returns RoutingPolicy.route()'s full result dict for this hospital, or
    None if this hospital has no calibrated policy (caller should then use
    clinical_preferred_department as both preferred and allocated, as
    combined_pipeline.py always did before this integration existed).
    """
    from triageguard_agent.hospital.hospital_load_controller import HospitalLoadController
    from triageguard_agent.hospital.hospital_registry import (
        DEFAULT_HOSPITAL_ID,
        get_default_registry,
    )

    artifact_hid = _artifact_hospital_id(hospital_id)
    if not artifacts.artifacts_exist(hospital_id=artifact_hid):
        return None

    config = PolicyConfig()
    bayesian_policy = artifacts.load_bayesian_policy(config, hospital_id=artifact_hid)

    registry = get_default_registry()
    try:
        ctx = registry.get(hospital_id or DEFAULT_HOSPITAL_ID)
    except KeyError:
        # Policy artifact exists but the hospital isn't registered for live
        # state — cannot safely route without real occupancy. Fall back.
        return None
    department_state = ctx.state_service.get_all()

    load_controller = HospitalLoadController()
    load_ratio = load_controller.calculate_load(department_state)
    operating_mode = load_controller.calculate_operating_mode(load_ratio)
    hospital_signal = HospitalSignal(
        department_state=department_state,
        operating_mode=operating_mode,
        load_ratio=load_ratio,
    )

    clinical_signal = ClinicalSignal.from_pipeline_output(
        reconciled=reconciled,
        xgb_output=xgb_output,
        preferred_department=clinical_preferred_department,
        rag_history_count=rag_history_count,
        rag_similar_count=rag_similar_count,
    )

    policy = RoutingPolicy(bayesian_policy=bayesian_policy, config=config)
    return policy.route(clinical_signal, hospital_signal)
