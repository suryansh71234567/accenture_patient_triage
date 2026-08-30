"""
reward.py
---------
Transparent, configurable reward function (Phase 10).

Every term below is a real, named component of RewardWeights (config.py) —
nothing is hidden inside arithmetic. Safety-scale penalties
(unsafe_downgrade, failure_to_allocate_critical, clinically_incompatible_allocation)
are an order of magnitude (or more) larger than every efficiency term, by
construction of the default config, so RL training can never learn to trade
safety for throughput (see tests/test_routing_policy.py's reward-dominance test).
"""

from __future__ import annotations

from typing import Dict, Tuple

from triageguard_router.policy.candidates import acuity_tier
from triageguard_router.policy.config import RewardWeights
from triageguard_router.policy.features import ClinicalSignal
from triageguard_router.policy.safety import AllocationDecision, is_clinically_compatible

# Thresholds used only to decide which shaping term applies — not clinical
# thresholds themselves (those live in router.py and are untouched here).
_HIGH_PRIORITY_THRESHOLD = 0.6
_LOW_PRIORITY_THRESHOLD = 0.35
_UNSAFE_DOWNGRADE_TIER_GAP = 2
_SHORT_WAIT_MINUTES = 15
_LONG_WAIT_MINUTES = 60


def compute_reward(
    clinical: ClinicalSignal,
    allocation: AllocationDecision,
    wait_minutes: float,
    max_step_down_tiers: int,
    weights: RewardWeights,
) -> Tuple[float, Dict[str, float]]:
    """Returns (total_reward, breakdown) — breakdown keys match RewardWeights fields."""
    breakdown: Dict[str, float] = {}

    if allocation.allocated_department is None:
        breakdown["resource_conflict"] = weights.resource_conflict
        if clinical.clinical_priority >= _HIGH_PRIORITY_THRESHOLD:
            breakdown["failure_to_allocate_critical"] = weights.failure_to_allocate_critical
        return sum(breakdown.values()), breakdown

    allocated = allocation.allocated_department
    preferred = allocation.preferred_department

    if not is_clinically_compatible(allocated, preferred, max_step_down_tiers):
        breakdown["clinically_incompatible_allocation"] = weights.clinically_incompatible_allocation

    tier_gap = acuity_tier(allocated) - acuity_tier(preferred)
    if clinical.clinical_priority >= _HIGH_PRIORITY_THRESHOLD and tier_gap >= _UNSAFE_DOWNGRADE_TIER_GAP:
        breakdown["unsafe_downgrade"] = weights.unsafe_downgrade

    if allocated == preferred:
        breakdown["clinically_appropriate_destination"] = weights.clinically_appropriate_destination
    breakdown["successful_allocation"] = weights.successful_allocation

    if wait_minutes <= _SHORT_WAIT_MINUTES:
        breakdown["reduced_waiting_time"] = weights.reduced_waiting_time
    elif wait_minutes > _LONG_WAIT_MINUTES and clinical.clinical_priority >= _HIGH_PRIORITY_THRESHOLD:
        breakdown["excessive_high_priority_waiting"] = weights.excessive_high_priority_waiting

    # Reward avoiding critical-care beds for patients who didn't clinically need them.
    if acuity_tier(allocated) > 0 or clinical.clinical_priority < _LOW_PRIORITY_THRESHOLD:
        breakdown["reduced_unnecessary_icu_usage"] = weights.reduced_unnecessary_icu_usage

    breakdown["throughput"] = weights.throughput

    return sum(breakdown.values()), breakdown
