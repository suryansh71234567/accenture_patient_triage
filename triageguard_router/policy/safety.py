"""
safety.py
---------
Hard safety / feasibility layer (Phase 9 & 12) — the boundary between
CLINICAL PRIORITY and RESOURCE-AWARE ALLOCATION.

Nothing in this module can change clinical_priority or preferred_department.
It only decides, given a FIXED clinical preference and the current real
hospital state, which departments a patient could physically be allocated
to right now, and enforces that an "unsafe downgrade" (skipping straight to
a department far below what's clinically acceptable, or DISCHARGE, purely
because something else is full) is never produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from triageguard_router.policy.candidates import (
    DISCHARGE_DEPARTMENT,
    _SIBLING_FALLBACKS,
    acuity_tier,
    candidate_departments,
)


@dataclass
class FeasibilityResult:
    candidates: List[str]
    feasible: List[str]
    infeasible: List[str]
    action_mask: np.ndarray          # bool, aligned with `candidates`
    resource_conflict: bool          # True: no feasible department exists at all


def is_clinically_compatible(
    department: str,
    preferred_department: str,
    max_step_down_tiers: int,
) -> bool:
    """
    Defense-in-depth check (used by both the runtime allocator below and the
    RL reward function): true only if `department` is the preferred one, its
    one-directional sibling fallback (see candidates._SIBLING_FALLBACKS —
    e.g. CICU -> ICU, never the reverse), or a safe step-down of it under
    the same rule candidate_departments() uses. DISCHARGE is compatible only
    when it IS the preference itself.
    """
    if department == preferred_department:
        return True
    if department == DISCHARGE_DEPARTMENT:
        return False
    if department in _SIBLING_FALLBACKS.get(preferred_department, []):
        return True
    gap = acuity_tier(department) - acuity_tier(preferred_department)
    return 0 < gap <= max_step_down_tiers


def compute_feasibility(
    candidates: List[str],
    hospital_state: Dict[str, Dict],
) -> FeasibilityResult:
    """
    Physical/operational feasibility: OPEN status and at least one available
    bed (DISCHARGE is always feasible when it is itself the only candidate,
    since it does not consume a bed).
    """
    feasible: List[str] = []
    infeasible: List[str] = []
    for dept in candidates:
        state = hospital_state.get(dept)
        if state is None:
            infeasible.append(dept)
            continue
        if state.get("status", "OPEN") != "OPEN":
            infeasible.append(dept)
            continue
        if dept == DISCHARGE_DEPARTMENT:
            feasible.append(dept)
            continue
        if int(state.get("available", 0)) > 0:
            feasible.append(dept)
        else:
            infeasible.append(dept)

    mask = np.array([d in feasible for d in candidates], dtype=bool)
    return FeasibilityResult(
        candidates=list(candidates),
        feasible=feasible,
        infeasible=infeasible,
        action_mask=mask,
        resource_conflict=(len(feasible) == 0),
    )


@dataclass
class AllocationDecision:
    preferred_department: str
    allocated_department: Optional[str]   # None only on resource_conflict
    resource_constraint: bool
    human_escalation: bool
    feasibility: FeasibilityResult


def allocate(
    preferred_department: str,
    candidates: List[str],
    hospital_state: Dict[str, Dict],
    department_scores: Dict[str, float],
) -> AllocationDecision:
    """
    The hard-constrained decision (Phase 9):

        clinical assessment -> policy scores -> hard safety constraints
        -> resource feasibility -> allocation

    1. If the clinically preferred department is physically feasible right
       now, allocate it — resource_constraint=False.
    2. Otherwise, among the remaining FEASIBLE, clinically-acceptable
       candidates, allocate whichever the policy scored highest —
       resource_constraint=True.
    3. If no candidate is feasible at all, do NOT fabricate a safe
       allocation: return allocated_department=None and human_escalation=True
       (Phase 17 Scenario C).
    """
    feasibility = compute_feasibility(candidates, hospital_state)

    if preferred_department in feasibility.feasible:
        return AllocationDecision(
            preferred_department=preferred_department,
            allocated_department=preferred_department,
            resource_constraint=False,
            human_escalation=False,
            feasibility=feasibility,
        )

    if feasibility.feasible:
        best = max(feasibility.feasible, key=lambda d: department_scores.get(d, float("-inf")))
        return AllocationDecision(
            preferred_department=preferred_department,
            allocated_department=best,
            resource_constraint=True,
            human_escalation=False,
            feasibility=feasibility,
        )

    return AllocationDecision(
        preferred_department=preferred_department,
        allocated_department=None,
        resource_constraint=True,
        human_escalation=True,
        feasibility=feasibility,
    )
