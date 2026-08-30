"""
candidates.py
-------------
Department acuity ordering and candidate-action generation (Phase 8).

Reads department names from whatever the current HospitalStateService
reports (see triageguard_agent/hospital/hospital_state_store.py) rather than
assuming a fixed count — but needs an ACUITY ORDER to know which departments
are safe step-downs from a clinically preferred one. That order matches
triageguard_router/router.py's own docstring
("CICU, ICU, ADMITTED_GEN, ED_OBS, DISCHARGE, descending acuity") — it is
not a new taxonomy, just that same ordering made explicit as data.

If a department appears in hospital state that isn't in _KNOWN_ACUITY_ORDER,
it is treated as tied with ADMITTED_GEN's tier (a safe, conservative
mid-acuity default) rather than silently excluded — see
_tier_for_department().
"""

from __future__ import annotations

from typing import Dict, Iterable, List

# Descending acuity order (tier 0 = most acute). CICU/ICU share a tier: they
# are both "critical care," differentiated by cardiac vs non-cardiac
# presentation (router.py::_is_cardiac), not by resource step-down logic.
_KNOWN_ACUITY_ORDER: List[List[str]] = [
    ["CICU", "ICU"],
    ["ADMITTED_GEN"],
    ["ED_OBS"],
    ["DISCHARGE"],
]

DISCHARGE_DEPARTMENT = "DISCHARGE"

_TIER_BY_DEPT: Dict[str, int] = {
    dept: tier for tier, group in enumerate(_KNOWN_ACUITY_ORDER) for dept in group
}
_FALLBACK_TIER = _TIER_BY_DEPT["ADMITTED_GEN"]
_DISCHARGE_TIER = _TIER_BY_DEPT[DISCHARGE_DEPARTMENT]

# Same-tier sibling fallbacks are intentionally ONE-DIRECTIONAL, not derived
# from _KNOWN_ACUITY_ORDER's grouping: a cardiac-critical patient whose CICU
# is full can safely flex into general ICU under cardiology consult (this
# direction is real and appears in the nurse demonstrations, see S05), but
# CICU is a small, specialized cardiac unit — a non-cardiac ICU-preferred
# patient must NEVER be offered CICU as a "sibling" fallback just because
# they share an acuity tier. Getting this backwards was caught by a smoke
# test: it caused the fitted policy to route a non-cardiac sepsis patient to
# CICU purely because CICU's one-hot weight generalized well from cardiac
# training examples it was never actually a candidate in.
_SIBLING_FALLBACKS: Dict[str, List[str]] = {
    "CICU": ["ICU"],
}


def acuity_tier(department: str) -> int:
    """Lower is more acute. Unknown departments default to the ADMITTED_GEN tier."""
    return _TIER_BY_DEPT.get(department, _FALLBACK_TIER)


def candidate_departments(
    preferred_department: str,
    available_departments: Iterable[str],
    max_step_down_tiers: int = 3,
) -> List[str]:
    """
    The candidate ACTION SET for the routing policy (Phase 8): the clinically
    preferred department, plus every department strictly less acute than it
    (up to `max_step_down_tiers` tiers down), ordered most-to-least acute.

    DISCHARGE is only ever a candidate when it IS the preferred department —
    it is never offered as a resource-driven "fallback" for a patient whose
    real clinical need is higher, which is exactly the "unsafe downgrade"
    the safety layer (safety.py) must never allow.

    A one-directional SIBLING fallback (see _SIBLING_FALLBACKS above) is
    included as a gap=0 candidate when the preferred department has one:
    a cardiac patient whose CICU is full can go to general ICU, not
    straight to a general ward. This is intentionally NOT symmetric.
    """
    available = set(available_departments)
    preferred_tier = acuity_tier(preferred_department)

    if preferred_department == DISCHARGE_DEPARTMENT:
        return [DISCHARGE_DEPARTMENT] if DISCHARGE_DEPARTMENT in available else []

    candidates = [preferred_department] if preferred_department in available else []
    for sibling in _SIBLING_FALLBACKS.get(preferred_department, []):
        if sibling in available and sibling not in candidates:
            candidates.append(sibling)
    for dept in available:
        if dept in candidates or dept == DISCHARGE_DEPARTMENT:
            continue
        tier = acuity_tier(dept)
        if preferred_tier < tier <= preferred_tier + max_step_down_tiers:
            candidates.append(dept)

    candidates.sort(key=acuity_tier)
    # Keep preferred_department first even if tier-sorting ties it with a sibling.
    if preferred_department in candidates:
        candidates.remove(preferred_department)
        candidates.insert(0, preferred_department)
    return candidates
