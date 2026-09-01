"""
demonstrations.py
------------------
Fixed nurse/staff expert demonstrations (Phase 2) — 18 hand-designed
scenarios, NOT randomly generated and NOT LLM-authored. Each `reason` string
is a fixed expert annotation, exactly as the schema requires.

clinical_preferred_department is kept consistent with the REAL clinical
router's own thresholds (triageguard_router/router.py:
ICU_RISK_THRESHOLD=0.35, ADMISSION_THRESHOLD=0.50, OBS_THRESHOLD=0.30, and
the cardiac-keyword CICU/ICU split) and stays CONSTANT across a resource-
availability ladder. preferred_department is the nurse's actual allocation
choice for that specific scenario's resource situation — the behavior-
cloning training label — and only diverges from clinical_preferred_department
when the clinically preferred department is unavailable (see schema.py's
NurseScenario docstring for the exact semantics).

Design (matches the required coverage axes):
  * urgency: low / moderate / high / critical
  * ICU: available / moderately occupied / nearly full / full
  * ADMITTED_GEN ("step-down 1"): available / full
  * ED_OBS ("step-down 2"): available
  * model agreement: agree / strong disagreement
  * confidence: high / low
  * explicit PAIRED ladders where only hospital occupancy changes between
    scenarios sharing the identical clinical state (tags "paired:*"):
      S01 -> S02 -> S03   (ICU ladder: ICU -> ADMITTED_GEN -> ED_OBS)
      S04 -> S05 -> S06   (CICU ladder: CICU -> ICU sibling -> ADMITTED_GEN)
      S07 -> S08          (ADMITTED_GEN -> ED_OBS)
      S11 -> S12          (disagreement-driven ICU escalation under resource pressure)
      S13 -> S14          (low-confidence ICU case under resource pressure)
      S17, S18            (never pre-emptively downgrade just because ONE bed remains)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from triageguard_router.policy.candidates import acuity_tier
from triageguard_router.policy.schema import ClinicalState, DepartmentState, NurseScenario

# The 5 department roles every baseline scenario template is written in
# terms of (see _hospital() below) — distinct from a specific hospital's own
# department names, which the onboarding flow lets a nurse choose freely.
_CANONICAL_DEPARTMENT_CODES = ["CICU", "ICU", "ADMITTED_GEN", "ED_OBS", "DISCHARGE"]


def _hospital(
    icu=(10, 8), cicu=(6, 4), gen=(50, 38), obs=(20, 12),
    icu_status="OPEN", cicu_status="OPEN", gen_status="OPEN", obs_status="OPEN",
):
    """Shorthand: (capacity, occupied) per department, DISCHARGE always open/unlimited."""
    return {
        "ICU": DepartmentState(capacity=icu[0], occupied=icu[1], status=icu_status),
        "CICU": DepartmentState(capacity=cicu[0], occupied=cicu[1], status=cicu_status),
        "ADMITTED_GEN": DepartmentState(capacity=gen[0], occupied=gen[1], status=gen_status),
        "ED_OBS": DepartmentState(capacity=obs[0], occupied=obs[1], status=obs_status),
        "DISCHARGE": DepartmentState(capacity=999, occupied=0, status="OPEN"),
    }


def _all_scenarios() -> List[NurseScenario]:
    """The original, unmodified 18 hand-designed scenarios (Phase 2). Builds
    a fresh list of fresh objects on every call — safe to filter/adapt the
    result without touching shared state."""
    scenarios: List[NurseScenario] = []

    # ================================================================
    # Ladder 1 — non-cardiac ICU-level patient (S01 -> S02 -> S03)
    # ================================================================
    shared_cs_1 = dict(
        icu_risk_2h=0.55, icu_risk_6h=0.62, icu_risk_12h=0.58, admission_risk=0.82,
        xgb_confidence=0.85, information_completeness=0.95,
        rag_urgency="urgent", rag_evidence_strength=3, branches_agree=True,
        top_diagnoses=["sepsis", "pneumonia"], red_flags=["tachycardia", "hypoxia"], cardiac=False,
    )
    scenarios.append(NurseScenario(
        scenario_id="S01_icu_available",
        description="Non-cardiac ICU-level sepsis patient; ICU has open beds.",
        clinical_state=ClinicalState(**shared_cs_1),
        hospital_state=_hospital(icu=(10, 7)),
        candidate_departments=["ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ICU",
        preferred_department="ICU",
        reason="ICU-level risk and ICU has open beds — admit directly to ICU.",
        tags=["paired:icu_ladder", "urgency:high", "icu:available", "confidence:high", "agreement:agree"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S02_icu_full_gen_available",
        description="Same patient as S01, but ICU is now completely full; general ward has room.",
        clinical_state=ClinicalState(**shared_cs_1),
        hospital_state=_hospital(icu=(10, 10)),
        candidate_departments=["ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ICU",
        preferred_department="ADMITTED_GEN",
        reason="ICU is full. General ward provides an appropriate step-down with close monitoring while a bed opens.",
        acceptable_departments=["ADMITTED_GEN", "ICU"],
        unacceptable_departments=["DISCHARGE"],
        tags=["paired:icu_ladder", "urgency:high", "icu:full", "hdu:available", "confidence:high"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S03_icu_and_gen_full_obs_available",
        description="Same patient again; ICU AND general ward are both full; ED observation has room.",
        clinical_state=ClinicalState(**shared_cs_1),
        hospital_state=_hospital(icu=(10, 10), gen=(50, 50)),
        candidate_departments=["ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ICU",
        preferred_department="ED_OBS",
        reason="ICU and general ward are both saturated. ED observation with 1:1 monitoring is the only remaining safe option.",
        acceptable_departments=["ED_OBS", "ICU"],
        unacceptable_departments=["DISCHARGE"],
        tags=["paired:icu_ladder", "urgency:high", "icu:full", "hdu:full", "ward:available"],
    ))

    # ================================================================
    # Ladder 2 — cardiac CICU-level patient (S04 -> S05 -> S06)
    # ================================================================
    shared_cs_2 = dict(
        icu_risk_2h=0.85, icu_risk_6h=0.88, icu_risk_12h=0.80, admission_risk=0.95,
        xgb_confidence=0.90, information_completeness=1.0,
        rag_urgency="emergent", rag_evidence_strength=4, branches_agree=True,
        top_diagnoses=["STEMI", "ACS"], red_flags=["crushing chest pain", "ST elevation"], cardiac=True,
    )
    scenarios.append(NurseScenario(
        scenario_id="S04_cicu_available",
        description="STEMI patient; CICU has open beds.",
        clinical_state=ClinicalState(**shared_cs_2),
        hospital_state=_hospital(cicu=(6, 3)),
        candidate_departments=["CICU", "ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="CICU",
        preferred_department="CICU",
        reason="Active STEMI — direct to CICU for cath-lab-adjacent cardiac monitoring.",
        tags=["paired:cicu_ladder", "urgency:critical", "icu:available", "confidence:high", "agreement:agree"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S05_cicu_full_icu_available",
        description="Same STEMI patient; CICU is full, general ICU has room.",
        clinical_state=ClinicalState(**shared_cs_2),
        hospital_state=_hospital(cicu=(6, 6), icu=(10, 8)),
        candidate_departments=["CICU", "ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="CICU",
        preferred_department="ICU",
        reason="CICU is full. General ICU with cardiology consult is an appropriate critical-care alternative — not a general ward.",
        acceptable_departments=["ICU", "CICU"],
        unacceptable_departments=["ADMITTED_GEN", "ED_OBS", "DISCHARGE"],
        tags=["paired:cicu_ladder", "urgency:critical", "icu:full", "confidence:high"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S06_cicu_and_icu_full_gen_available",
        description="Same STEMI patient; CICU AND general ICU are both full.",
        clinical_state=ClinicalState(**shared_cs_2),
        hospital_state=_hospital(cicu=(6, 6), icu=(10, 10)),
        candidate_departments=["CICU", "ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="CICU",
        preferred_department="ADMITTED_GEN",
        reason="All critical-care beds are full. Escalate to general ward with continuous telemetry pending transfer/critical-care bed.",
        acceptable_departments=["ADMITTED_GEN", "CICU"],
        unacceptable_departments=["ED_OBS", "DISCHARGE"],
        tags=["paired:cicu_ladder", "urgency:critical", "icu:full", "hdu:available"],
    ))

    # ================================================================
    # Ladder 3 — moderate-urgency admission patient (S07 -> S08)
    # ================================================================
    shared_cs_3 = dict(
        icu_risk_2h=0.08, icu_risk_6h=0.12, icu_risk_12h=0.15, admission_risk=0.62,
        xgb_confidence=0.75, information_completeness=0.9,
        rag_urgency="routine", rag_evidence_strength=2, branches_agree=True,
        top_diagnoses=["appendicitis"], red_flags=[], cardiac=False,
    )
    scenarios.append(NurseScenario(
        scenario_id="S07_gen_available",
        description="Moderate-acuity surgical admission; general ward has room.",
        clinical_state=ClinicalState(**shared_cs_3),
        hospital_state=_hospital(gen=(50, 30)),
        candidate_departments=["ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ADMITTED_GEN",
        preferred_department="ADMITTED_GEN",
        reason="Below ICU threshold, admission indicated. General ward has capacity.",
        tags=["paired:gen_ladder", "urgency:moderate", "hdu:available", "confidence:high"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S08_gen_full_obs_available",
        description="Same patient; general ward is full, ED observation has room.",
        clinical_state=ClinicalState(**shared_cs_3),
        hospital_state=_hospital(gen=(50, 50)),
        candidate_departments=["ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ADMITTED_GEN",
        preferred_department="ED_OBS",
        reason="General ward is full. ED observation is a safe holding placement for this moderate-acuity patient pending a ward bed.",
        acceptable_departments=["ED_OBS", "ADMITTED_GEN"],
        tags=["paired:gen_ladder", "urgency:moderate", "hdu:full", "ward:available"],
    ))

    # ================================================================
    # Low-urgency single-candidate coverage
    # ================================================================
    scenarios.append(NurseScenario(
        scenario_id="S09_low_urgency_discharge",
        description="Minor ankle sprain, normal vitals, everything available.",
        clinical_state=ClinicalState(
            icu_risk_2h=0.01, icu_risk_6h=0.01, icu_risk_12h=0.02, admission_risk=0.10,
            xgb_confidence=0.95, information_completeness=1.0,
            rag_urgency="routine", rag_evidence_strength=1, branches_agree=True,
            top_diagnoses=["ankle sprain"], red_flags=[], cardiac=False,
        ),
        hospital_state=_hospital(),
        candidate_departments=["DISCHARGE"],
        clinical_preferred_department="DISCHARGE",
        preferred_department="DISCHARGE",
        reason="Low risk across both branches — safe for discharge with home-care instructions.",
        tags=["urgency:low", "confidence:high", "agreement:agree"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S10_borderline_observation",
        description="Borderline abdominal pain, admission risk between discharge and admit thresholds.",
        clinical_state=ClinicalState(
            icu_risk_2h=0.03, icu_risk_6h=0.04, icu_risk_12h=0.05, admission_risk=0.38,
            xgb_confidence=0.70, information_completeness=0.8,
            rag_urgency="routine", rag_evidence_strength=2, branches_agree=True,
            top_diagnoses=["nonspecific abdominal pain"], red_flags=[], cardiac=False,
        ),
        hospital_state=_hospital(),
        candidate_departments=["ED_OBS"],
        clinical_preferred_department="ED_OBS",
        preferred_department="ED_OBS",
        reason="Borderline risk — observe and reassess rather than discharge or admit outright.",
        tags=["urgency:low-moderate", "confidence:moderate"],
    ))

    # ================================================================
    # Model disagreement, RAG-forced escalation (S11 -> S12)
    # ================================================================
    shared_cs_4 = dict(
        icu_risk_2h=0.25, icu_risk_6h=0.30, icu_risk_12h=0.28, admission_risk=0.55,
        xgb_confidence=0.55, information_completeness=0.5,
        rag_urgency="emergent", rag_evidence_strength=3, branches_agree=False,
        top_diagnoses=["atypical presentation, possible early sepsis"],
        red_flags=["rapidly rising lactate"], cardiac=False,
    )
    scenarios.append(NurseScenario(
        scenario_id="S11_disagreement_icu_available",
        description="XGBoost moderate risk but RAG flags emergent deterioration signs; branches disagree. ICU available.",
        clinical_state=ClinicalState(**shared_cs_4),
        hospital_state=_hospital(icu=(10, 6)),
        candidate_departments=["ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ICU",
        preferred_department="ICU",
        reason="Strong branch disagreement with an emergent RAG signal — trust the more cautious read and admit to ICU for close monitoring.",
        tags=["paired:disagreement_ladder", "urgency:high", "icu:available", "agreement:strong_disagreement"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S12_disagreement_icu_full",
        description="Same disagreement case; ICU is now full, general ward available.",
        clinical_state=ClinicalState(**shared_cs_4),
        hospital_state=_hospital(icu=(10, 10)),
        candidate_departments=["ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ICU",
        preferred_department="ADMITTED_GEN",
        reason="ICU is full. Admit to general ward with escalated monitoring frequency given the disagreement signal.",
        acceptable_departments=["ADMITTED_GEN", "ICU"],
        tags=["paired:disagreement_ladder", "urgency:high", "icu:full", "agreement:strong_disagreement"],
    ))

    # ================================================================
    # Low XGBoost confidence / low completeness (S13 -> S14)
    # ================================================================
    shared_cs_5 = dict(
        icu_risk_2h=0.40, icu_risk_6h=0.45, icu_risk_12h=0.42, admission_risk=0.78,
        xgb_confidence=0.35, information_completeness=0.25,
        rag_urgency="urgent", rag_evidence_strength=2, branches_agree=True,
        top_diagnoses=["undifferentiated shock"], red_flags=["hypotension"], cardiac=False,
    )
    scenarios.append(NurseScenario(
        scenario_id="S13_low_confidence_icu_available",
        description="Missing vitals reduce XGBoost confidence sharply; RAG fills the gap. ICU has room.",
        clinical_state=ClinicalState(**shared_cs_5),
        hospital_state=_hospital(icu=(10, 6)),
        candidate_departments=["ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ICU",
        preferred_department="ICU",
        reason="Data is incomplete but the available signal is concerning; ICU has capacity — admit and complete workup there.",
        tags=["paired:low_confidence_ladder", "urgency:high", "icu:available", "confidence:low"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S14_low_confidence_icu_full",
        description="Same low-confidence case; ICU is full, general ward has room.",
        clinical_state=ClinicalState(**shared_cs_5),
        hospital_state=_hospital(icu=(10, 10)),
        candidate_departments=["ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ICU",
        preferred_department="ADMITTED_GEN",
        reason="ICU is full. General ward admission with frequent reassessment given the incomplete data picture.",
        acceptable_departments=["ADMITTED_GEN", "ICU"],
        tags=["paired:low_confidence_ladder", "urgency:high", "icu:full", "confidence:low"],
    ))

    # ================================================================
    # Overall hospital pressure should not override an available preferred bed
    # ================================================================
    scenarios.append(NurseScenario(
        scenario_id="S15_high_load_but_icu_open",
        description="Hospital overall in CRITICAL load (ED obs and general ward nearly full), but ICU itself has room.",
        clinical_state=ClinicalState(
            icu_risk_2h=0.50, icu_risk_6h=0.55, icu_risk_12h=0.52, admission_risk=0.80,
            xgb_confidence=0.80, information_completeness=0.9,
            rag_urgency="urgent", rag_evidence_strength=3, branches_agree=True,
            top_diagnoses=["severe pneumonia"], red_flags=["hypoxia"], cardiac=False,
        ),
        hospital_state=_hospital(icu=(10, 6), gen=(50, 48), obs=(20, 19)),
        candidate_departments=["ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ICU",
        preferred_department="ICU",
        reason="ICU itself has open beds — overall hospital pressure elsewhere does not justify placing this patient anywhere but ICU.",
        tags=["urgency:high", "icu:available", "ed_pressure:severe"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S16_normal_day_baseline",
        description="Typical moderate-acuity admission on an ordinary, unremarkable day.",
        clinical_state=ClinicalState(
            icu_risk_2h=0.05, icu_risk_6h=0.07, icu_risk_12h=0.06, admission_risk=0.58,
            xgb_confidence=0.82, information_completeness=1.0,
            rag_urgency="routine", rag_evidence_strength=2, branches_agree=True,
            top_diagnoses=["community-acquired pneumonia"], red_flags=[], cardiac=False,
        ),
        hospital_state=_hospital(icu=(10, 5), cicu=(6, 2), gen=(50, 20), obs=(20, 8)),
        candidate_departments=["ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ADMITTED_GEN",
        preferred_department="ADMITTED_GEN",
        reason="Routine admission, ward has plenty of room — standard placement.",
        tags=["urgency:moderate", "hdu:available", "ed_pressure:low"],
    ))

    # ================================================================
    # Take the last available preferred bed rather than pre-emptively downgrade
    # ================================================================
    scenarios.append(NurseScenario(
        scenario_id="S17_last_cicu_bed_taken",
        description="Critical cardiac patient; CICU has exactly one bed left.",
        clinical_state=ClinicalState(
            icu_risk_2h=0.80, icu_risk_6h=0.85, icu_risk_12h=0.78, admission_risk=0.92,
            xgb_confidence=0.88, information_completeness=1.0,
            rag_urgency="emergent", rag_evidence_strength=4, branches_agree=True,
            top_diagnoses=["cardiogenic shock"], red_flags=["hypotension", "cool extremities"], cardiac=True,
        ),
        hospital_state=_hospital(cicu=(6, 5)),
        candidate_departments=["CICU", "ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="CICU",
        preferred_department="CICU",
        reason="One CICU bed remains — take it for this critical cardiac patient rather than pre-emptively reserving it.",
        tags=["urgency:critical", "icu:nearly_full", "confidence:high"],
    ))
    scenarios.append(NurseScenario(
        scenario_id="S18_last_icu_bed_high_load",
        description="High-acuity patient; ICU has one bed left and the hospital overall is in HIGH_LOAD.",
        clinical_state=ClinicalState(
            icu_risk_2h=0.45, icu_risk_6h=0.50, icu_risk_12h=0.48, admission_risk=0.79,
            xgb_confidence=0.80, information_completeness=0.9,
            rag_urgency="urgent", rag_evidence_strength=3, branches_agree=True,
            top_diagnoses=["diabetic ketoacidosis"], red_flags=["altered mental status"], cardiac=False,
        ),
        hospital_state=_hospital(icu=(10, 9), gen=(50, 40), obs=(20, 15)),
        candidate_departments=["ICU", "ADMITTED_GEN", "ED_OBS"],
        clinical_preferred_department="ICU",
        preferred_department="ICU",
        reason="ICU still has one open bed — admit there; do not downgrade this patient just because the hospital is busy overall.",
        tags=["urgency:high", "icu:nearly_full", "ed_pressure:moderate"],
    ))

    return scenarios


# ---------------------------------------------------------------------------
# Facility-adaptive selection (Step 4 — multi-hospital)
# ---------------------------------------------------------------------------
#
# Reuses the 18 scenarios above verbatim — never rewrites/regenerates them.
# A scenario is only INCLUDED if every department it actually offers as a
# routing candidate exists at this facility (e.g. no CICU -> no CICU-ladder
# scenarios); the departments it DOES use are then rescaled onto the
# facility's real bed counts, preserving each scenario's clinical intent
# (full stays full; "exactly one bed left" stays exactly one bed left).
# No ML, no learned scenario count — the count is simply how many of the 18
# templates are still clinically meaningful for that facility's resources.

def load_demonstrations(
    facility_config: Optional[Dict[str, Any]] = None,
) -> List[NurseScenario]:
    """
    Parameters
    ----------
    facility_config : the target hospital's departments dict — same shape as
        HospitalStateService.get_all() / hospital_config.json's
        "departments" (each value needs at least a "capacity" key).
        None (default) returns the original fixed 18-scenario baseline,
        unchanged — every existing caller (artifacts.py, training.py,
        tests) keeps working exactly as before.
    """
    scenarios = _all_scenarios()
    if facility_config is None:
        return scenarios

    available_departments = set(facility_config.keys())
    dept_map = _department_role_map(available_departments)
    selected: List[NurseScenario] = []
    for scenario in scenarios:
        if not set(scenario.candidate_departments).issubset(dept_map.keys()):
            continue
        selected.append(_adapt_scenario(scenario, facility_config, dept_map))
    return selected


def _department_role_map(available_departments: set) -> Dict[str, str]:
    """
    Canonical demonstration department code (CICU/ICU/ADMITTED_GEN/ED_OBS/
    DISCHARGE) -> the real hospital department name that satisfies it, for a
    hospital whose department names don't necessarily match those codes
    (onboarding lets a nurse name departments freely).

    A literal match — the hospital really does have a department named e.g.
    "ICU" — always wins. Otherwise, an unmatched canonical code is satisfied
    by any hospital department at the same acuity tier: the same
    "an unrecognized department name defaults to ADMITTED_GEN's tier"
    fallback candidates.py already uses for live routing (see its module
    docstring), applied here so a hospital using its own department names
    isn't silently excluded from every scenario that needs one. A canonical
    code with no literal match and no same-tier department available is left
    unmapped, and scenarios needing it are still excluded — this fills in
    what the existing acuity-tier philosophy can safely infer, it does not
    invent a specific clinical role (e.g. "ICU") for a department nothing
    ever labeled that way.
    """
    mapping: Dict[str, str] = {}
    for code in _CANONICAL_DEPARTMENT_CODES:
        if code in available_departments:
            mapping[code] = code

    claimed = set(mapping.values())
    unclaimed_by_tier: Dict[int, List[str]] = {}
    for dept in sorted(available_departments):
        if dept in claimed:
            continue
        unclaimed_by_tier.setdefault(acuity_tier(dept), []).append(dept)

    for code in _CANONICAL_DEPARTMENT_CODES:
        if code in mapping:
            continue
        pool = unclaimed_by_tier.get(acuity_tier(code))
        if pool:
            mapping[code] = pool.pop(0)
    return mapping


def _remap_department(name: str, dept_map: Dict[str, str]) -> str:
    return dept_map.get(name, name)


def _adapt_scenario(
    scenario: NurseScenario,
    facility_config: Dict[str, Any],
    dept_map: Dict[str, str],
) -> NurseScenario:
    """Retarget one scenario's department bed counts AND department-name
    references onto the real facility, via dept_map (canonical code -> real
    hospital department name — see _department_role_map). A canonical
    department the facility has no match for at all (e.g. a baseline CICU
    entry on a non-cardiac scenario) is dropped rather than shown to the
    nurse."""
    adapted_state = {
        dept_map[dept]: _rescale_department(state, int(facility_config[dept_map[dept]]["capacity"]))
        for dept, state in scenario.hospital_state.items()
        if dept in dept_map
    }
    return dataclasses.replace(
        scenario,
        hospital_state=adapted_state,
        candidate_departments=[dept_map[d] for d in scenario.candidate_departments],
        clinical_preferred_department=_remap_department(scenario.clinical_preferred_department, dept_map),
        preferred_department=_remap_department(scenario.preferred_department, dept_map),
        acceptable_departments=[_remap_department(d, dept_map) for d in scenario.acceptable_departments],
        unacceptable_departments=[_remap_department(d, dept_map) for d in scenario.unacceptable_departments],
    )


def _rescale_department(original: DepartmentState, target_capacity: int) -> DepartmentState:
    """Pure arithmetic rescale — no learned/ML component. Preserves the two
    clinically load-bearing edge cases (fully occupied; exactly one bed
    remaining) exactly; otherwise scales proportionally."""
    target_capacity = max(0, int(target_capacity))
    if target_capacity == 0:
        return DepartmentState(capacity=0, occupied=0, status="CLOSED")

    orig_available = original.capacity - original.occupied
    if original.capacity <= 0 or orig_available <= 0:
        new_occupied = target_capacity                # was full -> stays full
    elif orig_available == 1:
        new_occupied = max(0, target_capacity - 1)     # exactly one left -> stays exactly one left
    else:
        ratio = original.occupied / original.capacity
        new_occupied = min(target_capacity, round(target_capacity * ratio))

    return DepartmentState(capacity=target_capacity, occupied=new_occupied, status=original.status)
