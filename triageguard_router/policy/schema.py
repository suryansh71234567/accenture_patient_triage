"""
schema.py
---------
Structured representation for a nurse expert demonstration (Phase 2).

A NurseScenario pairs a realistic (clinical_state, hospital_state) situation
with an EXPERT-ANNOTATED routing decision. The `reason` field is a fixed,
human-written clinical justification — it is data, never LLM-generated, and
is never regenerated at runtime.

The schema keeps room for future ranking/preference data
(acceptable_departments / unacceptable_departments) even though the current
Bayesian/RL policy only consumes preferred_department directly, so
pairwise-preference learning (Phase 5) can be added without a schema change.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClinicalState:
    """
    The clinical-assessment inputs a nurse would see, in the same units as
    triageguard_router.reconciler.reconcile() / TriageGuardPredictor.predict()
    output. Not a new schema — a compact expert-facing subset of the real one.
    """
    icu_risk_2h: float
    icu_risk_6h: float
    icu_risk_12h: float
    admission_risk: float
    xgb_confidence: float
    information_completeness: float
    rag_urgency: str            # "emergent" | "urgent" | "routine" | "unknown"
    rag_evidence_strength: int  # count of supporting diagnoses/red flags/history docs
    branches_agree: bool
    top_diagnoses: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    cardiac: bool = False


@dataclass
class DepartmentState:
    capacity: int
    occupied: int
    status: str = "OPEN"

    @property
    def available(self) -> int:
        return max(0, self.capacity - self.occupied)

    @property
    def occupancy_ratio(self) -> float:
        return 0.0 if self.capacity <= 0 else min(1.0, self.occupied / self.capacity)


@dataclass
class NurseScenario:
    """
    One expert demonstration.

    IMPORTANT field semantics (matches the spec's own paired-scenario
    example: "nurse selects HDU but ICU remains clinically preferred"):

    * clinical_preferred_department — the department the CLINICAL pipeline
      (router.route()) would prefer, independent of resources. Constant
      across every scenario in a resource-availability ladder (e.g. ICU
      stays "ICU" in S01/S02/S03 even though the nurse's actual choice
      changes as ICU/general-ward availability changes). This is the
      reference point features.py's is_preferred_department/acuity_gap use.
    * preferred_department — the NURSE'S actual allocation choice for THIS
      specific (clinical_state, hospital_state) scenario — the behavior-
      cloning training label. Equal to clinical_preferred_department
      whenever the clinically preferred department is itself available.
    """
    scenario_id: str
    description: str
    clinical_state: ClinicalState
    hospital_state: Dict[str, DepartmentState]
    candidate_departments: List[str]
    clinical_preferred_department: str
    preferred_department: str
    reason: str
    acceptable_departments: List[str] = field(default_factory=list)
    unacceptable_departments: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)   # e.g. ["paired:A", "confidence:low"]

    def __post_init__(self) -> None:
        if self.preferred_department not in self.candidate_departments:
            raise ValueError(
                f"{self.scenario_id}: preferred_department {self.preferred_department!r} "
                f"must be one of candidate_departments {self.candidate_departments}."
            )
        if not self.acceptable_departments:
            self.acceptable_departments = [self.preferred_department]

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NurseScenario":
        cs = data["clinical_state"]
        hs = {
            dept: DepartmentState(**state)
            for dept, state in data["hospital_state"].items()
        }
        return cls(
            scenario_id=data["scenario_id"],
            description=data["description"],
            clinical_state=ClinicalState(**cs),
            hospital_state=hs,
            candidate_departments=list(data["candidate_departments"]),
            clinical_preferred_department=data["clinical_preferred_department"],
            preferred_department=data["preferred_department"],
            reason=data["reason"],
            acceptable_departments=list(data.get("acceptable_departments", [])),
            unacceptable_departments=list(data.get("unacceptable_departments", [])),
            tags=list(data.get("tags", [])),
        )
