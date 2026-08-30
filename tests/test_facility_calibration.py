"""
test_facility_calibration.py
------------------------------
Multi-hospital Step 4: facility-adaptive nurse calibration scenarios.

Covers
------
* Default (no facility_config) behavior is unchanged — still all 18.
* A hospital missing a department never receives scenarios needing it.
* A minimal-resource hospital gets a much smaller, still-meaningful set.
* Scenario count is not fixed at 18 — it responds to facility shape.
* Two same-department hospitals with different capacities get numerically
  different (adapted) scenarios, not identical copies.
* scenarios_for_hospital() keeps two hospitals' calibration sets distinct.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_router.policy.demonstrations import load_demonstrations
from triageguard_router.policy.facility_calibration import scenarios_for_hospital


def _dept(capacity, occupied, status="OPEN"):
    return {"capacity": capacity, "occupied": occupied, "status": status}


FULL_FACILITY = {
    "ICU": _dept(10, 7),
    "CICU": _dept(6, 4),
    "ADMITTED_GEN": _dept(50, 38),
    "ED_OBS": _dept(20, 12),
    "DISCHARGE": _dept(999, 0),
}

NO_CICU_FACILITY = {
    "ICU": _dept(8, 5),
    "ADMITTED_GEN": _dept(30, 20),
    "ED_OBS": _dept(10, 6),
    "DISCHARGE": _dept(999, 0),
}

MINIMAL_FACILITY = {
    "ED_OBS": _dept(6, 2),
    "DISCHARGE": _dept(999, 0),
}

SMALL_ICU_FACILITY = {
    "ICU": _dept(4, 0),
    "CICU": _dept(3, 0),
    "ADMITTED_GEN": _dept(20, 0),
    "ED_OBS": _dept(8, 0),
    "DISCHARGE": _dept(999, 0),
}


class TestDefaultBehaviorUnchanged:
    def test_no_facility_config_returns_all_18(self):
        assert len(load_demonstrations()) == 18

    def test_none_is_explicit_default(self):
        assert len(load_demonstrations(None)) == 18


class TestDepartmentPresenceFiltering:
    def test_hospital_without_cicu_excludes_cicu_scenarios(self):
        scenarios = load_demonstrations(NO_CICU_FACILITY)
        ids = {s.scenario_id for s in scenarios}
        assert not ids & {
            "S04_cicu_available",
            "S05_cicu_full_icu_available",
            "S06_cicu_and_icu_full_gen_available",
            "S17_last_cicu_bed_taken",
        }
        assert len(scenarios) == 14

    def test_no_returned_scenario_references_missing_department(self):
        scenarios = load_demonstrations(NO_CICU_FACILITY)
        for s in scenarios:
            assert "CICU" not in s.candidate_departments
            assert "CICU" not in s.hospital_state  # baseline entry dropped, not just unused

    def test_minimal_facility_gets_only_applicable_scenarios(self):
        scenarios = load_demonstrations(MINIMAL_FACILITY)
        ids = {s.scenario_id for s in scenarios}
        # Only scenarios whose candidate_departments fit in {ED_OBS, DISCHARGE}.
        assert ids == {"S09_low_urgency_discharge", "S10_borderline_observation"}
        for s in scenarios:
            assert set(s.hospital_state.keys()).issubset({"ED_OBS", "DISCHARGE"})


class TestScenarioCountRespondsToComplexity:
    def test_count_is_not_fixed_at_18(self):
        counts = {
            len(load_demonstrations(FULL_FACILITY)),
            len(load_demonstrations(NO_CICU_FACILITY)),
            len(load_demonstrations(MINIMAL_FACILITY)),
        }
        assert counts == {18, 14, 2}

    def test_simpler_facility_gets_strictly_fewer_scenarios(self):
        full = len(load_demonstrations(FULL_FACILITY))
        no_cicu = len(load_demonstrations(NO_CICU_FACILITY))
        minimal = len(load_demonstrations(MINIMAL_FACILITY))
        assert minimal < no_cicu < full


class TestCapacityAdaptation:
    def test_full_scenario_stays_full_after_rescale(self):
        scenarios = {s.scenario_id: s for s in load_demonstrations(SMALL_ICU_FACILITY)}
        s02 = scenarios["S02_icu_full_gen_available"]  # baseline ICU 10/10 = full
        icu = s02.hospital_state["ICU"]
        assert icu.capacity == 4
        assert icu.occupied == 4  # still full

    def test_exactly_one_bed_left_is_preserved_after_rescale(self):
        scenarios = {s.scenario_id: s for s in load_demonstrations(SMALL_ICU_FACILITY)}
        s17 = scenarios["S17_last_cicu_bed_taken"]  # baseline CICU 6/5 -> exactly 1 left
        cicu = s17.hospital_state["CICU"]
        assert cicu.capacity == 3
        assert cicu.occupied == 2  # exactly one bed still left

    def test_two_hospitals_same_departments_different_capacity_get_different_numbers(self):
        big = {"ICU": _dept(10, 7), "ADMITTED_GEN": _dept(50, 30), "ED_OBS": _dept(20, 12), "DISCHARGE": _dept(999, 0)}
        small = {"ICU": _dept(10, 7), "ADMITTED_GEN": _dept(10, 6), "ED_OBS": _dept(20, 12), "DISCHARGE": _dept(999, 0)}

        big_s07 = {s.scenario_id: s for s in load_demonstrations(big)}["S07_gen_available"]
        small_s07 = {s.scenario_id: s for s in load_demonstrations(small)}["S07_gen_available"]

        assert big_s07.hospital_state["ADMITTED_GEN"].capacity == 50
        assert small_s07.hospital_state["ADMITTED_GEN"].capacity == 10
        assert big_s07.hospital_state["ADMITTED_GEN"].occupied != small_s07.hospital_state["ADMITTED_GEN"].occupied

    def test_rescaled_occupied_never_exceeds_capacity(self):
        for s in load_demonstrations(SMALL_ICU_FACILITY):
            for dept, state in s.hospital_state.items():
                assert 0 <= state.occupied <= state.capacity


class TestScenariosForHospitalIdentity:
    def test_two_hospitals_calibration_sets_are_not_confused(self):
        result_a = scenarios_for_hospital("hosp_a", FULL_FACILITY)
        result_b = scenarios_for_hospital("hosp_b", NO_CICU_FACILITY)

        assert result_a["hospital_id"] == "hosp_a"
        assert result_b["hospital_id"] == "hosp_b"
        assert result_a["scenario_count"] == 18
        assert result_b["scenario_count"] == 14

        ids_a = {s.scenario_id for s in result_a["scenarios"]}
        ids_b = {s.scenario_id for s in result_b["scenarios"]}
        assert ids_a != ids_b
        assert "S04_cicu_available" in ids_a
        assert "S04_cicu_available" not in ids_b

    def test_empty_hospital_id_rejected(self):
        with pytest.raises(ValueError):
            scenarios_for_hospital("", FULL_FACILITY)
