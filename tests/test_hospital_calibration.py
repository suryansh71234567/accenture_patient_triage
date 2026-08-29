"""
test_hospital_calibration.py
-----------------------------
Unit tests for the hospital calibration workflow.

Tests
-----
1.  Normal calibration — valid occupancy update committed
2.  Ambiguous staff input — partial/unclear → validation error
3.  Contradictory occupancy — occupied > capacity rejected
4.  Capacity change — capacity update recalculates available
5.  Resource closure — CLOSED → available = 0 enforced
6.  Stale state detection — is_stale() threshold
7.  Lambda recalculation — λ updates after state change
8.  Confirmation accepted — pending action committed on "yes"
9.  Confirmation rejected — pending action discarded on "no"
10. Routing-after-calibration — route decision uses updated state
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_agent.hospital.hospital_state_store import HospitalStateStore, OPEN, CLOSED
from triageguard_agent.hospital.hospital_state_service import HospitalStateService
from triageguard_agent.hospital.hospital_load_controller import HospitalLoadController
from triageguard_agent.protocols.confirmation_protocol import ConfirmationProtocol
from triageguard_agent.state.agent_state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(departments: dict, tmp_path: Path) -> Path:
    """Write a minimal hospital_config.json and return its path."""
    cfg = {
        "departments": departments,
        "lambda_thresholds": {"normal_max": 0.70, "high_load_max": 0.90},
        "stale_threshold_minutes": 30,
    }
    p = tmp_path / "hospital_config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _make_service(departments: dict, tmp_path: Path) -> HospitalStateService:
    """Create a fresh (non-singleton) HospitalStateService with given departments."""
    cfg_path = _make_config(departments, tmp_path)
    store = HospitalStateStore(cfg_path)
    return HospitalStateService(store)


# ===========================================================================
# 1. Normal calibration
# ===========================================================================

class TestNormalCalibration:
    def test_valid_occupancy_update(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 8, "status": "OPEN"}},
            tmp_path,
        )
        validated = svc.validate_update("ICU", {"occupied": 9})
        svc.apply_update("ICU", validated)
        state = svc.get_state("ICU")
        assert state["occupied"] == 9
        assert state["available"] == 1

    def test_available_recalculated(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        validated = svc.validate_update("ICU", {"occupied": 7})
        svc.apply_update("ICU", validated)
        state = svc.get_state("ICU")
        assert state["available"] == 3

    def test_last_updated_refreshed(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        before = svc.get_state("ICU")["last_updated"]
        validated = svc.validate_update("ICU", {"occupied": 6})
        svc.apply_update("ICU", validated)
        after = svc.get_state("ICU")["last_updated"]
        assert after >= before


# ===========================================================================
# 2. Ambiguous / invalid staff input
# ===========================================================================

class TestAmbiguousStaffInput:
    def test_unrecognised_field_rejected(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        with pytest.raises(ValueError, match="Unrecognised update fields"):
            svc.validate_update("ICU", {"beds_used": 5})

    def test_unknown_department_rejected(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        with pytest.raises(ValueError, match="not in the hospital configuration"):
            svc.validate_update("NICU", {"occupied": 2})

    def test_non_integer_value_rejected(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        with pytest.raises(ValueError, match="must be an integer"):
            svc.validate_update("ICU", {"occupied": "many"})

    def test_invalid_status_rejected(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        with pytest.raises(ValueError, match="Invalid status"):
            svc.validate_update("ICU", {"status": "BROKEN"})


# ===========================================================================
# 3. Contradictory occupancy — occupied > capacity
# ===========================================================================

class TestContradictoryOccupancy:
    def test_occupied_exceeds_capacity_rejected(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        with pytest.raises(ValueError, match="cannot exceed"):
            svc.validate_update("ICU", {"occupied": 11})

    def test_occupied_equals_capacity_allowed(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        validated = svc.validate_update("ICU", {"occupied": 10})
        assert validated["occupied"] == 10
        assert validated["available"] == 0

    def test_negative_occupied_rejected(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        with pytest.raises(ValueError, match="cannot be negative"):
            svc.validate_update("ICU", {"occupied": -1})

    def test_capacity_below_current_occupancy_rejected(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 8, "status": "OPEN"}},
            tmp_path,
        )
        with pytest.raises(ValueError):
            svc.validate_update("ICU", {"capacity": 5})  # no new occupied given


# ===========================================================================
# 4. Capacity change
# ===========================================================================

class TestCapacityChange:
    def test_capacity_increase(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 8, "status": "OPEN"}},
            tmp_path,
        )
        validated = svc.validate_update("ICU", {"capacity": 12})
        svc.apply_update("ICU", validated)
        state = svc.get_state("ICU")
        assert state["capacity"] == 12
        assert state["available"] == 4

    def test_capacity_with_matching_occupied_update(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 8, "status": "OPEN"}},
            tmp_path,
        )
        validated = svc.validate_update("ICU", {"capacity": 6, "occupied": 5})
        svc.apply_update("ICU", validated)
        state = svc.get_state("ICU")
        assert state["capacity"] == 6
        assert state["occupied"] == 5
        assert state["available"] == 1


# ===========================================================================
# 5. Resource closure
# ===========================================================================

class TestResourceClosure:
    def test_close_empty_department(self, tmp_path):
        svc = _make_service(
            {"ED_OBS": {"capacity": 20, "occupied": 0, "status": "OPEN"}},
            tmp_path,
        )
        validated = svc.validate_update("ED_OBS", {"status": "CLOSED"})
        svc.apply_update("ED_OBS", validated)
        state = svc.get_state("ED_OBS")
        assert state["status"] == CLOSED
        assert state["available"] == 0

    def test_close_occupied_department_rejected(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        with pytest.raises(ValueError, match="CLOSED"):
            svc.validate_update("ICU", {"status": "CLOSED"})

    def test_available_forced_to_zero_on_close(self, tmp_path):
        svc = _make_service(
            {"ED_OBS": {"capacity": 20, "occupied": 0, "status": "OPEN"}},
            tmp_path,
        )
        validated = svc.validate_update("ED_OBS", {"status": "CLOSED"})
        assert validated["available"] == 0


# ===========================================================================
# 6. Stale state detection
# ===========================================================================

class TestStaleStateDetection:
    def test_fresh_state_not_stale(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        # Just loaded — should not be stale
        assert not svc.is_stale("ICU")

    def test_state_is_stale_after_threshold(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        # Manually backdate the last_updated timestamp
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
        svc._store._departments["ICU"]["last_updated"] = old_time
        assert svc.is_stale("ICU")

    def test_custom_threshold(self, tmp_path):
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"}},
            tmp_path,
        )
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        svc._store._departments["ICU"]["last_updated"] = old_time
        # 5-minute threshold — 10 min old should be stale
        assert svc.is_stale("ICU", threshold_minutes=5)
        # 60-minute threshold — 10 min old should not be stale
        assert not svc.is_stale("ICU", threshold_minutes=60)


# ===========================================================================
# 7. Lambda recalculation
# ===========================================================================

class TestLambdaRecalculation:
    def _controller(self) -> HospitalLoadController:
        return HospitalLoadController(normal_max=0.70, high_load_max=0.90)

    def test_normal_load_normal_mode(self):
        ctrl = self._controller()
        state = {
            "ICU": {"capacity": 10, "occupied": 5, "status": "OPEN"},
        }
        result = ctrl.recalculate(state)
        assert result["operating_mode"] == "NORMAL"
        assert result["lambda"] < 0.50

    def test_high_load_mode(self):
        ctrl = self._controller()
        state = {
            "ICU": {"capacity": 10, "occupied": 8, "status": "OPEN"},
        }
        result = ctrl.recalculate(state)
        assert result["operating_mode"] == "HIGH_LOAD"
        assert 0.50 <= result["lambda"] < 0.80

    def test_critical_mode(self):
        ctrl = self._controller()
        state = {
            "ICU": {"capacity": 10, "occupied": 10, "status": "OPEN"},
        }
        result = ctrl.recalculate(state)
        assert result["operating_mode"] == "CRITICAL"
        assert result["lambda"] >= 0.80

    def test_discharge_excluded_from_load(self):
        ctrl = self._controller()
        state = {
            "ICU":      {"capacity": 10, "occupied": 5, "status": "OPEN"},
            "DISCHARGE": {"capacity": 999, "occupied": 0, "status": "OPEN"},
        }
        load = ctrl.calculate_load(state)
        # DISCHARGE should not inflate total capacity
        assert load == pytest.approx(5 / 10, rel=1e-3)

    def test_closed_department_excluded(self):
        ctrl = self._controller()
        state = {
            "ICU":    {"capacity": 10, "occupied": 5, "status": "OPEN"},
            "ED_OBS": {"capacity": 20, "occupied": 0, "status": "CLOSED"},
        }
        load = ctrl.calculate_load(state)
        assert load == pytest.approx(5 / 10, rel=1e-3)

    def test_lambda_increases_after_calibration(self, tmp_path):
        svc = _make_service(
            {
                "ICU":          {"capacity": 10, "occupied": 5, "status": "OPEN"},
                "ADMITTED_GEN": {"capacity": 50, "occupied": 20, "status": "OPEN"},
            },
            tmp_path,
        )
        ctrl = self._controller()
        before = ctrl.recalculate(svc.get_all())["lambda"]

        # Simulate calibration: ICU jumps from 5 to 9
        validated = svc.validate_update("ICU", {"occupied": 9})
        svc.apply_update("ICU", validated)
        after = ctrl.recalculate(svc.get_all())["lambda"]
        assert after > before

    def test_lambda_in_valid_range(self):
        ctrl = self._controller()
        for occ in range(0, 11):
            state = {"ICU": {"capacity": 10, "occupied": occ, "status": "OPEN"}}
            lam = ctrl.recalculate(state)["lambda"]
            assert 0.0 <= lam <= 1.0


# ===========================================================================
# 8. Confirmation accepted
# ===========================================================================

class TestConfirmationAccepted:
    def test_yes_is_confirmed(self):
        proto = ConfirmationProtocol()
        assert proto.is_confirmed("yes")

    def test_confirm_is_confirmed(self):
        proto = ConfirmationProtocol()
        assert proto.is_confirmed("confirm")

    def test_ok_is_confirmed(self):
        proto = ConfirmationProtocol()
        assert proto.is_confirmed("ok")

    def test_case_insensitive(self):
        proto = ConfirmationProtocol()
        assert proto.is_confirmed("YES")
        assert proto.is_confirmed("  Ok  ")

    def test_resolve_returns_confirmed(self):
        proto = ConfirmationProtocol()
        assert proto.resolve("yes") == "confirmed"

    def test_pending_action_set_on_state(self):
        proto = ConfirmationProtocol()
        state = AgentState(session_id="s1")
        proto.require_confirmation(state, "commit_test", {"dept": "ICU"}, "Test action.")
        assert state.has_pending()
        assert state.pending_action["action_type"] == "commit_test"


# ===========================================================================
# 9. Confirmation rejected
# ===========================================================================

class TestConfirmationRejected:
    def test_no_is_rejected(self):
        proto = ConfirmationProtocol()
        assert proto.is_rejected("no")

    def test_cancel_is_rejected(self):
        proto = ConfirmationProtocol()
        assert proto.is_rejected("cancel")

    def test_abort_is_rejected(self):
        proto = ConfirmationProtocol()
        assert proto.is_rejected("abort")

    def test_resolve_returns_rejected(self):
        proto = ConfirmationProtocol()
        assert proto.resolve("no") == "rejected"

    def test_ambiguous_is_ambiguous(self):
        proto = ConfirmationProtocol()
        assert proto.resolve("maybe later") == "ambiguous"
        assert proto.resolve("what does that mean") == "ambiguous"

    def test_is_ambiguous(self):
        proto = ConfirmationProtocol()
        assert proto.is_ambiguous("perhaps")
        assert not proto.is_ambiguous("yes")
        assert not proto.is_ambiguous("no")


# ===========================================================================
# 10. Routing decision uses updated hospital state
# ===========================================================================

class TestRoutingAfterCalibration:
    def test_load_ratio_changes_after_calibration(self, tmp_path):
        """Verify load_ratio from controller increases after occupancy update."""
        svc = _make_service(
            {
                "ICU":    {"capacity": 10, "occupied": 5, "status": "OPEN"},
                "ED_OBS": {"capacity": 20, "occupied": 10, "status": "OPEN"},
            },
            tmp_path,
        )
        ctrl = HospitalLoadController(normal_max=0.70, high_load_max=0.90)

        before = ctrl.recalculate(svc.get_all())
        assert before["operating_mode"] == "NORMAL"

        # ICU now 9/10 and ED_OBS 18/20 — heavy load
        validated = svc.validate_update("ICU", {"occupied": 9})
        svc.apply_update("ICU", validated)
        validated = svc.validate_update("ED_OBS", {"occupied": 18})
        svc.apply_update("ED_OBS", validated)

        after = ctrl.recalculate(svc.get_all())
        assert after["load_ratio"] > before["load_ratio"]
        assert after["operating_mode"] in ("HIGH_LOAD", "CRITICAL")
        assert after["lambda"] > before["lambda"]

    def test_icu_available_correctly_updated(self, tmp_path):
        """After calibration, available beds reflect the new state."""
        svc = _make_service(
            {"ICU": {"capacity": 10, "occupied": 8, "status": "OPEN"}},
            tmp_path,
        )
        # Nurse says one patient moved in
        validated = svc.validate_update("ICU", {"occupied": 9})
        svc.apply_update("ICU", validated)
        state = svc.get_state("ICU")
        assert state["available"] == 1

    def test_full_calibration_and_lambda_pipeline(self, tmp_path):
        """End-to-end: calibrate → validate → apply → recalculate → check λ."""
        svc = _make_service(
            {
                "ICU":          {"capacity": 10, "occupied": 2, "status": "OPEN"},
                "ADMITTED_GEN": {"capacity": 50, "occupied": 10, "status": "OPEN"},
            },
            tmp_path,
        )
        ctrl = HospitalLoadController.from_config(svc.lambda_thresholds)

        # Step 1: get baseline
        baseline = ctrl.recalculate(svc.get_all())
        assert baseline["operating_mode"] == "NORMAL"

        # Step 2: simulate both departments near capacity
        for dept, new_occ in [("ICU", 9), ("ADMITTED_GEN", 46)]:
            v = svc.validate_update(dept, {"occupied": new_occ})
            svc.apply_update(dept, v)

        # Step 3: verify outcome
        result = ctrl.recalculate(svc.get_all())
        assert result["lambda"] > baseline["lambda"]
        assert result["operating_mode"] in ("HIGH_LOAD", "CRITICAL")
        # ICU: 9/10, GEN: 46/50 → total 55/60 ≈ 91.7% → CRITICAL
        assert result["load_ratio"] == pytest.approx(55 / 60, rel=1e-2)
