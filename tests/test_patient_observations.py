"""
test_patient_observations.py
-----------------------------
Tests for the timestamped-observation -> rerun-assessment workflow:
    triageguard_agent/tools/patient_tools.py :: add_patient_observation
    triageguard_agent/runtime/agent_runtime.py :: auto-reassessment on commit

These exercise the REAL handler, REAL ToolRegistry/ToolExecutor/
ConfirmationProtocol, and REAL file I/O against an isolated copy of the
patient-52 fixture (never the real triageguard_agent/data/patients/52.json).

The only thing stubbed anywhere in this file is TriageGuardPipeline itself
(triageguard_agent/tools/assessment_tools.py::_get_pipeline) — running it for
real requires loading XGBoost+FAISS and making a live OpenRouter call, which
no existing test in this repo does (see tests/test_llm_planning_loop.py,
which stubs only the LLM transport for the same reason). Every other layer
in these tests — the observation write, the WRITE approval gate, the
confirmation flow, and the run_triage_assessment tool wiring itself — is
real. The full, real, network-attached pipeline path is verified separately
via scripts/chat_with_agent.py.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_agent.tools import patient_tools
from triageguard_agent.tools.patient_tools import (
    add_patient_observation,
    add_patient_observation_spec,
    get_patient_record,
    build_assessment_input,
)
from triageguard_agent.tools.registry import ToolRegistry
from triageguard_agent.runtime.tool_executor import ToolExecutor
from triageguard_agent.runtime.agent_runtime import AgentRuntime
from triageguard_agent.state.agent_state import AgentState

_REAL_PATIENTS_DIR = _REPO / "triageguard_agent" / "data" / "patients"


# ---------------------------------------------------------------------------
# Isolated patient-file fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def patients_dir(tmp_path, monkeypatch):
    """
    Point patient_tools at an isolated tmp directory seeded with a copy of
    the real patient-52 fixture, so tests exercise real file I/O without
    ever mutating triageguard_agent/data/patients/52.json.
    """
    dst_dir = tmp_path / "patients"
    dst_dir.mkdir()
    src = _REAL_PATIENTS_DIR / "52.json"
    (dst_dir / "52.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(patient_tools, "_PATIENTS_DIR", dst_dir)
    return dst_dir


def _read(patients_dir: Path, patient_id: str = "52") -> dict:
    return json.loads((patients_dir / f"{patient_id}.json").read_text(encoding="utf-8"))


# ===========================================================================
# 1. Valid write — appends, preserves history, updates current state
# ===========================================================================

class TestAddPatientObservationValid:
    def test_appends_new_observation(self, patients_dir):
        before = _read(patients_dir)
        n_before = len(before["observations"])

        result = add_patient_observation("52", "heart_rate", 125)

        assert result.success
        after = _read(patients_dir)
        assert len(after["observations"]) == n_before + 1
        new_obs = after["observations"][-1]
        assert new_obs["heart_rate"] == 125
        assert new_obs["type"] == "heart_rate"
        assert "timestamp" in new_obs

    def test_updates_current_state_field(self, patients_dir):
        add_patient_observation("52", "heart_rate", 125)
        after = _read(patients_dir)
        assert after["heartrate"] == 125

    def test_previous_observation_preserved_in_history(self, patients_dir):
        before = _read(patients_dir)
        first_obs = before["observations"][0]

        add_patient_observation("52", "heart_rate", 125)

        after = _read(patients_dir)
        assert after["observations"][0] == first_obs

    def test_result_reports_previous_and_new_value_for_comparison(self, patients_dir):
        result = add_patient_observation("52", "heart_rate", 125)
        assert result.data["previous_value"] == 112
        assert result.data["new_value"] == 125

    def test_timestamp_comes_from_system_clock_not_the_caller(self, patients_dir):
        """
        Core requirement: never invent/trust a caller-supplied timestamp.
        Even if a (schema-violating) caller passes one, it must be ignored —
        the recorded timestamp must be "now", from the system clock.
        """
        before_call = datetime.now(timezone.utc)
        result = add_patient_observation(
            "52", "spo2", 89, timestamp="2001-01-01T00:00:00Z"
        )
        after_call = datetime.now(timezone.utc)

        recorded_ts = datetime.fromisoformat(result.data["timestamp"])
        assert before_call <= recorded_ts <= after_call

        stored_obs = _read(patients_dir)["observations"][-1]
        stored_ts = datetime.fromisoformat(stored_obs["timestamp"])
        assert before_call <= stored_ts <= after_call

    def test_note_is_stored_when_provided(self, patients_dir):
        add_patient_observation("52", "heart_rate", 125, note="Post-exertion reading")
        after = _read(patients_dir)
        assert after["observations"][-1]["note"] == "Post-exertion reading"

    @pytest.mark.parametrize(
        "obs_type,current_field,value",
        [
            ("heart_rate", "heartrate", 100),
            ("spo2", "o2sat", 97),
            ("resp_rate", "resprate", 18),
            ("sbp", "sbp", 130),
            ("dbp", "dbp", 80),
            ("temperature", "temperature", 37.0),
        ],
    )
    def test_each_observation_type_maps_to_its_current_field(
        self, patients_dir, obs_type, current_field, value
    ):
        result = add_patient_observation("52", obs_type, value)
        assert result.success
        after = _read(patients_dir)
        assert after[current_field] == value


# ===========================================================================
# 2. Invalid inputs — clean failures, no mutation
# ===========================================================================

class TestInvalidInputs:
    def test_invalid_patient_id(self, patients_dir):
        result = add_patient_observation("does-not-exist", "heart_rate", 100)
        assert not result.success
        assert result.error["code"] == "PATIENT_NOT_FOUND"
        assert not (patients_dir / "does-not-exist.json").exists()

    def test_invalid_observation_type(self, patients_dir):
        before = _read(patients_dir)
        result = add_patient_observation("52", "blood_sugar", 100)
        assert not result.success
        assert result.error["code"] == "INVALID_OBSERVATION_TYPE"
        assert _read(patients_dir) == before

    def test_value_out_of_range(self, patients_dir):
        before = _read(patients_dir)
        result = add_patient_observation("52", "heart_rate", 999)
        assert not result.success
        assert result.error["code"] == "INVALID_VALUE"
        assert _read(patients_dir) == before

    def test_value_non_numeric(self, patients_dir):
        before = _read(patients_dir)
        result = add_patient_observation("52", "heart_rate", "fast")
        assert not result.success
        assert result.error["code"] == "INVALID_VALUE"
        assert _read(patients_dir) == before

    def test_missing_patient_id(self, patients_dir):
        result = add_patient_observation("", "heart_rate", 100)
        assert not result.success
        assert result.error["code"] == "MISSING_PATIENT_ID"


# ===========================================================================
# 3. Duplicate handling
# ===========================================================================

class TestDuplicateHandling:
    def test_resubmitting_current_value_is_a_noop(self, patients_dir):
        before = _read(patients_dir)  # heartrate already 112

        result = add_patient_observation("52", "heart_rate", 112)

        assert result.success
        assert result.data["duplicate"] is True
        assert _read(patients_dir) == before  # nothing written

    def test_genuinely_changed_value_after_a_duplicate_still_records(self, patients_dir):
        add_patient_observation("52", "heart_rate", 112)  # duplicate, no-op
        result = add_patient_observation("52", "heart_rate", 130)  # real change

        assert result.success
        assert result.data["duplicate"] is False
        assert _read(patients_dir)["heartrate"] == 130


# ===========================================================================
# 4. WRITE approval gate
# ===========================================================================

class TestWriteApprovalGate:
    def test_no_approval_token_blocks_handler(self, patients_dir):
        before = _read(patients_dir)
        registry = ToolRegistry()
        registry.register(add_patient_observation_spec())
        executor = ToolExecutor(registry)

        result = executor.execute(
            "add_patient_observation",
            {"patient_id": "52", "observation_type": "heart_rate", "value": 125},
            approval_token=None,
        )

        assert not result.success
        assert result.error["code"] == "APPROVAL_REQUIRED"
        assert _read(patients_dir) == before  # handler never actually ran

    def test_with_approval_token_executes(self, patients_dir):
        registry = ToolRegistry()
        registry.register(add_patient_observation_spec())
        executor = ToolExecutor(registry)

        result = executor.execute(
            "add_patient_observation",
            {"patient_id": "52", "observation_type": "heart_rate", "value": 125},
            approval_token="nurse_confirmed",
        )

        assert result.success
        assert _read(patients_dir)["heartrate"] == 125


# ===========================================================================
# 5. Assessment-input builder
# ===========================================================================

class TestAssessmentInputBuilder:
    def test_drops_observations_key_keeps_current_fields(self, patients_dir):
        record = get_patient_record("52")
        data = build_assessment_input(record)
        assert "observations" not in data
        assert data["heartrate"] == 112
        assert data["patient_id"] == "52"

    def test_reflects_updated_value_after_write(self, patients_dir):
        add_patient_observation("52", "heart_rate", 125)
        record = get_patient_record("52")
        data = build_assessment_input(record)
        assert data["heartrate"] == 125


# ===========================================================================
# 6. Tool registration
# ===========================================================================

class TestToolRegistration:
    def test_add_patient_observation_registered_by_default(self):
        runtime = AgentRuntime(
            auto_register=True,
            llm_call_fn=lambda messages, tools, model=None: {"content": "x", "tool_calls": None},
        )
        assert "add_patient_observation" in runtime.tool_registry
        spec = runtime.tool_registry.get("add_patient_observation")
        assert spec.risk_level == "WRITE"
        assert spec.requires_approval is True
        assert spec.side_effect is True


# ===========================================================================
# 7. End-to-end via AgentRuntime: propose -> confirm -> auto-reassess
# ===========================================================================

class _FakePipeline:
    """Stand-in for TriageGuardPipeline — records what patient_data it received."""

    def __init__(self):
        self.calls = []

    def run(self, patient_data):
        self.calls.append(patient_data)
        return {
            "department": "ADMITTED_GEN",
            "department_reasoning": "stub reasoning",
            "acuity_tier": 2,
            "reconciled_admission_risk": 0.55,
            "reconciled_icu_risk": 0.12,
            "branches_agree": True,
            "confidence_note": "stub",
            "top_diagnoses": [],
            "red_flags": [],
            "structured_output": {"disposition": "ADMIT", "escalation_level": "routine"},
            "rag_response": "stub narrative",
            "xgb": {"information_completeness": 1.0},
        }


def _tool_call_message(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, messages, tools, model=None):
        self.calls.append({"messages": messages, "tools": tools, "model": model})
        return self._responses.pop(0)


class TestEndToEndObservationTriggersReassessment:
    def test_confirm_write_then_auto_reassess_uses_updated_value(self, patients_dir, monkeypatch):
        from triageguard_agent.tools import assessment_tools

        fake_pipeline = _FakePipeline()
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: fake_pipeline)

        llm = ScriptedLLM([
            _tool_call_message(
                "call_1",
                "add_patient_observation",
                {"patient_id": "52", "observation_type": "heart_rate", "value": 125},
            ),
        ])
        runtime = AgentRuntime(auto_register=True, llm_call_fn=llm)
        state = AgentState(session_id="s1")

        turn1 = runtime.process_turn("Patient 52's heart rate is now 125.", state)
        assert turn1.response_type == "approval_required"
        assert state.has_pending()
        assert fake_pipeline.calls == []  # not reassessed until confirmed

        turn2 = runtime.process_turn("yes", state)

        assert turn2.response_type == "confirmation"
        assert not state.has_pending()

        # Real file was actually written
        assert _read(patients_dir)["heartrate"] == 125

        # Reassessment ran exactly once, with the NEW value — not the stale 112
        assert len(fake_pipeline.calls) == 1
        assert fake_pipeline.calls[0]["heartrate"] == 125

        # Both the write and the reassessment are surfaced in the response
        tool_names = [a["tool"] for a in turn2.actions]
        assert "add_patient_observation" in tool_names
        assert "run_triage_assessment" in tool_names
        assert "ADMITTED_GEN" in turn2.message

    def test_rejecting_the_write_never_reassesses(self, patients_dir, monkeypatch):
        from triageguard_agent.tools import assessment_tools

        fake_pipeline = _FakePipeline()
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: fake_pipeline)

        llm = ScriptedLLM([
            _tool_call_message(
                "call_1",
                "add_patient_observation",
                {"patient_id": "52", "observation_type": "heart_rate", "value": 125},
            ),
        ])
        runtime = AgentRuntime(auto_register=True, llm_call_fn=llm)
        state = AgentState(session_id="s1")

        runtime.process_turn("Patient 52's heart rate is now 125.", state)
        response = runtime.process_turn("no", state)

        assert response.response_type == "information"
        assert fake_pipeline.calls == []
        before = json.loads((_REAL_PATIENTS_DIR / "52.json").read_text(encoding="utf-8"))
        assert _read(patients_dir)["heartrate"] == before["heartrate"]  # unchanged

    def test_duplicate_write_confirmed_but_not_reassessed(self, patients_dir, monkeypatch):
        """Confirming a write whose value equals the current value is a no-op,
        so it must not trigger a pointless reassessment."""
        from triageguard_agent.tools import assessment_tools

        fake_pipeline = _FakePipeline()
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: fake_pipeline)

        llm = ScriptedLLM([
            _tool_call_message(
                "call_1",
                "add_patient_observation",
                {"patient_id": "52", "observation_type": "heart_rate", "value": 112},  # already 112
            ),
        ])
        runtime = AgentRuntime(auto_register=True, llm_call_fn=llm)
        state = AgentState(session_id="s1")

        runtime.process_turn("Patient 52's heart rate is 112.", state)
        response = runtime.process_turn("yes", state)

        assert response.response_type == "confirmation"
        assert fake_pipeline.calls == []
        tool_names = [a["tool"] for a in response.actions]
        assert "run_triage_assessment" not in tool_names


# ===========================================================================
# 8. Regression: run_triage_assessment tool itself still works
# ===========================================================================

class TestRunTriageAssessmentRegression:
    def test_run_triage_assessment_still_callable(self, monkeypatch):
        from triageguard_agent.tools import assessment_tools

        fake_pipeline = _FakePipeline()
        monkeypatch.setattr(assessment_tools, "_get_pipeline", lambda: fake_pipeline)

        result = assessment_tools.run_triage_assessment({"patient_id": "52", "heartrate": 112})

        assert result.success
        assert result.data["department"] == "ADMITTED_GEN"
        assert len(fake_pipeline.calls) == 1
