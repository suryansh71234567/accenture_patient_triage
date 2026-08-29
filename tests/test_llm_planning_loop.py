"""
test_llm_planning_loop.py
--------------------------
Unit tests for the LLM tool-calling planning loop wired into AgentRuntime
(triageguard_agent/runtime/agent_runtime.py::_run_llm_loop and friends).

These tests NEVER call the real OpenRouter API. They inject a scripted fake
LLM transport (ScriptedLLM) via AgentRuntime(llm_call_fn=...), so the tests
are fully offline, deterministic, and fast — while exercising the REAL
ToolRegistry, ToolExecutor, ConfirmationProtocol, and real tool handlers
(e.g. get_patient_summary reads the real triageguard_agent/data/patients/52.json
fixture). Only the network-calling LLM boundary is faked.

Tests
-----
1.  Read-tool call -> final answer (normal request / tool selection)
2.  No tool call needed -> direct final answer
3.  Focused-patient context updates deterministically from tool I/O
4.  WRITE tool attempted directly -> short-circuits to approval_required,
    pending_action is set, and the write handler is NEVER actually invoked
5.  Multiple tool_calls in one LLM turn, second is a WRITE -> loop stops
    at the WRITE call; first (READ) call's result is still real/effective
6.  Confirmation "yes" on the next turn -> executes the previously-pending
    WRITE tool via the deterministic confirmation path, WITHOUT calling the
    LLM at all (hallucination-prevention: the actual execution is backend
    truth, not something the LLM can merely claim happened)
7.  Confirmation "no" -> action is cancelled, write never executes
8.  Ambiguous confirmation response -> asks again, does not execute
9.  Unknown tool name requested by LLM -> reported back as a tool failure,
    loop continues, never silently guessed
10. Malformed tool-call arguments (invalid JSON) -> reported back as a
    tool failure, loop continues
11. Iteration cap -> a model that never stops requesting tools gets cut off
    with a graceful error, not an infinite loop
12. OpenRouterError from the transport -> surfaces as a clean error
    AgentResponse, does not raise out of process_turn
13. End-to-end: two-turn conversation (ask -> confirm) via process_turn
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_agent.runtime.agent_runtime import AgentRuntime
from triageguard_agent.state.agent_state import AgentState
from triageguard_agent.llm.openrouter_client import OpenRouterError
from triageguard_agent.hospital.hospital_state_service import HospitalStateService


@pytest.fixture(autouse=True)
def reset_hospital_singleton():
    """
    commit_hospital_calibration goes through the process-wide
    HospitalStateService singleton, same as production code. Reset it
    before/after every test so these tests don't leak state into each
    other or into other test files (mirrors tests/test_dynamic_simulation.py).
    """
    HospitalStateService.reset_instance()
    yield
    HospitalStateService.reset_instance()


# ---------------------------------------------------------------------------
# Fake LLM transport
# ---------------------------------------------------------------------------

class ScriptedLLM:
    """
    Fake llm_call_fn: returns pre-scripted OpenAI-format assistant messages
    in order, one per call. Records every call for inspection so tests can
    assert on exactly what was sent to "the model" and how many times.
    """

    def __init__(self, responses: List[Dict[str, Any]]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, messages, tools, model=None):
        self.calls.append({"messages": messages, "tools": tools, "model": model})
        if not self._responses:
            raise AssertionError(
                f"ScriptedLLM ran out of scripted responses after {len(self.calls)} calls."
            )
        return self._responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _tool_call_message(call_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
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


def _final_message(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": None}


def _raising_llm(*args, **kwargs):
    raise OpenRouterError("simulated network failure")


@pytest.fixture
def runtime_factory():
    """Return a factory so each test builds a fresh AgentRuntime with its own script."""
    def _make(scripted_responses):
        llm = ScriptedLLM(scripted_responses)
        runtime = AgentRuntime(auto_register=True, llm_call_fn=llm)
        return runtime, llm
    return _make


# ===========================================================================
# 1. Read-tool call -> final answer
# ===========================================================================

class TestReadToolThenAnswer:
    def test_calls_tool_then_returns_final_text(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message("call_1", "get_patient_summary", {"patient_id": "52"}),
            _final_message("Patient 52 is a 62-year-old male with chest pain, HR 112."),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Show me patient 52.", state)

        # get_patient_summary is a plain lookup, not a clinical assessment tool
        assert response.response_type == "information"
        assert "62-year-old" in response.message
        assert llm.call_count == 2
        assert response.actions[0]["tool"] == "get_patient_summary"
        assert response.actions[0]["status"] == "executed"
        # The real handler actually ran — data reflects the real fixture file
        assert response.actions[0]["data"]["patient_id"] == "52"

    def test_clinical_assessment_tool_labelled_assessment(self, runtime_factory):
        """
        A turn that used a genuine clinical-assessment tool should be
        labelled 'assessment', distinct from a plain lookup, so a future
        UI cannot mistake a factual lookup for a clinical judgment.
        """
        runtime, llm = runtime_factory([
            _tool_call_message("call_1", "get_xgb_explanation", {"patient_data": {"patient_id": "52"}}),
            _final_message("XGBoost weighed the missing vitals heavily here."),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Why did XGBoost score this patient this way?", state)

        assert response.response_type == "assessment"

    def test_tool_result_fed_back_to_second_llm_call(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message("call_1", "get_patient_summary", {"patient_id": "52"}),
            _final_message("Done."),
        ])
        state = AgentState(session_id="s1")
        runtime.process_turn("Show me patient 52.", state)

        second_call_messages = llm.calls[1]["messages"]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        payload = json.loads(tool_messages[0]["content"])
        assert payload["success"] is True
        assert payload["data"]["patient_id"] == "52"


# ===========================================================================
# 2. No tool call needed
# ===========================================================================

class TestDirectAnswer:
    def test_no_tool_calls_returns_immediately(self, runtime_factory):
        runtime, llm = runtime_factory([_final_message("Hello, how can I help?")])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("hi", state)

        assert response.message == "Hello, how can I help?"
        assert response.response_type == "information"
        assert response.actions == []
        assert llm.call_count == 1


# ===========================================================================
# 3. Focused-patient context
# ===========================================================================

class TestFocusedPatientTracking:
    def test_active_patient_set_after_successful_lookup(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message("call_1", "get_patient_summary", {"patient_id": "52"}),
            _final_message("Here she is."),
        ])
        state = AgentState(session_id="s1")
        assert state.active_patient_id is None

        runtime.process_turn("Open patient 52.", state)

        assert state.active_patient_id == "52"

    def test_active_patient_not_set_on_failed_lookup(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message("call_1", "get_patient_summary", {"patient_id": "does-not-exist"}),
            _final_message("I couldn't find that patient."),
        ])
        state = AgentState(session_id="s1")

        runtime.process_turn("Open patient does-not-exist.", state)

        assert state.active_patient_id is None


# ===========================================================================
# 4. WRITE tool -> approval required, never silently executed
# ===========================================================================

class TestWriteToolRequiresApproval:
    def test_write_tool_short_circuits_to_approval(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1", "admit_simulated_patient", {"patient_id": "PAT-1", "department": "ICU"}
            ),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Admit PAT-1 to ICU.", state)

        assert response.response_type == "approval_required"
        assert response.human_approval_required is True
        assert state.has_pending()
        assert state.pending_action["action_type"] == "admit_simulated_patient"
        # Only one LLM call — the loop stopped the instant approval was required
        assert llm.call_count == 1

    def test_pending_description_reflects_actual_payload(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1", "admit_simulated_patient", {"patient_id": "PAT-9", "department": "CICU"}
            ),
        ])
        state = AgentState(session_id="s1")
        response = runtime.process_turn("Admit PAT-9 to CICU.", state)

        assert "PAT-9" in response.message
        assert "CICU" in response.message


# ===========================================================================
# 5. Multiple tool_calls in one turn, second is WRITE
# ===========================================================================

class TestMixedReadWriteInSameTurn:
    def test_read_executes_write_halts_loop(self, runtime_factory):
        mixed_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_patient_summary",
                        "arguments": json.dumps({"patient_id": "52"}),
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "admit_simulated_patient",
                        "arguments": json.dumps({"patient_id": "52", "department": "ICU"}),
                    },
                },
            ],
        }
        runtime, llm = runtime_factory([mixed_message])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Check patient 52 and admit to ICU.", state)

        # Loop halted on the WRITE call — never reached a 2nd LLM round
        assert llm.call_count == 1
        assert response.response_type == "approval_required"
        # But the READ call before it genuinely executed (real side-effect-free read)
        assert state.active_patient_id == "52"


# ===========================================================================
# 6 & 7 & 8. Confirmation flow
# ===========================================================================

class TestConfirmationFlow:
    """
    Uses commit_hospital_calibration (rather than admit_simulated_patient) as
    the WRITE tool under test, because it can actually execute successfully
    against real backend state (ICU exists in hospital_config.json) without
    depending on a live simulated patient already existing in the queue —
    the earlier approval-gate tests already cover admit_simulated_patient's
    short-circuit behavior, which happens before its handler ever runs.
    """

    _CALIBRATION_CALL = {
        "department": "ICU",
        "validated_update": {"capacity": 10, "occupied": 8, "status": "OPEN"},
    }

    def test_confirm_executes_without_calling_llm(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message("call_1", "commit_hospital_calibration", self._CALIBRATION_CALL),
        ])
        state = AgentState(session_id="s1")
        runtime.process_turn("Update ICU occupancy to 8.", state)
        assert state.has_pending()
        calls_before = llm.call_count

        response = runtime.process_turn("yes", state)

        # Confirmation path never touches the LLM — execution is deterministic
        assert llm.call_count == calls_before
        assert not state.has_pending()
        assert response.response_type == "confirmation"
        assert "executed successfully" in response.message
        # Verify the backend actually changed state — not just an LLM claim
        real_state = HospitalStateService.instance().get_state("ICU")
        assert real_state["occupied"] == 8

    def test_reject_cancels_without_executing(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message("call_1", "commit_hospital_calibration", self._CALIBRATION_CALL),
        ])
        state = AgentState(session_id="s1")
        before_state = dict(HospitalStateService.instance().get_state("ICU"))
        runtime.process_turn("Update ICU occupancy to 8.", state)

        response = runtime.process_turn("no", state)

        assert not state.has_pending()
        assert response.response_type == "information"
        assert "cancelled" in response.message.lower()
        # Nothing actually changed in the backend
        after_state = HospitalStateService.instance().get_state("ICU")
        assert after_state["occupied"] == before_state["occupied"]

    def test_ambiguous_response_asks_again(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message("call_1", "commit_hospital_calibration", self._CALIBRATION_CALL),
        ])
        state = AgentState(session_id="s1")
        runtime.process_turn("Update ICU occupancy to 8.", state)

        response = runtime.process_turn("maybe later", state)

        assert state.has_pending()  # still pending — nothing executed
        assert response.response_type == "approval_required"


# ===========================================================================
# 9. Unknown tool name
# ===========================================================================

class TestUnknownTool:
    def test_unknown_tool_reported_and_loop_continues(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message("call_1", "delete_all_patients", {}),
            _final_message("That action isn't available, so I didn't perform it."),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Delete all patients.", state)

        assert "isn't available" in response.message
        # Verify the failure was actually reported back to the model
        second_call_messages = llm.calls[1]["messages"]
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        payload = json.loads(tool_msgs[0]["content"])
        assert payload["success"] is False
        assert payload["error"]["code"] == "TOOL_NOT_FOUND"


# ===========================================================================
# 10. Malformed tool-call arguments
# ===========================================================================

class TestMalformedArguments:
    def test_invalid_json_arguments_handled_gracefully(self, runtime_factory):
        bad_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_patient_summary", "arguments": "{not valid json"},
                }
            ],
        }
        runtime, llm = runtime_factory([
            bad_call_message,
            _final_message("Let me try that again with the correct format."),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Show me the patient.", state)

        assert "try that again" in response.message
        second_call_messages = llm.calls[1]["messages"]
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        payload = json.loads(tool_msgs[0]["content"])
        assert payload["success"] is False
        assert payload["error"]["code"] == "MALFORMED_ARGUMENTS"


# ===========================================================================
# 10b. Duplicated tool-call arguments (regression for get_xgb_explanation bug)
# ===========================================================================
#
# Real bug, reproduced live against OpenRouter (meta-llama/llama-3.1-8b-instruct):
# asking "Why is patient 52 flagged?" in a follow-up turn produced a
# get_xgb_explanation tool call whose `arguments` string was a valid JSON
# object immediately followed by an identical duplicate, e.g.:
#   {"patient_data": "..."}{"patient_data": "..."}
# This is a generic small-model repetition artifact (not specific to this
# tool or to patient 52) — these tests use synthetic data and directly
# exercise the parser, so the fix is verified structurally rather than by
# hardcoding around one patient.

from triageguard_agent.runtime.agent_runtime import _parse_tool_call_arguments  # noqa: E402
from triageguard_agent.tools.assessment_tools import get_xgb_explanation_spec  # noqa: E402


def _duplicate(json_text: str) -> str:
    """Build the exact 'valid JSON immediately followed by itself' shape observed live."""
    return json_text + json_text


class TestDuplicatedArgumentRecovery:
    def test_parser_recovers_first_value_from_duplicated_json(self):
        single = json.dumps({"patient_data": {"patient_id": "7", "hr_current": 100}})
        duplicated = _duplicate(single)

        recovered = _parse_tool_call_arguments(duplicated)

        assert recovered == {"patient_data": {"patient_id": "7", "hr_current": 100}}

    def test_parser_still_rejects_genuinely_truncated_json(self):
        # Not "valid JSON + extra data" — this is an incomplete object, and
        # must still be treated as malformed rather than guessed at.
        truncated = '{"patient_data": {"patient_id": "7"'
        with pytest.raises(json.JSONDecodeError):
            _parse_tool_call_arguments(truncated)

    def test_parser_rejects_recovered_non_object_value(self):
        # '"x"' is a complete, self-delimited JSON string token, so
        # duplicating it produces "valid JSON followed by extra data" (unlike
        # bare digits, which would just merge into a different number). The
        # recovered value is valid JSON but not a JSON object, so this must
        # still surface as malformed rather than silently coerced.
        with pytest.raises(ValueError):
            _parse_tool_call_arguments(_duplicate('"x"'))

    def test_agent_runtime_recovers_duplicated_xgb_explanation_call(self, runtime_factory):
        """
        End-to-end: an LLM turn whose get_xgb_explanation arguments are
        duplicated (the exact shape reproduced live) must still result in
        the REAL tool executing successfully — not a MALFORMED_ARGUMENTS
        failure and not a fabricated success.
        """
        patient_data = {
            "patient_id": "52",
            "age": 62,
            "sex": "M",
            "hr_current": 112,
            "spo2_current": 94,
            "triage_complaint": "chest pain and shortness of breath",
        }
        single = json.dumps({"patient_data": patient_data})
        duplicated_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_xgb_explanation", "arguments": _duplicate(single)},
                }
            ],
        }
        runtime, llm = runtime_factory([
            duplicated_call_message,
            _final_message("XGBoost weighed the available vitals for patient 52."),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Why is patient 52 flagged?", state)

        assert response.actions[0]["tool"] == "get_xgb_explanation"
        assert response.actions[0]["status"] == "executed"
        # Prove the REAL handler ran (real XGBoost model), not a stub:
        # its output always includes information_completeness and icu_risks.
        assert "information_completeness" in response.actions[0]["data"]
        assert "icu_risks" in response.actions[0]["data"]

    def test_patient_data_schema_has_named_properties(self):
        """
        Schema regression: patient_data must no longer be a bare
        {"type": "object"} with zero properties — that under-specification
        is what left the model guessing field names in the first place.
        """
        spec = get_xgb_explanation_spec()
        patient_data_schema = spec.input_schema["properties"]["patient_data"]

        assert patient_data_schema["type"] == "object"
        properties = patient_data_schema.get("properties", {})
        assert "hr_current" in properties
        assert "spo2_current" in properties
        assert "triage_complaint" in properties


# ===========================================================================
# 11. Iteration cap
# ===========================================================================

class TestIterationCap:
    def test_never_ending_tool_requests_are_cut_off(self):
        # Always requests a (harmless, real) read tool — never gives a final answer
        infinite_responses = [
            _tool_call_message(f"call_{i}", "get_patient_summary", {"patient_id": "52"})
            for i in range(50)
        ]
        llm = ScriptedLLM(infinite_responses)
        runtime = AgentRuntime(auto_register=True, llm_call_fn=llm, max_tool_iterations=3)
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Loop forever.", state)

        assert response.response_type == "error"
        assert llm.call_count == 3
        assert "allowed number of steps" in response.message


# ===========================================================================
# 12. Transport failure
# ===========================================================================

class TestLLMTransportFailure:
    def test_openrouter_error_becomes_clean_error_response(self):
        runtime = AgentRuntime(auto_register=True, llm_call_fn=_raising_llm)
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Anything.", state)

        assert response.response_type == "error"
        assert "couldn't reach" in response.message.lower()


# ===========================================================================
# 13. End-to-end two-turn conversation
# ===========================================================================

class TestEndToEndConversation:
    def test_ask_then_confirm_flow(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {
                    "department": "ICU",
                    "validated_update": {"capacity": 10, "occupied": 8, "status": "OPEN"},
                },
            ),
        ])
        state = AgentState(session_id="s1")

        turn1 = runtime.process_turn("Please update ICU occupancy to 8.", state)
        assert turn1.human_approval_required is True

        turn2 = runtime.process_turn("confirm", state)
        assert turn2.response_type == "confirmation"
        assert not state.has_pending()

        # Conversation context recorded both turns for both exchanges
        roles = [t["role"] for t in state.conversation_context]
        assert roles.count("user") == 2
        assert roles.count("assistant") == 2


# ===========================================================================
# 14. Hospital calibration tool-calling regression
# ===========================================================================
#
# Real bug, reproduced live against OpenRouter (meta-llama/llama-3.1-8b-instruct):
# "Set ICU occupied beds to 9." produced
#   propose_hospital_calibration(department="ICU", update={})
# — a syntactically valid call with a structurally empty "update" — because
# its schema declared "update" as a bare {"type": "object"} with no
# properties. The call failed ("update must be a non-empty dict"), and the
# model retried the IDENTICAL empty-update call 5 times in a row (measured
# live) before the iteration cap terminated the turn. These tests cover the
# schema fix and the generic repeated-identical-failure safeguard, using
# synthetic data — no patient/department-specific detection in the fix
# itself.

from triageguard_agent.tools.hospital_tools import (  # noqa: E402
    propose_hospital_calibration_spec,
    commit_hospital_calibration_spec,
)


class TestHospitalCalibrationSchema:
    def test_propose_update_schema_has_named_properties(self):
        """
        Schema regression: "update" must no longer be a bare
        {"type": "object"} with zero properties — that under-specification
        is exactly what left the model sending update={}.
        """
        spec = propose_hospital_calibration_spec()
        update_schema = spec.input_schema["properties"]["update"]

        assert update_schema["type"] == "object"
        properties = update_schema.get("properties", {})
        assert "capacity" in properties
        assert "occupied" in properties
        assert "status" in properties

    def test_commit_validated_update_schema_has_named_properties(self):
        spec = commit_hospital_calibration_spec()
        update_schema = spec.input_schema["properties"]["validated_update"]

        properties = update_schema.get("properties", {})
        assert "capacity" in properties
        assert "occupied" in properties
        assert "status" in properties


class TestHospitalCalibrationFullFlow:
    """
    End-to-end: propose (freely, no confirmation) -> commit (confirmation
    required) -> real backend state change, exactly matching the intended
    nurse-request -> propose -> confirmation -> commit-only-after-confirmation
    workflow. Every assertion here checks REAL HospitalStateService state,
    not the agent's text.
    """

    def test_propose_runs_freely_and_returns_real_proposal(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "propose_hospital_calibration",
                {"department": "ICU", "update": {"occupied": 9}},
            ),
            _final_message("I can update ICU occupancy to 9 — shall I proceed?"),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Set ICU occupied beds to 9.", state)

        assert response.actions[0]["tool"] == "propose_hospital_calibration"
        assert response.actions[0]["status"] == "executed"
        assert response.actions[0]["data"]["proposed_update"]["occupied"] == 9
        # Proposing must NOT itself require confirmation or change state
        assert not state.has_pending()
        real_state = HospitalStateService.instance().get_state("ICU")
        assert real_state["occupied"] == 8  # unchanged — still the config default

    def test_commit_requires_confirmation_and_state_unchanged_until_then(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"capacity": 10, "occupied": 9, "status": "OPEN"}},
            ),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Set ICU occupied beds to 9.", state)

        assert response.response_type == "approval_required"
        assert state.has_pending()
        real_state = HospitalStateService.instance().get_state("ICU")
        assert real_state["occupied"] == 8  # NOT changed yet — confirmation still pending

    def test_yes_commits_real_state_change(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"capacity": 10, "occupied": 9, "status": "OPEN"}},
            ),
        ])
        state = AgentState(session_id="s1")
        runtime.process_turn("Set ICU occupied beds to 9.", state)

        response = runtime.process_turn("yes", state)

        assert response.response_type == "confirmation"
        assert not state.has_pending()
        real_state = HospitalStateService.instance().get_state("ICU")
        assert real_state["occupied"] == 9  # actually changed in the real backend

    def test_no_leaves_state_unchanged(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"capacity": 10, "occupied": 9, "status": "OPEN"}},
            ),
        ])
        state = AgentState(session_id="s1")
        runtime.process_turn("Set ICU occupied beds to 9.", state)

        response = runtime.process_turn("no", state)

        assert response.response_type == "information"
        real_state = HospitalStateService.instance().get_state("ICU")
        assert real_state["occupied"] == 8  # untouched


class TestRepeatedFailingToolCallDoesNotLoopIndefinitely:
    def test_identical_failing_call_stops_after_second_attempt(self, runtime_factory):
        """
        Regression for the exact live-measured behavior: the model retried
        propose_hospital_calibration(department="ICU", update={}) 5 times
        (once per available iteration) with the OLD under-specified schema.
        Even if a model still does this for some other reason, the loop
        must stop on the SECOND identical failure rather than exhausting
        the full iteration budget.
        """
        empty_update_call = _tool_call_message(
            "call_x", "propose_hospital_calibration", {"department": "ICU", "update": {}}
        )
        # Script 5 identical (still-empty) attempts available — if the
        # safeguard works, only 2 are ever actually consumed.
        runtime, llm = runtime_factory([empty_update_call] * 5)
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Set ICU occupied beds to 9.", state)

        assert llm.call_count == 2  # stopped after the repeat, not all 5
        assert response.response_type == "error"
        assert "same information" in response.message or "same way" in response.message
        assert "clarify" in response.message.lower()

    def test_different_arguments_are_not_treated_as_repeats(self, runtime_factory):
        """A second attempt with genuinely different (still-empty-of-value) arguments
        is a different signature and should NOT be short-circuited as a repeat —
        only exact repeats are."""
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1", "propose_hospital_calibration", {"department": "ICU", "update": {}}
            ),
            _tool_call_message(
                "call_2", "propose_hospital_calibration", {"department": "CICU", "update": {}}
            ),
            _final_message("Neither update had any fields — could you specify what to change?"),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("Update ICU and CICU.", state)

        # Different department => different signature => not short-circuited;
        # the loop continued to a 3rd (final-answer) LLM call.
        assert llm.call_count == 3
        assert response.response_type == "information"


# ===========================================================================
# 15. Tool-selection safety net (regression for the "wrong tool selected"
#     planning bug)
# ===========================================================================
#
# Real bug, reproduced live against OpenRouter (meta-llama/llama-3.1-8b-instruct)
# in a multi-turn session: after an earlier hospital-calibration interaction
# (even a REJECTED one), later, completely unrelated requests —
# "how many icu beds are pending" (a READ request) and "delete patient 52"
# (a request with no matching tool at all) — got mapped onto
# commit_hospital_calibration, reusing whatever ICU occupancy number was
# most recently seen in conversation context.
#
# This is fundamentally a real-model judgment/planning-quality problem, not
# something a scripted fake LLM can be made to "get right" or "get wrong" in
# a unit test — these tests instead verify the two things that actually can
# and must be deterministically guaranteed regardless of what the LLM
# proposes:
#   1. The corrective guidance is actually present in what gets sent to the
#      model (system prompt content + tool description content) — a
#      regression guard against silently losing this fix.
#   2. No matter which tool a (possibly still-wrong) LLM decision proposes,
#      the existing deterministic safety net still holds: a WRITE action
#      never executes without confirmation, rejecting it changes nothing in
#      the real backend, and the nurse always sees their own request
#      alongside the proposed action so a mismatch is visible.
# These use synthetic, generic scenarios (a made-up unsupported capability,
# a generic informational phrasing) rather than the exact reported strings.

from triageguard_agent.llm.system_prompt import SYSTEM_PROMPT as _SYSTEM_PROMPT_TEXT  # noqa: E402


class TestSystemPromptGuardsToolSelection:
    """Regression guards: these phrases must not silently disappear from the prompt."""

    def test_prompt_forbids_write_tool_for_read_requests(self):
        assert "READ request" in _SYSTEM_PROMPT_TEXT
        assert "never" in _SYSTEM_PROMPT_TEXT.lower()
        assert "WRITE tool" in _SYSTEM_PROMPT_TEXT

    def test_prompt_forbids_substituting_unrelated_tool(self):
        assert "repurpose" in _SYSTEM_PROMPT_TEXT.lower() or "substitut" in _SYSTEM_PROMPT_TEXT.lower()

    def test_prompt_requires_grounding_in_current_message(self):
        assert "CURRENT message" in _SYSTEM_PROMPT_TEXT

    def test_prompt_requires_clarification_for_ambiguity(self):
        assert "clarifying question" in _SYSTEM_PROMPT_TEXT.lower() or "ambiguous" in _SYSTEM_PROMPT_TEXT.lower()

    def test_hospital_calibration_tool_descriptions_state_their_boundary(self):
        propose_spec = propose_hospital_calibration_spec()
        commit_spec = commit_hospital_calibration_spec()
        assert "get_hospital_state instead" in propose_spec.description
        assert "no patient-deletion" in commit_spec.description.lower() or "does not exist" in commit_spec.description.lower()


class TestToolSelectionSafetyNetHoldsRegardlessOfLLMChoice:
    """
    Even if a real model (still) proposes the wrong tool for a request, the
    deterministic architecture around it must make that harmless: the write
    never executes without confirmation, and rejecting it changes nothing.
    """

    def test_write_proposed_for_a_read_shaped_request_is_still_gated(self, runtime_factory):
        # Simulates the exact bug shape: an informational request, but the
        # (adversarial/still-imperfect) LLM proposes a WRITE tool anyway.
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"occupied": 7}},
            ),
        ])
        state = AgentState(session_id="s1")
        before = dict(HospitalStateService.instance().get_state("ICU"))

        response = runtime.process_turn("how many icu beds are pending", state)

        assert response.response_type == "approval_required"
        assert state.has_pending()
        # Never executed just because it was proposed
        after = HospitalStateService.instance().get_state("ICU")
        assert after["occupied"] == before["occupied"]

        # And rejecting it changes nothing either
        reject_response = runtime.process_turn("no", state)
        assert not state.has_pending()
        final_state = HospitalStateService.instance().get_state("ICU")
        assert final_state["occupied"] == before["occupied"]

    def test_confirmation_prompt_echoes_the_actual_nurse_request(self, runtime_factory):
        """
        Defense-in-depth: whatever the nurse actually said must appear in the
        confirmation prompt, so a mismatch between request and proposed
        action is visible to the human reviewing it, not hidden.
        """
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"occupied": 8}},
            ),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("delete patient 52", state)

        assert response.response_type == "approval_required"
        assert "delete patient 52" in response.message

    def test_write_proposed_for_unsupported_capability_request_is_gated_and_rejectable(self, runtime_factory):
        # Generic stand-in for "a capability that doesn't exist" — not the
        # literal reported phrase, to avoid a hardcoded-string test.
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"occupied": 3}},
            ),
        ])
        state = AgentState(session_id="s1")
        before = dict(HospitalStateService.instance().get_state("ICU"))

        response = runtime.process_turn("please archive patient 999's old chart", state)

        assert response.response_type == "approval_required"
        reject_response = runtime.process_turn("no", state)
        assert reject_response.response_type == "information"
        after = HospitalStateService.instance().get_state("ICU")
        assert after["occupied"] == before["occupied"]

    def test_generic_write_tool_rejection_leaves_no_state_mutation(self, runtime_factory):
        """Same guarantee, but for a different WRITE tool entirely (not hospital
        calibration) — the safety net is generic, not specific to one tool."""
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1", "admit_simulated_patient", {"patient_id": "some-unrelated-id", "department": "ICU"}
            ),
        ])
        state = AgentState(session_id="s1")
        before = dict(HospitalStateService.instance().get_state("ICU"))

        runtime.process_turn("what's today's weather", state)
        response = runtime.process_turn("no", state)

        assert response.response_type == "information"
        after = HospitalStateService.instance().get_state("ICU")
        assert after["occupied"] == before["occupied"]


class TestAmbiguousRequestAsksForClarification:
    def test_text_only_clarifying_response_has_no_side_effects(self, runtime_factory):
        runtime, llm = runtime_factory([
            _final_message("Which patient did you mean — could you give me a patient ID or full name?"),
        ])
        state = AgentState(session_id="s1")

        response = runtime.process_turn("open the patient", state)

        assert response.response_type == "information"
        assert response.actions == []
        assert not state.has_pending()


class TestValidHospitalCalibrationStillWorksAfterPromptChanges:
    """
    Confirms the legitimate path wasn't collateral damage from tightening
    tool-selection guidance: a genuine calibration request still flows
    propose -> confirmation -> commit, with the real backend only changing
    after explicit confirmation.
    """

    def test_full_propose_confirm_commit_path(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "propose_hospital_calibration",
                {"department": "ICU", "update": {"occupied": 9}},
            ),
            _final_message("ICU occupancy would become 9/10 — shall I proceed?"),
        ])
        state = AgentState(session_id="s1")

        propose_response = runtime.process_turn("Set ICU occupied beds to 9.", state)
        assert propose_response.actions[0]["tool"] == "propose_hospital_calibration"
        assert not state.has_pending()
        mid_state = HospitalStateService.instance().get_state("ICU")
        assert mid_state["occupied"] == 8  # proposing alone never changes state

        runtime2, llm2 = runtime_factory([
            _tool_call_message(
                "call_2",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"capacity": 10, "occupied": 9, "status": "OPEN"}},
            ),
        ])
        state2 = AgentState(session_id="s2")
        runtime2.process_turn("Set ICU occupied beds to 9.", state2)
        assert state2.has_pending()

        confirm_response = runtime2.process_turn("yes", state2)
        assert confirm_response.response_type == "confirmation"
        final_state = HospitalStateService.instance().get_state("ICU")
        assert final_state["occupied"] == 9  # only now, after explicit confirmation


# ===========================================================================
# 16. Confirmation description persists across an ambiguous re-prompt
# ===========================================================================
#
# Found while verifying the tool-selection safety net: the rich description
# built at proposal time (which includes the "You asked: ..." echo added in
# the previous fix) was shown on the FIRST approval_required message, but
# AgentState.set_pending() only ever stored {action_type, payload} — not the
# description — so a later ambiguous reply's re-prompt fell back to the
# generic "the proposed action" placeholder instead of repeating the real
# description. Fixed by threading the description through
# ConfirmationProtocol.require_confirmation() -> AgentState.set_pending().

class TestConfirmationDescriptionPersistsAcrossAmbiguousRetry:
    def test_reprompt_after_ambiguous_reply_repeats_the_original_description(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"occupied": 9}},
            ),
        ])
        state = AgentState(session_id="s1")

        first = runtime.process_turn("Set ICU occupied beds to 9.", state)
        assert "Set ICU occupied beds to 9." in first.message

        # An ambiguous reply must NOT lose the original description — it
        # should still show what was actually asked, not a generic fallback.
        reprompt = runtime.process_turn("hmm not sure", state)

        assert state.has_pending()  # still pending, nothing executed
        assert reprompt.response_type == "approval_required"
        assert "Set ICU occupied beds to 9." in reprompt.message
        assert "the proposed action" not in reprompt.message

    def test_pending_action_dict_carries_description(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"occupied": 9}},
            ),
        ])
        state = AgentState(session_id="s1")
        runtime.process_turn("Set ICU occupied beds to 9.", state)

        assert state.pending_action.get("description") is not None
        assert "Set ICU occupied beds to 9." in state.pending_action["description"]

    def test_multiple_ambiguous_replies_keep_repeating_the_same_real_description(self, runtime_factory):
        runtime, llm = runtime_factory([
            _tool_call_message(
                "call_1",
                "commit_hospital_calibration",
                {"department": "ICU", "validated_update": {"occupied": 9}},
            ),
        ])
        state = AgentState(session_id="s1")
        runtime.process_turn("Set ICU occupied beds to 9.", state)

        for _ in range(3):
            r = runtime.process_turn("what do you mean", state)
            assert "Set ICU occupied beds to 9." in r.message
            assert state.has_pending()
