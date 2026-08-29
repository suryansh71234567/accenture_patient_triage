"""
test_agent_foundation.py
------------------------
Unit tests for the TriageGuard agent foundation layer.

Tests
-----
1.  ToolResult.ok() — success serialisation
2.  ToolResult.fail() — failure serialisation
3.  AgentResponse — field validation and type guard
4.  AgentState — defaults and mutation
5.  WorkingMemory — add/clear lifecycle
6.  ToolRegistry — register and lookup
7.  ToolRegistry — WRITE tool requires approval_gate in ToolExecutor
8.  SkillRegistry — lazy load of SKILL.md
9.  ContextManager — context dict structure
"""

import sys
import os
from pathlib import Path

# Ensure repo root is on path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_agent.schemas.tool_result import ToolResult
from triageguard_agent.schemas.agent_response import AgentResponse
from triageguard_agent.state.agent_state import AgentState
from triageguard_agent.state.working_memory import WorkingMemory
from triageguard_agent.tools.registry import ToolRegistry, ToolSpec, READ, COMPUTE, WRITE
from triageguard_agent.runtime.tool_executor import ToolExecutor
from triageguard_agent.skills.registry import SkillRegistry
from triageguard_agent.context.context_manager import ContextManager


# ===========================================================================
# 1. ToolResult.ok()
# ===========================================================================

class TestToolResultSuccess:
    def test_success_flag(self):
        result = ToolResult.ok("my_tool", {"value": 42})
        assert result.success is True

    def test_tool_name(self):
        result = ToolResult.ok("my_tool", {"value": 42})
        assert result.tool == "my_tool"

    def test_data_preserved(self):
        result = ToolResult.ok("my_tool", {"x": 1, "y": 2})
        assert result.data == {"x": 1, "y": 2}

    def test_error_is_none(self):
        result = ToolResult.ok("my_tool", {})
        assert result.error is None

    def test_metadata_default_empty(self):
        result = ToolResult.ok("my_tool", {})
        assert result.metadata == {}

    def test_metadata_preserved(self):
        result = ToolResult.ok("my_tool", {}, metadata={"source": "test"})
        assert result.metadata["source"] == "test"

    def test_to_dict_keys(self):
        d = ToolResult.ok("t", {"a": 1}).to_dict()
        assert set(d.keys()) == {"success", "tool", "data", "error", "metadata"}

    def test_to_dict_success_true(self):
        d = ToolResult.ok("t", {}).to_dict()
        assert d["success"] is True


# ===========================================================================
# 2. ToolResult.fail()
# ===========================================================================

class TestToolResultFailure:
    def test_success_flag(self):
        result = ToolResult.fail("t", "ERR_CODE", "Something went wrong.")
        assert result.success is False

    def test_data_is_none(self):
        result = ToolResult.fail("t", "ERR_CODE", "msg")
        assert result.data is None

    def test_error_code(self):
        result = ToolResult.fail("t", "MY_ERROR", "msg")
        assert result.error["code"] == "MY_ERROR"

    def test_error_message(self):
        result = ToolResult.fail("t", "CODE", "Detailed message.")
        assert result.error["message"] == "Detailed message."

    def test_to_dict_success_false(self):
        d = ToolResult.fail("t", "C", "M").to_dict()
        assert d["success"] is False

    def test_repr_contains_code(self):
        result = ToolResult.fail("t", "BOOM", "msg")
        assert "BOOM" in repr(result)


# ===========================================================================
# 3. AgentResponse
# ===========================================================================

class TestAgentResponse:
    def test_valid_response_type(self):
        r = AgentResponse(message="Hello", response_type="information")
        assert r.response_type == "information"

    def test_invalid_response_type_raises(self):
        with pytest.raises(ValueError):
            AgentResponse(message="Hello", response_type="invalid_type")

    def test_defaults(self):
        r = AgentResponse(message="Hi", response_type="assessment")
        assert r.patient_id is None
        assert r.actions == []
        assert r.evidence == []
        assert r.human_approval_required is False

    def test_to_dict_keys(self):
        r = AgentResponse(message="x", response_type="error")
        d = r.to_dict()
        assert "message" in d
        assert "response_type" in d
        assert "human_approval_required" in d

    def test_error_response_factory(self):
        r = AgentResponse.error_response("Something failed.")
        assert r.response_type == "error"
        assert r.human_approval_required is False

    def test_approval_required_factory(self):
        r = AgentResponse.approval_required(
            "Confirm this?", actions=[{"action": "do_thing"}], patient_id="42"
        )
        assert r.response_type == "approval_required"
        assert r.human_approval_required is True
        assert r.patient_id == "42"

    def test_all_valid_types(self):
        for t in ["information", "assessment", "confirmation", "approval_required", "error"]:
            r = AgentResponse(message="x", response_type=t)
            assert r.response_type == t


# ===========================================================================
# 4. AgentState
# ===========================================================================

class TestAgentState:
    def test_defaults(self):
        state = AgentState(session_id="s1")
        assert state.user_role == "nurse"
        assert state.active_patient_id is None
        assert state.active_task == "idle"
        assert state.pending_action is None
        assert state.conversation_context == []

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            AgentState(session_id="s1", user_role="janitor")

    def test_add_turn(self):
        state = AgentState(session_id="s1")
        state.add_turn("user", "Hello")
        assert len(state.conversation_context) == 1
        assert state.conversation_context[0]["role"] == "user"

    def test_context_trimmed_at_max(self):
        state = AgentState(session_id="s1")
        for i in range(15):
            state.add_turn("user", f"msg {i}")
        assert len(state.conversation_context) <= 10

    def test_set_pending(self):
        state = AgentState(session_id="s1")
        state.set_pending("commit_something", {"key": "val"})
        assert state.has_pending()
        assert state.pending_action["action_type"] == "commit_something"

    def test_clear_pending(self):
        state = AgentState(session_id="s1")
        state.set_pending("x", {})
        state.clear_pending()
        assert not state.has_pending()

    def test_to_dict(self):
        state = AgentState(session_id="abc", user_role="doctor")
        d = state.to_dict()
        assert d["session_id"] == "abc"
        assert d["user_role"] == "doctor"


# ===========================================================================
# 5. WorkingMemory
# ===========================================================================

class TestWorkingMemory:
    def test_starts_empty(self):
        wm = WorkingMemory()
        assert len(wm.tool_results) == 0
        assert wm.active_skill is None

    def test_add_tool_result(self):
        wm = WorkingMemory()
        wm.add_tool_result({"tool": "t", "success": True, "data": {}})
        assert len(wm.tool_results) == 1

    def test_successful_results_filter(self):
        wm = WorkingMemory()
        wm.add_tool_result({"tool": "a", "success": True})
        wm.add_tool_result({"tool": "b", "success": False})
        assert len(wm.successful_results()) == 1

    def test_failed_results_filter(self):
        wm = WorkingMemory()
        wm.add_tool_result({"tool": "a", "success": False})
        assert len(wm.failed_results()) == 1

    def test_clear_resets_all(self):
        wm = WorkingMemory()
        wm.add_tool_result({"tool": "x", "success": True})
        wm.active_skill = "triage_assessment"
        wm.add_note("test note")
        wm.clear()
        assert len(wm.tool_results) == 0
        assert wm.active_skill is None
        assert len(wm.notes) == 0

    def test_get_last_tool_result(self):
        wm = WorkingMemory()
        wm.add_tool_result({"tool": "get_hospital_state", "success": True, "data": {"x": 1}})
        r = wm.get_last_tool_result("get_hospital_state")
        assert r is not None
        assert r["data"]["x"] == 1

    def test_get_last_tool_result_missing(self):
        wm = WorkingMemory()
        assert wm.get_last_tool_result("nonexistent") is None


# ===========================================================================
# 6. ToolRegistry — register and lookup
# ===========================================================================

class TestToolRegistry:
    def _make_spec(self, name: str, risk: str = READ) -> ToolSpec:
        side = risk == WRITE
        return ToolSpec(
            name=name,
            description=f"Test tool {name}",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda **kw: ToolResult.ok(name, {}),
            risk_level=risk,
            side_effect=side,
            requires_approval=False,
        )

    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(self._make_spec("my_tool"))
        assert reg.get("my_tool") is not None

    def test_get_missing_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("no_such_tool") is None

    def test_require_raises_on_missing(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.require("ghost_tool")

    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        reg.register(self._make_spec("tool_a"))
        with pytest.raises(ValueError):
            reg.register(self._make_spec("tool_a"))

    def test_list_tools_includes_all(self):
        reg = ToolRegistry()
        reg.register(self._make_spec("t1"))
        reg.register(self._make_spec("t2"))
        names = [t["name"] for t in reg.list_tools()]
        assert "t1" in names and "t2" in names

    def test_list_by_risk(self):
        reg = ToolRegistry()
        reg.register(self._make_spec("read_tool", READ))
        reg.register(self._make_spec("write_tool", WRITE))
        reads = reg.list_by_risk(READ)
        assert all(s.risk_level == READ for s in reads)

    def test_contains(self):
        reg = ToolRegistry()
        reg.register(self._make_spec("check_me"))
        assert "check_me" in reg
        assert "nope" not in reg

    def test_len(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(self._make_spec("one"))
        assert len(reg) == 1


# ===========================================================================
# 7. ToolExecutor — WRITE approval gate
# ===========================================================================

class TestToolExecutorApprovalGate:
    def _make_write_spec(self) -> ToolSpec:
        return ToolSpec(
            name="dangerous_write",
            description="A write tool",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda **kw: ToolResult.ok("dangerous_write", {"done": True}),
            risk_level=WRITE,
            side_effect=True,
            requires_approval=True,
        )

    def test_write_without_token_fails(self):
        reg = ToolRegistry()
        reg.register(self._make_write_spec())
        executor = ToolExecutor(reg)
        result = executor.execute("dangerous_write", {}, approval_token=None)
        assert result.success is False
        assert result.error["code"] == "APPROVAL_REQUIRED"

    def test_write_with_token_succeeds(self):
        reg = ToolRegistry()
        reg.register(self._make_write_spec())
        executor = ToolExecutor(reg)
        result = executor.execute("dangerous_write", {}, approval_token="nurse_confirmed")
        assert result.success is True

    def test_unknown_tool_fails(self):
        reg = ToolRegistry()
        executor = ToolExecutor(reg)
        result = executor.execute("not_a_tool", {})
        assert result.success is False
        assert result.error["code"] == "TOOL_NOT_FOUND"

    def test_read_tool_no_token_needed(self):
        spec = ToolSpec(
            name="safe_read",
            description="Read",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda **kw: ToolResult.ok("safe_read", {"data": "ok"}),
            risk_level=READ,
            side_effect=False,
            requires_approval=False,
        )
        reg = ToolRegistry()
        reg.register(spec)
        executor = ToolExecutor(reg)
        result = executor.execute("safe_read", {})
        assert result.success is True

    def test_missing_required_field_fails(self):
        spec = ToolSpec(
            name="needs_arg",
            description="Needs patient_id",
            input_schema={
                "type": "object",
                "properties": {"patient_id": {"type": "string"}},
                "required": ["patient_id"],
            },
            handler=lambda **kw: ToolResult.ok("needs_arg", {}),
            risk_level=READ,
            side_effect=False,
            requires_approval=False,
        )
        reg = ToolRegistry()
        reg.register(spec)
        executor = ToolExecutor(reg)
        result = executor.execute("needs_arg", {})  # missing patient_id
        assert result.success is False
        assert result.error["code"] == "MISSING_REQUIRED_FIELDS"


# ===========================================================================
# 8. SkillRegistry — lazy load
# ===========================================================================

class TestSkillRegistry:
    def test_register_and_load(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            "---\nname: test_skill\ndescription: A test skill.\n---\n\n# Body text.",
            encoding="utf-8",
        )
        reg = SkillRegistry()
        reg.register("test_skill", skill_file)
        text = reg.load("test_skill")
        assert text is not None
        assert "Body text" in text

    def test_description_from_frontmatter(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            "---\nname: s\ndescription: My skill description.\n---\n\n# Body.",
            encoding="utf-8",
        )
        reg = SkillRegistry()
        reg.register("s", skill_file)
        assert reg.get_description("s") == "My skill description."

    def test_load_missing_returns_none(self):
        reg = SkillRegistry()
        result = reg.load("not_registered")
        assert result is None

    def test_register_missing_file_raises(self, tmp_path):
        reg = SkillRegistry()
        with pytest.raises(FileNotFoundError):
            reg.register("ghost", tmp_path / "nonexistent.md")

    def test_list_skills(self, tmp_path):
        for name in ["skill_a", "skill_b"]:
            p = tmp_path / f"{name}.md"
            p.write_text(f"---\nname: {name}\ndescription: Desc.\n---\n# body", encoding="utf-8")
            pass
        reg = SkillRegistry()
        for name in ["skill_a", "skill_b"]:
            reg.register(name, tmp_path / f"{name}.md")
        listing = reg.list_skills()
        names = [s["name"] for s in listing]
        assert "skill_a" in names
        assert "skill_b" in names

    def test_default_registry_loads_standard_skills(self):
        from triageguard_agent.skills.registry import build_default_registry
        reg = build_default_registry()
        assert len(reg) >= 8
        assert reg.is_registered("triage_assessment")
        assert reg.is_registered("hospital_status")
        assert reg.is_registered("human_review")

    def test_cache_prevents_double_read(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\nname: c\ndescription: Cached.\n---\nBody.", encoding="utf-8")
        reg = SkillRegistry()
        reg.register("c", skill_file)
        t1 = reg.load("c")
        # Modify file after first load
        skill_file.write_text("CHANGED", encoding="utf-8")
        t2 = reg.load("c")
        assert t1 == t2  # cached, not re-read


# ===========================================================================
# 9. ContextManager — structure
# ===========================================================================

class TestContextManager:
    def _make_skill_registry(self, tmp_path) -> SkillRegistry:
        p = tmp_path / "test_skill.md"
        p.write_text(
            "---\nname: test_skill\ndescription: Test.\n---\n# Instructions here.",
            encoding="utf-8",
        )
        reg = SkillRegistry()
        reg.register("test_skill", p)
        return reg

    def test_context_has_required_keys(self, tmp_path):
        reg = self._make_skill_registry(tmp_path)
        cm = ContextManager(skill_registry=reg)
        state = AgentState(session_id="x", active_patient_id="42")
        wm = WorkingMemory()
        ctx = cm.build_context(state, wm)
        for key in ["session_info", "active_patient_ref", "skill_context",
                    "tool_results", "pending_confirmation", "conversation_context", "warnings"]:
            assert key in ctx, f"Missing key: {key}"

    def test_skill_context_loaded_when_specified(self, tmp_path):
        reg = self._make_skill_registry(tmp_path)
        cm = ContextManager(skill_registry=reg)
        state = AgentState(session_id="x")
        wm = WorkingMemory()
        ctx = cm.build_context(state, wm, active_skill="test_skill")
        assert ctx["skill_context"] is not None
        assert "Instructions here" in ctx["skill_context"]

    def test_skill_context_none_when_not_specified(self, tmp_path):
        reg = self._make_skill_registry(tmp_path)
        cm = ContextManager(skill_registry=reg)
        state = AgentState(session_id="x")
        wm = WorkingMemory()
        ctx = cm.build_context(state, wm)
        assert ctx["skill_context"] is None

    def test_warnings_include_failed_tools(self, tmp_path):
        reg = self._make_skill_registry(tmp_path)
        cm = ContextManager(skill_registry=reg)
        state = AgentState(session_id="x")
        wm = WorkingMemory()
        wm.add_tool_result({
            "tool": "bad_tool",
            "success": False,
            "error": {"code": "FAIL", "message": "it broke"},
        })
        ctx = cm.build_context(state, wm)
        assert any("bad_tool" in w for w in ctx["warnings"])

    def test_tool_results_bounded(self, tmp_path):
        reg = self._make_skill_registry(tmp_path)
        cm = ContextManager(skill_registry=reg, max_tool_results=2)
        state = AgentState(session_id="x")
        wm = WorkingMemory()
        for i in range(5):
            wm.add_tool_result({"tool": f"t{i}", "success": True, "data": {}})
        ctx = cm.build_context(state, wm)
        assert len(ctx["tool_results"]) <= 2

    def test_pending_confirmation_included(self, tmp_path):
        reg = self._make_skill_registry(tmp_path)
        cm = ContextManager(skill_registry=reg)
        state = AgentState(session_id="x")
        state.set_pending("commit_something", {"dept": "ICU"})
        wm = WorkingMemory()
        ctx = cm.build_context(state, wm)
        assert ctx["pending_confirmation"] is not None
        assert ctx["pending_confirmation"]["action_type"] == "commit_something"
