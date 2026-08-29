"""
agent_runtime.py
----------------
AgentRuntime — the central orchestrator for the TriageGuard agent.

Scope of this implementation (foundation phase)
------------------------------------------------
This module wires together all the agent components:
    ToolRegistry + ToolExecutor
    SkillRegistry
    ContextManager
    WorkingMemory
    AgentState
    ConfirmationProtocol

It does NOT yet implement the full LLM planning loop (that is the next phase).
The runtime is designed so the LLM loop can be dropped in without changing
any of the surrounding infrastructure.

Lifecycle of one turn
---------------------
1. Receive user input + agent state.
2. Clear working memory from the prior turn.
3. Detect if there is a pending confirmation action.
4. Build the context for this turn.
5. [NEXT PHASE: call LLM with context and tools list]
6. [NEXT PHASE: parse tool calls from LLM response]
7. Execute requested tools via ToolExecutor.
8. Collect tool results into WorkingMemory.
9. [NEXT PHASE: generate final response text from LLM]
10. Return structured AgentResponse.
"""

from __future__ import annotations
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from triageguard_agent.schemas.tool_result import ToolResult
from triageguard_agent.schemas.agent_response import AgentResponse
from triageguard_agent.state.agent_state import AgentState
from triageguard_agent.state.working_memory import WorkingMemory
from triageguard_agent.tools.registry import ToolRegistry, ToolSpec, READ, COMPUTE, WRITE
from triageguard_agent.runtime.tool_executor import ToolExecutor
from triageguard_agent.skills.registry import SkillRegistry, build_default_registry
from triageguard_agent.context.context_manager import ContextManager
from triageguard_agent.protocols.confirmation_protocol import ConfirmationProtocol

logger = logging.getLogger(__name__)


class AgentRuntime:
    """
    Central orchestrator for the TriageGuard conversational agent.

    Parameters
    ----------
    tool_registry   : Pre-built ToolRegistry. If None, an empty one is created.
    skill_registry  : Pre-built SkillRegistry. If None, the default one is built.
    auto_register   : If True, register all standard tools automatically on init.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        skill_registry: Optional[SkillRegistry] = None,
        auto_register: bool = True,
    ) -> None:
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or build_default_registry()
        self.context_manager = ContextManager(skill_registry=self.skill_registry)
        self.executor = ToolExecutor(self.tool_registry)
        self.confirmation = ConfirmationProtocol()

        # Per-session working memory (cleared each turn)
        self._working_memory = WorkingMemory()

        if auto_register:
            self._register_standard_tools()

        logger.info(
            "AgentRuntime initialised with %d tools, %d skills.",
            len(self.tool_registry),
            len(self.skill_registry),
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def new_session(self, user_role: str = "nurse") -> AgentState:
        """Create a new AgentState for a fresh session."""
        return AgentState(
            session_id=str(uuid.uuid4()),
            user_role=user_role,
        )

    # ------------------------------------------------------------------
    # Main entry point (foundation — LLM loop not yet wired)
    # ------------------------------------------------------------------

    def process_turn(
        self,
        user_input: str,
        agent_state: AgentState,
        active_skill: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process one conversational turn.

        In this foundation phase:
        - Clears working memory.
        - Checks for pending confirmation.
        - Builds context dict.
        - Appends the turn to conversation context.
        - Returns an AgentResponse (message is stub — LLM not yet wired).

        Parameters
        ----------
        user_input   : The nurse/staff's raw input text.
        agent_state  : Current session state (mutated in-place).
        active_skill : Override which skill to load for context.
        """
        # ── 1. Clear working memory from prior turn ───────────────────
        self._working_memory.clear()

        # ── 2. Check pending confirmation ─────────────────────────────
        if agent_state.has_pending():
            return self._handle_pending_confirmation(user_input, agent_state)

        # ── 3. Build context ──────────────────────────────────────────
        ctx = self.context_manager.build_context(
            agent_state, self._working_memory, active_skill
        )

        # ── 4. Record this turn in conversation context ───────────────
        agent_state.add_turn("user", user_input)

        # ── 5. [PLACEHOLDER] LLM planning loop ───────────────────────
        # In the next phase: call LLM with ctx + tool list → parse tool calls
        # → execute via self.executor → collect results → call LLM for response
        logger.debug(
            "Context built for turn: patient=%s task=%s",
            agent_state.active_patient_id,
            agent_state.active_task,
        )

        # Foundation response — the LLM will replace this in the next phase
        response = AgentResponse(
            message=(
                "[AgentRuntime foundation mode] "
                "Context assembled. LLM planning loop not yet wired. "
                f"Active patient: {agent_state.active_patient_id or 'none'}. "
                f"Task: {agent_state.active_task}. "
                f"Tools available: {len(self.tool_registry)}."
            ),
            response_type="information",
            patient_id=agent_state.active_patient_id,
        )

        agent_state.add_turn("assistant", response.message)
        return response

    # ------------------------------------------------------------------
    # Tool execution (called by the LLM loop in the next phase,
    # and available for direct use in tests / scripts)
    # ------------------------------------------------------------------

    def run_tool(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        agent_state: Optional[AgentState] = None,
        approval_token: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute a tool and store the result in working memory.

        Parameters
        ----------
        tool_name      : Registered tool name.
        kwargs         : Tool arguments.
        agent_state    : If provided, updates hospital_state_timestamp after
                         hospital state reads.
        approval_token : Required for WRITE tools.
        """
        result = self.executor.execute(tool_name, kwargs, approval_token)
        self._working_memory.add_tool_result(result.to_dict())

        # Update hospital state timestamp after a successful hospital read
        if (
            result.success
            and tool_name == "get_hospital_state"
            and agent_state is not None
        ):
            agent_state.hospital_state_timestamp = datetime.now(timezone.utc)

        if not result.success:
            err = result.error or {}
            self._working_memory.add_note(
                f"Tool {tool_name!r} failed: [{err.get('code')}] {err.get('message')}"
            )

        return result

    # ------------------------------------------------------------------
    # Pending confirmation handler
    # ------------------------------------------------------------------

    def _handle_pending_confirmation(
        self,
        user_input: str,
        agent_state: AgentState,
    ) -> AgentResponse:
        """
        Resolve a pending WRITE action confirmation.

        Returns an AgentResponse reporting whether the action was
        confirmed, rejected, or the input was ambiguous.
        """
        pending = agent_state.pending_action
        resolution = self.confirmation.resolve(user_input)

        if resolution == "confirmed":
            action_type = pending["action_type"]
            payload = pending["payload"]
            agent_state.clear_pending()

            # Execute the approved WRITE tool
            result = self.run_tool(
                action_type,
                payload,
                agent_state=agent_state,
                approval_token="nurse_confirmed",
            )

            if result.success:
                agent_state.add_turn("user", user_input)
                agent_state.add_turn("assistant", "Action confirmed and executed.")
                return AgentResponse(
                    message=(
                        f"Action confirmed. {action_type} executed successfully. "
                        f"Result: {result.data}"
                    ),
                    response_type="confirmation",
                    patient_id=agent_state.active_patient_id,
                    actions=[{"tool": action_type, "status": "executed", "data": result.data}],
                )
            else:
                err = result.error or {}
                agent_state.add_turn("user", user_input)
                agent_state.add_turn("assistant", "Action failed after confirmation.")
                return AgentResponse.error_response(
                    f"Action {action_type} failed after confirmation: "
                    f"[{err.get('code')}] {err.get('message')}"
                )

        elif resolution == "rejected":
            pending_desc = pending.get("payload", {})
            agent_state.clear_pending()
            agent_state.add_turn("user", user_input)
            agent_state.add_turn("assistant", "Action cancelled.")
            return AgentResponse(
                message="Action cancelled. No changes were made.",
                response_type="information",
                patient_id=agent_state.active_patient_id,
            )

        else:
            # Ambiguous — ask again
            prompt = self.confirmation.build_confirmation_prompt(pending)
            agent_state.add_turn("user", user_input)
            agent_state.add_turn("assistant", prompt)
            return AgentResponse.approval_required(
                message=f"I didn't understand that response. {prompt}",
                actions=[pending],
                patient_id=agent_state.active_patient_id,
            )

    # ------------------------------------------------------------------
    # Tool registration helpers
    # ------------------------------------------------------------------

    def _register_standard_tools(self) -> None:
        """Register all standard TriageGuard tools."""
        from triageguard_agent.tools.patient_tools import (
            get_patient_summary_spec,
            get_patient_observations_spec,
        )
        from triageguard_agent.tools.assessment_tools import (
            run_triage_assessment_spec,
            get_xgb_explanation_spec,
        )
        from triageguard_agent.tools.hospital_tools import (
            get_hospital_state_spec,
            propose_hospital_calibration_spec,
            commit_hospital_calibration_spec,
        )
        from triageguard_agent.tools.simulation_tools import (
            get_live_simulation_dashboard_spec,
            step_simulation_time_spec,
            trigger_patient_arrival_spec,
            triage_simulated_patient_spec,
            admit_simulated_patient_spec,
        )

        specs = [
            get_patient_summary_spec(),
            get_patient_observations_spec(),
            run_triage_assessment_spec(),
            get_xgb_explanation_spec(),
            get_hospital_state_spec(),
            propose_hospital_calibration_spec(),
            commit_hospital_calibration_spec(),
            get_live_simulation_dashboard_spec(),
            step_simulation_time_spec(),
            trigger_patient_arrival_spec(),
            triage_simulated_patient_spec(),
            admit_simulated_patient_spec(),
        ]
        for spec in specs:
            self.tool_registry.register(spec)

        logger.info("Registered %d standard tools.", len(specs))

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Return the LLM-safe tool catalogue."""
        return self.tool_registry.list_tools()

    @property
    def working_memory(self) -> WorkingMemory:
        return self._working_memory
