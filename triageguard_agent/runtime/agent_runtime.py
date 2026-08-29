"""
agent_runtime.py
----------------
AgentRuntime — the central orchestrator for the TriageGuard agent.

Scope of this implementation
-----------------------------
This module wires together all the agent components:
    ToolRegistry + ToolExecutor
    SkillRegistry
    ContextManager
    WorkingMemory
    AgentState
    ConfirmationProtocol
    LLM planning loop (OpenRouter tool-calling)

Lifecycle of one turn
---------------------
1. Receive user input + agent state.
2. Clear working memory from the prior turn.
3. Detect if there is a pending confirmation action.
4. Record this turn in conversation context, then build the context for this turn
   (so the LLM actually sees the current message).
5. Call the LLM with context + tool catalogue.
6. Parse tool call(s) from the LLM response.
7. Execute requested tools via ToolExecutor (WRITE tools without prior
   approval short-circuit into a pending confirmation instead of running).
8. Collect tool results into WorkingMemory and feed them back to the LLM.
9. Repeat 5-8 until the LLM returns a final answer with no further tool
   calls, or the iteration cap is reached.
10. Return structured AgentResponse.
"""

from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from triageguard_agent.schemas.tool_result import ToolResult
from triageguard_agent.schemas.agent_response import AgentResponse
from triageguard_agent.state.agent_state import AgentState
from triageguard_agent.state.working_memory import WorkingMemory
from triageguard_agent.tools.registry import ToolRegistry, ToolSpec, READ, COMPUTE, WRITE
from triageguard_agent.runtime.tool_executor import ToolExecutor
from triageguard_agent.skills.registry import SkillRegistry, build_default_registry
from triageguard_agent.context.context_manager import ContextManager
from triageguard_agent.protocols.confirmation_protocol import ConfirmationProtocol
from triageguard_agent.llm.openrouter_client import (
    call_chat_with_tools,
    to_openai_tools,
    tool_result_message,
    OpenRouterError,
)
from triageguard_agent.llm.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Hard cap on tool-call round-trips within a single turn. Prevents an
# LLM stuck in a tool-selection loop from running indefinitely; a well
# behaved turn resolves in 1-3 iterations.
DEFAULT_MAX_TOOL_ITERATIONS = 5

# Tool result fields that, on success, identify the patient the LLM is
# now discussing. Used to deterministically update the focused-patient
# context rather than relying on the LLM to track identity itself.
_PATIENT_ID_RESULT_KEYS = ("patient_id",)


class AgentRuntime:
    """
    Central orchestrator for the TriageGuard conversational agent.

    Parameters
    ----------
    tool_registry   : Pre-built ToolRegistry. If None, an empty one is created.
    skill_registry  : Pre-built SkillRegistry. If None, the default one is built.
    auto_register   : If True, register all standard tools automatically on init.
    llm_call_fn     : Callable(messages, tools, model) -> assistant message dict.
                      Defaults to the real OpenRouter tool-calling client.
                      Overridable for tests so no network call is made.
    llm_model       : OpenRouter model name. Defaults to
                      openrouter_client.get_model() (env-configurable).
    max_tool_iterations : Cap on tool-call round-trips per turn.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        skill_registry: Optional[SkillRegistry] = None,
        auto_register: bool = True,
        llm_call_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        llm_model: Optional[str] = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ) -> None:
        self.tool_registry = tool_registry or ToolRegistry()
        self.skill_registry = skill_registry or build_default_registry()
        self.context_manager = ContextManager(skill_registry=self.skill_registry)
        self.executor = ToolExecutor(self.tool_registry)
        self.confirmation = ConfirmationProtocol()

        # LLM transport — swappable for tests; defaults to real OpenRouter call.
        self._llm_call_fn = llm_call_fn or call_chat_with_tools
        self._llm_model = llm_model
        self._max_tool_iterations = max_tool_iterations

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

        - Clears working memory.
        - Checks for pending confirmation (handled without invoking the LLM).
        - Records this turn, then builds context so the LLM sees it.
        - Runs the LLM tool-calling loop (see _run_llm_loop).
        - Returns the resulting AgentResponse.

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

        # ── 3. Record this turn BEFORE building context, so the LLM's
        #      message history actually includes what the nurse just said ──
        agent_state.add_turn("user", user_input)

        # ── 4. Build context ──────────────────────────────────────────
        ctx = self.context_manager.build_context(
            agent_state, self._working_memory, active_skill
        )

        logger.debug(
            "Context built for turn: patient=%s task=%s",
            agent_state.active_patient_id,
            agent_state.active_task,
        )

        # ── 5-9. LLM planning loop ─────────────────────────────────────
        try:
            response = self._run_llm_loop(agent_state, ctx, user_input)
        except OpenRouterError as exc:
            logger.error("LLM call failed: %s", exc)
            response = AgentResponse.error_response(
                "I couldn't reach the reasoning service right now "
                f"({exc}). Please try again, or use the direct patient/"
                "hospital-state tools if this persists."
            )

        # ── 10. Record assistant turn and return ───────────────────────
        agent_state.add_turn("assistant", response.message)
        return response

    # ------------------------------------------------------------------
    # LLM planning loop
    # ------------------------------------------------------------------

    def _run_llm_loop(
        self,
        agent_state: AgentState,
        ctx: Dict[str, Any],
        user_input: str,
    ) -> AgentResponse:
        """
        Drive the tool-calling conversation with the LLM until it returns a
        final answer (no further tool calls) or the iteration cap is hit.

        Safety properties this loop enforces (do not weaken without
        discussion — see MASTER_TRIAGEGUARD_KNOWLEDGE_BASE.md Part III):
        * The LLM never receives a raw handler or bypasses ToolExecutor.
        * A WRITE tool call without prior approval NEVER executes — it is
          converted into a pending confirmation and the loop stops
          immediately, before any other tool calls in the same LLM turn
          are executed. The nurse must respond on the next turn.
        * Unknown tool names / malformed arguments are reported back to
          the LLM as tool failures, never silently guessed or skipped.
        * The loop is capped so a confused model cannot spin forever.
        * If the model calls the same tool with the exact same arguments
          and it fails again, the loop does not let it burn through the
          full iteration budget repeating a call that cannot succeed
          differently — it stops on the repeat and asks for clarification
          (generic: keyed only on tool name + arguments, not any specific
          tool or patient).
        """
        messages = self.context_manager.to_llm_messages(ctx, system_prompt=SYSTEM_PROMPT)
        openai_tools = to_openai_tools(self.tool_registry.list_tools())
        actions: List[Dict[str, Any]] = []
        failed_call_signatures: set = set()

        for _iteration in range(self._max_tool_iterations):
            assistant_message = self._llm_call_fn(
                messages=messages,
                tools=openai_tools,
                model=self._llm_model,
            )
            messages.append(_strip_to_openai_message(assistant_message))

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                text = assistant_message.get("content") or (
                    "I don't have a response for that — could you rephrase?"
                )
                return AgentResponse(
                    message=text,
                    response_type=_infer_response_type(actions),
                    patient_id=agent_state.active_patient_id,
                    actions=actions,
                )

            for call in tool_calls:
                interrupt = self._execute_one_tool_call(
                    call, agent_state, messages, actions, failed_call_signatures, user_input
                )
                if interrupt is not None:
                    # Either a WRITE tool needs confirmation, or the same
                    # call failed identically twice — stop the loop now.
                    # Do not execute any further tool calls from this same
                    # LLM turn once either condition is hit.
                    return interrupt

        logger.warning(
            "LLM tool loop exceeded %d iterations without a final answer.",
            self._max_tool_iterations,
        )
        return AgentResponse.error_response(
            "I wasn't able to finish that request in the allowed number of "
            "steps. Please try a narrower request (e.g. ask about one "
            "patient or one action at a time)."
        )

    def _execute_one_tool_call(
        self,
        call: Dict[str, Any],
        agent_state: AgentState,
        messages: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        failed_call_signatures: set,
        user_input: str,
    ) -> Optional[AgentResponse]:
        """
        Execute a single LLM tool call and append its result to `messages`.

        Returns an AgentResponse ONLY if this call must interrupt the loop
        (a WRITE tool awaiting confirmation, OR the exact same tool+arguments
        already failed once earlier this turn); otherwise returns None and
        the loop continues.
        """
        call_id = call.get("id", "")
        function = call.get("function", {}) or {}
        tool_name = function.get("name", "")
        raw_args = function.get("arguments") or "{}"

        try:
            kwargs = _parse_tool_call_arguments(raw_args)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Malformed tool-call arguments for %r: %r", tool_name, raw_args)
            messages.append(
                tool_result_message(
                    call_id,
                    tool_name,
                    {
                        "success": False,
                        "error": {
                            "code": "MALFORMED_ARGUMENTS",
                            "message": "Arguments were not valid JSON. Retry with a valid object.",
                        },
                    },
                )
            )
            return None

        if tool_name not in self.tool_registry:
            messages.append(
                tool_result_message(
                    call_id,
                    tool_name,
                    {
                        "success": False,
                        "error": {
                            "code": "TOOL_NOT_FOUND",
                            "message": f"{tool_name!r} is not a registered tool.",
                        },
                    },
                )
            )
            return None

        # Execute without an approval token — WRITE tools requiring approval
        # will fail with APPROVAL_REQUIRED, which we handle explicitly below.
        result = self.run_tool(tool_name, kwargs, agent_state=agent_state)

        if not result.success and result.error and result.error.get("code") == "APPROVAL_REQUIRED":
            description = _describe_pending_action(tool_name, kwargs, user_input)
            pending = self.confirmation.require_confirmation(
                agent_state, tool_name, kwargs, description
            )
            return AgentResponse.approval_required(
                message=self.confirmation.build_confirmation_prompt(pending),
                actions=[{"tool": tool_name, "status": "awaiting_confirmation", "payload": kwargs}],
                patient_id=agent_state.active_patient_id,
            )

        if not result.success:
            signature = (tool_name, json.dumps(kwargs, sort_keys=True, default=str))
            if signature in failed_call_signatures:
                # The model already tried this exact tool call with these
                # exact arguments earlier this turn and it failed the same
                # way. Retrying again cannot succeed differently — stop
                # here instead of burning the rest of the iteration budget,
                # and tell the nurse plainly rather than a generic timeout.
                logger.warning(
                    "Tool %r repeated with identical failing arguments — "
                    "stopping instead of retrying.",
                    tool_name,
                )
                return AgentResponse.error_response(_describe_repeated_failure(tool_name, result))
            failed_call_signatures.add(signature)

        # Deterministically track the focused patient from successful tool
        # I/O, rather than relying on the LLM to remember identity itself.
        _update_focused_patient(agent_state, tool_name, kwargs, result)

        actions.append(
            {
                "tool": tool_name,
                "status": "executed" if result.success else "failed",
                "data": result.data,
                "error": result.error,
            }
        )
        messages.append(tool_result_message(call_id, tool_name, result.to_dict()))
        return None

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
        from triageguard_agent.tools.ingestion_tools import (
            ingest_hospital_records_spec,
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
            ingest_hospital_records_spec(),
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


# ---------------------------------------------------------------------------
# Module-level helpers for the LLM planning loop
# ---------------------------------------------------------------------------

# Tools whose results represent an actual clinical assessment/prediction —
# a final answer that used one of these is labelled "assessment"; a plain
# lookup (get_patient_summary, get_hospital_state, ...) stays "information"
# so a UI cannot mistake a factual lookup for a clinical judgment.
_CLINICAL_ASSESSMENT_TOOLS = frozenset(
    {"run_triage_assessment", "get_xgb_explanation", "triage_simulated_patient"}
)


def _infer_response_type(actions: List[Dict[str, Any]]) -> str:
    """Pick the AgentResponse.response_type for a final (non-approval) answer."""
    if any(a.get("tool") in _CLINICAL_ASSESSMENT_TOOLS for a in actions):
        return "assessment"
    return "information"


def _parse_tool_call_arguments(raw_args: Any) -> Dict[str, Any]:
    """
    Parse one LLM tool-call's raw `arguments` field into a kwargs dict.

    Generic hardening — applies to every tool, every model, not any one
    tool or patient: some smaller OpenRouter-hosted models occasionally
    repeat their own function-call arguments verbatim with no separator
    between the two copies (observed directly: a valid JSON object followed
    immediately by an identical duplicate). A plain json.loads() rejects
    this whole string as "Extra data" and would discard an argument set the
    model actually got right the first time. Instead, decode only the first
    complete JSON value from the start of the string via
    json.JSONDecoder.raw_decode, and ignore the trailing duplicate.

    This does NOT paper over genuine malformation: any JSONDecodeError other
    than "Extra data" (e.g. truncated/invalid JSON) is re-raised unchanged,
    and a recovered first value that isn't a JSON object also raises — both
    are still reported as MALFORMED_ARGUMENTS by the caller, same as before.
    """
    if not isinstance(raw_args, str):
        return dict(raw_args or {})

    text = raw_args.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg != "Extra data":
            raise

        decoder = json.JSONDecoder()
        first_value, end = decoder.raw_decode(text)
        leftover = text[end:].strip()
        logger.warning(
            "Tool-call arguments had %d trailing character(s) after a valid "
            "JSON value (duplicated model output) — using the first value "
            "and discarding the rest.",
            len(leftover),
        )
        if not isinstance(first_value, dict):
            raise ValueError(
                f"Recovered tool-call argument value is not a JSON object: {type(first_value)!r}"
            )
        return first_value


def _strip_to_openai_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only the fields OpenAI-compatible chat history expects on an
    assistant message (role, content, tool_calls) — drop any extra fields
    an OpenRouter response might include, so replaying `messages` on the
    next iteration stays well-formed.
    """
    stripped: Dict[str, Any] = {
        "role": message.get("role", "assistant"),
        "content": message.get("content"),
    }
    if message.get("tool_calls"):
        stripped["tool_calls"] = message["tool_calls"]
    return stripped


def _describe_repeated_failure(tool_name: str, result: ToolResult) -> str:
    """
    Nurse-facing message when the same tool call has failed identically
    twice in one turn. Generic — built only from the tool name and the
    tool's own real error message, never from special-cased knowledge of
    any particular tool.
    """
    err = result.error or {}
    detail = err.get("message", "the request could not be completed")
    return (
        f"I tried {tool_name} again with the same information and it failed "
        f"the same way: {detail} I need more specific details before trying "
        "again — could you clarify exactly what should change?"
    )


def _describe_pending_action(tool_name: str, kwargs: Dict[str, Any], user_input: str) -> str:
    """
    Build a nurse-facing one-line description of a WRITE action awaiting
    confirmation. Deterministic — never LLM-generated — so the confirmation
    prompt cannot drift from what will actually be executed.

    Always echoes the nurse's own current request alongside the proposed
    action (using only data already available this turn — no new heuristics
    or mismatch detection). Confirmation is not a substitute for correct
    tool selection: if the LLM ever proposes a WRITE action that does not
    match what was actually asked, showing the nurse their own words next
    to the proposal is what makes that mismatch visible enough to reject.
    """
    if tool_name == "commit_hospital_calibration":
        dept = kwargs.get("department", "unknown department")
        update = kwargs.get("validated_update", {})
        action_desc = f"Apply hospital state update to {dept}: {update}."
    elif tool_name == "admit_simulated_patient":
        pid = kwargs.get("patient_id", "unknown patient")
        dept = kwargs.get("department") or "the recommended department"
        action_desc = f"Admit patient {pid} to {dept}, occupying a bed."
    else:
        action_desc = f"Proposed action: {tool_name}({kwargs})."

    return f'You asked: "{user_input}"\n{action_desc}'


def _update_focused_patient(
    agent_state: AgentState,
    tool_name: str,
    kwargs: Dict[str, Any],
    result: ToolResult,
) -> None:
    """
    Deterministically track which patient is currently in focus, based on
    tool I/O — never inferred by the LLM. This lets follow-up turns like
    "why is she high risk?" resolve pronouns to the right patient without
    the model having to remember/guess an identifier.

    Only updates on a *successful* tool call that clearly identifies a
    single patient, so a failed lookup or an ambiguous/ unrelated tool call
    never silently changes focus.
    """
    if not result.success:
        return

    patient_id = kwargs.get("patient_id")
    if not patient_id and isinstance(result.data, dict):
        patient_id = result.data.get("patient_id")

    if patient_id:
        agent_state.active_patient_id = str(patient_id)
