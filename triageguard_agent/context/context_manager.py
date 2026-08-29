"""
context_manager.py
------------------
Assembles the per-turn LLM context from multiple sources.

What the ContextManager does NOT do
------------------------------------
* It does NOT inject the full hospital state.
* It does NOT inject all skills.
* It does NOT inline patient clinical records.
* It does NOT duplicate information that tools can provide on demand.

What it DOES provide per turn
------------------------------
* A compact active-patient reference (IDs + task only, not full data).
* The text of the relevant skill (loaded by the SkillRegistry, lazily).
* Successful tool results from this turn's WorkingMemory.
* A pending confirmation prompt (if a WRITE action is waiting).
* The last N conversation turns from AgentState.
* A hospital state staleness warning (if stale dept detected).

The agent's system prompt and per-turn context together must be small
enough to leave room for the LLM's response and tool planning.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from triageguard_agent.state.agent_state import AgentState
from triageguard_agent.state.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Assembles a context dict for the current agent turn.

    Parameters
    ----------
    skill_registry  : Optional SkillRegistry for loading skill text.
    max_tool_results: Maximum number of recent tool results to include in context.
    """

    def __init__(
        self,
        skill_registry=None,
        max_tool_results: int = 3,
    ) -> None:
        self._skill_registry = skill_registry
        self._max_tool_results = max_tool_results

    def build_context(
        self,
        agent_state: AgentState,
        working_memory: WorkingMemory,
        active_skill: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a structured context dict for the current turn.

        Returns
        -------
        {
            "session_info":         {...},   # role, session ID
            "active_patient_ref":   {...},   # patient ID, encounter ID, task
            "skill_context":        str,     # SKILL.md text or None
            "tool_results":         [...],   # recent successful tool results
            "pending_confirmation": {...},   # pending WRITE action or None
            "conversation_context": [...],   # recent turns
            "warnings":             [...],   # staleness or missing-data warnings
        }
        """
        ctx: Dict[str, Any] = {}

        # ── Session info ──────────────────────────────────────────────
        ctx["session_info"] = {
            "session_id": agent_state.session_id,
            "user_role":  agent_state.user_role,
        }

        # ── Active patient reference ──────────────────────────────────
        ctx["active_patient_ref"] = {
            "patient_id":   agent_state.active_patient_id,
            "encounter_id": agent_state.active_encounter_id,
            "active_task":  agent_state.active_task,
            "note": (
                "Use get_patient_summary to retrieve patient clinical data."
                if agent_state.active_patient_id
                else "No active patient — ask nurse for patient identity."
            ),
        }

        # ── Skill context (lazy-loaded) ───────────────────────────────
        skill_to_load = active_skill or working_memory.active_skill
        skill_text = None
        if skill_to_load and self._skill_registry:
            skill_text = self._skill_registry.load(skill_to_load)
            if skill_text:
                logger.debug("Skill loaded for context: %s", skill_to_load)
            else:
                logger.warning("Skill not found: %s", skill_to_load)

        ctx["skill_context"] = skill_text

        # ── Tool results (most recent successful, bounded) ────────────
        successful = working_memory.successful_results()
        ctx["tool_results"] = successful[-self._max_tool_results:]

        # ── Pending confirmation ──────────────────────────────────────
        ctx["pending_confirmation"] = agent_state.pending_action

        # ── Recent conversation ───────────────────────────────────────
        ctx["conversation_context"] = list(agent_state.conversation_context)

        # ── Warnings ─────────────────────────────────────────────────
        warnings = []
        if working_memory.notes:
            warnings.extend(working_memory.notes)

        # Check failed tool results
        for fail in working_memory.failed_results():
            err = fail.get("error", {})
            warnings.append(
                f"Tool {fail.get('tool')!r} failed: "
                f"[{err.get('code')}] {err.get('message')}"
            )

        ctx["warnings"] = warnings

        return ctx

    def to_llm_messages(
        self,
        ctx: Dict[str, Any],
        system_prompt: str,
    ) -> list:
        """
        Convert the context dict to an OpenRouter-compatible message list.

        The system prompt is kept static (no dynamic hospital state injected).
        Tool results and skill context are injected as user-turn context.
        """
        messages = [{"role": "system", "content": system_prompt}]

        # Build a compact context block for this turn
        context_parts = []

        if ctx.get("active_patient_ref", {}).get("patient_id"):
            ref = ctx["active_patient_ref"]
            context_parts.append(
                f"[ACTIVE PATIENT] ID={ref['patient_id']} "
                f"Encounter={ref.get('encounter_id', 'N/A')} "
                f"Task={ref.get('active_task', 'idle')}"
            )

        if ctx.get("skill_context"):
            context_parts.append(
                f"[ACTIVE SKILL]\n{ctx['skill_context']}"
            )

        if ctx.get("tool_results"):
            import json
            for result in ctx["tool_results"]:
                tool_name = result.get("tool", "unknown")
                data = result.get("data", {})
                context_parts.append(
                    f"[TOOL RESULT: {tool_name}]\n{json.dumps(data, indent=2)}"
                )

        if ctx.get("pending_confirmation"):
            import json
            context_parts.append(
                f"[PENDING CONFIRMATION]\n"
                f"{json.dumps(ctx['pending_confirmation'], indent=2)}\n"
                "Waiting for nurse to confirm or reject this action."
            )

        if ctx.get("warnings"):
            context_parts.append(
                "[WARNINGS]\n" + "\n".join(f"- {w}" for w in ctx["warnings"])
            )

        if context_parts:
            context_block = "\n\n".join(context_parts)
            messages.append({"role": "user", "content": context_block})

        # Append recent conversation turns
        for turn in ctx.get("conversation_context", []):
            messages.append(turn)

        return messages
