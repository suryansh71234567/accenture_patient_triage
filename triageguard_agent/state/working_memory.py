"""
working_memory.py
-----------------
Ephemeral, per-turn memory buffer.

WorkingMemory accumulates tool results, skill context, and pending
confirmation info for the *current* agent turn. It is cleared at the
start of each new turn. It is NOT persisted and NOT part of AgentState.

Purpose: Avoid stuffing tool results into AgentState (which is session-scoped)
or into the system prompt (which is per-session and static).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorkingMemory:
    """
    Per-turn scratch buffer for the AgentRuntime.

    Attributes
    ----------
    tool_results    : Ordered list of ToolResult dicts from this turn.
    active_skill    : Name of the skill currently in use (if any).
    skill_context   : The loaded SKILL.md text for the current turn.
    pending_confirmation : A proposed action dict awaiting nurse confirmation.
    notes           : Free-form internal notes (debugging, warnings).
    """

    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    active_skill: Optional[str] = None
    skill_context: Optional[str] = None
    pending_confirmation: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Tool result helpers
    # ------------------------------------------------------------------

    def add_tool_result(self, result_dict: Dict[str, Any]) -> None:
        """Append a serialised ToolResult dict."""
        self.tool_results.append(result_dict)

    def get_last_tool_result(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Return the most recent result from a named tool, or None."""
        for result in reversed(self.tool_results):
            if result.get("tool") == tool_name:
                return result
        return None

    def successful_results(self) -> List[Dict[str, Any]]:
        return [r for r in self.tool_results if r.get("success")]

    def failed_results(self) -> List[Dict[str, Any]]:
        return [r for r in self.tool_results if not r.get("success")]

    # ------------------------------------------------------------------
    # Confirmation helpers
    # ------------------------------------------------------------------

    def set_pending_confirmation(self, action: Dict[str, Any]) -> None:
        self.pending_confirmation = action

    def clear_pending_confirmation(self) -> None:
        self.pending_confirmation = None

    def has_pending_confirmation(self) -> bool:
        return self.pending_confirmation is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset for a new turn."""
        self.tool_results.clear()
        self.active_skill = None
        self.skill_context = None
        self.pending_confirmation = None
        self.notes.clear()

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_results": self.tool_results,
            "active_skill": self.active_skill,
            "skill_context": self.skill_context,
            "pending_confirmation": self.pending_confirmation,
            "notes": self.notes,
        }

    def __repr__(self) -> str:
        return (
            f"WorkingMemory("
            f"results={len(self.tool_results)}, "
            f"skill={self.active_skill!r}, "
            f"pending={self.has_pending_confirmation()})"
        )
