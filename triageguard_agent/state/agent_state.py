"""
agent_state.py
--------------
Persistent conversational state for one agent session.

The AgentState records *who* is talking, *what patient* is active,
and *what the agent is trying to do*. It does NOT store:
  - Full hospital state (dynamic — obtained via tools at query time)
  - Patient clinical data (obtained via tools)
  - XGBoost/RAG model internals

hospital_state_timestamp records *when* the agent last fetched hospital
state, so the runtime can detect staleness and prompt for recalibration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


VALID_ROLES = frozenset({"nurse", "doctor", "admin", "system"})

VALID_TASKS = frozenset(
    {
        "triage_assessment",
        "triage_reassessment",
        "patient_lookup",
        "patient_update",
        "hospital_calibration",
        "routing",
        "xgb_explanation",
        "rag_reasoning",
        "human_review",
        "idle",
    }
)


@dataclass
class AgentState:
    """
    Session-scoped state for the TriageGuard agent.

    Fields
    ------
    session_id              : Unique identifier for this session.
    user_role               : Role of the authenticated user.
    active_patient_id       : Patient currently being discussed, if any.
    active_encounter_id     : Encounter ID for the current patient, if any.
    active_task             : What workflow is currently active.
    hospital_id             : Which registered hospital this session is
                               currently scoped to (None -> "default", same
                               resolution every hospital-aware tool already
                               uses). Stamped by api_server.py's /api/chat
                               handler from the frontend's selected hospital
                               on every turn — never inferred from tool
                               output, so it never goes stale mid-session.
    hospital_state_timestamp: When hospital state was last fetched (UTC).
    last_assessment_reference: Key/ID of the last triage result (for reference).
    pending_action          : A proposed WRITE action awaiting confirmation.
    conversation_context    : Recent turns (bounded ring-buffer, most recent last).
    """

    session_id: str
    user_role: str = "nurse"
    active_patient_id: Optional[str] = None
    active_encounter_id: Optional[str] = None
    active_task: Optional[str] = "idle"
    hospital_id: Optional[str] = None
    hospital_state_timestamp: Optional[datetime] = None
    last_assessment_reference: Optional[str] = None
    pending_action: Optional[Dict[str, Any]] = None
    conversation_context: List[Dict[str, str]] = field(default_factory=list)

    # Maximum turns kept in conversation_context
    _MAX_CONTEXT_TURNS: int = field(default=10, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.user_role not in VALID_ROLES:
            raise ValueError(
                f"Invalid user_role {self.user_role!r}. "
                f"Must be one of: {sorted(VALID_ROLES)}"
            )

    # ------------------------------------------------------------------
    # Conversation context helpers
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn and trim to the last N turns."""
        self.conversation_context.append({"role": role, "content": content})
        if len(self.conversation_context) > self._MAX_CONTEXT_TURNS:
            self.conversation_context = self.conversation_context[
                -self._MAX_CONTEXT_TURNS :
            ]

    def clear_context(self) -> None:
        self.conversation_context.clear()

    # ------------------------------------------------------------------
    # Pending action helpers
    # ------------------------------------------------------------------

    def set_pending(
        self,
        action_type: str,
        payload: Dict[str, Any],
        description: Optional[str] = None,
    ) -> None:
        """
        description is the nurse-facing confirmation text built at proposal
        time (e.g. by ConfirmationProtocol.create_pending). Storing it here
        lets a later ambiguous-response re-prompt show the same description
        again, instead of falling back to a generic placeholder.
        """
        self.pending_action = {
            "action_type": action_type,
            "payload": payload,
            "description": description,
        }

    def clear_pending(self) -> None:
        self.pending_action = None

    def has_pending(self) -> bool:
        return self.pending_action is not None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_role": self.user_role,
            "active_patient_id": self.active_patient_id,
            "active_encounter_id": self.active_encounter_id,
            "active_task": self.active_task,
            "hospital_id": self.hospital_id,
            "hospital_state_timestamp": (
                self.hospital_state_timestamp.isoformat()
                if self.hospital_state_timestamp
                else None
            ),
            "last_assessment_reference": self.last_assessment_reference,
            "pending_action": self.pending_action,
            "conversation_context": self.conversation_context,
        }

    def __repr__(self) -> str:
        return (
            f"AgentState(session={self.session_id!r}, "
            f"role={self.user_role!r}, "
            f"patient={self.active_patient_id!r}, "
            f"task={self.active_task!r})"
        )
