"""
confirmation_protocol.py
------------------------
Human-in-the-loop confirmation workflow.

When a WRITE tool requires approval, the runtime uses this protocol to:
1. Create a pending action record.
2. Ask the nurse for confirmation.
3. Parse the nurse's response.
4. Either commit or discard the action.

Rules
-----
* The agent NEVER auto-confirms a WRITE action.
* A pending action is stored on AgentState.pending_action.
* Only explicit confirmation tokens trigger a commit.
* Ambiguous responses are treated as neither confirmed nor rejected —
  the agent asks again.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Words/phrases that constitute a confirmation
_CONFIRM_TOKENS = frozenset({
    "yes", "y", "confirm", "confirmed", "ok", "okay",
    "approved", "approve", "do it", "proceed", "go ahead",
    "correct", "right", "affirmative",
})

# Words/phrases that constitute a rejection
_REJECT_TOKENS = frozenset({
    "no", "n", "cancel", "cancelled", "reject", "rejected",
    "stop", "abort", "never mind", "nevermind", "don't",
    "do not", "incorrect", "wrong",
})


class ConfirmationProtocol:
    """
    Manages the human-in-the-loop confirmation flow for WRITE actions.

    Typical usage
    -------------
    proto = ConfirmationProtocol()

    # 1. Agent proposes an action
    pending = proto.create_pending("commit_hospital_calibration", proposed_update)
    agent_state.set_pending(pending["action_type"], pending["payload"])

    # 2. Nurse responds
    confirmed = proto.is_confirmed(nurse_input)
    rejected  = proto.is_rejected(nurse_input)

    # 3. Act on response
    if confirmed:
        agent_state.clear_pending()
        # ... call commit tool
    elif rejected:
        agent_state.clear_pending()
        # ... inform nurse
    else:
        # ... ask again
    """

    # ------------------------------------------------------------------
    # Pending action creation
    # ------------------------------------------------------------------

    def create_pending(
        self,
        action_type: str,
        payload: Dict[str, Any],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a pending action record.

        Parameters
        ----------
        action_type  : The tool name that will be called on confirmation.
        payload      : The validated arguments for the tool.
        description  : Human-readable summary shown to the nurse for confirmation.

        Returns
        -------
        A dict suitable for AgentState.pending_action.
        """
        return {
            "action_type": action_type,
            "payload": payload,
            "description": description or f"Proposed action: {action_type}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "awaiting_confirmation",
        }

    def build_confirmation_prompt(self, pending: Dict[str, Any]) -> str:
        """
        Return the message the agent should show the nurse asking for confirmation.
        """
        description = pending.get("description", "the proposed action")
        return (
            f"{description}\n\n"
            "Please confirm to proceed (yes/confirm/ok) or cancel (no/cancel)."
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def is_confirmed(self, user_input: str) -> bool:
        """Return True if the user's input is an explicit confirmation."""
        return _normalise(user_input) in _CONFIRM_TOKENS

    def is_rejected(self, user_input: str) -> bool:
        """Return True if the user's input is an explicit rejection."""
        normalised = _normalise(user_input)
        words = set(normalised.split())
        # Check exact word match first, then multi-word token substring match
        for token in _REJECT_TOKENS:
            if " " in token:
                # Multi-word tokens (e.g. "do not", "never mind") — substring ok
                if token in normalised:
                    return True
            else:
                # Single-word tokens — require whole-word match to avoid
                # matching "n" inside "mean", "any", etc.
                if token in words:
                    return True
        return False

    def is_ambiguous(self, user_input: str) -> bool:
        """Return True if the input is neither a clear confirm nor reject."""
        return not self.is_confirmed(user_input) and not self.is_rejected(user_input)

    def resolve(self, user_input: str) -> str:
        """
        Resolve user input to one of: 'confirmed' | 'rejected' | 'ambiguous'.
        """
        if self.is_confirmed(user_input):
            return "confirmed"
        if self.is_rejected(user_input):
            return "rejected"
        return "ambiguous"

    # ------------------------------------------------------------------
    # State helpers for AgentState integration
    # ------------------------------------------------------------------

    def require_confirmation(
        self,
        agent_state,   # AgentState — avoid circular import
        action_type: str,
        payload: Dict[str, Any],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set a pending action on agent_state and return the pending dict.
        Convenience method for use inside the runtime.
        """
        pending = self.create_pending(action_type, payload, description)
        agent_state.set_pending(action_type, payload)
        logger.info("Pending confirmation set: action_type=%s", action_type)
        return pending


def _normalise(text: str) -> str:
    """Lowercase + strip for consistent token comparison."""
    return text.strip().lower()
