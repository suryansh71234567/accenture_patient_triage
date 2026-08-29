"""
agent_response.py
-----------------
Structured internal response produced by the AgentRuntime each turn.

The `message` field contains the human-facing text (eventually LLM-generated).
The remaining fields are structured metadata the runtime preserves regardless
of what the LLM says — this is what downstream systems act on.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


VALID_RESPONSE_TYPES = frozenset(
    {"information", "assessment", "confirmation", "approval_required", "error"}
)


@dataclass
class AgentResponse:
    """
    Structured output from one agent turn.

    Fields
    ------
    message                : Human-facing text for the nurse/staff.
    response_type          : One of VALID_RESPONSE_TYPES.
    patient_id             : Patient this response relates to, if any.
    actions                : List of structured actions taken or proposed.
    evidence               : Source references backing this response.
    human_approval_required: True when a WRITE action is waiting for confirmation.
    """

    message: str
    response_type: str
    patient_id: Optional[str] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    human_approval_required: bool = False

    def __post_init__(self) -> None:
        if self.response_type not in VALID_RESPONSE_TYPES:
            raise ValueError(
                f"Invalid response_type {self.response_type!r}. "
                f"Must be one of: {sorted(VALID_RESPONSE_TYPES)}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "response_type": self.response_type,
            "patient_id": self.patient_id,
            "actions": self.actions,
            "evidence": self.evidence,
            "human_approval_required": self.human_approval_required,
        }

    @classmethod
    def error_response(cls, message: str) -> "AgentResponse":
        """Convenience constructor for error responses."""
        return cls(
            message=message,
            response_type="error",
            human_approval_required=False,
        )

    @classmethod
    def approval_required(
        cls,
        message: str,
        actions: List[Dict[str, Any]],
        patient_id: Optional[str] = None,
    ) -> "AgentResponse":
        """Convenience constructor for responses awaiting human approval."""
        return cls(
            message=message,
            response_type="approval_required",
            patient_id=patient_id,
            actions=actions,
            human_approval_required=True,
        )

    def __repr__(self) -> str:
        return (
            f"AgentResponse(type={self.response_type!r}, "
            f"patient={self.patient_id!r}, "
            f"approval={self.human_approval_required})"
        )
