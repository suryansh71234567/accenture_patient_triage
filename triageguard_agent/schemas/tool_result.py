"""
tool_result.py
--------------
Common structured envelope for every tool invocation.

Every tool handler MUST return a ToolResult. The runtime never passes
raw dicts or natural-language strings to the LLM to indicate success/failure.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """
    Structured envelope returned by every tool handler.

    Fields
    ------
    success  : True if the tool ran without error.
    tool     : Name of the tool that produced this result.
    data     : Payload dict on success, None on failure.
    error    : {"code": str, "message": str} on failure, None on success.
    metadata : Optional extra info (timing, source, staleness, etc.).
    """

    success: bool
    tool: str
    data: Optional[Dict[str, Any]]
    error: Optional[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def ok(
        cls,
        tool: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Create a successful ToolResult."""
        return cls(
            success=True,
            tool=tool,
            data=data,
            error=None,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        tool: str,
        code: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Create a failed ToolResult."""
        return cls(
            success=False,
            tool=tool,
            data=None,
            error={"code": code, "message": message},
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool": self.tool,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        if self.success:
            return f"ToolResult(tool={self.tool!r}, success=True)"
        code = self.error.get("code", "UNKNOWN") if self.error else "UNKNOWN"
        return f"ToolResult(tool={self.tool!r}, success=False, code={code!r})"
