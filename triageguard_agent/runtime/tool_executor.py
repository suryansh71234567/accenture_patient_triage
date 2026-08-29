"""
tool_executor.py
----------------
Executes registered tools and enforces the approval gate for WRITE tools.

Responsibilities
----------------
* Look up the ToolSpec in the ToolRegistry.
* Enforce: WRITE tools with requires_approval=True cannot be called without
  an explicit approval token.
* Call the handler with the provided kwargs.
* Catch all exceptions and convert them to ToolResult.fail() — never
  let a raw exception propagate to the LLM layer.
* Return a ToolResult in all cases.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from triageguard_agent.schemas.tool_result import ToolResult
from triageguard_agent.tools.registry import ToolRegistry, WRITE

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Safe tool execution wrapper.

    Parameters
    ----------
    registry : The ToolRegistry to look tools up from.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        approval_token: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute a named tool with the given kwargs.

        Parameters
        ----------
        tool_name      : Name of the tool to execute.
        kwargs         : Arguments passed to the tool handler.
        approval_token : Required for WRITE tools with requires_approval=True.
                         Pass the pending_action key or a session-scoped token.

        Returns
        -------
        ToolResult — always, even on exception.
        """
        # ── 1. Look up the tool spec ──────────────────────────────────
        spec = self._registry.get(tool_name)
        if spec is None:
            logger.error("Unknown tool requested: %r", tool_name)
            return ToolResult.fail(
                tool_name,
                "TOOL_NOT_FOUND",
                f"Tool {tool_name!r} is not registered. "
                "Check the tool name or contact the system administrator.",
            )

        # ── 2. Enforce WRITE approval gate ────────────────────────────
        if spec.risk_level == WRITE and spec.requires_approval:
            if not approval_token:
                logger.warning(
                    "Attempted WRITE tool %r without approval token.", tool_name
                )
                return ToolResult.fail(
                    tool_name,
                    "APPROVAL_REQUIRED",
                    f"Tool {tool_name!r} is a WRITE tool that requires "
                    "human approval before execution. "
                    "Use the confirmation protocol to obtain approval first.",
                )

        # ── 3. Validate kwargs against the tool's input schema ────────
        missing = _check_required(spec.input_schema, kwargs)
        if missing:
            return ToolResult.fail(
                tool_name,
                "MISSING_REQUIRED_FIELDS",
                f"Missing required fields for {tool_name!r}: {missing}.",
            )

        # ── 4. Call the handler ───────────────────────────────────────
        try:
            logger.debug("Executing tool %r with kwargs=%s", tool_name, list(kwargs.keys()))
            result = spec.handler(**kwargs)

            # Ensure the handler returned a ToolResult
            if not isinstance(result, ToolResult):
                logger.error(
                    "Tool %r handler returned %r instead of ToolResult.",
                    tool_name, type(result),
                )
                return ToolResult.fail(
                    tool_name,
                    "HANDLER_CONTRACT_VIOLATION",
                    f"Tool {tool_name!r} handler did not return a ToolResult. "
                    "This is a bug in the tool implementation.",
                )

            return result

        except Exception as exc:
            logger.exception("Unhandled exception in tool %r.", tool_name)
            return ToolResult.fail(
                tool_name,
                "HANDLER_EXCEPTION",
                f"Tool {tool_name!r} raised an exception: {type(exc).__name__}: {exc}",
            )


def _check_required(schema: Dict[str, Any], kwargs: Dict[str, Any]) -> list:
    """
    Check that all required fields in the JSON-Schema-style schema are present.
    Returns a list of missing field names (empty if all present).
    """
    required = schema.get("required", [])
    return [field for field in required if field not in kwargs]
