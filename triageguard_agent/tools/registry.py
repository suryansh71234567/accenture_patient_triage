"""
registry.py  (tools)
--------------------
ToolRegistry — central catalogue of all available tools.

Design rules
------------
* Every tool is registered with a ToolSpec before use.
* WRITE tools require explicit approval — the executor enforces this.
* The LLM sees only tool names + descriptions + input schemas.
  It never sees the handler callable directly.
* Low-level implementation details (PCA, model loading, embeddings)
  are never exposed as tools. Only high-level workflow tools are registered.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# Risk levels for tools
READ = "READ"
COMPUTE = "COMPUTE"
WRITE = "WRITE"

VALID_RISK_LEVELS = frozenset({READ, COMPUTE, WRITE})


@dataclass
class ToolSpec:
    """
    Specification for a single registered tool.

    Fields
    ------
    name             : Unique snake_case tool identifier.
    description      : One-sentence description shown to the LLM.
    input_schema     : JSON-Schema-style dict describing required inputs.
    handler          : Callable that receives kwargs and returns ToolResult.
    risk_level       : READ | COMPUTE | WRITE
    side_effect      : True if the tool modifies external state.
    requires_approval: True if a human must confirm before execution.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    risk_level: str = READ
    side_effect: bool = False
    requires_approval: bool = False

    def __post_init__(self) -> None:
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(
                f"Invalid risk_level {self.risk_level!r}. "
                f"Must be one of: {VALID_RISK_LEVELS}"
            )
        if self.risk_level == WRITE and not self.side_effect:
            raise ValueError("WRITE tools must have side_effect=True.")

    def to_llm_spec(self) -> Dict[str, Any]:
        """Return the subset of metadata suitable for sending to an LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
        }


class ToolRegistry:
    """
    Central catalogue of all registered tools.

    Usage
    -----
    registry = ToolRegistry()
    registry.register(ToolSpec(...))
    spec = registry.get("get_patient_summary")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool. Raises if name already registered."""
        if spec.name in self._tools:
            raise ValueError(f"Tool {spec.name!r} is already registered.")
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """Return the ToolSpec for the given name, or None."""
        return self._tools.get(name)

    def require(self, name: str) -> ToolSpec:
        """Return the ToolSpec or raise KeyError."""
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Tool {name!r} is not registered.")
        return spec

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return LLM-safe tool specs for all registered tools."""
        return [spec.to_llm_spec() for spec in self._tools.values()]

    def list_by_risk(self, level: str) -> List[ToolSpec]:
        """Return all tools of a given risk level."""
        return [s for s in self._tools.values() if s.risk_level == level]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry({list(self._tools.keys())})"
