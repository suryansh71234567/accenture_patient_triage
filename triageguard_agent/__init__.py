"""
triageguard_agent
-----------------
Conversational orchestration layer for TriageGuard.

The agent is the primary interaction surface for nurses/staff. It wraps
the existing XGBoost + RAG + reconcile + route pipeline behind a
structured tool/skill system with human-in-the-loop confirmation.

The LLM decides *what* to do.
The tools *do* it.
The deterministic backend *validates* everything.
"""

__version__ = "0.1.0"
