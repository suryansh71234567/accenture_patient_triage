"""
triageguard_agent.llm
----------------------
LLM tool-calling client used by AgentRuntime's planning loop.

Kept separate from triageguard_rag.src.reasoning.llm_reasoner, which is the
RAG branch's single-shot clinical reasoning call and belongs to that
component. This package is specific to the conversational agent's
tool-selection loop (function/tool calling), not clinical narrative
generation.
"""
