"""
openrouter_client.py
---------------------
Tool-calling LLM client for the AgentRuntime planning loop, via OpenRouter's
OpenAI-compatible chat completions API.

This module owns exactly two responsibilities:
    1. Convert ToolRegistry-style tool specs into OpenAI "tools" schema.
    2. Send one chat-completion request (with tools attached) and return the
       raw assistant message dict (role / content / tool_calls) unmodified.

It does NOT decide which tool to call, does NOT execute tools, and does NOT
interpret results — that is AgentRuntime's job. Keeping this module dumb is
deliberate: the LLM boundary should be a thin, swappable transport, not a
place where orchestration logic accumulates.

Configuration
-------------
Reads the same environment variable convention as the rest of the project:
    OPENROUTER_API_KEY        (required to actually call the API. May be a
                               single key, or a JSON list of keys — e.g.
                               ["sk-or-...", "sk-or-..."] — to spread the
                               free-tier rate limit across multiple OpenRouter
                               accounts. When a list is given, get_api_key()
                               round-robins one key per call.)
    TRIAGEGUARD_AGENT_MODEL   (optional, defaults to a tool-calling-capable
                               OpenRouter model; independent from the RAG
                               branch's model configured in
                               triageguard_rag/config/config.yaml)
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter call fails or returns an unusable response."""


_key_lock = threading.Lock()
_key_cycle: Optional["itertools.cycle[str]"] = None
_key_cycle_source: Optional[str] = None


def _parse_api_keys(raw: str) -> List[str]:
    """Accept either a single key or a JSON list of keys in OPENROUTER_API_KEY."""
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("OPENROUTER_API_KEY looks like a list but isn't valid JSON; using it as a single key.")
            return [raw]
        if isinstance(parsed, list):
            return [str(k).strip() for k in parsed if str(k).strip()]
    return [raw]


def get_api_key() -> Optional[str]:
    """
    Read the OpenRouter API key from the environment. Never hardcode it.

    OPENROUTER_API_KEY may hold several keys as a JSON list. When it does,
    this rotates round-robin — a different key on every call — so a single
    free-tier key's rate limit isn't hit on every request.
    """
    global _key_cycle, _key_cycle_source
    raw = os.environ.get("OPENROUTER_API_KEY", "")
    if not raw:
        return None

    with _key_lock:
        if raw != _key_cycle_source:
            keys = _parse_api_keys(raw)
            if not keys:
                return None
            _key_cycle = itertools.cycle(keys)
            _key_cycle_source = raw
        return next(_key_cycle)


def get_model() -> str:
    """Read the agent's LLM model name from the environment, with a safe default."""
    return os.environ.get("TRIAGEGUARD_AGENT_MODEL", DEFAULT_MODEL)


def to_openai_tools(tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert ToolRegistry.list_tools() output (name/description/input_schema/...)
    into the OpenAI-compatible "tools" array OpenRouter expects.

    Only name, description, and input_schema are forwarded — risk_level and
    requires_approval are internal governance metadata the LLM never sees or
    needs; approval enforcement happens in ToolExecutor, not in the prompt.
    """
    tools = []
    for spec in tool_specs:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def call_chat_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Send one chat-completion request with tools attached.

    Returns
    -------
    The raw assistant message dict, e.g.:
        {"role": "assistant", "content": "...", "tool_calls": [...] | None}

    Raises
    ------
    OpenRouterError if the API key is missing, the HTTP call fails, or the
    response does not contain a usable message. Never returns a fabricated
    message on failure — the caller must handle this explicitly.
    """
    key = api_key or get_api_key()
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Set it in the environment or a "
            ".env file (see .env.example) before starting the agent."
        )

    payload: Dict[str, Any] = {
        "model": model or get_model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/TriageGuard",
        "X-Title": "TriageGuard-Agent",
    }

    logger.info("Calling OpenRouter model=%s with %d tool(s).", payload["model"], len(tools))
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

    try:
        data = response.json()
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, ValueError) as exc:
        raise OpenRouterError(
            f"OpenRouter returned an unexpected response shape: {exc}"
        ) from exc

    return message


def tool_result_message(tool_call_id: str, tool_name: str, result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build the OpenAI-format 'tool' role message that reports a tool's result back to the LLM."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(result_dict, default=str),
    }
