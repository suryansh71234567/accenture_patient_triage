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


# Start from this key index on first boot (0-based).
# Keys before this index are intentionally skipped at startup because they
# hit the free-tier rate limit quickly; the cycle still wraps back to them.
INITIAL_KEY_INDEX = 2

_key_lock = threading.Lock()
_key_cycle: Optional["itertools.cycle[str]"] = None
_key_cycle_source: Optional[str] = None
_all_keys: List[str] = []  # kept so callers can iterate the full pool


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


def _build_cycle(keys: List[str]) -> "itertools.cycle[str]":
    """Return a cycle that starts at INITIAL_KEY_INDEX (wraps if pool is smaller)."""
    if len(keys) <= 1:
        return itertools.cycle(keys)
    start = INITIAL_KEY_INDEX % len(keys)
    # Rotate the list so the desired key comes first, then cycle.
    rotated = keys[start:] + keys[:start]
    logger.info(
        "Key pool has %d key(s); starting rotation at index %d (key ending …%s).",
        len(keys), start, rotated[0][-6:],
    )
    return itertools.cycle(rotated)


def get_api_key() -> Optional[str]:
    """
    Read the OpenRouter API key from the environment. Never hardcode it.

    OPENROUTER_API_KEY may hold several keys as a JSON list. When it does,
    this rotates round-robin — a different key on every call — starting from
    INITIAL_KEY_INDEX so the most-rate-limited keys are skipped at startup.
    """
    global _key_cycle, _key_cycle_source, _all_keys
    raw = os.environ.get("OPENROUTER_API_KEY", "")
    if not raw:
        return None

    with _key_lock:
        if raw != _key_cycle_source:
            keys = _parse_api_keys(raw)
            if not keys:
                return None
            _all_keys = keys
            _key_cycle = _build_cycle(keys)
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

    # ------------------------------------------------------------------ #
    # Retry loop: try every key in the pool before giving up.             #
    # A 429 (rate-limit) or a response without 'choices' both trigger a   #
    # rotation to the next key automatically.                             #
    # ------------------------------------------------------------------ #
    with _key_lock:
        total_keys = len(_all_keys) if _all_keys else 1

    last_error: Optional[Exception] = None
    for attempt in range(total_keys):
        current_key = key if attempt == 0 else get_api_key()
        attempt_headers = dict(headers)
        attempt_headers["Authorization"] = f"Bearer {current_key}"

        logger.info(
            "Calling OpenRouter model=%s with %d tool(s) [attempt %d/%d, key …%s].",
            payload["model"], len(tools), attempt + 1, total_keys, current_key[-6:],
        )

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(OPENROUTER_URL, headers=attempt_headers, json=payload)
        except httpx.HTTPError as exc:
            last_error = OpenRouterError(f"OpenRouter request failed: {exc}")
            logger.warning("Network error on attempt %d: %s — trying next key.", attempt + 1, exc)
            continue

        if response.status_code == 429:
            logger.warning(
                "Rate-limited (429) on attempt %d (key …%s) — rotating to next key.",
                attempt + 1, current_key[-6:],
            )
            last_error = OpenRouterError(f"Rate-limited on key …{current_key[-6:]}")
            continue

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            last_error = OpenRouterError(f"OpenRouter HTTP {response.status_code}: {exc}")
            logger.warning("HTTP error %d on attempt %d — trying next key.", response.status_code, attempt + 1)
            continue

        try:
            data = response.json()
            if "error" in data:
                err_msg = data["error"].get("message", str(data["error"]))
                logger.warning(
                    "OpenRouter API error on attempt %d (key …%s): %s — trying next key.",
                    attempt + 1, current_key[-6:], err_msg,
                )
                last_error = OpenRouterError(f"OpenRouter API error: {err_msg}")
                continue
            message = data["choices"][0]["message"]
            return message  # success
        except (KeyError, IndexError, ValueError) as exc:
            last_error = OpenRouterError(
                f"OpenRouter returned an unexpected response shape: {exc}"
            )
            logger.warning(
                "Unexpected response shape on attempt %d (key …%s): %s — trying next key.",
                attempt + 1, current_key[-6:], exc,
            )
            continue

    raise last_error or OpenRouterError("All API keys exhausted without a successful response.")


def tool_result_message(tool_call_id: str, tool_name: str, result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build the OpenAI-format 'tool' role message that reports a tool's result back to the LLM."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(result_dict, default=str),
    }
