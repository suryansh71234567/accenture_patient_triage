"""
chat_with_agent.py
-------------------
Interactive terminal chat for manually exercising the TriageGuard nurse-facing
agent (AgentRuntime + the LLM tool-calling planning loop).

This is NOT a frontend. It is a manual-testing entry point — the same role
scripts/live_hospital_demo.py plays for the simulation layer — so the agent's
conversational behavior can be verified end-to-end before any UI exists.

Requires OPENROUTER_API_KEY to be set (see .env.example). Without it, every
turn will return a clean error response rather than crashing.

Usage
-----
  .venv\\Scripts\\python.exe scripts/chat_with_agent.py
  .venv\\Scripts\\python.exe scripts/chat_with_agent.py --role doctor
  .venv\\Scripts\\python.exe scripts/chat_with_agent.py --model openai/gpt-4o-mini

In-chat commands
----------------
  /reset   — start a fresh session (clears conversation + pending actions)
  /tools   — list every tool currently registered
  /state   — dump the current AgentState (active patient, pending action, etc.)
  /quit    — exit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles (mirrors live_hospital_demo.py)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    _dotenv_path = _REPO_ROOT / ".env"
    # override=True: this project's own .env is the authoritative source for
    # this script's configuration. Without it, python-dotenv's default
    # (override=False) silently keeps whatever OPENROUTER_API_KEY may already
    # be sitting in the calling shell's environment (e.g. an empty value
    # exported during earlier testing before .env existed) instead of the
    # real value in .env — which is exactly the failure mode this fixes.
    load_dotenv(_dotenv_path, override=True)
except ImportError:
    pass

from triageguard_agent.runtime.agent_runtime import AgentRuntime
from triageguard_agent.llm.openrouter_client import get_api_key


def print_response(response) -> None:
    print(f"\nAgent: {response.message}")
    if response.human_approval_required:
        print("       [awaiting your confirmation — reply yes/no]")
    if response.actions:
        for action in response.actions:
            status = action.get("status")
            tool = action.get("tool")
            print(f"       (tool: {tool} -> {status})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the TriageGuard nurse-facing agent")
    parser.add_argument("--role", default="nurse", choices=["nurse", "doctor", "admin"])
    parser.add_argument("--model", default=None, help="Override TRIAGEGUARD_AGENT_MODEL for this session")
    args = parser.parse_args()

    if not get_api_key():
        print(
            "WARNING: OPENROUTER_API_KEY is not set. Every turn will return a "
            "clean error response instead of a real reply. Set it in .env "
            "(see .env.example) to actually talk to the model.\n"
        )

    runtime = AgentRuntime(auto_register=True, llm_model=args.model)
    state = runtime.new_session(user_role=args.role)

    print("=" * 70)
    print("  TriageGuard Agent — interactive chat")
    print(f"  Role: {args.role} | Tools registered: {len(runtime.tool_registry)}")
    print("  Commands: /reset  /tools  /state  /quit")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            print("Exiting.")
            break
        if user_input == "/reset":
            state = runtime.new_session(user_role=args.role)
            print(">> Session reset.")
            continue
        if user_input == "/tools":
            for t in runtime.get_tools_for_llm():
                print(f"  - {t['name']} [{t['risk_level']}]: {t['description']}")
            continue
        if user_input == "/state":
            print(json.dumps(state.to_dict(), indent=2, default=str))
            continue

        response = runtime.process_turn(user_input, state)
        print_response(response)


if __name__ == "__main__":
    main()
