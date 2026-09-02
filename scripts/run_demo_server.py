"""
run_demo_server.py
-------------------
Runs prepare_demo.py's setup (clear pre-simulated pool, register the
curated EVAL-* cohort as ARRIVED) and then starts uvicorn IN THE SAME
PROCESS, so the in-memory simulation state prepare_demo.py builds is
exactly what the frontend talks to.

Deliberately does not use --reload: reload spawns a fresh subprocess that
re-imports api_server from scratch, which would silently discard this
prepared state on the first file save. Restart this script to rebuild the
same curated state if needed; go back to `uvicorn api_server:app --reload`
for normal development once the demo recording is done.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import prepare_demo

prepare_demo.main()

import uvicorn
import api_server

uvicorn.run(api_server.app, host="127.0.0.1", port=8000)
