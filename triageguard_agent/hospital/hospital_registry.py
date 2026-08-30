"""
hospital_registry.py
---------------------
Hospital identity + registry foundation (multi-hospital Step 2).

Introduces `hospital_id` as a first-class boundary WITHOUT touching the
existing single-hospital classes: HospitalStateStore and
HospitalStateService already accept an arbitrary config/store in their
constructors, so a new hospital is just "point the same classes at a
different config file" — never a copied/modified class.

Model
-----
hospital_id -> HospitalContext -> HospitalStateService -> HospitalStateStore

The "default" hospital_id is special-cased to wrap the pre-existing
process-wide HospitalStateService.instance() singleton, so every caller
that still uses that singleton directly (hospital_tools.py,
hospital_simulator.py, existing tests) keeps working unchanged. Any other
hospital_id gets its own independent HospitalStateService, backed by its
own HospitalStateStore, and is fully isolated from "default" and from
every other registered hospital.

Persistence is a single JSON manifest (hospital_id -> name/config path) —
no database, consistent with the rest of this prototype's storage style.

Scope note
----------
This module ONLY establishes identity + isolated state. RAG scoping and
routing-policy scoping are deliberately NOT wired here (multi-hospital
Steps 3+) — see MASTER_TRIAGEGUARD_KNOWLEDGE_BASE.md Part II.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from triageguard_agent.hospital.hospital_state_service import HospitalStateService
from triageguard_agent.hospital.hospital_state_store import HospitalStateStore

logger = logging.getLogger(__name__)

DEFAULT_HOSPITAL_ID = "default"

_HOSPITALS_DIR = Path(__file__).resolve().parents[1] / "data" / "hospitals"
_DEFAULT_MANIFEST_PATH = _HOSPITALS_DIR / "registry.json"
_DEFAULT_HOSPITAL_CONFIG = Path(__file__).resolve().parents[1] / "data" / "hospital_config.json"


@dataclass
class HospitalContext:
    """Everything scoped to one hospital. Minimal by design — Step 2 only
    binds identity + isolated operational state; RAG/policy references are
    added in later steps without changing this shape's meaning."""

    hospital_id: str
    hospital_name: str
    config_path: Path
    state_service: HospitalStateService


class HospitalRegistry:
    """
    Registry of known hospitals, keyed by hospital_id.

    Adding a hospital is always: register(...) -> config file written/
    reused -> isolated HospitalStateService created -> manifest updated.
    Never: copy a class, write hospital-specific code.
    """

    def __init__(self, manifest_path: Optional[Path] = None) -> None:
        self._manifest_path = manifest_path or _DEFAULT_MANIFEST_PATH
        # Per-hospital config files live alongside the manifest (not a
        # hardcoded module-level path) so a registry pointed at a test
        # tmp_path never writes into the real repo data directory.
        self._base_dir = self._manifest_path.parent
        self._hospitals: Dict[str, HospitalContext] = {}

        self._load_manifest()
        self._ensure_default_registered()

    # ------------------------------------------------------------------
    # Manifest persistence (hospital_id -> {hospital_name, config_path})
    # ------------------------------------------------------------------

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            return
        with open(self._manifest_path, encoding="utf-8") as fh:
            entries = json.load(fh)
        for hospital_id, entry in entries.items():
            if hospital_id == DEFAULT_HOSPITAL_ID:
                continue  # default is always re-bound to the live singleton below
            config_path = Path(entry["config_path"])
            self._hospitals[hospital_id] = HospitalContext(
                hospital_id=hospital_id,
                hospital_name=entry["hospital_name"],
                config_path=config_path,
                state_service=HospitalStateService(HospitalStateStore(config_path)),
            )
            logger.info("Loaded hospital %r from manifest.", hospital_id)

    def _save_manifest(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        entries = {
            hid: {"hospital_name": ctx.hospital_name, "config_path": str(ctx.config_path)}
            for hid, ctx in self._hospitals.items()
            if hid != DEFAULT_HOSPITAL_ID
        }
        with open(self._manifest_path, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)

    def _ensure_default_registered(self) -> None:
        if DEFAULT_HOSPITAL_ID in self._hospitals:
            return
        self._hospitals[DEFAULT_HOSPITAL_ID] = HospitalContext(
            hospital_id=DEFAULT_HOSPITAL_ID,
            hospital_name="Default Hospital",
            config_path=_DEFAULT_HOSPITAL_CONFIG,
            # Bound to the pre-existing singleton, NOT a fresh instance —
            # every existing caller of HospitalStateService.instance()
            # keeps reading/writing this exact same state.
            state_service=HospitalStateService.instance(),
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        hospital_id: str,
        hospital_name: str,
        config_path: Optional[Path] = None,
        config_dict: Optional[dict] = None,
    ) -> HospitalContext:
        """
        Register a new hospital and give it isolated operational state.

        Exactly one of `config_path` / `config_dict` may be given:
            config_path : reuse an existing hospital_config.json-shaped file.
            config_dict : write a new one (same schema as hospital_config.json).
            neither     : clone the default hospital's config as a starting point.

        Raises ValueError if hospital_id is empty, reserved ("default"),
        already registered, or both config_path and config_dict are given.
        """
        hospital_id = str(hospital_id).strip()
        hospital_name = str(hospital_name).strip()
        if not hospital_id:
            raise ValueError("hospital_id must not be empty.")
        if not hospital_name:
            raise ValueError("hospital_name must not be empty.")
        if hospital_id == DEFAULT_HOSPITAL_ID:
            raise ValueError(f"{DEFAULT_HOSPITAL_ID!r} is reserved for the default hospital.")
        if hospital_id in self._hospitals:
            raise ValueError(
                f"Hospital {hospital_id!r} is already registered. "
                "Choose a different hospital_id."
            )
        if config_path is not None and config_dict is not None:
            raise ValueError("Provide config_path or config_dict, not both.")

        target_path = self._base_dir / hospital_id / "hospital_config.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if config_dict is not None:
            with open(target_path, "w", encoding="utf-8") as fh:
                json.dump(config_dict, fh, indent=2)
        elif config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise ValueError(f"config_path {config_path} does not exist.")
            shutil.copyfile(config_path, target_path)
        else:
            # No configuration supplied — clone the default hospital's
            # config as a sane starting point the new hospital can edit.
            shutil.copyfile(_DEFAULT_HOSPITAL_CONFIG, target_path)

        context = HospitalContext(
            hospital_id=hospital_id,
            hospital_name=hospital_name,
            config_path=target_path,
            state_service=HospitalStateService(HospitalStateStore(target_path)),
        )
        self._hospitals[hospital_id] = context
        self._save_manifest()
        logger.info("Registered hospital %r (%s).", hospital_id, hospital_name)
        return context

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, hospital_id: str) -> HospitalContext:
        if hospital_id not in self._hospitals:
            raise KeyError(
                f"Hospital {hospital_id!r} is not registered. "
                f"Known hospitals: {list(self._hospitals.keys())}"
            )
        if hospital_id == DEFAULT_HOSPITAL_ID:
            # Re-resolve on every call rather than returning a cached
            # HospitalContext: other code (existing tests, tool callers)
            # calls HospitalStateService.reset_instance()/.instance()
            # independently of this registry's own lifecycle, and "default"
            # must always mirror whatever that live singleton currently is.
            cached = self._hospitals[DEFAULT_HOSPITAL_ID]
            return HospitalContext(
                hospital_id=DEFAULT_HOSPITAL_ID,
                hospital_name=cached.hospital_name,
                config_path=cached.config_path,
                state_service=HospitalStateService.instance(),
            )
        return self._hospitals[hospital_id]

    def exists(self, hospital_id: str) -> bool:
        return hospital_id in self._hospitals

    def list_hospitals(self) -> List[Dict[str, str]]:
        return [
            {
                "hospital_id": ctx.hospital_id,
                "hospital_name": ctx.hospital_name,
                "config_path": str(ctx.config_path),
            }
            for ctx in self._hospitals.values()
        ]


# ---------------------------------------------------------------------------
# Process-wide default registry accessor (one registry holding many
# hospitals — this is not a reintroduction of the single-hospital
# assumption, it mirrors HospitalStateService's own singleton-of-a-
# container pattern).
# ---------------------------------------------------------------------------

_default_registry: Optional[HospitalRegistry] = None


def get_default_registry() -> HospitalRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = HospitalRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Reset the process-wide registry (for testing only)."""
    global _default_registry
    _default_registry = None
