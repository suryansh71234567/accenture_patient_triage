"""
artifacts.py
------------
Persist / load calibrated and trained policy artifacts (Phase 20).

triageguard_router/data/routing_policy/
    nurse_demonstrations.json   — the 18 scenarios, serialized
    bayesian_policy.json        — w_map, posterior_cov, demo stats
    rl_policy.json              — RL policy weights (if trained)
    training_metadata.json      — seed, dates, counts, config snapshot

triageguard_router/data/evaluation/
    baseline_metrics.json
    nurse_policy_metrics.json
    rl_policy_metrics.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from triageguard_router.policy.bayesian_policy import BayesianLinearPolicy
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.demonstrations import load_demonstrations
from triageguard_router.policy.features import FEATURE_NAMES
from triageguard_router.policy.schema import NurseScenario

_POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "routing_policy"
_EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "evaluation"


def save_demonstrations(scenarios=None, path: Optional[Path] = None) -> Path:
    scenarios = scenarios or load_demonstrations()
    path = path or (_POLICY_DIR / "nurse_demonstrations.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([s.to_dict() for s in scenarios], fh, indent=2, default=str)
    return path


def save_bayesian_policy(policy: BayesianLinearPolicy, path: Optional[Path] = None) -> Path:
    path = path or (_POLICY_DIR / "bayesian_policy.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "feature_names": FEATURE_NAMES,
        "w_map": policy.w_map.tolist(),
        "posterior_cov": policy.posterior_cov.tolist(),
        "demo_state_mean": policy.demo_state_mean.tolist() if policy.demo_state_mean is not None else None,
        "demo_state_std": policy.demo_state_std.tolist() if policy.demo_state_std is not None else None,
        "training_metadata": policy.training_metadata,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


def load_bayesian_policy(config: PolicyConfig, path: Optional[Path] = None) -> BayesianLinearPolicy:
    path = path or (_POLICY_DIR / "bayesian_policy.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    policy = BayesianLinearPolicy(n_features=len(data["feature_names"]), config=config.bayesian)
    policy.w_map = np.array(data["w_map"])
    policy.posterior_cov = np.array(data["posterior_cov"])
    if data.get("demo_state_mean") is not None:
        policy.demo_state_mean = np.array(data["demo_state_mean"])
        policy.demo_state_std = np.array(data["demo_state_std"])
    policy.training_metadata = data.get("training_metadata", {})
    policy.fitted = True
    return policy


def save_rl_policy(rl_policy, path: Optional[Path] = None) -> Path:
    path = path or (_POLICY_DIR / "rl_policy.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "feature_names": FEATURE_NAMES,
        "w": rl_policy.w.detach().numpy().tolist(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


def load_rl_policy(config: PolicyConfig, path: Optional[Path] = None):
    import torch
    from triageguard_router.policy.rl_policy import RLRoutingPolicy

    path = path or (_POLICY_DIR / "rl_policy.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rl = RLRoutingPolicy(n_features=len(data["feature_names"]), config=config.rl)
    rl.w = torch.tensor(data["w"], dtype=torch.float64, requires_grad=True)
    rl.initialized_from_nurse = True
    return rl


def save_training_metadata(
    metadata: Dict[str, Any],
    path: Optional[Path] = None,
) -> Path:
    path = path or (_POLICY_DIR / "training_metadata.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    return path


def save_evaluation_metrics(name: str, metrics: Dict[str, Any]) -> Path:
    path = _EVAL_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    return path


def artifacts_exist() -> bool:
    return (_POLICY_DIR / "bayesian_policy.json").exists()
