"""
evaluate.py
-----------
Ablation study (Phase 16): existing heuristic router vs nurse (Bayesian)
policy vs nurse+RL policy, all run through the SAME RoutingEnv (same
simulator, same seeds, same reward accounting) so the comparison is apples
to apples.

Baseline A is not reimplemented from scratch — it mirrors the existing
inline resource-fallback heuristic already in
triageguard_agent/simulation/hospital_simulator.py::HospitalSimulator.triage_patient()
(ICU/CICU full -> jump straight to ED_OBS, skipping any intermediate
step-down): generic, resource-blind, and exactly what a learned policy
should be able to improve on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from triageguard_router.policy.bayesian_policy import BayesianLinearPolicy
from triageguard_router.policy.candidates import DISCHARGE_DEPARTMENT
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.simulation_env import EpisodeResult, RoutingEnv


def heuristic_baseline_policy_fn(phi_mat: np.ndarray, mask: np.ndarray, candidates: List[str]) -> int:
    """
    The existing simulator's own inline heuristic (Phase 16 baseline A):
    take the preferred department if available, otherwise jump straight to
    ED_OBS (or DISCHARGE, if that's genuinely the only feasible thing),
    ignoring any intermediate step-down that might exist.
    """
    preferred_slot = _first_action_feature_index()  # ACTION_FEATURE_NAMES[0] == "is_preferred_department"

    for i, dept in enumerate(candidates):
        if mask[i] and phi_mat[i][preferred_slot] == 1.0:
            return i

    for i, dept in enumerate(candidates):
        if mask[i] and dept == "ED_OBS":
            return i

    feasible = [i for i in range(len(candidates)) if mask[i]]
    return feasible[0] if feasible else 0


def _first_action_feature_index() -> int:
    from triageguard_router.policy.features import STATE_FEATURE_NAMES
    return len(STATE_FEATURE_NAMES)  # ACTION features start right after STATE features


def bayesian_policy_fn(policy: BayesianLinearPolicy):
    def _fn(phi_mat: np.ndarray, mask: np.ndarray, candidates: List[str]) -> int:
        means = phi_mat @ policy.w_map
        means = np.where(mask, means, -np.inf)
        return int(np.argmax(means))
    return _fn


def rl_policy_fn(rl_policy):
    def _fn(phi_mat: np.ndarray, mask: np.ndarray, candidates: List[str]) -> int:
        return rl_policy.select_action(phi_mat, mask, deterministic=True)
    return _fn


@dataclass
class AblationMetrics:
    total_reward: float = 0.0
    n_episodes: int = 0
    n_decisions: int = 0
    unsafe_allocation_count: int = 0
    resource_conflict_count: int = 0
    avg_high_priority_wait_minutes: float = 0.0
    department_admission_counts: Dict[str, int] = field(default_factory=dict)
    throughput: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": round(self.total_reward, 2),
            "n_episodes": self.n_episodes,
            "n_decisions": self.n_decisions,
            "unsafe_allocation_count": self.unsafe_allocation_count,
            "resource_conflict_count": self.resource_conflict_count,
            "avg_high_priority_wait_minutes": round(self.avg_high_priority_wait_minutes, 2),
            "department_admission_counts": self.department_admission_counts,
            "throughput": self.throughput,
        }


def evaluate_policy(
    env: RoutingEnv,
    policy_fn,
    n_episodes: int,
    scenario_name: str = "NORMAL_DAY",
    seed_offset: int = 0,
) -> AblationMetrics:
    metrics = AblationMetrics(n_episodes=n_episodes)
    all_waits: List[float] = []

    for ep in range(n_episodes):
        result: EpisodeResult = env.run_episode(policy_fn, scenario_name=scenario_name, seed=env.config.random_seed + seed_offset + ep)
        metrics.total_reward += result.total_reward
        metrics.n_decisions += len(result.transitions)
        metrics.unsafe_allocation_count += result.unsafe_allocation_count
        metrics.resource_conflict_count += result.resource_conflict_count
        metrics.throughput += sum(result.department_admission_counts.values())
        all_waits.extend(result.high_priority_wait_minutes)
        for dept, count in result.department_admission_counts.items():
            metrics.department_admission_counts[dept] = metrics.department_admission_counts.get(dept, 0) + count

    metrics.avg_high_priority_wait_minutes = float(np.mean(all_waits)) if all_waits else 0.0
    return metrics


def run_ablation(
    config: PolicyConfig,
    bayesian_policy: BayesianLinearPolicy,
    rl_policy=None,
    n_episodes: int = 20,
    scenario_name: str = "NORMAL_DAY",
) -> Dict[str, Dict[str, Any]]:
    env = RoutingEnv(config=config)

    results = {
        "heuristic_baseline": evaluate_policy(env, heuristic_baseline_policy_fn, n_episodes, scenario_name, seed_offset=0).to_dict(),
        "nurse_bayesian_policy": evaluate_policy(env, bayesian_policy_fn(bayesian_policy), n_episodes, scenario_name, seed_offset=1000).to_dict(),
    }
    if rl_policy is not None:
        results["nurse_rl_policy"] = evaluate_policy(env, rl_policy_fn(rl_policy), n_episodes, scenario_name, seed_offset=2000).to_dict()
    return results
