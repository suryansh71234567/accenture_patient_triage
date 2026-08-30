"""
config.py
---------
All tunable knobs for the routing-policy subsystem, in one place, loadable
from / savable to YAML — mirroring triageguard_rag/config/config.yaml's
convention of keeping tunables out of code.

Nothing in this module touches clinical risk. Every field here only affects
HOW resource allocation is scored/optimized among already clinically-
acceptable departments (see safety.py), never WHICH department is clinically
preferred (that is router.route()'s decision, upstream and untouched).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "policy_config.yaml"


@dataclass
class RewardWeights:
    """
    Transparent, configurable reward weights (Phase 10).

    Safety-violation penalties are deliberately an order of magnitude (or
    more) larger than any efficiency reward/penalty, so a training run can
    never learn to trade safety for throughput.
    """
    # Positive
    clinically_appropriate_destination: float = 1.0
    successful_allocation: float = 0.5
    reduced_waiting_time: float = 0.3
    reasonable_utilization: float = 0.2
    reduced_unnecessary_icu_usage: float = 0.3
    throughput: float = 0.2

    # Negative (efficiency-scale)
    excessive_high_priority_waiting: float = -0.6
    resource_conflict: float = -1.0

    # Negative (safety-scale — must dominate all of the above)
    unsafe_downgrade: float = -25.0
    failure_to_allocate_critical: float = -25.0
    clinically_incompatible_allocation: float = -50.0


@dataclass
class SafetyPenalties:
    """Penalties applied inside the RL environment for masked/invalid actions (Phase 12)."""
    invalid_action_penalty: float = -50.0
    action_masking_enabled: bool = True


@dataclass
class BayesianConfig:
    """Phase 4/19: Bayesian linear-utility policy hyperparameters."""
    prior_mean: float = 0.0
    prior_variance: float = 4.0
    map_learning_rate: float = 0.05
    map_iterations: int = 400
    hessian_jitter: float = 1e-4
    # Risk-averse scoring: department_scores use (utility_mean - weight *
    # utility_std) rather than the raw mean, so a high-uncertainty option
    # never wins purely on an untrustworthy point estimate. Also the
    # "uncertainty adjustment" term surfaced in explain.py's decomposition.
    uncertainty_penalty_weight: float = 0.15


@dataclass
class RLConfig:
    """Phase 11/19: lightweight PPO-style policy-gradient hyperparameters."""
    algorithm: str = "lightweight_ppo"
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    clip_range: float = 0.2
    epochs: int = 4
    episodes: int = 120
    steps_per_episode: int = 24
    minutes_per_step: int = 15
    behavior_cloning_weight: float = 0.5   # BC loss weight during warm start
    policy_kl_weight: float = 0.05         # beta: RL objective - beta * KL(pi_RL || pi_nurse)


@dataclass
class UncertaintyConfig:
    """Phase 6/19: OOD / uncertainty knobs."""
    high_uncertainty_threshold: float = 0.55
    utility_std_weight: float = 0.5
    feature_distance_weight: float = 0.5


@dataclass
class PolicyConfig:
    number_of_nurse_scenarios: int = 18
    random_seed: int = 1337

    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    safety: SafetyPenalties = field(default_factory=SafetyPenalties)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

    max_step_down_tiers: int = 3  # how many acuity tiers below preferred are ever candidates

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyConfig":
        cfg = cls()
        for key in ("number_of_nurse_scenarios", "random_seed", "max_step_down_tiers"):
            if key in data:
                setattr(cfg, key, data[key])
        if "bayesian" in data:
            cfg.bayesian = BayesianConfig(**data["bayesian"])
        if "rl" in data:
            cfg.rl = RLConfig(**data["rl"])
        if "reward" in data:
            cfg.reward = RewardWeights(**data["reward"])
        if "safety" in data:
            cfg.safety = SafetyPenalties(**data["safety"])
        if "uncertainty" in data:
            cfg.uncertainty = UncertaintyConfig(**data["uncertainty"])
        return cfg

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PolicyConfig":
        path = path or _DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or _DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)


def default_config() -> PolicyConfig:
    return PolicyConfig.load()
