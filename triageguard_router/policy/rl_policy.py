"""
rl_policy.py
------------
Lightweight PPO-style policy-gradient optimizer (Phase 11/12), implemented
directly with torch — no stable_baselines3/gymnasium dependency (neither is
installed in this environment; adding that dependency chain for a single
~25-parameter linear policy would be disproportionate — see the config
docstring and the final report's "deviations" section).

The RL policy is the SAME shallow linear utility form as the Bayesian
policy (U(s,a) = w^T phi(s,a)) — it is explicitly INITIALIZED from the
Bayesian policy's MAP weights (never from random init), and is kept close
to that nurse-informed policy via:

  1. A PPO clipped-surrogate objective (standard, prevents any single
     update from moving the weights too far).
  2. An explicit KL(pi_RL(.|s) || pi_nurse(.|s)) penalty evaluated at the
     states actually visited during simulated rollouts.
  3. A behavior-cloning ANCHOR: a quadratic penalty pulling w directly
     toward the nurse policy's own w_map. (A first version of this pulled
     toward maximizing raw likelihood on the demonstrations instead — but
     w_map already balances that likelihood against the Bayesian prior, so
     chasing pure likelihood HARDER actually pushed weights AWAY from
     w_map, the opposite of "stay close to the nurse policy." Caught by a
     test asserting a higher regularization weight produces less drift.)

Both regularizers are weighted by RLConfig.policy_kl_weight /
behavior_cloning_weight (config.py) — never hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from triageguard_router.policy.bayesian_policy import BayesianLinearPolicy
from triageguard_router.policy.config import RLConfig
from triageguard_router.policy.simulation_env import RoutingEnv

logger = logging.getLogger(__name__)


class RLRoutingPolicy:
    """A linear utility policy trained with a lightweight PPO-style update, warm-started from a Bayesian nurse policy."""

    def __init__(self, n_features: int, config: Optional[RLConfig] = None):
        self.n_features = n_features
        self.config = config or RLConfig()
        self.w = torch.zeros(n_features, dtype=torch.float64, requires_grad=True)
        self._nurse_w: Optional[torch.Tensor] = None
        self.initialized_from_nurse = False

    def init_from_bayesian(self, bayesian_policy: BayesianLinearPolicy) -> None:
        if not bayesian_policy.fitted:
            raise ValueError("Bayesian policy must be fitted before RL initialization.")
        self.w = torch.tensor(bayesian_policy.w_map.copy(), dtype=torch.float64, requires_grad=True)
        self._nurse_w = torch.tensor(bayesian_policy.w_map.copy(), dtype=torch.float64)  # frozen reference
        self.initialized_from_nurse = True

    # ------------------------------------------------------------------
    # Action selection (used both for training rollouts and, via
    # deterministic=True, at inference time in routing_policy.py)
    # ------------------------------------------------------------------

    def _masked_log_softmax(self, phi_mat: torch.Tensor, mask: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        utils = phi_mat @ w
        utils = utils.masked_fill(~mask, float("-inf"))
        return torch.log_softmax(utils, dim=-1)

    def select_action(self, phi_candidates: np.ndarray, action_mask: np.ndarray, deterministic: bool = False) -> int:
        with torch.no_grad():
            phi_t = torch.tensor(phi_candidates, dtype=torch.float64)
            mask_t = torch.tensor(action_mask, dtype=torch.bool)
            logp = self._masked_log_softmax(phi_t, mask_t, self.w)
            probs = torch.exp(logp).numpy()
        if deterministic:
            return int(np.argmax(probs))
        probs = np.nan_to_num(probs, nan=0.0)
        probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def utility(self, phi_vec: np.ndarray) -> float:
        with torch.no_grad():
            return float(torch.tensor(phi_vec, dtype=torch.float64) @ self.w)

    # ------------------------------------------------------------------
    # Training (Phase 11/12)
    # ------------------------------------------------------------------

    def train(self, env: RoutingEnv) -> Dict[str, Any]:
        if not self.initialized_from_nurse:
            raise RuntimeError(
                "RLRoutingPolicy must be initialized from a fitted BayesianLinearPolicy "
                "via init_from_bayesian() before training — random init is not permitted."
            )
        cfg = self.config
        optimizer = torch.optim.Adam([self.w], lr=cfg.learning_rate)
        history: List[Dict[str, Any]] = []

        for episode in range(cfg.episodes):
            def policy_fn(phi_mat, mask, candidates):
                return self.select_action(phi_mat, mask, deterministic=False)

            ep_result = env.run_episode(policy_fn, seed=env.config.random_seed + episode)
            if not ep_result.transitions:
                history.append({"episode": episode, "total_reward": 0.0, "n_transitions": 0, "unsafe": 0})
                continue

            old_w = self.w.detach().clone()
            rewards = np.array([t.reward for t in ep_result.transitions])
            # Simple running-mean baseline (no separate value network — kept
            # deliberately lightweight, as instructed).
            baseline = rewards.mean()
            advantages = rewards - baseline
            std = advantages.std()
            if std > 1e-6:
                advantages = advantages / std

            for _epoch in range(cfg.epochs):
                optimizer.zero_grad()
                ppo_loss = torch.zeros((), dtype=torch.float64)
                kl_loss = torch.zeros((), dtype=torch.float64)

                for t, adv in zip(ep_result.transitions, advantages):
                    phi_t = torch.tensor(t.phi_candidates, dtype=torch.float64)
                    mask_t = torch.tensor(t.action_mask, dtype=torch.bool)

                    logp_new_full = self._masked_log_softmax(phi_t, mask_t, self.w)
                    logp_new = logp_new_full[t.chosen_index]
                    with torch.no_grad():
                        logp_old = self._masked_log_softmax(phi_t, mask_t, old_w)[t.chosen_index]

                    ratio = torch.exp(logp_new - logp_old)
                    adv_t = torch.tensor(float(adv), dtype=torch.float64)
                    surrogate1 = ratio * adv_t
                    surrogate2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv_t
                    ppo_loss = ppo_loss - torch.min(surrogate1, surrogate2)

                    logp_nurse_full = self._masked_log_softmax(phi_t, mask_t, self._nurse_w)
                    probs_new = torch.exp(logp_new_full)
                    valid = mask_t
                    kl = (probs_new[valid] * (logp_new_full[valid] - logp_nurse_full[valid])).sum()
                    kl_loss = kl_loss + kl

                n = len(ep_result.transitions)
                bc_anchor = torch.sum((self.w - self._nurse_w) ** 2) / self.n_features
                loss = (
                    ppo_loss / n
                    + cfg.policy_kl_weight * (kl_loss / n)
                    + cfg.behavior_cloning_weight * bc_anchor
                )

                loss.backward()
                optimizer.step()

            history.append({
                "episode": episode,
                "total_reward": float(ep_result.total_reward),
                "n_transitions": len(ep_result.transitions),
                "unsafe": ep_result.unsafe_allocation_count,
                "resource_conflicts": ep_result.resource_conflict_count,
            })

        return {"history": history, "final_w": self.w.detach().numpy().copy()}
