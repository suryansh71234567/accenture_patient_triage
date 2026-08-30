"""
bayesian_policy.py
-------------------
Lightweight Bayesian linear-utility routing policy (Phase 4/5).

Model
-----
For candidate department a in state s:
    U(s,a) = w^T phi(s,a)                      (shallow linear utility)
    pi(a|s) = softmax_a( U(s,a) )               (softmax policy)

Prior:
    w ~ Normal(prior_mean, prior_variance * I)

There are only ~15-20 expert demonstrations, so an exact softmax posterior
has no closed form and a full MCMC sampler would be overkill (and slow) for
a hackathon system. Instead this uses a LAPLACE APPROXIMATION, a standard,
numerically simple approximate-Bayesian method:

    1. Find w_MAP by maximizing log p(w | data) = log-likelihood + log-prior
       via gradient ascent (Adam) on the negative log posterior.
    2. Approximate the posterior as Normal(w_MAP, H^-1), where H is the
       Hessian of the NEGATIVE log posterior at w_MAP (computed via
       torch.autograd.functional.hessian — exact for this small parameter
       count, ~20-30 dims). The Gaussian prior's contribution to H
       guarantees H stays well-conditioned even with few datapoints.

This gives closed-form utility_mean / utility_std per (s,a) without any
sampling loop, which is what routing_policy.py and explain.py expose to
callers (Phase 4's required output shape).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from triageguard_router.policy.config import BayesianConfig
from triageguard_router.policy.features import (
    STATE_FEATURE_NAMES,
    FEATURE_NAMES,
    ClinicalSignal,
    HospitalSignal,
    phi,
    phi_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class DemonstrationExample:
    """One (state, candidate-set, chosen-action) training example."""
    phi_candidates: np.ndarray   # (n_candidates, n_features)
    candidate_departments: List[str]
    chosen_index: int
    state_features: np.ndarray   # (n_state_features,) — for OOD distance


class BayesianLinearPolicy:
    """A Bayesian softmax policy over a shared linear utility function."""

    def __init__(self, n_features: int = len(FEATURE_NAMES), config: Optional[BayesianConfig] = None):
        self.n_features = n_features
        self.config = config or BayesianConfig()
        self.w_map: np.ndarray = np.zeros(n_features)
        self.posterior_cov: np.ndarray = np.eye(n_features) * self.config.prior_variance
        self.demo_state_mean: Optional[np.ndarray] = None
        self.demo_state_std: Optional[np.ndarray] = None
        self.fitted = False
        self.training_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Fitting (Phase 5: behavior cloning via MAP + Laplace posterior)
    # ------------------------------------------------------------------

    def fit(self, examples: List[DemonstrationExample]) -> Dict[str, Any]:
        if not examples:
            raise ValueError("BayesianLinearPolicy.fit() requires at least one demonstration.")

        cfg = self.config
        torch.manual_seed(0)
        w = torch.zeros(self.n_features, dtype=torch.float64, requires_grad=True)
        phis = [torch.tensor(ex.phi_candidates, dtype=torch.float64) for ex in examples]
        chosen = [ex.chosen_index for ex in examples]

        def neg_log_posterior(w_: torch.Tensor) -> torch.Tensor:
            nll = torch.zeros((), dtype=torch.float64)
            for phi_mat, c in zip(phis, chosen):
                utils = phi_mat @ w_
                logp = torch.log_softmax(utils, dim=0)
                nll = nll - logp[c]
            neg_log_prior = torch.sum((w_ - cfg.prior_mean) ** 2) / (2.0 * cfg.prior_variance)
            return nll + neg_log_prior

        optimizer = torch.optim.Adam([w], lr=cfg.map_learning_rate)
        final_loss = None
        for _ in range(cfg.map_iterations):
            optimizer.zero_grad()
            loss = neg_log_posterior(w)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        w_map = w.detach()

        hessian = torch.autograd.functional.hessian(neg_log_posterior, w_map)
        hessian_np = hessian.numpy() + np.eye(self.n_features) * cfg.hessian_jitter
        try:
            cov = np.linalg.inv(hessian_np)
        except np.linalg.LinAlgError:
            logger.warning("Posterior Hessian singular — falling back to pseudo-inverse.")
            cov = np.linalg.pinv(hessian_np)
        cov = (cov + cov.T) / 2.0  # numerical symmetrization

        self.w_map = w_map.numpy()
        self.posterior_cov = cov
        self.fitted = True

        state_feats = np.stack([ex.state_features for ex in examples])
        self.demo_state_mean = state_feats.mean(axis=0)
        self.demo_state_std = state_feats.std(axis=0) + 1e-6  # avoid div-by-zero

        train_accuracy = self._imitation_accuracy(examples)
        self.training_metadata = {
            "n_demonstrations": len(examples),
            "final_neg_log_posterior": final_loss,
            "map_iterations": cfg.map_iterations,
            "training_imitation_accuracy": train_accuracy,
        }
        logger.info(
            "BayesianLinearPolicy fit on %d demonstrations — imitation accuracy=%.2f",
            len(examples), train_accuracy,
        )
        return self.training_metadata

    def _imitation_accuracy(self, examples: List[DemonstrationExample]) -> float:
        correct = 0
        for ex in examples:
            utils = ex.phi_candidates @ self.w_map
            if int(np.argmax(utils)) == ex.chosen_index:
                correct += 1
        return correct / len(examples)

    # ------------------------------------------------------------------
    # Inference (Phase 4's required output shape)
    # ------------------------------------------------------------------

    def utility(self, phi_vec: np.ndarray) -> Tuple[float, float]:
        """Return (utility_mean, utility_std) for one phi(s,a) vector."""
        mean = float(phi_vec @ self.w_map)
        var = float(phi_vec @ self.posterior_cov @ phi_vec)
        std = float(np.sqrt(max(var, 0.0)))
        return mean, std

    def predict(
        self,
        departments: List[str],
        clinical: ClinicalSignal,
        hospital: HospitalSignal,
    ) -> Dict[str, Dict[str, float]]:
        """
        {
          "ICU": {"probability": 0.71, "utility_mean": 1.42, "utility_std": 0.18},
          ...
        }
        """
        if not departments:
            return {}
        phi_mat = phi_matrix(departments, clinical, hospital)
        means = phi_mat @ self.w_map
        probs = _softmax(means)

        out: Dict[str, Dict[str, float]] = {}
        for i, dept in enumerate(departments):
            _, std = self.utility(phi_mat[i])
            out[dept] = {
                "probability": float(probs[i]),
                "utility_mean": float(means[i]),
                "utility_std": std,
            }
        return out

    def state_ood_distance(self, state_features: np.ndarray) -> float:
        """
        Normalized distance of a state's feature vector from the training-
        demonstration distribution (mean absolute z-score across STATE
        features), clipped to [0, 1]. 0 = right at the demo mean, 1 = far
        from anything seen during nurse calibration.
        """
        if self.demo_state_mean is None or self.demo_state_std is None:
            return 1.0  # unfitted policy — treat everything as maximally OOD
        z = np.abs((state_features - self.demo_state_mean) / self.demo_state_std)
        return float(np.clip(z.mean() / 3.0, 0.0, 1.0))  # 3 std devs -> fully OOD


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / exp.sum()
