"""
uncertainty.py
--------------
OOD / policy-uncertainty mechanism (Phase 6).

Combines two simple, interpretable signals — deliberately NOT a deep
uncertainty model:

  1. utility_std of the chosen action from the Bayesian posterior
     (bayesian_policy.py's Laplace-approximated posterior covariance):
     how confident the fitted policy itself is about this utility estimate.
  2. state_ood_distance: how far the current clinical/hospital state is
     from the nurse-demonstration distribution the policy was calibrated on.

policy_uncertainty = clip(w1 * normalized_utility_std + w2 * state_ood_distance, 0, 1)

This affects ROUTING CONFIDENCE ONLY. It is never applied to clinical_priority
or to icu_risk/admission_risk anywhere in this codebase.
"""

from __future__ import annotations

from typing import Dict

from triageguard_router.policy.config import UncertaintyConfig

# A utility_std of this magnitude or more is treated as "fully uncertain" for
# normalization purposes. With only 18 demonstrations spread across a
# 23-dimensional feature vector (BayesianConfig.prior_variance=4.0), the
# Laplace-approximated posterior std for a typical candidate is empirically
# ~5-6 (many weight directions stay close to the prior after so few
# examples) — calibrated here so normalized_std spans a meaningful [0,1]
# range instead of saturating at 1.0 for nearly every prediction.
_UTILITY_STD_SATURATION = 8.0


def compute_policy_uncertainty(
    chosen_utility_std: float,
    state_ood_distance: float,
    config: UncertaintyConfig,
) -> float:
    normalized_std = min(1.0, chosen_utility_std / _UTILITY_STD_SATURATION)
    raw = (
        config.utility_std_weight * normalized_std
        + config.feature_distance_weight * state_ood_distance
    )
    return float(max(0.0, min(1.0, raw)))


def uncertainty_requires_review(policy_uncertainty: float, config: UncertaintyConfig) -> bool:
    return policy_uncertainty >= config.high_uncertainty_threshold
