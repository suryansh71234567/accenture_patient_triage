"""
scripts/calibrate_policy.py
----------------------------
Fit the Bayesian nurse-aligned routing policy from the fixed nurse
demonstrations and persist it (Phase 21: `python -m` equivalent of
`triageguard.routing.calibrate`, adapted to this repo's existing
`triageguard_router/scripts/*.py` convention — see scripts/run_triage.py).

Usage
-----
    .venv\\Scripts\\python.exe triageguard_router/scripts/calibrate_policy.py
"""
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from triageguard_router.policy import artifacts
from triageguard_router.policy.bayesian_policy import BayesianLinearPolicy
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.demonstrations import load_demonstrations
from triageguard_router.policy.features import FEATURE_NAMES
from triageguard_router.policy.training import build_training_examples


def main() -> None:
    cfg = PolicyConfig.load()
    cfg.save()  # ensure policy_config.yaml exists with current defaults documented

    scenarios = load_demonstrations()
    print(f"Loaded {len(scenarios)} nurse demonstrations.")
    examples = build_training_examples(scenarios)

    policy = BayesianLinearPolicy(n_features=len(FEATURE_NAMES), config=cfg.bayesian)
    meta = policy.fit(examples)

    demo_path = artifacts.save_demonstrations(scenarios)
    policy_path = artifacts.save_bayesian_policy(policy)
    meta_path = artifacts.save_training_metadata({
        "stage": "calibrate",
        "random_seed": cfg.random_seed,
        "n_nurse_scenarios": len(scenarios),
        "feature_names": FEATURE_NAMES,
        "bayesian_training_metadata": meta,
        "config": cfg.to_dict(),
    })

    print(f"Nurse demonstrations saved to {demo_path}")
    print(f"Bayesian policy saved to {policy_path}")
    print(f"Training metadata saved to {meta_path}")
    print(f"Imitation accuracy on demonstrations: {meta['training_imitation_accuracy']:.2%}")


if __name__ == "__main__":
    main()
