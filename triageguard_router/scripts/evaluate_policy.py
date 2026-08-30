"""
scripts/evaluate_policy.py
----------------------------
Ablation study: heuristic baseline vs nurse (Bayesian) policy vs
nurse+RL policy (Phase 16/21: equivalent of `triageguard.routing.evaluate`).

Usage
-----
    .venv\\Scripts\\python.exe triageguard_router/scripts/evaluate_policy.py
"""
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from triageguard_router.policy import artifacts
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.evaluate import run_ablation


def main() -> None:
    cfg = PolicyConfig.load()

    if not artifacts.artifacts_exist():
        print("No calibrated Bayesian policy found — run calibrate_policy.py first.")
        sys.exit(1)

    bayesian_policy = artifacts.load_bayesian_policy(cfg)
    rl_policy = None
    rl_path = artifacts._POLICY_DIR / "rl_policy.json"
    if rl_path.exists():
        rl_policy = artifacts.load_rl_policy(cfg, rl_path)
    else:
        print("No trained RL policy found — evaluating heuristic vs nurse-Bayesian only.")

    results = run_ablation(cfg, bayesian_policy, rl_policy, n_episodes=20)

    for name, metrics in results.items():
        artifacts.save_evaluation_metrics(
            {"heuristic_baseline": "baseline_metrics", "nurse_bayesian_policy": "nurse_policy_metrics", "nurse_rl_policy": "rl_policy_metrics"}[name],
            metrics,
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
