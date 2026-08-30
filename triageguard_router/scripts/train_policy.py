"""
scripts/train_policy.py
------------------------
Optimize the nurse-informed Bayesian policy with simulation-based RL
(Phase 21: equivalent of `triageguard.routing.train`).

Requires calibrate_policy.py to have been run first (loads its saved
Bayesian policy as the RL warm-start / KL reference).

Usage
-----
    .venv\\Scripts\\python.exe triageguard_router/scripts/train_policy.py
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from triageguard_router.policy import artifacts
from triageguard_router.policy.config import PolicyConfig
from triageguard_router.policy.features import FEATURE_NAMES
from triageguard_router.policy.rl_policy import RLRoutingPolicy
from triageguard_router.policy.simulation_env import RoutingEnv


def main() -> None:
    cfg = PolicyConfig.load()

    if not artifacts.artifacts_exist():
        print("No calibrated Bayesian policy found — run calibrate_policy.py first.")
        sys.exit(1)

    bayesian_policy = artifacts.load_bayesian_policy(cfg)

    env = RoutingEnv(config=cfg)
    rl_policy = RLRoutingPolicy(n_features=len(FEATURE_NAMES), config=cfg.rl)
    rl_policy.init_from_bayesian(bayesian_policy)

    print(f"Training RL policy: {cfg.rl.episodes} episodes x {cfg.rl.epochs} epochs, warm-started from nurse policy.")
    result = rl_policy.train(env)

    rl_path = artifacts.save_rl_policy(rl_policy)
    artifacts.save_training_metadata({
        "stage": "train",
        "random_seed": cfg.random_seed,
        "rl_config": cfg.to_dict()["rl"],
        "episode_history": result["history"],
    }, path=artifacts._POLICY_DIR / "rl_training_metadata.json")

    final = result["history"][-1] if result["history"] else {}
    print(f"RL policy saved to {rl_path}")
    print(f"Final episode reward: {final.get('total_reward')}, unsafe: {final.get('unsafe')}")


if __name__ == "__main__":
    main()
