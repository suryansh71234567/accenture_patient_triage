"""
triageguard_router.policy
--------------------------
Nurse-guided Bayesian routing policy + simulation-based RL optimization.

Pipeline (see routing_policy.py for the orchestrator):

    nurse demonstrations
            -> Bayesian nurse-aligned policy (bayesian_policy.py)
            -> hospital simulation (simulation_env.py, reuses triageguard_agent.simulation)
            -> policy-gradient optimization (rl_policy.py)
            -> final routing policy
            -> runtime routing (routing_policy.py)

This package NEVER modifies clinical risk. It consumes the existing
reconciler/router clinical output as a fixed input and only decides how to
ALLOCATE a patient among clinically-acceptable departments under real
resource constraints. See safety.py for the hard boundary between the two.
"""
