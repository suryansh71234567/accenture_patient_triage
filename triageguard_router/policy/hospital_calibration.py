"""
hospital_calibration.py
------------------------
Multi-hospital Step 5: nurse response collection + policy fitting.

Lifecycle
---------
    scenarios_for_hospital(hospital_id, facility)   [Step 4]
            -> nurse answers each scenario_id       (NurseResponses, this module)
            -> apply_responses(...)                 (override template's baked-in choice)
            -> fit_hospital_policy(...)              -> BayesianLinearPolicy
            -> artifacts.save_bayesian_policy(policy, hospital_id=...)   [caller's choice, Step 5]

Reuses existing fitting code unmodified:
    training.build_training_examples()  (NurseScenario -> DemonstrationExample)
    BayesianLinearPolicy.fit()          (Laplace-approximated MAP)
No new ML algorithm, no learned questionnaire size — this module only wires
existing pieces together per hospital_id.

Scope note
----------
RL refinement (RLRoutingPolicy) is intentionally NOT triggered here: its
training environment (RoutingEnv/HospitalSimulator) still always simulates
against the single global hospital state (Step 7 is where simulation
becomes hospital-scoped). The Bayesian policy fitted here is exactly the
same "nurse-calibrated policy" artifact the existing single-hospital system
already produces — it just no longer has to be global.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from triageguard_router.policy.bayesian_policy import BayesianLinearPolicy
from triageguard_router.policy.config import BayesianConfig
from triageguard_router.policy.facility_calibration import scenarios_for_hospital
from triageguard_router.policy.schema import NurseScenario
from triageguard_router.policy.training import build_training_examples


@dataclasses.dataclass
class NurseResponses:
    """
    A nurse's answers for one hospital's calibration session.

    responses : scenario_id -> chosen department. A scenario_id with no
        entry falls back to that scenario template's own baked-in
        preferred_department (partial calibration is allowed — fitting
        only requires at least one demonstration).
    """

    hospital_id: str
    responses: Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NurseResponses":
        return cls(hospital_id=data["hospital_id"], responses=dict(data.get("responses", {})))


def apply_responses(
    scenarios: List[NurseScenario],
    responses: NurseResponses,
) -> List[NurseScenario]:
    """
    Override each scenario's preferred_department with the nurse's actual
    choice for THIS hospital, where one was given. Re-validates via
    NurseScenario's own __post_init__ (chosen department must be one of
    that scenario's candidate_departments) — raises ValueError otherwise,
    the same error the schema already enforces for the hardcoded templates.
    """
    calibrated = []
    for scenario in scenarios:
        chosen = responses.responses.get(scenario.scenario_id)
        if chosen is None or chosen == scenario.preferred_department:
            calibrated.append(scenario)
        else:
            calibrated.append(dataclasses.replace(scenario, preferred_department=chosen))
    return calibrated


def fit_hospital_policy(
    hospital_id: str,
    facility_departments: Dict[str, Any],
    responses: NurseResponses,
    bayesian_config: Optional[BayesianConfig] = None,
) -> BayesianLinearPolicy:
    """
    Full Step 5 fit: facility-adapted scenarios (Step 4) + this hospital's
    nurse responses -> a fitted BayesianLinearPolicy. Does not save
    anything — persistence is a separate, explicit artifacts.py call.
    """
    if responses.hospital_id != hospital_id:
        raise ValueError(
            f"responses belong to hospital_id={responses.hospital_id!r}, "
            f"not {hospital_id!r} — refusing to mix calibration data across hospitals."
        )

    scenarios = scenarios_for_hospital(hospital_id, facility_departments)["scenarios"]
    calibrated_scenarios = apply_responses(scenarios, responses)
    examples = build_training_examples(calibrated_scenarios)

    policy = BayesianLinearPolicy(config=bayesian_config)
    policy.fit(examples)  # mutates policy.training_metadata in place
    policy.training_metadata["hospital_id"] = hospital_id
    policy.training_metadata["scenario_count"] = len(calibrated_scenarios)
    return policy


def train_hospital_rl_policy(
    hospital_id: str,
    bayesian_policy: BayesianLinearPolicy,
    config=None,
) -> "RLRoutingPolicy":  # noqa: F821 (imported lazily below)
    """
    Multi-hospital Step 7 (Part D): RL refinement scoped to one hospital's
    own facility shape, reusing the existing PPO-style RLRoutingPolicy and
    RoutingEnv UNCHANGED — this only supplies hospital_id so the env's
    per-episode HospitalStateStore is re-read from that hospital's own
    config_path instead of the global singleton (see simulation_env.py).
    No new RL algorithm; no math changes.

    Caller is responsible for persisting the result, e.g.:
        artifacts.save_rl_policy(rl_policy, hospital_id=hospital_id)
    """
    from triageguard_router.policy.config import PolicyConfig
    from triageguard_router.policy.rl_policy import RLRoutingPolicy
    from triageguard_router.policy.simulation_env import RoutingEnv

    cfg = config or PolicyConfig()
    rl_policy = RLRoutingPolicy(n_features=len(bayesian_policy.w_map), config=cfg.rl)
    rl_policy.init_from_bayesian(bayesian_policy)
    env = RoutingEnv(config=cfg, hospital_id=hospital_id)
    rl_policy.train(env)
    return rl_policy
