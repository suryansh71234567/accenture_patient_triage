"""
test_hospital_registry.py
--------------------------
Multi-hospital Step 2: identity + registry foundation.

Verifies:
* "default" hospital is pre-registered and bound to the existing
  HospitalStateService.instance() singleton (no behavior change).
* A newly registered hospital gets fully isolated state.
* Mutating one hospital's state never affects another's.
* Manifest persistence survives a fresh HospitalRegistry instance.
* Sane error handling (duplicate id, unknown id, reserved id).
"""

import json

import pytest

from triageguard_agent.hospital.hospital_registry import (
    DEFAULT_HOSPITAL_ID,
    HospitalRegistry,
)
from triageguard_agent.hospital.hospital_state_service import HospitalStateService


@pytest.fixture(autouse=True)
def _reset_default_singleton():
    """Isolate from other test files that also use the process-wide
    HospitalStateService singleton (test_dynamic_simulation.py,
    test_llm_planning_loop.py follow the same pattern)."""
    HospitalStateService.reset_instance()
    yield
    HospitalStateService.reset_instance()


@pytest.fixture
def registry(tmp_path):
    return HospitalRegistry(manifest_path=tmp_path / "registry.json")


_MINIMAL_CONFIG = {
    "departments": {
        "ICU": {"capacity": 4, "occupied": 1, "available": 3, "status": "OPEN"},
        "DISCHARGE": {"capacity": 999, "occupied": 0, "available": 999, "status": "OPEN"},
    },
    "lambda_thresholds": {"normal_max": 0.70, "high_load_max": 0.90},
    "stale_threshold_minutes": 30,
}


class TestDefaultHospital:
    def test_default_is_preregistered(self, registry):
        assert registry.exists(DEFAULT_HOSPITAL_ID)
        ctx = registry.get(DEFAULT_HOSPITAL_ID)
        assert ctx.hospital_id == DEFAULT_HOSPITAL_ID

    def test_default_is_bound_to_existing_singleton(self, registry):
        ctx = registry.get(DEFAULT_HOSPITAL_ID)
        assert ctx.state_service is HospitalStateService.instance()


class TestRegistration:
    def test_register_with_config_dict_creates_isolated_state(self, registry):
        ctx = registry.register("hosp_a", "Hospital A", config_dict=_MINIMAL_CONFIG)
        assert ctx.state_service.get_state("ICU")["capacity"] == 4
        assert ctx.config_path.exists()

    def test_register_without_config_clones_default(self, registry):
        ctx = registry.register("hosp_b", "Hospital B")
        # Default hospital_config.json has ICU capacity 10 — proves the clone happened.
        assert ctx.state_service.get_state("ICU")["capacity"] == 10

    def test_duplicate_hospital_id_rejected(self, registry):
        registry.register("hosp_a", "Hospital A", config_dict=_MINIMAL_CONFIG)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("hosp_a", "Hospital A Again", config_dict=_MINIMAL_CONFIG)

    def test_reserved_default_id_rejected(self, registry):
        with pytest.raises(ValueError, match="reserved"):
            registry.register(DEFAULT_HOSPITAL_ID, "Nope", config_dict=_MINIMAL_CONFIG)

    def test_both_config_sources_rejected(self, registry, tmp_path):
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps(_MINIMAL_CONFIG), encoding="utf-8")
        with pytest.raises(ValueError, match="not both"):
            registry.register("hosp_c", "Hospital C", config_path=cfg, config_dict=_MINIMAL_CONFIG)

    def test_unknown_hospital_id_raises(self, registry):
        with pytest.raises(KeyError, match="not registered"):
            registry.get("does_not_exist")


class TestIsolation:
    def test_mutating_one_hospital_never_affects_another_or_default(self, registry):
        hosp_a = registry.register("hosp_a", "Hospital A", config_dict=_MINIMAL_CONFIG)
        hosp_b = registry.register("hosp_b", "Hospital B", config_dict=_MINIMAL_CONFIG)
        default_ctx = registry.get(DEFAULT_HOSPITAL_ID)

        before_b = hosp_b.state_service.get_state("ICU")["occupied"]
        before_default = default_ctx.state_service.get_state("ICU")["occupied"]

        patch = hosp_a.state_service.validate_update("ICU", {"occupied": 3})
        hosp_a.state_service.apply_update("ICU", patch)

        assert hosp_a.state_service.get_state("ICU")["occupied"] == 3
        assert hosp_b.state_service.get_state("ICU")["occupied"] == before_b
        assert default_ctx.state_service.get_state("ICU")["occupied"] == before_default

    def test_list_hospitals_includes_default_and_registered(self, registry):
        registry.register("hosp_a", "Hospital A", config_dict=_MINIMAL_CONFIG)
        ids = {h["hospital_id"] for h in registry.list_hospitals()}
        assert ids == {DEFAULT_HOSPITAL_ID, "hosp_a"}


class TestManifestPersistence:
    def test_registered_hospital_survives_new_registry_instance(self, tmp_path):
        manifest = tmp_path / "registry.json"
        r1 = HospitalRegistry(manifest_path=manifest)
        r1.register("hosp_a", "Hospital A", config_dict=_MINIMAL_CONFIG)

        r2 = HospitalRegistry(manifest_path=manifest)
        assert r2.exists("hosp_a")
        assert r2.get("hosp_a").state_service.get_state("ICU")["capacity"] == 4

    def test_second_instance_has_independent_state_objects(self, tmp_path):
        """Reloading from the manifest creates a fresh HospitalStateService
        (isolation is per-process-registry, not shared mutable global state
        beyond the intentional 'default' singleton binding)."""
        manifest = tmp_path / "registry.json"
        r1 = HospitalRegistry(manifest_path=manifest)
        r1.register("hosp_a", "Hospital A", config_dict=_MINIMAL_CONFIG)

        r2 = HospitalRegistry(manifest_path=manifest)
        assert r2.get("hosp_a").state_service is not r1.get("hosp_a").state_service
