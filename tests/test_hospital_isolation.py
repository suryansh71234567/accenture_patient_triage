"""
test_hospital_isolation.py
---------------------------
Multi-hospital Step 3: hospital-specific operational state + hospital-scoped
RAG retrieval.

Covers
------
1. Retriever.retrieve() never mixes hospitals, including the tricky case of
   two hospitals reusing the same patient_id.
2. Documents with no hospital_id field (legacy/base MIMIC corpus) are only
   visible under the "default" hospital, never under a named one.
3. hospital_id=None preserves the old unscoped behavior exactly.
4. RAGPipeline.run() and TriageGuardPipeline.run() thread hospital_id
   through correctly (including falling back to patient_state["hospital_id"]).
5. hospital_tools.py routes get/propose/commit through the hospital
   registry, and one hospital's state changes never affect another's.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from triageguard_rag.src.embeddings.embedder import Embedder
from triageguard_rag.src.retrieval.retriever import Retriever, build_index


@pytest.fixture(scope="session")
def embedder():
    return Embedder("sentence-transformers/all-MiniLM-L6-v2")


def _doc(text, patient_id, hospital_id=None, **extra_meta):
    meta = {"patient_id": patient_id, **extra_meta}
    if hospital_id is not None:
        meta["hospital_id"] = hospital_id
    return {"document_text": text, "metadata": meta}


@pytest.fixture()
def two_hospital_index(tmp_path, embedder):
    """
    5 documents:
      doc0 patient 1 @ hosp_a  — "sepsis fever tachycardia"
      doc1 patient 1 @ hosp_b  — "sepsis fever tachycardia"   (colliding patient_id!)
      doc2 patient 2 @ hosp_a  — "sepsis fever tachycardia"   (similar case for hosp_a)
      doc3 patient 3 @ hosp_b  — "sepsis fever tachycardia"   (similar case for hosp_b)
      doc4 patient 4 @ (none)  — "sepsis fever tachycardia"   (legacy/base corpus doc)
    """
    docs = [
        _doc("Patient with sepsis, fever, and tachycardia.", 1, "hosp_a"),
        _doc("Patient with sepsis, fever, and tachycardia.", 1, "hosp_b"),
        _doc("Another patient with sepsis, fever, and tachycardia.", 2, "hosp_a"),
        _doc("Another patient with sepsis, fever, and tachycardia.", 3, "hosp_b"),
        _doc("Another patient with sepsis, fever, and tachycardia.", 4),
    ]
    vs_dir = tmp_path / "vector_store"
    vs_dir.mkdir()
    build_index(docs, embedder, vs_dir)
    return Retriever(vs_dir, embedder)


class TestRetrieverHospitalScoping:
    def test_same_patient_id_across_hospitals_never_mixes(self, two_hospital_index):
        history_a, _ = two_hospital_index.retrieve(
            "sepsis fever", patient_id=1, hospital_id="hosp_a", top_k_self=5, top_k_similar=5
        )
        history_b, _ = two_hospital_index.retrieve(
            "sepsis fever", patient_id=1, hospital_id="hosp_b", top_k_self=5, top_k_similar=5
        )
        assert len(history_a) == 1
        assert history_a[0]["metadata"]["hospital_id"] == "hosp_a"
        assert len(history_b) == 1
        assert history_b[0]["metadata"]["hospital_id"] == "hosp_b"

    def test_similar_cases_never_cross_hospitals(self, two_hospital_index):
        _, similar_a = two_hospital_index.retrieve(
            "sepsis fever", patient_id=999, hospital_id="hosp_a", top_k_self=5, top_k_similar=10
        )
        hospital_ids_seen = {d["metadata"].get("hospital_id") for d in similar_a}
        assert hospital_ids_seen == {"hosp_a"}

    def test_legacy_docs_without_hospital_id_are_default_only(self, two_hospital_index):
        _, similar_default = two_hospital_index.retrieve(
            "sepsis fever", patient_id=999, hospital_id="default", top_k_self=5, top_k_similar=10
        )
        patient_ids_seen = {d["metadata"]["patient_id"] for d in similar_default}
        assert patient_ids_seen == {4}  # only the legacy doc, never hosp_a/hosp_b docs

        _, similar_a = two_hospital_index.retrieve(
            "sepsis fever", patient_id=999, hospital_id="hosp_a", top_k_self=5, top_k_similar=10
        )
        assert 4 not in {d["metadata"]["patient_id"] for d in similar_a}

    def test_hospital_id_none_is_unscoped_legacy_behavior(self, two_hospital_index):
        history, similar = two_hospital_index.retrieve(
            "sepsis fever", patient_id=1, hospital_id=None, top_k_self=5, top_k_similar=10
        )
        # Unscoped: both hospitals' patient-1 docs count as "same patient".
        assert len(history) == 2
        # Unscoped similar-case pool spans every hospital plus the legacy doc.
        all_patient_ids = {d["metadata"]["patient_id"] for d in similar}
        assert {2, 3, 4}.issubset(all_patient_ids)


class TestPipelineHospitalIdThreading:
    def test_rag_pipeline_threads_hospital_id_to_retriever(self, monkeypatch):
        from triageguard_rag.src.pipeline import rag_pipeline as rp

        pipeline = object.__new__(rp.RAGPipeline)
        pipeline.top_k_self = 3
        pipeline.top_k_sim = 5
        pipeline.api_key = "fake"
        pipeline.model = "fake-model"
        pipeline.temperature = 0.1
        pipeline.max_tokens = 100

        captured = {}

        class _FakeRetriever:
            def retrieve(self, **kwargs):
                captured.update(kwargs)
                return [], []

        pipeline.retriever = _FakeRetriever()
        monkeypatch.setattr(rp, "reason", lambda **kw: {"prompt": "", "response": "", "structured_output": {}})

        pipeline.run({"patient_id": 1, "chiefcomplaint": "fever"}, hospital_id="hosp_a")
        assert captured["hospital_id"] == "hosp_a"

    def test_rag_pipeline_falls_back_to_patient_state_hospital_id(self, monkeypatch):
        from triageguard_rag.src.pipeline import rag_pipeline as rp

        pipeline = object.__new__(rp.RAGPipeline)
        pipeline.top_k_self = 3
        pipeline.top_k_sim = 5
        pipeline.api_key = "fake"
        pipeline.model = "fake-model"
        pipeline.temperature = 0.1
        pipeline.max_tokens = 100

        captured = {}

        class _FakeRetriever:
            def retrieve(self, **kwargs):
                captured.update(kwargs)
                return [], []

        pipeline.retriever = _FakeRetriever()
        monkeypatch.setattr(rp, "reason", lambda **kw: {"prompt": "", "response": "", "structured_output": {}})

        pipeline.run({"patient_id": 1, "chiefcomplaint": "fever", "hospital_id": "hosp_b"})
        assert captured["hospital_id"] == "hosp_b"

    def test_combined_pipeline_threads_hospital_id_to_rag_only(self):
        from triageguard_router import combined_pipeline as cp

        pipeline = object.__new__(cp.TriageGuardPipeline)

        rag_calls = []

        class _FakeXgb:
            def predict(self, patient):
                return {"admission_risk": 0.5, "information_completeness": 1.0}

        class _FakeRag:
            def run(self, patient, hospital_id=None):
                rag_calls.append(hospital_id)
                return {"response": "", "structured_output": {}}

        pipeline.xgb = _FakeXgb()
        pipeline.rag = _FakeRag()

        pipeline.run({"patient_id": 1, "age": 40}, hospital_id="hosp_a")
        assert rag_calls == ["hosp_a"]

    def test_combined_pipeline_falls_back_to_patient_dict_hospital_id(self):
        from triageguard_router import combined_pipeline as cp

        pipeline = object.__new__(cp.TriageGuardPipeline)
        rag_calls = []

        class _FakeXgb:
            def predict(self, patient):
                return {"admission_risk": 0.5, "information_completeness": 1.0}

        class _FakeRag:
            def run(self, patient, hospital_id=None):
                rag_calls.append(hospital_id)
                return {"response": "", "structured_output": {}}

        pipeline.xgb = _FakeXgb()
        pipeline.rag = _FakeRag()

        pipeline.run({"patient_id": 1, "age": 40, "hospital_id": "hosp_b"})
        assert rag_calls == ["hosp_b"]


_MINIMAL_CONFIG = {
    "departments": {
        "ICU": {"capacity": 4, "occupied": 1, "available": 3, "status": "OPEN"},
        "DISCHARGE": {"capacity": 999, "occupied": 0, "available": 999, "status": "OPEN"},
    },
    "lambda_thresholds": {"normal_max": 0.70, "high_load_max": 0.90},
    "stale_threshold_minutes": 30,
}


@pytest.fixture()
def isolated_registry(tmp_path, monkeypatch):
    """
    A HospitalRegistry scoped to tmp_path, substituted in place of the
    process-wide default registry for the duration of the test — never
    writes into the real triageguard_agent/data/hospitals/ directory.
    """
    from triageguard_agent.hospital import hospital_registry as hr
    from triageguard_agent.hospital.hospital_state_service import HospitalStateService

    HospitalStateService.reset_instance()
    test_registry = hr.HospitalRegistry(manifest_path=tmp_path / "registry.json")
    monkeypatch.setattr(hr, "get_default_registry", lambda: test_registry)
    yield test_registry
    HospitalStateService.reset_instance()


class TestHospitalToolsRouting:
    def test_get_state_defaults_to_default_hospital(self, isolated_registry):
        from triageguard_agent.tools.hospital_tools import get_hospital_state

        result = get_hospital_state(department="ICU")
        assert result.success
        assert result.metadata["hospital_id"] is None  # caller omitted it -> default used

    def test_propose_and_commit_scoped_to_named_hospital(self, isolated_registry):
        from triageguard_agent.tools.hospital_tools import (
            propose_hospital_calibration,
            commit_hospital_calibration,
            get_hospital_state,
        )

        isolated_registry.register("hosp_a", "Hospital A", config_dict=_MINIMAL_CONFIG)
        isolated_registry.register("hosp_b", "Hospital B", config_dict=_MINIMAL_CONFIG)

        proposal = propose_hospital_calibration("ICU", {"occupied": 3}, hospital_id="hosp_a")
        assert proposal.success
        commit = commit_hospital_calibration(
            "ICU", proposal.data["proposed_update"], hospital_id="hosp_a"
        )
        assert commit.success

        state_a = get_hospital_state(department="ICU", hospital_id="hosp_a")
        state_b = get_hospital_state(department="ICU", hospital_id="hosp_b")
        assert state_a.data["state"]["occupied"] == 3
        assert state_b.data["state"]["occupied"] == 1  # untouched

    def test_unknown_hospital_id_fails_cleanly(self, isolated_registry):
        from triageguard_agent.tools.hospital_tools import get_hospital_state

        result = get_hospital_state(department="ICU", hospital_id="does_not_exist")
        assert not result.success
        assert result.error["code"] == "HOSPITAL_NOT_FOUND"
