"""
Tests for theriad.workflow.detector.

MockEmbeddingBackend is used throughout to avoid model downloads.
Because mock embeddings are hash-deterministic-random, we do not assert the
*direction* of embedding-layer decisions — only that the schema is correct
and that the rule guardrail decides correctly for clear-cut prompts.
"""
from __future__ import annotations

import pytest

from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
from theriad.workflow.detector import (
    MatryoshkaEmbeddingDetector,
    RuleGuardrailDetector,
    WorkflowDetector,
)
from theriad.workflow.types import WorkflowDetection


# ── RuleGuardrailDetector ─────────────────────────────────────────────────────

class TestRuleGuardrailDetector:

    def setup_method(self):
        self.det = RuleGuardrailDetector()

    # Clear single-prompt cases
    def test_summarize_is_single_prompt(self):
        r = self.det.detect("Summarize this paragraph in three sentences.")
        assert r.is_workflow is False
        assert r.decision_source == "rule_guardrail"
        assert r.confidence >= 0.75

    def test_translate_is_single_prompt(self):
        r = self.det.detect("Translate this sentence into Spanish.")
        assert r.is_workflow is False
        assert r.decision_source == "rule_guardrail"

    def test_rewrite_is_single_prompt(self):
        r = self.det.detect("Rewrite this email in a more professional tone.")
        assert r.is_workflow is False
        assert r.confidence >= 0.75

    def test_extract_is_single_prompt(self):
        r = self.det.detect("Extract all email addresses from this text.")
        assert r.is_workflow is False

    def test_explain_is_single_prompt(self):
        r = self.det.detect("Explain this concept briefly.")
        assert r.is_workflow is False

    def test_classify_is_single_prompt(self):
        r = self.det.detect("Classify this text into one of three categories.")
        assert r.is_workflow is False

    # "Explain how to build" — starts with single-verb; should stay v1
    def test_explain_how_to_build_is_single_prompt(self):
        r = self.det.detect("Explain how to build a React app.")
        assert r.is_workflow is False
        assert r.decision_source == "rule_guardrail"
        assert r.confidence >= 0.75

    # Clear workflow cases
    def test_design_build_test_deploy_is_workflow(self):
        r = self.det.detect("Design, build, test, and deploy a React app.")
        assert r.is_workflow is True
        assert r.decision_source == "rule_guardrail"
        assert r.confidence >= 0.85

    def test_refactor_add_fix_document_is_workflow(self):
        r = self.det.detect(
            "Refactor this backend, add tests, fix failing routes, and document the API."
        )
        assert r.is_workflow is True
        assert r.confidence >= 0.85

    def test_implement_run_debug_is_workflow(self):
        r = self.det.detect(
            "Implement the feature, run tests, and debug any failures."
        )
        assert r.is_workflow is True

    def test_build_with_strong_tech_cluster_is_workflow(self):
        r = self.det.detect(
            "Build a full app with frontend, backend, database, and auth."
        )
        assert r.is_workflow is True

    # Ambiguous case: passes to embedding
    def test_ambiguous_returns_low_confidence(self):
        r = self.det.detect("Help me with something.")
        assert r.confidence <= 0.70

    # Schema validation
    def test_returns_workflow_detection_dataclass(self):
        r = self.det.detect("Summarize this.")
        assert isinstance(r, WorkflowDetection)
        assert isinstance(r.signals, list)
        assert r.semantic_margin is None  # rule layer doesn't use embedding
        assert r.embedding_model is None
        assert r.embedding_dims_used is None


# ── MatryoshkaEmbeddingDetector ───────────────────────────────────────────────

class TestMatryoshkaEmbeddingDetector:

    def _make(self, dim: int = 384):
        backend = MockEmbeddingBackend(dim=dim)
        return MatryoshkaEmbeddingDetector(backend=backend)

    def test_returns_workflow_detection(self):
        det = self._make()
        r = det.detect("Help me get this repo ready to ship.")
        assert isinstance(r, WorkflowDetection)

    def test_decision_source_is_mrl_embedding(self):
        det = self._make()
        r = det.detect("Build a full app.")
        assert r.decision_source == "mrl_embedding"

    def test_semantic_margin_is_float(self):
        det = self._make()
        r = det.detect("Summarize this.")
        assert r.semantic_margin is not None
        assert isinstance(r.semantic_margin, float)

    def test_embedding_model_is_set(self):
        det = self._make()
        r = det.detect("Translate this sentence.")
        assert r.embedding_model is not None
        assert "mock" in r.embedding_model

    def test_embedding_dims_used_is_set(self):
        det = self._make(dim=64)
        r = det.detect("Build a full-stack app with auth and deployment.")
        assert r.embedding_dims_used is not None
        assert isinstance(r.embedding_dims_used, int)

    def test_small_dim_backend(self):
        # Should not raise even with dim < smallest MRL slice
        det = self._make(dim=32)
        r = det.detect("Create a new service with tests and CI.")
        assert isinstance(r, WorkflowDetection)

    def test_is_workflow_is_bool(self):
        det = self._make()
        r = det.detect("Design and deploy a microservice.")
        assert isinstance(r.is_workflow, bool)


# ── WorkflowDetector (orchestrator) ──────────────────────────────────────────

class TestWorkflowDetector:

    def _make(self, dim: int = 64):
        backend = MockEmbeddingBackend(dim=dim)
        return WorkflowDetector(embedding_backend=backend)

    def test_summarize_is_v1(self):
        det = self._make()
        r = det.detect("Summarize this paragraph.")
        assert r.is_workflow is False

    def test_explain_how_to_build_is_v1(self):
        det = self._make()
        r = det.detect("Explain how to build a React app.")
        assert r.is_workflow is False

    def test_build_deploy_react_is_workflow(self):
        det = self._make()
        r = det.detect("Design, build, test, and deploy a React app.")
        assert r.is_workflow is True

    def test_refactor_add_test_is_workflow(self):
        det = self._make()
        r = det.detect("Refactor this backend, add tests, fix routes, and document the API.")
        assert r.is_workflow is True

    def test_project_signal_escalates_ambiguous_to_workflow(self):
        # "repo" is a project signal → should escalate to workflow after ambiguous rule+embedding
        det = self._make()
        r = det.detect("Help me get this repo ready.")
        assert r.is_workflow is True

    def test_no_signals_defaults_to_v1(self):
        det = self._make()
        r = det.detect("Help me with something.")
        assert r.is_workflow is False

    def test_tool_signal_escalates_to_workflow(self):
        # "docker" and "pipeline" are tool/tech signals; no single-verb ambiguity
        det = self._make()
        r = det.detect("Get the docker pipeline ready.")
        assert r.is_workflow is True

    def test_result_is_workflow_detection(self):
        det = self._make()
        r = det.detect("Summarize this.")
        assert isinstance(r, WorkflowDetection)

    def test_decision_source_set(self):
        det = self._make()
        r = det.detect("Summarize this paragraph.")
        assert r.decision_source in ("rule_guardrail", "mrl_embedding", "default")

    @pytest.mark.parametrize("prompt, expect_workflow", [
        ("Summarize this article.", False),
        ("Translate this sentence.", False),
        ("Rewrite this paragraph.", False),
        ("Define the term 'entropy'.", False),
        ("Design, build, test, deploy a new API.", True),
        ("Refactor, add tests, fix bugs, document.", True),
    ])
    def test_parametrized_classification(self, prompt, expect_workflow):
        det = self._make()
        r = det.detect(prompt)
        assert r.is_workflow is expect_workflow, (
            f"Expected is_workflow={expect_workflow} for: {prompt!r}\n"
            f"Got: is_workflow={r.is_workflow}, source={r.decision_source}, "
            f"reason={r.reason!r}"
        )
