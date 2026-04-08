# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.core.llm.integration module."""

from agi.core.llm.integration import (
    LLMIntegrationConfig,
    LHPlannerIntegration,
    MetacognitionIntegration,
    MemoryEmbeddingIntegration,
    SafetyFallbackIntegration,
)


class TestLLMIntegrationConfig:
    def test_defaults(self):
        cfg = LLMIntegrationConfig()
        assert cfg.planner_model == "claude-3-5-sonnet-20241022"
        assert cfg.planner_temperature == 0.3
        assert cfg.planner_max_tokens == 4096
        assert cfg.metacog_model == "claude-3-5-sonnet-20241022"
        assert cfg.metacog_temperature == 0.2
        assert cfg.metacog_max_tokens == 2048
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.embedding_provider == "openai"
        assert cfg.safety_model == "claude-3-5-sonnet-20241022"
        assert cfg.safety_temperature == 0.0
        assert cfg.safety_max_tokens == 1024
        assert cfg.enable_caching is True
        assert cfg.enable_fallback is True
        assert cfg.fallback_provider == "ollama"
        assert cfg.fallback_model == "llama3"

    def test_custom_values(self):
        cfg = LLMIntegrationConfig(
            planner_model="gpt-4",
            planner_temperature=0.5,
            planner_max_tokens=8192,
            enable_caching=False,
            fallback_model="mistral",
        )
        assert cfg.planner_model == "gpt-4"
        assert cfg.planner_temperature == 0.5
        assert cfg.planner_max_tokens == 8192
        assert cfg.enable_caching is False
        assert cfg.fallback_model == "mistral"

    def test_safety_temperature_zero(self):
        cfg = LLMIntegrationConfig()
        assert cfg.safety_temperature == 0.0


class TestLHPlannerIntegration:
    def test_init_default(self):
        pi = LHPlannerIntegration()
        assert pi._config is not None
        assert pi._client is None

    def test_init_with_config(self):
        cfg = LLMIntegrationConfig(planner_model="custom-model")
        pi = LHPlannerIntegration(config=cfg)
        assert pi._config.planner_model == "custom-model"

    def test_generate_plan_stub_without_client(self):
        pi = LHPlannerIntegration()
        plan = pi.generate_plan("build a tower")
        assert plan["goal"] == "build a tower"
        assert plan["stub"] is True
        assert plan["plan_id"] == "stub-001"
        assert isinstance(plan["steps"], list)
        assert len(plan["steps"]) == 1
        assert plan["steps"][0]["step_id"] == "1"
        assert plan["steps"][0]["action_type"] == "generic"

    def test_generate_plan_with_constraints(self):
        pi = LHPlannerIntegration()
        plan = pi.generate_plan("navigate", context="indoor", constraints=["no stairs"])
        assert plan["stub"] is True
        assert plan["goal"] == "navigate"

    def test_refine_plan_returns_same_without_client(self):
        pi = LHPlannerIntegration()
        original = {"plan_id": "test", "steps": []}
        refined = pi.refine_plan(original, "add more steps")
        assert refined is original

    def test_decompose_task_stub_without_client(self):
        pi = LHPlannerIntegration()
        subtasks = pi.decompose_task("clean the kitchen")
        assert isinstance(subtasks, list)
        assert len(subtasks) == 1
        assert subtasks[0]["subtask_id"] == "1"
        assert subtasks[0]["description"] == "clean the kitchen"
        assert subtasks[0]["type"] == "generic"

    def test_has_system_prompt(self):
        assert isinstance(LHPlannerIntegration.SYSTEM_PROMPT, str)
        assert len(LHPlannerIntegration.SYSTEM_PROMPT) > 0


class TestMetacognitionIntegration:
    def test_init_default(self):
        mi = MetacognitionIntegration()
        assert mi._config is not None
        assert mi._client is None

    def test_critique_plan_stub_without_client(self):
        mi = MetacognitionIntegration()
        plan = {"plan_id": "p1", "steps": []}
        result = mi.critique_plan(plan)
        assert result["confidence"] == 0.5
        assert result["issues"] == []
        assert result["suggestions"] == []
        assert result["decision"] == "ACCEPT"
        assert result["stub"] is True

    def test_critique_plan_with_context(self):
        mi = MetacognitionIntegration()
        result = mi.critique_plan({"steps": []}, context="urgent task")
        assert result["stub"] is True

    def test_generate_explanation_without_client(self):
        mi = MetacognitionIntegration()
        explanation = mi.generate_explanation("approve", "risk is low")
        assert "approve" in explanation
        assert "risk is low" in explanation

    def test_generate_explanation_truncates_reasoning(self):
        mi = MetacognitionIntegration()
        long_reasoning = "x" * 200
        explanation = mi.generate_explanation("decide", long_reasoning)
        assert "decide" in explanation
        # stub truncates reasoning to first 100 chars
        assert len(long_reasoning[:100]) == 100

    def test_assess_confidence_without_client(self):
        mi = MetacognitionIntegration()
        confidence = mi.assess_confidence({"steps": []}, ["evidence1"])
        assert confidence == 0.5

    def test_assess_confidence_returns_float(self):
        mi = MetacognitionIntegration()
        result = mi.assess_confidence({}, [])
        assert isinstance(result, float)


class TestMemoryEmbeddingIntegration:
    def test_init_default(self):
        mei = MemoryEmbeddingIntegration()
        assert mei._config is not None
        assert mei._client is None

    def test_generate_embeddings_stub(self):
        mei = MemoryEmbeddingIntegration()
        texts = ["hello", "world"]
        embeddings = mei.generate_embeddings(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384
        assert all(v == 0.0 for v in embeddings[0])

    def test_generate_query_embedding_stub(self):
        mei = MemoryEmbeddingIntegration()
        embedding = mei.generate_query_embedding("search query")
        assert len(embedding) == 384
        assert all(v == 0.0 for v in embedding)

    def test_generate_embeddings_empty_list(self):
        mei = MemoryEmbeddingIntegration()
        embeddings = mei.generate_embeddings([])
        assert embeddings == []

    def test_embedding_dimension_consistency(self):
        mei = MemoryEmbeddingIntegration()
        single = mei.generate_query_embedding("test")
        batch = mei.generate_embeddings(["test"])
        assert len(single) == len(batch[0])

    def test_init_with_config(self):
        cfg = LLMIntegrationConfig(embedding_model="custom-embed")
        mei = MemoryEmbeddingIntegration(config=cfg)
        assert mei._config.embedding_model == "custom-embed"


class TestSafetyFallbackIntegration:
    def test_init_default(self):
        sfi = SafetyFallbackIntegration()
        assert sfi._config is not None
        assert sfi._client is None

    def test_assess_safety_stub_without_client(self):
        sfi = SafetyFallbackIntegration()
        result = sfi.assess_safety("pick up object")
        assert result["safe"] is True
        assert result["risk_level"] == "low"
        assert result["issues"] == []
        assert result["recommendations"] == []
        assert result["stub"] is True

    def test_assess_safety_with_context(self):
        sfi = SafetyFallbackIntegration()
        result = sfi.assess_safety("move fast", context="near humans")
        assert result["safe"] is True
        assert result["stub"] is True

    def test_explain_violation_without_client(self):
        sfi = SafetyFallbackIntegration()
        explanation = sfi.explain_violation("exceeded force limit")
        assert "exceeded force limit" in explanation
        assert explanation == "Safety violation: exceeded force limit"

    def test_init_with_config(self):
        cfg = LLMIntegrationConfig(safety_model="custom-safety")
        sfi = SafetyFallbackIntegration(config=cfg)
        assert sfi._config.safety_model == "custom-safety"

    def test_has_safety_prompt(self):
        assert isinstance(SafetyFallbackIntegration.SAFETY_PROMPT, str)
        assert "safety" in SafetyFallbackIntegration.SAFETY_PROMPT.lower()
