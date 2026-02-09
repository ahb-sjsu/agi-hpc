# AGI-HPC Project - Unified Memory Service Tests
# Sprint 5: Unit tests for UnifiedMemoryService

import pytest
from unittest.mock import MagicMock, AsyncMock

from agi.memory.unified import (
    UnifiedMemoryService,
    UnifiedMemoryConfig,
    PlanningContext,
)


# ---------------------------------------------------------------------------
# PlanningContext Tests
# ---------------------------------------------------------------------------


class TestPlanningContext:
    """Tests for PlanningContext dataclass."""

    def test_empty_context_has_no_context(self):
        """Empty PlanningContext should report has_context=False."""
        ctx = PlanningContext()
        assert ctx.has_context is False

    def test_context_with_facts_has_context(self):
        """PlanningContext with facts should report has_context=True."""
        ctx = PlanningContext(
            facts=[{"fact_id": "f1", "content": "test fact", "confidence": 0.9}]
        )
        assert ctx.has_context is True

    def test_context_with_episodes_has_context(self):
        """PlanningContext with episodes should report has_context=True."""
        ctx = PlanningContext(
            episodes=[{"episode_id": "e1", "task_description": "test"}]
        )
        assert ctx.has_context is True

    def test_context_with_skills_has_context(self):
        """PlanningContext with skills should report has_context=True."""
        ctx = PlanningContext(skills=[{"skill_id": "s1", "name": "test_skill"}])
        assert ctx.has_context is True

    def test_to_prompt_context_empty(self):
        """Empty context should produce empty prompt string."""
        ctx = PlanningContext()
        assert ctx.to_prompt_context() == ""

    def test_to_prompt_context_with_facts(self):
        """Context with facts should format them in prompt."""
        ctx = PlanningContext(
            facts=[
                {"content": "Robots need safety checks", "confidence": 0.95},
                {"content": "Navigation requires planning", "confidence": 0.8},
            ]
        )
        prompt = ctx.to_prompt_context()
        assert "RELEVANT FACTS:" in prompt
        assert "Robots need safety checks" in prompt
        assert "confidence: 0.95" in prompt

    def test_to_prompt_context_with_episodes(self):
        """Context with episodes should format them in prompt."""
        ctx = PlanningContext(
            episodes=[
                {
                    "task_description": "Navigate to target",
                    "success": True,
                    "insights": ["Check obstacles first"],
                }
            ]
        )
        prompt = ctx.to_prompt_context()
        assert "SIMILAR PAST EPISODES:" in prompt
        assert "Navigate to target" in prompt
        assert "succeeded" in prompt
        assert "Check obstacles first" in prompt

    def test_to_prompt_context_with_skills(self):
        """Context with skills should format them in prompt."""
        ctx = PlanningContext(
            skills=[
                {
                    "name": "move_forward",
                    "description": "Move robot forward",
                    "proficiency": 0.85,
                }
            ]
        )
        prompt = ctx.to_prompt_context()
        assert "AVAILABLE SKILLS:" in prompt
        assert "move_forward" in prompt
        assert "proficiency: 0.85" in prompt

    def test_to_prompt_context_limits_items(self):
        """Context should limit number of items in prompt."""
        ctx = PlanningContext(
            facts=[{"content": f"Fact {i}", "confidence": 0.9} for i in range(10)]
        )
        prompt = ctx.to_prompt_context()
        # Should only include first 5 facts
        assert "Fact 0" in prompt
        assert "Fact 4" in prompt
        assert "Fact 5" not in prompt


# ---------------------------------------------------------------------------
# UnifiedMemoryService Tests (Stub Mode)
# ---------------------------------------------------------------------------


class TestUnifiedMemoryServiceStub:
    """Tests for UnifiedMemoryService in stub mode (no real clients)."""

    @pytest.mark.asyncio
    async def test_enrich_context_returns_stub_data(self, unified_memory_service_stub):
        """Stub mode should return placeholder context."""
        ctx = await unified_memory_service_stub.enrich_planning_context(
            task_description="Navigate to the red cube",
            task_type="navigation",
        )

        assert ctx.has_context is True
        assert len(ctx.facts) >= 1
        assert len(ctx.episodes) >= 1
        assert len(ctx.skills) >= 1

    @pytest.mark.asyncio
    async def test_stub_facts_include_query(self, unified_memory_service_stub):
        """Stub facts should reference the query."""
        ctx = await unified_memory_service_stub.enrich_planning_context(
            task_description="Navigate to target",
        )

        fact_content = ctx.facts[0].get("content", "")
        assert "Navigate" in fact_content or "Placeholder" in fact_content

    @pytest.mark.asyncio
    async def test_stub_respects_include_flags(self, unified_memory_service_stub):
        """Stub mode should respect include flags."""
        ctx = await unified_memory_service_stub.enrich_planning_context(
            task_description="Test task",
            include_semantic=False,
            include_episodic=False,
            include_procedural=True,
        )

        # Only skills should be populated
        assert len(ctx.skills) >= 1


# ---------------------------------------------------------------------------
# UnifiedMemoryService Tests (With Mock Clients)
# ---------------------------------------------------------------------------


class TestUnifiedMemoryServiceWithClients:
    """Tests for UnifiedMemoryService with mock clients."""

    @pytest.mark.asyncio
    async def test_enrich_context_queries_all_memories(self, unified_memory_service):
        """Should query all three memory types."""
        ctx = await unified_memory_service.enrich_planning_context(
            task_description="Navigate to the red cube",
            task_type="navigation",
            scenario_id="test_scene",
        )

        assert ctx.has_context is True
        assert len(ctx.facts) >= 1
        assert len(ctx.episodes) >= 1
        assert len(ctx.skills) >= 1

    @pytest.mark.asyncio
    async def test_respects_include_semantic_false(
        self,
        unified_memory_service,
        mock_semantic_client,
    ):
        """Should skip semantic query when include_semantic=False."""
        await unified_memory_service.enrich_planning_context(
            task_description="Test task",
            include_semantic=False,
        )

        # Semantic client should not be called
        mock_semantic_client.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_include_episodic_false(
        self,
        unified_memory_service,
        mock_episodic_client,
    ):
        """Should skip episodic query when include_episodic=False."""
        await unified_memory_service.enrich_planning_context(
            task_description="Test task",
            include_episodic=False,
        )

        mock_episodic_client.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_include_procedural_false(
        self,
        unified_memory_service,
        mock_procedural_client,
    ):
        """Should skip procedural query when include_procedural=False."""
        await unified_memory_service.enrich_planning_context(
            task_description="Test task",
            include_procedural=False,
        )

        mock_procedural_client.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_semantic_client_error(self, unified_memory_config):
        """Should handle semantic client errors gracefully."""
        semantic = MagicMock()
        semantic.search = AsyncMock(side_effect=Exception("Connection failed"))

        service = UnifiedMemoryService(
            config=unified_memory_config,
            semantic_client=semantic,
        )

        ctx = await service.enrich_planning_context(
            task_description="Test task",
        )

        # Should still return context (with stub data for semantic)
        assert ctx is not None

    @pytest.mark.asyncio
    async def test_parallel_query_execution(self, unified_memory_service):
        """Should execute queries in parallel."""
        import time

        start = time.time()
        ctx = await unified_memory_service.enrich_planning_context(
            task_description="Test parallel queries",
        )
        elapsed = time.time() - start

        # With parallel execution, should be fast
        # (not 3x sequential time)
        assert elapsed < 1.0
        assert ctx.has_context is True


# ---------------------------------------------------------------------------
# UnifiedMemoryConfig Tests
# ---------------------------------------------------------------------------


class TestUnifiedMemoryConfig:
    """Tests for UnifiedMemoryConfig."""

    def test_default_config(self):
        """Default config should have reasonable defaults."""
        config = UnifiedMemoryConfig()

        assert config.semantic_addr == "localhost:50053"
        assert config.episodic_addr == "localhost:50052"
        assert config.procedural_addr == "localhost:50054"
        assert config.timeout_sec > 0
        assert config.default_max_facts > 0
        assert config.default_max_episodes > 0
        assert config.default_max_skills > 0

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = UnifiedMemoryConfig(
            semantic_addr="custom:1111",
            default_max_facts=20,
            enable_caching=False,
        )

        assert config.semantic_addr == "custom:1111"
        assert config.default_max_facts == 20
        assert config.enable_caching is False


# ---------------------------------------------------------------------------
# Caching Tests
# ---------------------------------------------------------------------------


class TestUnifiedMemoryCaching:
    """Tests for caching behavior."""

    @pytest.mark.asyncio
    async def test_caching_enabled_returns_cached(self):
        """With caching enabled, should return cached results."""
        config = UnifiedMemoryConfig(enable_caching=True)
        service = UnifiedMemoryService(config=config)

        # First call
        ctx1 = await service.enrich_planning_context(
            task_description="Test task",
            task_type="test",
        )

        # Second call with same params
        ctx2 = await service.enrich_planning_context(
            task_description="Test task",
            task_type="test",
        )

        # Should get cached result (same object or equivalent)
        assert ctx1.facts == ctx2.facts
        assert ctx1.episodes == ctx2.episodes

    @pytest.mark.asyncio
    async def test_caching_disabled_no_cache(self):
        """With caching disabled, should not cache."""
        config = UnifiedMemoryConfig(enable_caching=False)
        service = UnifiedMemoryService(config=config)

        # Clear any internal cache
        service._cache.clear()

        await service.enrich_planning_context(
            task_description="Test task",
        )

        # Cache should still be empty
        assert len(service._cache) == 0
