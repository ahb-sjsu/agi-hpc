# AGI-HPC Research Loop Tests
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from agi.metacognition.research_loop import (
    ResearchGoal,
    ResearchLoop,
    ResearchLoopConfig,
)


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.query_entity.return_value = None
    graph.query_relationships.return_value = []
    graph.store.return_value = "doc-id"
    return graph


@pytest.fixture
def mock_semantic():
    semantic = MagicMock()
    semantic.search.return_value = []
    return semantic


@pytest.fixture
def mock_extractor():
    from agi.memory.knowledge.extractor import ExtractedKnowledge

    extractor = MagicMock()
    extractor.extract_from_text.return_value = ExtractedKnowledge()
    return extractor


@pytest.fixture
def loop(mock_graph, mock_semantic, mock_extractor):
    config = ResearchLoopConfig(
        max_cycles=2,
        min_confidence=0.5,
        cooldown_seconds=0,
        enabled=True,
    )
    return ResearchLoop(
        config=config,
        knowledge_graph=mock_graph,
        semantic_memory=mock_semantic,
        extractor=mock_extractor,
    )


class TestResearch:
    @pytest.mark.asyncio
    async def test_basic_cycle(self, loop):
        goal = ResearchGoal(query="What is PCA?", source="user")
        result = await loop.research(goal)
        assert result.goal == goal
        assert result.cycles_used >= 1
        assert result.elapsed_s >= 0

    @pytest.mark.asyncio
    async def test_stops_on_confidence(self, loop, mock_semantic):
        # Return enough findings to reach confidence threshold
        mock_result = MagicMock()
        mock_result.text = (
            "PCA is Principal Component Analysis for dimensionality reduction"
        )
        mock_semantic.search.return_value = [mock_result] * 5
        goal = ResearchGoal(query="PCA dimensionality reduction", source="user")
        result = await loop.research(goal)
        assert result.confidence >= 0.5
        assert result.cycles_used == 1  # stopped early

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        config = ResearchLoopConfig(enabled=False)
        loop = ResearchLoop(config=config)
        goal = ResearchGoal(query="test", source="user")
        result = await loop.research(goal)
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_telemetry_updated(self, loop):
        goal = ResearchGoal(query="test query", source="user")
        await loop.research(goal)
        assert loop.telemetry.goals_completed == 1
        assert loop.telemetry.cycles_total >= 1


class TestDetectGaps:
    def test_executive_inhibition(self, loop):
        @dataclass
        class MockDecision:
            inhibit: bool = True
            inhibit_reason: str = "Need info about quantum computing"

        goals = loop.detect_gaps(executive_decisions=[MockDecision()])
        assert len(goals) == 1
        assert goals[0].query == "Need info about quantum computing"
        assert goals[0].source == "executive_function"
        assert goals[0].priority == 4

    def test_no_duplicate_gaps(self, loop):
        @dataclass
        class MockDecision:
            inhibit: bool = True
            inhibit_reason: str = "same gap"

        goals1 = loop.detect_gaps(executive_decisions=[MockDecision()])
        goals2 = loop.detect_gaps(executive_decisions=[MockDecision()])
        assert len(goals1) == 1
        assert len(goals2) == 0  # already in history

    def test_failed_lookups(self, loop):
        goals = loop.detect_gaps(failed_lookups=["how to deploy kubernetes"])
        assert len(goals) == 1
        assert goals[0].source == "procedural_memory"
        assert goals[0].priority == 2

    def test_low_hit_rate(self, loop):
        goals = loop.detect_gaps(memory_hit_rate=0.1)
        assert len(goals) == 1
        assert goals[0].source == "anomaly_detector"

    def test_normal_hit_rate_no_gap(self, loop):
        goals = loop.detect_gaps(memory_hit_rate=0.8)
        assert len(goals) == 0

    def test_no_inhibition_no_gap(self, loop):
        @dataclass
        class MockDecision:
            inhibit: bool = False
            inhibit_reason: str = ""

        goals = loop.detect_gaps(executive_decisions=[MockDecision()])
        assert len(goals) == 0

    def test_telemetry_counts_detected(self, loop):
        loop.detect_gaps(failed_lookups=["a", "b", "c"])
        assert loop.telemetry.goals_detected == 3


class TestIdleCycle:
    @pytest.mark.asyncio
    async def test_runs_when_gaps_exist(self, loop):
        @dataclass
        class MockDecision:
            inhibit: bool = True
            inhibit_reason: str = "gap"

        results = await loop.run_idle_cycle(executive_decisions=[MockDecision()])
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_respects_cooldown(self, loop):
        loop._config.cooldown_seconds = 9999
        loop._last_cycle_time = __import__("time").time()
        results = await loop.run_idle_cycle(failed_lookups=["test"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        config = ResearchLoopConfig(enabled=False)
        loop = ResearchLoop(config=config)
        results = await loop.run_idle_cycle(failed_lookups=["test"])
        assert len(results) == 0
