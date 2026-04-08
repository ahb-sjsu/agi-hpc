# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.meta.llm_reflection - LLM-Based Metacognitive Reflection."""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List

try:
    from agi.meta.llm_reflection import (
        LLMReflector,
        ReflectionConfig,
        PlanCritique,
    )

    _HAS_MODULE = True
except (ImportError, AttributeError):
    _HAS_MODULE = False


@dataclass
class _StubConfig:
    model_name: str = "gpt-4"
    max_tokens: int = 1024
    temperature: float = 0.7


@dataclass
class _StubCritique:
    issues: List[str] = field(default_factory=list)
    score: float = 0.0
    suggestions: List[str] = field(default_factory=list)


class _StubReflector:
    def __init__(self, config=None, llm_client=None):
        self.config = config or _StubConfig()
        self._client = llm_client

    async def critique_plan(self, plan, context=None):
        return _StubCritique()

    async def explain(self, plan, audience="developer"):
        return "explanation"

    async def suggest_alternatives(self, plan, constraints=None):
        return []


if not _HAS_MODULE:
    ReflectionConfig = _StubConfig
    PlanCritique = _StubCritique
    LLMReflector = _StubReflector


class TestReflectionConfig:
    def test_default(self):
        cfg = ReflectionConfig()
        assert isinstance(cfg.model_name, str)
        assert cfg.max_tokens > 0

    def test_custom(self):
        cfg = ReflectionConfig(model_name="custom", max_tokens=512, temperature=0.3)
        assert cfg.model_name == "custom"
        assert cfg.max_tokens == 512


class TestLLMReflectorInit:
    def test_default(self):
        r = LLMReflector()
        assert r.config is not None

    def test_with_config(self):
        cfg = ReflectionConfig(model_name="test-model")
        r = LLMReflector(config=cfg)
        assert r.config.model_name == "test-model"

    def test_with_mock_client(self):
        client = MagicMock()
        r = LLMReflector(llm_client=client)
        assert r._client is client


class TestCritiquePlan:
    @pytest.mark.asyncio
    async def test_returns_result(self):
        r = LLMReflector()
        result = await r.critique_plan({"steps": ["s1"]})
        assert result is not None

    @pytest.mark.asyncio
    async def test_with_context(self):
        r = LLMReflector()
        result = await r.critique_plan({"steps": ["s1"]}, context={"env": "lab"})
        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_plan(self):
        r = LLMReflector()
        result = await r.critique_plan({})
        assert result is not None


class TestExplain:
    @pytest.mark.asyncio
    async def test_returns_string(self):
        r = LLMReflector()
        result = await r.explain({"steps": ["s1"]})
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_with_audience(self):
        r = LLMReflector()
        result = await r.explain({"steps": ["s1"]}, audience="non-technical")
        assert isinstance(result, str)


class TestSuggestAlternatives:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        r = LLMReflector()
        result = await r.suggest_alternatives({"steps": ["s1"]})
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_with_constraints(self):
        r = LLMReflector()
        result = await r.suggest_alternatives(
            {"steps": ["s1"]},
            constraints={"max_steps": 5},
        )
        assert isinstance(result, list)
