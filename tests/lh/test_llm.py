# AGI-HPC Project - Tests for LLM Module
# Copyright (c) 2025 Andrew H. Bond

"""
Unit tests for the LLM adapters and LLM planner.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from agi.lh.llm import (
    LLMAdapter,
    LLMConfig,
    LLMResponse,
    BaseLLMAdapter,
    create_adapter,
    OllamaAdapter,
    AnthropicAdapter,
    OpenAIAdapter,
    LLMPlanner,
    LLMPlannerConfig,
)
from agi.lh.planner import PlanGraph, PlanStep
from agi.proto_gen import plan_pb2


# ---------------------------------------------------------------------------
# LLMConfig Tests
# ---------------------------------------------------------------------------


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()
            assert config.provider == "ollama"
            assert config.model == "llama3.2"
            assert config.api_key is None
            assert config.base_url == "http://localhost:11434"
            assert config.temperature == 0.7
            assert config.max_tokens == 2048
            assert config.timeout == 60.0

    def test_env_override(self):
        """Test configuration from environment variables."""
        env = {
            "AGI_LH_LLM_PROVIDER": "anthropic",
            "AGI_LH_LLM_MODEL": "claude-3-5-sonnet-20241022",
            "AGI_LH_LLM_API_KEY": "test-key-123",
            "AGI_LH_LLM_TEMPERATURE": "0.5",
            "AGI_LH_LLM_MAX_TOKENS": "4096",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig()
            assert config.provider == "anthropic"
            assert config.model == "claude-3-5-sonnet-20241022"
            assert config.api_key == "test-key-123"
            assert config.temperature == 0.5
            assert config.max_tokens == 4096


# ---------------------------------------------------------------------------
# LLMResponse Tests
# ---------------------------------------------------------------------------


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_basic_response(self):
        """Test basic response creation."""
        response = LLMResponse(
            content="Hello, world!",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert response.content == "Hello, world!"
        assert response.model == "test-model"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.total_tokens == 15

    def test_empty_usage(self):
        """Test response with empty usage stats."""
        response = LLMResponse(content="test")
        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0
        assert response.total_tokens == 0


# ---------------------------------------------------------------------------
# Adapter Factory Tests
# ---------------------------------------------------------------------------


class TestCreateAdapter:
    """Tests for create_adapter factory function."""

    def test_create_ollama_adapter(self):
        """Test creating Ollama adapter."""
        config = LLMConfig(provider="ollama")
        adapter = create_adapter(config)
        assert isinstance(adapter, OllamaAdapter)

    def test_create_anthropic_adapter(self):
        """Test creating Anthropic adapter."""
        config = LLMConfig(provider="anthropic", api_key="test-key")
        adapter = create_adapter(config)
        assert isinstance(adapter, AnthropicAdapter)

    def test_create_openai_adapter(self):
        """Test creating OpenAI adapter."""
        config = LLMConfig(provider="openai", api_key="test-key")
        adapter = create_adapter(config)
        assert isinstance(adapter, OpenAIAdapter)

    def test_unsupported_provider(self):
        """Test error on unsupported provider."""
        config = LLMConfig(provider="unknown")
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_adapter(config)


# ---------------------------------------------------------------------------
# OllamaAdapter Tests
# ---------------------------------------------------------------------------


class TestOllamaAdapter:
    """Tests for OllamaAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        config = LLMConfig(provider="ollama", model="llama3.2")
        adapter = OllamaAdapter(config)
        assert adapter.model_name == "llama3.2"

    def test_default_base_url(self):
        """Test default base URL."""
        adapter = OllamaAdapter()
        assert adapter._base_url == "http://localhost:11434"

    def test_custom_base_url(self):
        """Test custom base URL."""
        config = LLMConfig(base_url="http://custom:11434")
        adapter = OllamaAdapter(config)
        assert adapter._base_url == "http://custom:11434"

    @patch("urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "response": "Generated text",
                "model": "llama3.2",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        adapter = OllamaAdapter()
        result = adapter.generate("Test prompt")

        assert result.content == "Generated text"
        assert result.model == "llama3.2"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

    @patch("urllib.request.urlopen")
    def test_is_available_success(self, mock_urlopen):
        """Test availability check success."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        adapter = OllamaAdapter()
        assert adapter.is_available() is True

    @patch("urllib.request.urlopen")
    def test_is_available_failure(self, mock_urlopen):
        """Test availability check failure."""
        mock_urlopen.side_effect = Exception("Connection refused")

        adapter = OllamaAdapter()
        assert adapter.is_available() is False


# ---------------------------------------------------------------------------
# AnthropicAdapter Tests
# ---------------------------------------------------------------------------


class TestAnthropicAdapter:
    """Tests for AnthropicAdapter."""

    def test_initialization_with_key(self):
        """Test adapter initialization with API key."""
        config = LLMConfig(provider="anthropic", api_key="test-key")
        adapter = AnthropicAdapter(config)
        assert adapter.model_name == "claude-3-5-sonnet-20241022"

    def test_initialization_without_key(self):
        """Test adapter initialization without API key (warning only)."""
        config = LLMConfig(provider="anthropic", api_key=None)
        adapter = AnthropicAdapter(config)
        assert adapter.is_available() is False

    def test_generate_without_key(self):
        """Test generate raises error without API key."""
        config = LLMConfig(provider="anthropic", api_key=None)
        adapter = AnthropicAdapter(config)
        with pytest.raises(ValueError, match="API key not configured"):
            adapter.generate("Test prompt")

    @patch("urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "content": [{"type": "text", "text": "Claude response"}],
                "model": "claude-3-5-sonnet-20241022",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = LLMConfig(provider="anthropic", api_key="test-key")
        adapter = AnthropicAdapter(config)
        result = adapter.generate("Test prompt")

        assert result.content == "Claude response"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20


# ---------------------------------------------------------------------------
# OpenAIAdapter Tests
# ---------------------------------------------------------------------------


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter."""

    def test_initialization_with_key(self):
        """Test adapter initialization with API key."""
        config = LLMConfig(provider="openai", api_key="test-key")
        adapter = OpenAIAdapter(config)
        assert adapter.model_name == "gpt-4o"

    def test_generate_without_key(self):
        """Test generate raises error without API key."""
        config = LLMConfig(provider="openai", api_key=None)
        adapter = OpenAIAdapter(config)
        with pytest.raises(ValueError, match="API key not configured"):
            adapter.generate("Test prompt")

    @patch("urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "GPT response"}}],
                "model": "gpt-4o",
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40,
                },
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = LLMConfig(provider="openai", api_key="test-key")
        adapter = OpenAIAdapter(config)
        result = adapter.generate("Test prompt")

        assert result.content == "GPT response"
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 25
        assert result.total_tokens == 40


# ---------------------------------------------------------------------------
# LLMPlannerConfig Tests
# ---------------------------------------------------------------------------


class TestLLMPlannerConfig:
    """Tests for LLMPlannerConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = LLMPlannerConfig()
        assert config.use_llm is True
        assert config.fallback_on_error is True
        assert config.max_retries == 2
        assert config.temperature == 0.3
        assert len(config.available_tools) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMPlannerConfig(
            use_llm=False,
            fallback_on_error=False,
            max_retries=5,
        )
        assert config.use_llm is False
        assert config.fallback_on_error is False
        assert config.max_retries == 5


# ---------------------------------------------------------------------------
# LLMPlanner Tests
# ---------------------------------------------------------------------------


class TestLLMPlanner:
    """Tests for LLMPlanner."""

    def _create_plan_request(
        self, description: str = "Test goal", goal_id: str = "test-001"
    ) -> plan_pb2.PlanRequest:
        """Create a test PlanRequest."""
        request = plan_pb2.PlanRequest()
        request.task.description = description
        request.task.goal_id = goal_id
        request.task.task_type = "test"
        request.environment.scenario_id = "test-scenario"
        return request

    def test_fallback_when_llm_disabled(self):
        """Test that planner falls back when LLM is disabled."""
        config = LLMPlannerConfig(use_llm=False)
        planner = LLMPlanner(config=config)

        request = self._create_plan_request()
        graph = planner.generate_plan(request)

        assert isinstance(graph, PlanGraph)
        assert len(graph.steps) > 0
        # Should use deterministic planner
        assert graph.metadata.get("source") != "llm"

    def test_fallback_when_llm_unavailable(self):
        """Test fallback when LLM adapter not available."""
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = False

        planner = LLMPlanner(llm_adapter=mock_adapter)

        request = self._create_plan_request()
        graph = planner.generate_plan(request)

        assert isinstance(graph, PlanGraph)
        assert len(graph.steps) > 0

    def test_llm_plan_generation(self):
        """Test successful LLM plan generation."""
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.model_name = "test-model"
        mock_adapter.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "plan_id": "plan_test",
                    "steps": [
                        {
                            "step_id": "mission_001",
                            "level": 0,
                            "kind": "mission",
                            "description": "Test mission",
                            "parent_id": None,
                            "requires_simulation": False,
                            "safety_tags": ["high_level"],
                        },
                        {
                            "step_id": "subgoal_001",
                            "level": 1,
                            "kind": "subgoal",
                            "description": "Test subgoal",
                            "parent_id": "mission_001",
                            "requires_simulation": False,
                            "safety_tags": ["planning"],
                        },
                        {
                            "step_id": "action_001",
                            "level": 2,
                            "kind": "action",
                            "description": "Test action",
                            "parent_id": "subgoal_001",
                            "requires_simulation": True,
                            "safety_tags": [],
                        },
                    ],
                }
            ),
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        )

        planner = LLMPlanner(llm_adapter=mock_adapter)
        request = self._create_plan_request()
        graph = planner.generate_plan(request)

        assert isinstance(graph, PlanGraph)
        assert len(graph.steps) == 3
        assert graph.metadata.get("source") == "llm"
        assert graph.steps[0].kind == "mission"
        assert graph.steps[1].kind == "subgoal"
        assert graph.steps[2].kind == "action"

    def test_llm_plan_with_markdown_json(self):
        """Test parsing LLM response with JSON in markdown code block."""
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.model_name = "test-model"
        mock_adapter.generate.return_value = LLMResponse(
            content="""Here is the plan:

```json
{
    "plan_id": "plan_test",
    "steps": [
        {"step_id": "m1", "level": 0, "kind": "mission", "description": "Mission", "parent_id": null, "requires_simulation": false, "safety_tags": []},
        {"step_id": "s1", "level": 1, "kind": "subgoal", "description": "Subgoal", "parent_id": "m1", "requires_simulation": false, "safety_tags": []}
    ]
}
```
""",
            model="test-model",
        )

        planner = LLMPlanner(llm_adapter=mock_adapter)
        request = self._create_plan_request()
        graph = planner.generate_plan(request)

        assert len(graph.steps) == 2
        assert graph.metadata.get("source") == "llm"

    def test_fallback_on_invalid_json(self):
        """Test fallback when LLM returns invalid JSON."""
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.model_name = "test-model"
        mock_adapter.generate.return_value = LLMResponse(
            content="This is not valid JSON at all",
            model="test-model",
        )

        config = LLMPlannerConfig(max_retries=0)
        planner = LLMPlanner(config=config, llm_adapter=mock_adapter)
        request = self._create_plan_request()
        graph = planner.generate_plan(request)

        # Should fall back to deterministic planner
        assert isinstance(graph, PlanGraph)
        assert len(graph.steps) > 0
        assert graph.metadata.get("source") != "llm"

    def test_fallback_on_empty_steps(self):
        """Test fallback when LLM returns empty steps."""
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.model_name = "test-model"
        mock_adapter.generate.return_value = LLMResponse(
            content=json.dumps({"plan_id": "test", "steps": []}),
            model="test-model",
        )

        config = LLMPlannerConfig(max_retries=0)
        planner = LLMPlanner(config=config, llm_adapter=mock_adapter)
        request = self._create_plan_request()
        graph = planner.generate_plan(request)

        # Should fall back
        assert len(graph.steps) > 0
        assert graph.metadata.get("source") != "llm"

    def test_validation_requires_mission_and_subgoal(self):
        """Test that validation fails without mission or subgoal levels."""
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.model_name = "test-model"
        # Only level 2 steps, no mission or subgoal
        mock_adapter.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "plan_id": "test",
                    "steps": [
                        {
                            "step_id": "a1",
                            "level": 2,
                            "kind": "action",
                            "description": "Action",
                        },
                    ],
                }
            ),
            model="test-model",
        )

        config = LLMPlannerConfig(max_retries=0)
        planner = LLMPlanner(config=config, llm_adapter=mock_adapter)
        request = self._create_plan_request()
        graph = planner.generate_plan(request)

        # Should fall back due to validation failure
        assert graph.metadata.get("source") != "llm"

    def test_validation_checks_parent_references(self):
        """Test that validation catches invalid parent references."""
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.model_name = "test-model"
        mock_adapter.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "plan_id": "test",
                    "steps": [
                        {
                            "step_id": "m1",
                            "level": 0,
                            "kind": "mission",
                            "description": "Mission",
                        },
                        {
                            "step_id": "s1",
                            "level": 1,
                            "kind": "subgoal",
                            "description": "Subgoal",
                            "parent_id": "nonexistent",
                        },
                    ],
                }
            ),
            model="test-model",
        )

        config = LLMPlannerConfig(max_retries=0)
        planner = LLMPlanner(config=config, llm_adapter=mock_adapter)
        request = self._create_plan_request()
        graph = planner.generate_plan(request)

        # Should fall back due to invalid parent reference
        assert graph.metadata.get("source") != "llm"

    def test_retry_on_failure(self):
        """Test retry mechanism on LLM failures."""
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.model_name = "test-model"

        # First call fails, second succeeds
        call_count = [0]

        def generate_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Network error")
            return LLMResponse(
                content=json.dumps(
                    {
                        "plan_id": "test",
                        "steps": [
                            {
                                "step_id": "m1",
                                "level": 0,
                                "kind": "mission",
                                "description": "M",
                            },
                            {
                                "step_id": "s1",
                                "level": 1,
                                "kind": "subgoal",
                                "description": "S",
                                "parent_id": "m1",
                            },
                        ],
                    }
                ),
                model="test-model",
            )

        mock_adapter.generate.side_effect = generate_side_effect

        config = LLMPlannerConfig(max_retries=2)
        planner = LLMPlanner(config=config, llm_adapter=mock_adapter)
        request = self._create_plan_request()
        graph = planner.generate_plan(request)

        assert call_count[0] == 2  # First failed, second succeeded
        assert graph.metadata.get("source") == "llm"

    def test_no_fallback_raises_error(self):
        """Test that disabling fallback raises error on LLM failure."""
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        mock_adapter.model_name = "test-model"
        mock_adapter.generate.side_effect = ConnectionError("Network error")

        config = LLMPlannerConfig(fallback_on_error=False, max_retries=0)
        planner = LLMPlanner(config=config, llm_adapter=mock_adapter)
        request = self._create_plan_request()

        with pytest.raises(RuntimeError, match="LLM planning failed"):
            planner.generate_plan(request)

    def test_is_llm_available_property(self):
        """Test is_llm_available property."""
        # With available adapter
        mock_adapter = MagicMock()
        mock_adapter.is_available.return_value = True
        planner = LLMPlanner(llm_adapter=mock_adapter)
        assert planner.is_llm_available is True

        # With unavailable adapter
        mock_adapter.is_available.return_value = False
        assert planner.is_llm_available is False

        # With no adapter (use_llm=False skips adapter initialization)
        config = LLMPlannerConfig(use_llm=False)
        planner_no_llm = LLMPlanner(config=config)
        # When use_llm is False, adapter may not be initialized
        assert planner_no_llm._llm is None or not planner_no_llm.is_llm_available
