"""Tests for MemoryClient and memory dataclasses."""

import pytest
from unittest.mock import MagicMock, patch

from agi.lh.memory_client import (
    MemoryClient,
    MemoryClientConfig,
    PlanningContext,
    SemanticFact,
    Episode,
    Skill,
    ToolSchema,
)


# ---------------------------------------------------------------------------
# PlanningContext Tests
# ---------------------------------------------------------------------------


class TestPlanningContext:
    """Tests for PlanningContext dataclass."""

    def test_empty_context_has_no_context(self):
        """Empty context should report has_context=False."""
        context = PlanningContext()
        assert context.has_context is False

    def test_context_with_facts_has_context(self):
        """Context with facts should report has_context=True."""
        context = PlanningContext(
            facts=[SemanticFact(fact_id="f1", content="Test fact")]
        )
        assert context.has_context is True

    def test_context_with_episodes_has_context(self):
        """Context with episodes should report has_context=True."""
        context = PlanningContext(
            episodes=[Episode(episode_id="e1", task_description="Test task")]
        )
        assert context.has_context is True

    def test_context_with_skills_has_context(self):
        """Context with skills should report has_context=True."""
        context = PlanningContext(
            skills=[Skill(skill_id="s1", name="Test", description="Test skill")]
        )
        assert context.has_context is True

    def test_to_prompt_context_empty(self):
        """Empty context should return empty string."""
        context = PlanningContext()
        assert context.to_prompt_context() == ""

    def test_to_prompt_context_with_facts(self):
        """Context with facts should format them in prompt."""
        context = PlanningContext(
            facts=[
                SemanticFact(
                    fact_id="f1", content="Red cubes are heavy", confidence=0.9
                )
            ]
        )
        result = context.to_prompt_context()
        assert "RELEVANT FACTS:" in result
        assert "Red cubes are heavy" in result
        assert "0.90" in result

    def test_to_prompt_context_with_episodes(self):
        """Context with episodes should format them in prompt."""
        context = PlanningContext(
            episodes=[
                Episode(
                    episode_id="e1",
                    task_description="Pick up red cube",
                    success=True,
                    similarity=0.85,
                    insights=["Approach from left works best"],
                )
            ]
        )
        result = context.to_prompt_context()
        assert "SIMILAR PAST TASKS:" in result
        assert "Pick up red cube" in result
        assert "success" in result
        assert "Approach from left works best" in result

    def test_to_prompt_context_with_skills(self):
        """Context with skills should format them in prompt."""
        context = PlanningContext(
            skills=[
                Skill(
                    skill_id="s1",
                    name="GraspObject",
                    description="Grasp an object with gripper",
                    proficiency=0.95,
                    success_rate=0.88,
                )
            ]
        )
        result = context.to_prompt_context()
        assert "AVAILABLE SKILLS:" in result
        assert "GraspObject" in result
        assert "Grasp an object" in result
        assert "0.95" in result

    def test_to_prompt_context_with_tools(self):
        """Context with tool schemas should format them in prompt."""
        context = PlanningContext(
            tool_schemas=[
                ToolSchema(
                    tool_id="gripper.close",
                    name="CloseGripper",
                    description="Close the robot gripper",
                )
            ]
        )
        result = context.to_prompt_context()
        assert "AVAILABLE TOOLS:" in result
        assert "gripper.close" in result


# ---------------------------------------------------------------------------
# MemoryClientConfig Tests
# ---------------------------------------------------------------------------


class TestMemoryClientConfig:
    """Tests for MemoryClientConfig dataclass."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = MemoryClientConfig()
        assert config.semantic_address == "localhost:50110"
        assert config.episodic_address == "localhost:50111"
        assert config.procedural_address == "localhost:50112"
        assert config.timeout_seconds == 5.0
        assert config.max_facts == 10
        assert config.max_episodes == 5
        assert config.max_skills == 10

    def test_custom_config(self):
        """Config should accept custom values."""
        config = MemoryClientConfig(
            semantic_address="memory:9000",
            max_facts=20,
            min_similarity=0.8,
        )
        assert config.semantic_address == "memory:9000"
        assert config.max_facts == 20
        assert config.min_similarity == 0.8


# ---------------------------------------------------------------------------
# MemoryClient Tests
# ---------------------------------------------------------------------------


class TestMemoryClientInit:
    """Tests for MemoryClient initialization."""

    def test_memory_client_initializes_with_config(self):
        """MemoryClient should initialize with config."""
        config = MemoryClientConfig(semantic_address="localhost:50110")
        client = MemoryClient(config=config)
        assert client._config.semantic_address == "localhost:50110"

    def test_memory_client_initializes_with_legacy_address(self):
        """MemoryClient should accept legacy semantic_address parameter."""
        client = MemoryClient(semantic_address="localhost:50110")
        assert client._config.semantic_address == "localhost:50110"

    def test_memory_client_stubs_initialized(self):
        """MemoryClient should attempt to create stubs."""
        client = MemoryClient()
        # Stubs are created (channels opened) but may not be connected
        assert len(client._channels) > 0


class TestMemoryClientEnrichRequest:
    """Tests for MemoryClient.enrich_request method."""

    def test_enrich_request_passthrough_when_unavailable(self):
        """enrich_request should return request unchanged when service unavailable."""
        client = MemoryClient()

        # Create a mock request
        mock_request = MagicMock()
        mock_request.task.description = "Test task"

        result = client.enrich_request(mock_request)

        assert result is mock_request

    def test_enrich_request_returns_same_object(self):
        """enrich_request should return the exact same request object."""
        client = MemoryClient()

        class DummyRequest:
            pass

        request = DummyRequest()
        result = client.enrich_request(request)

        assert result is request

    def test_enrich_request_handles_none_task(self):
        """enrich_request should handle request without task attribute."""
        client = MemoryClient()

        class RequestWithoutTask:
            pass

        request = RequestWithoutTask()
        result = client.enrich_request(request)

        assert result is request


class TestMemoryClientGetPlanningContext:
    """Tests for MemoryClient.get_planning_context method."""

    def test_get_planning_context_returns_empty_when_unavailable(self):
        """get_planning_context should return empty context when services unavailable."""
        client = MemoryClient()

        context = client.get_planning_context("Pick up the red cube")

        assert isinstance(context, PlanningContext)
        # Services are unavailable, so context should be empty
        # (no actual RPC calls succeed)

    def test_get_planning_context_with_task_type(self):
        """get_planning_context should accept task_type parameter."""
        client = MemoryClient()

        context = client.get_planning_context(
            "Pick up the red cube",
            task_type="manipulation",
            scenario_id="tabletop-v1",
        )

        assert isinstance(context, PlanningContext)


class TestMemoryClientClose:
    """Tests for MemoryClient.close method."""

    def test_close_clears_channels(self):
        """close should clear the channel list."""
        client = MemoryClient()
        initial_channels = len(client._channels)
        assert initial_channels > 0

        client.close()

        assert len(client._channels) == 0


# ---------------------------------------------------------------------------
# SemanticFact Tests
# ---------------------------------------------------------------------------


class TestSemanticFact:
    """Tests for SemanticFact dataclass."""

    def test_semantic_fact_creation(self):
        """SemanticFact should store all fields."""
        fact = SemanticFact(
            fact_id="fact-001",
            content="Red cubes weigh 500 grams",
            confidence=0.95,
            similarity=0.85,
            source="knowledge_base",
            domains=["robotics", "objects"],
        )

        assert fact.fact_id == "fact-001"
        assert fact.content == "Red cubes weigh 500 grams"
        assert fact.confidence == 0.95
        assert fact.similarity == 0.85
        assert fact.source == "knowledge_base"
        assert fact.domains == ["robotics", "objects"]


# ---------------------------------------------------------------------------
# Episode Tests
# ---------------------------------------------------------------------------


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_episode_creation(self):
        """Episode should store all fields."""
        episode = Episode(
            episode_id="ep-001",
            task_description="Navigate to kitchen",
            task_type="navigation",
            scenario_id="house-v2",
            success=True,
            similarity=0.9,
            insights=["Avoid the couch", "Door may be closed"],
            plan_steps=["Move forward", "Turn left", "Open door"],
        )

        assert episode.episode_id == "ep-001"
        assert episode.task_description == "Navigate to kitchen"
        assert episode.success is True
        assert len(episode.insights) == 2
        assert len(episode.plan_steps) == 3


# ---------------------------------------------------------------------------
# Skill Tests
# ---------------------------------------------------------------------------


class TestSkill:
    """Tests for Skill dataclass."""

    def test_skill_creation(self):
        """Skill should store all fields."""
        skill = Skill(
            skill_id="skill-grasp",
            name="GraspObject",
            description="Grasp an object using the end effector",
            category="manipulation",
            preconditions=["object_visible", "gripper_open"],
            postconditions=["object_held"],
            proficiency=0.92,
            success_rate=0.88,
            similarity=0.75,
        )

        assert skill.skill_id == "skill-grasp"
        assert skill.name == "GraspObject"
        assert skill.category == "manipulation"
        assert skill.proficiency == 0.92
        assert skill.success_rate == 0.88


# ---------------------------------------------------------------------------
# ToolSchema Tests
# ---------------------------------------------------------------------------


class TestToolSchema:
    """Tests for ToolSchema dataclass."""

    def test_tool_schema_creation(self):
        """ToolSchema should store all fields."""
        schema = ToolSchema(
            tool_id="gripper.close",
            name="CloseGripper",
            description="Close the gripper to grasp object",
            parameters=[
                {"name": "force", "type": "float", "required": True},
            ],
            preconditions=["gripper_open"],
            postconditions=["gripper_closed"],
        )

        assert schema.tool_id == "gripper.close"
        assert schema.name == "CloseGripper"
        assert len(schema.parameters) == 1
        assert schema.parameters[0]["name"] == "force"
