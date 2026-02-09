# AGI-HPC Project - Memory Subsystem Tests
# Fixtures for memory service testing.

import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Mock Memory Clients
# ---------------------------------------------------------------------------


@dataclass
class MockSemanticFact:
    """Mock semantic fact for testing."""

    fact_id: str = "fact_001"
    content: str = "Test fact content"
    confidence: float = 0.9
    similarity: float = 0.85
    source: str = "test"
    domains: List[str] = None

    def __post_init__(self):
        if self.domains is None:
            self.domains = ["general"]


@dataclass
class MockEpisode:
    """Mock episode for testing."""

    episode_id: str = "ep_001"
    task_description: str = "Test task"
    task_type: str = "navigation"
    scenario_id: str = "test_scenario"
    similarity: float = 0.8
    insights: List[str] = None
    outcome: Any = None

    def __post_init__(self):
        if self.insights is None:
            self.insights = ["Test insight"]
        if self.outcome is None:
            self.outcome = MagicMock(success=True, completion_percentage=0.95)

    def HasField(self, field):
        return field == "outcome" and self.outcome is not None


@dataclass
class MockSkill:
    """Mock skill for testing."""

    skill_id: str = "skill_001"
    name: str = "test_skill"
    description: str = "A test skill"
    category: str = "navigation"
    proficiency: float = 0.75
    success_rate: float = 0.9
    execution_count: int = 10
    preconditions: List[str] = None
    postconditions: List[str] = None
    actions: List[Any] = None
    similarity: float = 0.7

    def __post_init__(self):
        if self.preconditions is None:
            self.preconditions = []
        if self.postconditions is None:
            self.postconditions = []
        if self.actions is None:
            self.actions = []


@pytest.fixture
def mock_semantic_client():
    """Create a mock semantic memory client."""
    client = MagicMock()

    async def mock_search(text, max_results=10, **kwargs):
        result = MagicMock()
        result.facts = [MockSemanticFact(content=f"Fact about: {text[:20]}")]
        return result

    async def mock_get_tool_schema(tool_id):
        schema = MagicMock()
        schema.tool_id = tool_id
        schema.name = f"Tool {tool_id}"
        schema.description = "Test tool"
        schema.parameters = []
        schema.preconditions = []
        schema.postconditions = []
        return schema

    client.search = AsyncMock(side_effect=mock_search)
    client.get_tool_schema = AsyncMock(side_effect=mock_get_tool_schema)
    return client


@pytest.fixture
def mock_episodic_client():
    """Create a mock episodic memory client."""
    client = MagicMock()

    async def mock_search(situation_description, task_type="", max_results=5, **kwargs):
        result = MagicMock()
        result.episodes = [
            MockEpisode(task_description=f"Similar to: {situation_description[:20]}")
        ]
        return result

    client.search = AsyncMock(side_effect=mock_search)
    return client


@pytest.fixture
def mock_procedural_client():
    """Create a mock procedural memory client."""
    client = MagicMock()

    async def mock_search(
        capability_description, task_type="", max_results=10, **kwargs
    ):
        result = MagicMock()
        result.skills = [
            MockSkill(description=f"Skill for: {capability_description[:20]}")
        ]
        return result

    client.search = AsyncMock(side_effect=mock_search)
    return client


# ---------------------------------------------------------------------------
# Unified Memory Service Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unified_memory_config():
    """Create unified memory config for testing."""
    from agi.memory.unified import UnifiedMemoryConfig

    return UnifiedMemoryConfig(
        semantic_addr="localhost:50053",
        episodic_addr="localhost:50052",
        procedural_addr="localhost:50054",
        timeout_sec=5.0,
        default_max_facts=5,
        default_max_episodes=3,
        default_max_skills=5,
        enable_caching=False,
    )


@pytest.fixture
def unified_memory_service(
    unified_memory_config,
    mock_semantic_client,
    mock_episodic_client,
    mock_procedural_client,
):
    """Create UnifiedMemoryService with mock clients."""
    from agi.memory.unified import UnifiedMemoryService

    return UnifiedMemoryService(
        config=unified_memory_config,
        semantic_client=mock_semantic_client,
        episodic_client=mock_episodic_client,
        procedural_client=mock_procedural_client,
    )


@pytest.fixture
def unified_memory_service_stub(unified_memory_config):
    """Create UnifiedMemoryService in stub mode (no clients)."""
    from agi.memory.unified import UnifiedMemoryService

    return UnifiedMemoryService(config=unified_memory_config)
