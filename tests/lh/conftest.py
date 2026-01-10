"""
Pytest fixtures for LH unit tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional, List

from agi.lh.planner import Planner, PlanGraph, PlanStep
from agi.proto_gen import plan_pb2


@pytest.fixture
def planner():
    """Create a fresh Planner instance."""
    return Planner()


@pytest.fixture
def sample_plan_request():
    """Create a sample PlanRequest for testing."""
    request = plan_pb2.PlanRequest()
    request.task.goal_id = "test-goal-001"
    request.task.description = "Navigate to the red cube and pick it up"
    request.task.task_type = "manipulation"
    request.environment.scenario_id = "tabletop-v1"
    return request


@pytest.fixture
def empty_plan_request():
    """Create an empty PlanRequest."""
    return plan_pb2.PlanRequest()


@dataclass
class MockSafetyResult:
    approved: bool = True
    issues: Optional[List[str]] = None


@dataclass
class MockMetaResult:
    decision: str = "ACCEPT"
    issues: Optional[List[str]] = None
    confidence: float = 1.0


class MockMemoryClient:
    """Mock MemoryClient that passes through requests."""

    def enrich_request(self, request):
        return request


class MockSafetyClient:
    """Mock SafetyClient with configurable behavior."""

    def __init__(self, approved: bool = True, issues: Optional[List[str]] = None):
        self._approved = approved
        self._issues = issues or []

    def check_plan(self, plan_graph):
        return MockSafetyResult(approved=self._approved, issues=self._issues)


class MockMetacogClient:
    """Mock MetacognitionClient with configurable behavior."""

    def __init__(self, decision: str = "ACCEPT", issues: Optional[List[str]] = None):
        self._decision = decision
        self._issues = issues or []

    def review_plan(self, plan_graph):
        return MockMetaResult(decision=self._decision, issues=self._issues)

    def revise_plan(self, plan_graph, review):
        return plan_graph


class MockEventFabric:
    """Mock EventFabric that records published events."""

    def __init__(self):
        self.published_events = []

    def publish(self, topic: str, payload: dict):
        self.published_events.append((topic, payload))

    def subscribe(self, topic: str, handler):
        pass

    def close(self):
        pass


@pytest.fixture
def mock_memory():
    return MockMemoryClient()


@pytest.fixture
def mock_safety():
    return MockSafetyClient(approved=True)


@pytest.fixture
def mock_safety_reject():
    return MockSafetyClient(approved=False, issues=["test_rejection"])


@pytest.fixture
def mock_metacog():
    return MockMetacogClient(decision="ACCEPT")


@pytest.fixture
def mock_metacog_reject():
    return MockMetacogClient(decision="REJECT", issues=["test_rejection"])


@pytest.fixture
def mock_fabric():
    return MockEventFabric()
