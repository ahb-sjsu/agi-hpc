"""
Pytest fixtures for RH (Right Hemisphere) unit tests.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from agi.rh.perception import Perception
from agi.rh.world_model import WorldModel, RolloutResult
from agi.rh.control_service import ControlService, ControlConfig, ActionResult
from agi.proto_gen import plan_pb2

# ---------------------------------------------------------------------------
# Perception Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perception():
    """Create a fresh Perception instance."""
    return Perception(model_name="dummy_encoder", device="cpu")


@pytest.fixture
def sample_frame():
    """Create a sample frame for perception testing."""
    return np.random.rand(480, 640, 3).astype(np.float32)


@pytest.fixture
def empty_frame():
    """Create an empty frame."""
    return np.zeros((480, 640, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# World Model Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def world_model():
    """Create a fresh WorldModel instance."""
    return WorldModel(model_name="dummy_world_model", horizon=5)


@pytest.fixture
def world_model_short_horizon():
    """Create a WorldModel with short horizon."""
    return WorldModel(model_name="dummy_world_model", horizon=2)


@pytest.fixture
def initial_state():
    """Create initial perception state for rollout."""
    return {
        "objects": [
            {"id": "obj_1", "label": "cube", "position": [1.0, 0.0, 0.5]},
            {"id": "obj_2", "label": "sphere", "position": [0.0, 1.0, 0.5]},
        ],
        "agent_pose": [0.0, 0.0, 0.0],
        "embedding": "test_embedding",
    }


@pytest.fixture
def safe_actions():
    """Create a list of safe actions for rollout."""
    return [
        {"type": "move", "target": [1.0, 0.0, 0.0], "magnitude": 0.3, "duration": 0.5},
        {"type": "reach", "magnitude": 0.2, "duration": 0.3},
        {"type": "grasp", "magnitude": 0.1, "duration": 0.2},
    ]


@pytest.fixture
def risky_actions():
    """Create actions that will trigger violations."""
    return [
        {
            "type": "move",
            "target": [1.0, 0.0, 0.0],
            "magnitude": 2.0,
            "duration": 0.5,
        },  # overspeed
    ]


@pytest.fixture
def forbidden_actions():
    """Create actions with forbidden type."""
    return [
        {"type": "forbidden_action", "magnitude": 0.1, "duration": 0.1},
    ]


# ---------------------------------------------------------------------------
# Control Service Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def control_config():
    """Create default control configuration."""
    return ControlConfig(
        controller_type="rule_based",
        default_env="mock:simple",
        action_timeout_sec=10.0,
        max_retries=3,
        safety_enabled=True,
    )


@pytest.fixture
def control_service(control_config):
    """Create ControlService with default config."""
    return ControlService(config=control_config)


@pytest.fixture
def control_service_disabled():
    """Create ControlService with disabled controller."""
    config = ControlConfig(controller_type="disabled")
    return ControlService(config=config)


@pytest.fixture
def mock_environment():
    """Create a mock environment."""

    class MockActionSpace:
        shape = (4,)

        def sample(self):
            return np.random.rand(4).astype(np.float32)

    env = MagicMock()
    env.name = "mock_env"
    env.action_space = MockActionSpace()
    env.step = AsyncMock(
        return_value=MagicMock(
            observation=np.zeros(10),
            reward=1.0,
            done=False,
            info={},
        )
    )
    env.reset = AsyncMock(
        return_value=MagicMock(
            observation=np.zeros(10),
        )
    )
    env.observe = AsyncMock(return_value=np.zeros(10))
    env.close = AsyncMock()

    return env


@pytest.fixture
def sample_plan_step():
    """Create a sample PlanStep for testing."""
    step = plan_pb2.PlanStep()
    step.step_id = "step_001"
    step.kind = "action"
    step.description = "Move to the red cube"
    step.params["magnitude"] = "0.5"
    step.params["target"] = "[1.0, 2.0, 0.0]"
    return step


@pytest.fixture
def manipulation_plan_step():
    """Create a manipulation PlanStep."""
    step = plan_pb2.PlanStep()
    step.step_id = "step_002"
    step.kind = "action"
    step.description = "Pick up the object"
    return step


@pytest.fixture
def scan_plan_step():
    """Create a scan/observation PlanStep."""
    step = plan_pb2.PlanStep()
    step.step_id = "step_003"
    step.kind = "observation"
    step.description = "Scan the environment"
    return step


# ---------------------------------------------------------------------------
# Event Fabric Mock
# ---------------------------------------------------------------------------


class MockEventFabric:
    """Mock EventFabric for testing."""

    def __init__(self):
        self.published_events = []
        self.subscriptions = {}

    def publish(self, topic: str, payload: dict):
        self.published_events.append((topic, payload))

    async def publish_async(self, topic: str, payload: dict):
        self.published_events.append((topic, payload))

    def subscribe(self, topic: str, handler):
        self.subscriptions[topic] = handler

    def close(self):
        pass


@pytest.fixture
def mock_fabric():
    """Create mock EventFabric."""
    return MockEventFabric()


# ---------------------------------------------------------------------------
# Integration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rh_components(perception, world_model, control_service):
    """Create all RH components together."""
    return {
        "perception": perception,
        "world_model": world_model,
        "control": control_service,
    }
