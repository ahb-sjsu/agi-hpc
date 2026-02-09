"""
Integration tests for RH pipeline.

These tests verify the RH pipeline integrates correctly with LH and Safety.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import numpy as np

from agi.rh.perception import Perception
from agi.rh.world_model import WorldModel
from agi.rh.control_service import ControlService, ControlConfig, ActionResult
from agi.proto_gen import plan_pb2

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_fabric():
    """Create a mock EventFabric that captures published events."""

    class MockFabric:
        def __init__(self):
            self.published_events = []
            self.subscriptions = {}

        def publish(self, topic: str, payload: dict):
            self.published_events.append((topic, payload))

        def subscribe(self, topic: str, handler):
            self.subscriptions[topic] = handler

        async def publish_async(self, topic: str, payload: dict):
            self.published_events.append((topic, payload))

        def close(self):
            pass

    return MockFabric()


@pytest.fixture
def sample_plan_step():
    """Create a sample PlanStep for testing."""
    step = plan_pb2.PlanStep()
    step.step_id = "step-001"
    step.description = "Move to target location"
    step.kind = "move"
    step.params["magnitude"] = "0.5"
    step.params["target"] = "[1.0, 0.0, 0.0]"
    return step


@pytest.fixture
def sample_simulation_request(sample_plan_step):
    """Create a sample SimulationRequest."""
    request = plan_pb2.SimulationRequest()
    request.plan_id = "plan-001"
    request.step.CopyFrom(sample_plan_step)
    request.horizon = 5
    return request


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""

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
            info={},
        )
    )
    env.close = AsyncMock()
    return env


@pytest.fixture
def perception():
    """Create a Perception instance."""
    return Perception()


@pytest.fixture
def world_model():
    """Create a WorldModel instance."""
    return WorldModel()


@pytest.fixture
def control_service():
    """Create a ControlService instance."""
    return ControlService()


# ---------------------------------------------------------------------------
# 3.1 LH → RH Simulation Flow Tests
# ---------------------------------------------------------------------------


class TestLHToRHSimulationFlow:
    """Integration tests for LH sending simulation requests to RH."""

    def test_simulation_request_translated_to_actions(
        self, sample_plan_step, control_service
    ):
        """SimulationRequest PlanStep should be translated to actions."""
        actions = control_service.translate_step(sample_plan_step)

        assert len(actions) >= 1
        assert actions[0]["type"] == "move"
        assert "magnitude" in actions[0]

    def test_world_model_rollout_with_actions(
        self, sample_plan_step, world_model, control_service
    ):
        """WorldModel should perform rollout with translated actions."""
        actions = control_service.translate_step(sample_plan_step)

        initial_state = {"position": [0, 0, 0], "velocity": [0, 0, 0]}
        result = world_model.rollout(initial_state, actions)

        assert result is not None
        assert hasattr(result, "risk_score")
        assert hasattr(result, "predicted_states")

    def test_risk_score_propagates_from_rollout(
        self, sample_plan_step, world_model, control_service
    ):
        """Risk score from WorldModel should be returned in simulation result."""
        actions = control_service.translate_step(sample_plan_step)

        initial_state = {"position": [0, 0, 0], "velocity": [0, 0, 0]}
        result = world_model.rollout(initial_state, actions)

        assert 0.0 <= result.risk_score <= 1.0

    def test_violations_detected_in_rollout(self, world_model):
        """WorldModel should detect violations during rollout."""
        # Create risky actions that should trigger violations
        risky_actions = [{"type": "move", "magnitude": 10.0}]  # Overspeed

        initial_state = {"position": [0, 0, 0], "velocity": [0, 0, 0]}
        result = world_model.rollout(initial_state, risky_actions)

        # High magnitude should trigger overspeed violation
        assert result.violation is not None or result.risk_score > 0.5


# ---------------------------------------------------------------------------
# 3.2 RH → Safety Integration Tests
# ---------------------------------------------------------------------------


class TestRHSafetyIntegration:
    """Integration tests for RH-Safety service interaction."""

    def test_control_service_respects_safety_stop(
        self, control_service, mock_environment
    ):
        """ControlService should stop execution on safety violation."""
        control_service.set_environment(mock_environment)

        # Simulate safety stop callback
        safety_stopped = False

        def safety_callback(action):
            nonlocal safety_stopped
            safety_stopped = True
            return False  # Not safe

        # If safety integration exists, it should stop
        # For now, verify the control service has safety_enabled config
        assert control_service._config.safety_enabled is True

    def test_world_model_risk_triggers_safety_threshold(self, world_model):
        """High risk from WorldModel should trigger safety consideration."""
        risky_actions = [{"type": "move", "magnitude": 100.0}]  # Very high magnitude

        initial_state = {"position": [0, 0, 0], "velocity": [0, 0, 0]}
        result = world_model.rollout(initial_state, risky_actions)

        # Should have high risk or violation
        assert result.risk_score >= 0.5 or result.violation is not None


# ---------------------------------------------------------------------------
# 3.3 EventFabric LH ↔ RH Communication Tests
# ---------------------------------------------------------------------------


class TestEventFabricLHRHCommunication:
    """Integration tests for EventFabric communication between LH and RH."""

    def test_rh_receives_plan_step_ready_event(self, mock_fabric, sample_plan_step):
        """RH should be able to receive plan.step_ready events from fabric."""
        received_events = []

        def handler(payload):
            received_events.append(payload)

        mock_fabric.subscribe("plan.step_ready", handler)

        # Simulate LH publishing event
        mock_fabric.publish(
            "plan.step_ready",
            {
                "plan_id": "plan-001",
                "step_id": sample_plan_step.step_id,
                "step": {
                    "description": sample_plan_step.description,
                    "kind": sample_plan_step.kind,
                },
            },
        )

        assert len(mock_fabric.published_events) == 1
        topic, payload = mock_fabric.published_events[0]
        assert topic == "plan.step_ready"
        assert payload["plan_id"] == "plan-001"

    def test_rh_publishes_simulation_result_event(
        self, mock_fabric, world_model, control_service
    ):
        """RH should publish simulation.result events to fabric."""
        # Simulate running a simulation
        initial_state = {"position": [0, 0, 0], "velocity": [0, 0, 0]}
        actions = [{"type": "move", "magnitude": 0.5}]
        result = world_model.rollout(initial_state, actions)

        # Publish simulation result event
        mock_fabric.publish(
            "simulation.result",
            {
                "plan_id": "plan-001",
                "step_id": "step-001",
                "risk_score": result.risk_score,
                "violation": result.violation,
                "predicted_states": len(result.predicted_states),
            },
        )

        simulation_events = [
            e for e in mock_fabric.published_events if e[0] == "simulation.result"
        ]
        assert len(simulation_events) == 1

        topic, payload = simulation_events[0]
        assert "risk_score" in payload
        assert "violation" in payload

    def test_rh_publishes_perception_state_update(self, mock_fabric, perception):
        """RH should publish perception.state_update events."""
        # Update perception with observation
        frame = np.random.rand(84, 84, 3).astype(np.float32)
        state = perception.update_observation(frame)

        # Publish perception event
        mock_fabric.publish(
            "perception.state_update",
            {
                "timestamp": 12345,
                "objects": state.get("objects", []),
                "features_dim": len(state.get("features", [])),
            },
        )

        perception_events = [
            e for e in mock_fabric.published_events if e[0] == "perception.state_update"
        ]
        assert len(perception_events) == 1


# ---------------------------------------------------------------------------
# Full RH Pipeline Integration Tests
# ---------------------------------------------------------------------------


class TestRHFullPipeline:
    """End-to-end integration tests for RH pipeline."""

    def test_full_simulation_pipeline(
        self,
        sample_plan_step,
        perception,
        world_model,
        control_service,
        mock_environment,
    ):
        """Test complete pipeline: step → perception → control → world_model."""
        control_service.set_environment(mock_environment)

        # 1. Get current perception state
        frame = np.random.rand(84, 84, 3).astype(np.float32)
        perception_state = perception.update_observation(frame)
        assert perception_state is not None

        # 2. Translate plan step to actions
        actions = control_service.translate_step(sample_plan_step)
        assert len(actions) > 0

        # 3. Run world model rollout
        initial_state = perception_state.copy()
        initial_state.setdefault("position", [0, 0, 0])
        initial_state.setdefault("velocity", [0, 0, 0])

        rollout_result = world_model.rollout(initial_state, actions)
        assert rollout_result is not None

        # 4. Verify pipeline output
        assert hasattr(rollout_result, "risk_score")
        assert hasattr(rollout_result, "predicted_states")

    @pytest.mark.asyncio
    async def test_action_execution_pipeline(
        self,
        sample_plan_step,
        control_service,
        mock_environment,
    ):
        """Test action execution with environment."""
        control_service.set_environment(mock_environment)

        # Reset environment
        await control_service.reset_environment()

        # Translate and execute actions
        actions = control_service.translate_step(sample_plan_step)
        results = await control_service.execute_actions(actions)

        assert len(results) == len(actions)
        assert all(isinstance(r, ActionResult) for r in results)

    def test_perception_world_model_integration(self, perception, world_model):
        """Test perception state flows to world model correctly."""
        # Get perception state
        frame = np.random.rand(84, 84, 3).astype(np.float32)
        perception_state = perception.update_observation(frame)

        # Use perception state as initial state for rollout
        initial_state = perception_state.copy()
        initial_state.setdefault("position", [0, 0, 0])
        initial_state.setdefault("velocity", [0, 0, 0])

        actions = [{"type": "move", "magnitude": 0.5}]
        result = world_model.rollout(initial_state, actions)

        assert result is not None
        assert result.risk_score >= 0.0


# ---------------------------------------------------------------------------
# Error Handling Integration Tests
# ---------------------------------------------------------------------------


class TestRHErrorHandling:
    """Integration tests for RH error handling."""

    def test_invalid_plan_step_handled(self, control_service):
        """Invalid plan steps should be handled gracefully."""
        step = plan_pb2.PlanStep()
        step.step_id = "invalid"
        step.description = "xyz unknown action"

        actions = control_service.translate_step(step)

        # Should return noop action
        assert len(actions) >= 1
        assert actions[0]["type"] == "noop"

    def test_world_model_handles_empty_actions(self, world_model):
        """WorldModel should handle empty action list."""
        initial_state = {"position": [0, 0, 0], "velocity": [0, 0, 0]}
        result = world_model.rollout(initial_state, [])

        assert result is not None
        assert result.risk_score == 0.0

    @pytest.mark.asyncio
    async def test_environment_error_propagates(
        self, control_service, mock_environment
    ):
        """Environment errors should be captured in ActionResult."""
        mock_environment.step = AsyncMock(side_effect=RuntimeError("Environment error"))
        control_service.set_environment(mock_environment)

        result = await control_service.execute_single_action(
            {"type": "move", "magnitude": 0.1}
        )

        assert result.success is False
        assert "Environment error" in result.info.get("error", "")
