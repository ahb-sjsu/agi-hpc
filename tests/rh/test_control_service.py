"""
Unit tests for the RH ControlService module.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

from agi.rh.control_service import ControlService, ControlConfig, ActionResult
from agi.proto_gen import plan_pb2


class TestControlServiceInit:
    """Tests for ControlService initialization."""

    def test_control_service_initializes_with_defaults(self):
        """ControlService should initialize with default config."""
        cs = ControlService()
        assert cs._config.controller_type == "rule_based"
        assert cs._config.default_env == "mock:simple"

    def test_control_service_accepts_custom_config(self, control_config):
        """ControlService should accept custom config."""
        cs = ControlService(config=control_config)
        assert cs._config.action_timeout_sec == 10.0
        assert cs._config.max_retries == 3

    def test_control_service_starts_without_environment(self, control_service):
        """ControlService should start with no environment."""
        assert control_service.get_environment() is None


class TestControlServiceEnvironment:
    """Tests for environment management."""

    def test_set_environment(self, control_service, mock_environment):
        """set_environment should set the environment."""
        control_service.set_environment(mock_environment)
        assert control_service.get_environment() is mock_environment

    @pytest.mark.asyncio
    async def test_reset_environment_returns_observation(
        self, control_service, mock_environment
    ):
        """reset_environment should return initial observation."""
        control_service.set_environment(mock_environment)
        obs = await control_service.reset_environment()

        assert obs is not None
        mock_environment.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_environment_with_seed(self, control_service, mock_environment):
        """reset_environment should pass seed to environment."""
        control_service.set_environment(mock_environment)
        await control_service.reset_environment(seed=42)

        mock_environment.reset.assert_called_with(seed=42, options=None)

    @pytest.mark.asyncio
    async def test_reset_without_environment_returns_none(self, control_service):
        """reset_environment without environment should return None."""
        obs = await control_service.reset_environment()
        assert obs is None


class TestControlServiceTranslateStep:
    """Tests for translate_step method."""

    def test_translate_move_step(self, control_service, sample_plan_step):
        """translate_step should create move actions for move description."""
        actions = control_service.translate_step(sample_plan_step)

        assert len(actions) >= 1
        assert actions[0]["type"] == "move"
        assert "magnitude" in actions[0]

    def test_translate_manipulation_step(self, control_service, manipulation_plan_step):
        """translate_step should create manipulation sequence."""
        actions = control_service.translate_step(manipulation_plan_step)

        # Manipulation should produce reach, grasp, lift sequence
        assert len(actions) == 3
        action_types = [a["type"] for a in actions]
        assert "reach" in action_types
        assert "grasp" in action_types
        assert "lift" in action_types

    def test_translate_scan_step(self, control_service, scan_plan_step):
        """translate_step should create scan actions."""
        actions = control_service.translate_step(scan_plan_step)

        action_types = [a["type"] for a in actions]
        assert "rotate_camera" in action_types or "capture_frame" in action_types

    def test_translate_unknown_step_returns_noop(self, control_service):
        """Unknown step description should return noop."""
        step = plan_pb2.PlanStep()
        step.step_id = "unknown"
        step.description = "some completely unknown action type xyz"

        actions = control_service.translate_step(step)

        assert len(actions) >= 1
        assert actions[0]["type"] == "noop"

    def test_translate_extracts_magnitude_param(
        self, control_service, sample_plan_step
    ):
        """translate_step should extract magnitude from params."""
        actions = control_service.translate_step(sample_plan_step)

        assert actions[0]["magnitude"] == 0.5  # From fixture params


class TestControlServiceExecuteActions:
    """Tests for action execution."""

    @pytest.mark.asyncio
    async def test_execute_actions_without_env_returns_stub(self, control_service):
        """execute_actions without environment returns stub results."""
        actions = [{"type": "move", "magnitude": 0.1}]
        results = await control_service.execute_actions(actions)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].info.get("stub") is True

    @pytest.mark.asyncio
    async def test_execute_actions_with_env(self, control_service, mock_environment):
        """execute_actions with environment should call env.step."""
        control_service.set_environment(mock_environment)
        actions = [{"type": "move", "magnitude": 0.1}]

        results = await control_service.execute_actions(actions)

        assert len(results) == 1
        assert results[0].success is True
        mock_environment.step.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_actions_stops_on_done(
        self, control_service, mock_environment
    ):
        """execute_actions should stop when episode is done."""
        # Make second step return done=True
        call_count = [0]

        async def step_side_effect(action):
            call_count[0] += 1
            done = call_count[0] >= 2
            return MagicMock(
                observation=np.zeros(10),
                reward=1.0,
                done=done,
                info={},
            )

        mock_environment.step = AsyncMock(side_effect=step_side_effect)
        control_service.set_environment(mock_environment)

        actions = [
            {"type": "move", "magnitude": 0.1},
            {"type": "move", "magnitude": 0.1},
            {"type": "move", "magnitude": 0.1},  # Should not execute
        ]

        results = await control_service.execute_actions(actions)

        # Should stop after second action
        assert len(results) == 2
        assert results[-1].done is True


class TestControlServiceSingleAction:
    """Tests for execute_single_action."""

    @pytest.mark.asyncio
    async def test_single_action_updates_observation(
        self, control_service, mock_environment
    ):
        """Single action should update last observation."""
        control_service.set_environment(mock_environment)

        result = await control_service.execute_single_action(
            {"type": "move", "magnitude": 0.1}
        )

        assert result.success is True
        assert control_service.get_observation() is not None

    @pytest.mark.asyncio
    async def test_single_action_accumulates_reward(
        self, control_service, mock_environment
    ):
        """Single action should accumulate episode reward."""
        control_service.set_environment(mock_environment)

        await control_service.execute_single_action({"type": "move", "magnitude": 0.1})
        await control_service.execute_single_action({"type": "move", "magnitude": 0.1})

        assert control_service.get_episode_reward() == 2.0

    @pytest.mark.asyncio
    async def test_single_action_increments_step_count(
        self, control_service, mock_environment
    ):
        """Single action should increment step count."""
        control_service.set_environment(mock_environment)

        await control_service.execute_single_action({"type": "move", "magnitude": 0.1})
        await control_service.execute_single_action({"type": "move", "magnitude": 0.1})

        assert control_service.get_step_count() == 2


class TestControlServiceActionConversion:
    """Tests for action conversion to environment format."""

    def test_convert_noop_action(self, control_service, mock_environment):
        """Noop action should convert to zeros."""
        control_service.set_environment(mock_environment)
        action = {"type": "noop", "magnitude": 0.0}

        env_action = control_service._convert_to_env_action(action)

        assert isinstance(env_action, np.ndarray)
        assert np.allclose(env_action, 0.0)

    def test_convert_move_action(self, control_service, mock_environment):
        """Move action should convert based on target and magnitude."""
        control_service.set_environment(mock_environment)
        action = {"type": "move", "target": [1.0, 0.0, 0.0, 0.0], "magnitude": 0.5}

        env_action = control_service._convert_to_env_action(action)

        assert isinstance(env_action, np.ndarray)
        assert env_action.shape == (4,)

    def test_convert_action_without_env_returns_scalar(self, control_service):
        """Conversion without environment should return scalar array."""
        action = {"type": "move", "magnitude": 0.5}

        env_action = control_service._convert_to_env_action(action)

        assert isinstance(env_action, np.ndarray)
        assert env_action[0] == 0.0


class TestControlServiceErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_action_timeout_returns_failure(
        self, control_service, mock_environment
    ):
        """Timeout should return failure result."""
        import asyncio

        async def timeout_step(action):
            raise asyncio.TimeoutError()

        mock_environment.step = AsyncMock(side_effect=timeout_step)
        control_service.set_environment(mock_environment)

        result = await control_service.execute_single_action(
            {"type": "move", "magnitude": 0.1}
        )

        assert result.success is False
        assert "timeout" in result.info.get("error", "")

    @pytest.mark.asyncio
    async def test_action_exception_returns_failure(
        self, control_service, mock_environment
    ):
        """Exception should return failure result."""
        mock_environment.step = AsyncMock(side_effect=RuntimeError("Test error"))
        control_service.set_environment(mock_environment)

        result = await control_service.execute_single_action(
            {"type": "move", "magnitude": 0.1}
        )

        assert result.success is False
        assert "Test error" in result.info.get("error", "")


class TestControlServiceCleanup:
    """Tests for cleanup."""

    @pytest.mark.asyncio
    async def test_close_closes_environment(self, control_service, mock_environment):
        """close should close the environment."""
        control_service.set_environment(mock_environment)

        await control_service.close()

        mock_environment.close.assert_called_once()
        assert control_service.get_environment() is None


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_defaults(self):
        """ActionResult should have sensible defaults."""
        result = ActionResult(success=True)

        assert result.success is True
        assert result.observation is None
        assert result.reward == 0.0
        assert result.done is False
        assert result.info == {}

    def test_action_result_with_values(self):
        """ActionResult should store all values."""
        obs = np.zeros(10)
        result = ActionResult(
            success=True,
            observation=obs,
            reward=1.5,
            done=True,
            info={"test": "value"},
        )

        assert result.success is True
        assert np.array_equal(result.observation, obs)
        assert result.reward == 1.5
        assert result.done is True
        assert result.info["test"] == "value"
