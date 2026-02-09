"""
Unit tests for the RH WorldModel module.
"""

import pytest

from agi.rh.world_model import WorldModel, RolloutResult


class TestWorldModelInit:
    """Tests for WorldModel initialization."""

    def test_world_model_initializes_with_defaults(self):
        """WorldModel should initialize with default parameters."""
        wm = WorldModel()
        assert wm._name == "dummy_world_model"
        assert wm._horizon == 5

    def test_world_model_accepts_custom_params(self):
        """WorldModel should accept custom parameters."""
        wm = WorldModel(model_name="mujoco", horizon=10)
        assert wm._name == "mujoco"
        assert wm._horizon == 10


class TestWorldModelRollout:
    """Tests for rollout method."""

    def test_rollout_returns_rollout_result(
        self, world_model, initial_state, safe_actions
    ):
        """rollout should return RolloutResult."""
        result = world_model.rollout(initial_state, safe_actions)

        assert isinstance(result, RolloutResult)
        assert hasattr(result, "risk_score")
        assert hasattr(result, "violation")
        assert hasattr(result, "predicted_states")

    def test_rollout_produces_predicted_states(
        self, world_model, initial_state, safe_actions
    ):
        """rollout should produce predicted state trajectory."""
        result = world_model.rollout(initial_state, safe_actions)

        assert result.predicted_states is not None
        assert len(result.predicted_states) > 0
        assert len(result.predicted_states) <= len(safe_actions)

    def test_rollout_updates_agent_pose(self, world_model, initial_state, safe_actions):
        """rollout should update agent pose in predicted states."""
        result = world_model.rollout(initial_state, safe_actions)

        initial_pose = initial_state["agent_pose"]
        final_pose = result.predicted_states[-1]["agent_pose"]

        # Pose should have changed
        assert final_pose != initial_pose

    def test_rollout_safe_actions_low_risk(
        self, world_model, initial_state, safe_actions
    ):
        """Safe actions should result in low risk score."""
        result = world_model.rollout(initial_state, safe_actions)

        # Safe actions with small magnitude should have low risk
        assert result.risk_score < 0.5
        assert result.violation == ""

    def test_rollout_empty_actions_returns_valid_result(
        self, world_model, initial_state
    ):
        """Empty action list should return valid result."""
        result = world_model.rollout(initial_state, [])

        assert isinstance(result, RolloutResult)
        assert result.risk_score == 0.0
        assert result.violation == ""
        assert result.predicted_states == []


class TestWorldModelRiskEstimation:
    """Tests for risk estimation in rollout."""

    def test_high_magnitude_increases_risk(self, world_model, initial_state):
        """High magnitude actions should increase risk."""
        low_mag_actions = [{"type": "move", "magnitude": 0.1}]
        high_mag_actions = [{"type": "move", "magnitude": 1.0}]

        low_result = world_model.rollout(initial_state, low_mag_actions)
        high_result = world_model.rollout(initial_state, high_mag_actions)

        assert high_result.risk_score > low_result.risk_score

    def test_overspeed_triggers_violation(
        self, world_model, initial_state, risky_actions
    ):
        """Magnitude > 1.5 should trigger overspeed violation."""
        result = world_model.rollout(initial_state, risky_actions)

        assert result.violation == "overspeed"
        assert result.risk_score == 1.0

    def test_forbidden_action_triggers_violation(
        self, world_model, initial_state, forbidden_actions
    ):
        """Forbidden action type should trigger violation."""
        result = world_model.rollout(initial_state, forbidden_actions)

        assert result.violation == "forbidden_action"
        assert result.risk_score == 1.0

    def test_violation_stops_rollout(self, world_model, initial_state):
        """Violation should stop the rollout early."""
        actions = [
            {"type": "move", "magnitude": 0.1},  # safe
            {"type": "forbidden_action", "magnitude": 0.1},  # violation
            {"type": "move", "magnitude": 0.1},  # should not execute
        ]

        result = world_model.rollout(initial_state, actions)

        # Should have stopped at second action
        assert len(result.predicted_states) == 2


class TestWorldModelHorizon:
    """Tests for horizon limiting."""

    def test_rollout_respects_horizon(self, world_model_short_horizon, initial_state):
        """rollout should respect horizon limit."""
        many_actions = [{"type": "move", "magnitude": 0.1} for _ in range(10)]

        result = world_model_short_horizon.rollout(initial_state, many_actions)

        # Should only process up to horizon
        assert len(result.predicted_states) <= 2

    def test_rollout_processes_all_if_under_horizon(
        self, world_model, initial_state, safe_actions
    ):
        """rollout should process all actions if under horizon."""
        result = world_model.rollout(initial_state, safe_actions)

        # All 3 safe actions should be processed
        assert len(result.predicted_states) == len(safe_actions)


class TestRolloutResult:
    """Tests for RolloutResult dataclass."""

    def test_rollout_result_defaults(self):
        """RolloutResult should have sensible defaults."""
        result = RolloutResult(risk_score=0.5)

        assert result.risk_score == 0.5
        assert result.violation == ""
        assert result.predicted_states is None

    def test_rollout_result_with_states(self):
        """RolloutResult can include predicted states."""
        states = [{"agent_pose": [1, 0, 0]}, {"agent_pose": [2, 0, 0]}]
        result = RolloutResult(
            risk_score=0.3,
            violation="",
            predicted_states=states,
        )

        assert len(result.predicted_states) == 2
        assert result.predicted_states[0]["agent_pose"] == [1, 0, 0]


class TestWorldModelStatePreservation:
    """Tests for state handling in rollout."""

    def test_rollout_preserves_original_state(
        self, world_model, initial_state, safe_actions
    ):
        """rollout should not modify the original state."""
        original_pose = initial_state["agent_pose"].copy()

        world_model.rollout(initial_state, safe_actions)

        # Original should be unchanged
        assert initial_state["agent_pose"] == original_pose

    def test_predicted_states_include_last_action(
        self, world_model, initial_state, safe_actions
    ):
        """Each predicted state should include the action that led to it."""
        result = world_model.rollout(initial_state, safe_actions)

        for i, state in enumerate(result.predicted_states):
            assert "last_action" in state
            assert state["last_action"]["type"] == safe_actions[i]["type"]
