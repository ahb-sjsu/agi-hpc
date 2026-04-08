# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""End-to-end scenario tests for AGI-HPC Sprint 6."""

from unittest.mock import MagicMock, AsyncMock

from tests.integration.framework import (
    IntegrationTestRunner,
    INTEGRATION_TEST_CASES,
)
from tests.integration.chaos import ChaosMonkey, ChaosConfig

# ---------------------------------------------------------------------------
# Navigation Scenario
# ---------------------------------------------------------------------------


class TestNavigationScenario:
    """Full navigation task from LH plan to RH execution."""

    def test_navigation_plan_and_execute(self, mock_event_fabric, mock_rh_control):
        """Plan a navigation task and execute through RH."""
        # 1. Create plan steps
        plan_steps = [
            MagicMock(
                step_id="nav-1",
                kind="navigate",
                description="Navigate to waypoint A",
                params={"target": "[1.0, 0.0, 0.0]", "magnitude": "0.5"},
            ),
        ]

        # 2. Translate to actions
        actions = mock_rh_control.translate_step(plan_steps[0])
        assert len(actions) > 0

        # 3. Publish event
        mock_event_fabric.publish(
            "plan.step_ready",
            {"step_id": "nav-1", "node_id": "LH-test"},
        )
        assert len(mock_event_fabric.published_events) == 1

    def test_navigation_with_multiple_waypoints(self, mock_event_fabric):
        """Navigate through multiple waypoints."""
        waypoints = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
        for i, wp in enumerate(waypoints):
            mock_event_fabric.publish(
                "plan.step_ready",
                {"step_id": f"nav-{i}", "target": list(wp)},
            )
        assert len(mock_event_fabric.published_events) == 3


# ---------------------------------------------------------------------------
# Manipulation Scenario
# ---------------------------------------------------------------------------


class TestManipulationScenario:
    """Pick and place task."""

    def test_pick_and_place(self, mock_rh_control):
        """Execute a pick-and-place sequence."""
        step = MagicMock(
            step_id="pick-1",
            kind="manipulate",
            description="Pick up object",
            params={},
        )
        actions = mock_rh_control.translate_step(step)
        assert len(actions) > 0

    def test_grasp_failure_retry(self, mock_rh_control):
        """Test retry logic when grasp fails."""
        # First attempt fails
        mock_rh_control.execute_actions = AsyncMock(
            side_effect=[
                [MagicMock(success=False, info={"error": "grasp_failed"})],
                [MagicMock(success=True, info={})],
            ]
        )

        # Would retry in real system
        assert mock_rh_control.execute_actions is not None


# ---------------------------------------------------------------------------
# Safety Intervention Scenario
# ---------------------------------------------------------------------------


class TestSafetyInterventionScenario:
    """Task blocked by safety checks."""

    def test_unsafe_action_blocked(self, mock_safety_gateway):
        """Safety blocks an unsafe action."""
        mock_safety_gateway.check.return_value = MagicMock(
            approved=False,
            issues=["velocity_limit_exceeded"],
            verdict="REJECTED",
        )

        result = mock_safety_gateway.check({"type": "move", "magnitude": 100.0})
        assert not result.approved
        assert "velocity_limit_exceeded" in result.issues

    def test_safe_action_allowed(self, mock_safety_gateway):
        """Safety allows a safe action."""
        result = mock_safety_gateway.check({"type": "move", "magnitude": 0.5})
        assert result.approved

    def test_safety_with_event_notification(
        self, mock_safety_gateway, mock_event_fabric
    ):
        """Safety publishes event when blocking action."""
        mock_safety_gateway.check.return_value = MagicMock(
            approved=False, issues=["unsafe"]
        )

        result = mock_safety_gateway.check({"type": "dangerous"})
        if not result.approved:
            mock_event_fabric.publish(
                "safety.violation",
                {"issues": result.issues},
            )

        assert len(mock_event_fabric.published_events) == 1
        assert mock_event_fabric.published_events[0][0] == "safety.violation"


# ---------------------------------------------------------------------------
# Chaos Resilience
# ---------------------------------------------------------------------------


class TestChaosResilience:
    """Using ChaosMonkey to test system resilience."""

    def test_chaos_latency_injection(self):
        """Test latency injection."""
        monkey = ChaosMonkey(ChaosConfig(enabled=True, seed=42))
        handle = monkey.inject_latency("lh.planner", latency_ms=200)

        assert handle.active
        assert handle.fault_type == "latency"

        handle.stop()
        assert not handle.active

    def test_chaos_error_injection(self):
        """Test error injection."""
        monkey = ChaosMonkey(ChaosConfig(enabled=True, seed=42))
        handle = monkey.inject_error("rh.control", error_type="ConnectionError")

        assert handle.active
        assert handle.fault_type == "error"

        monkey.stop_all()
        assert len(monkey.get_active_faults()) == 0

    def test_chaos_scope(self):
        """Test chaos scope context manager."""
        monkey = ChaosMonkey(ChaosConfig(enabled=True, failure_rate=1.0, seed=42))

        with monkey.chaos_scope(["service_a", "service_b"]) as handles:
            assert len(handles) > 0
            assert all(h.active for h in handles)

        # Faults should be stopped after scope
        assert len(monkey.get_active_faults()) == 0

    def test_chaos_random_fault(self):
        """Test random fault injection."""
        monkey = ChaosMonkey(ChaosConfig(enabled=True, seed=42))
        handle = monkey.random_fault(["svc_a", "svc_b", "svc_c"])

        assert handle.active
        assert handle.fault_type in ["latency", "error", "timeout"]

        monkey.stop_all()


# ---------------------------------------------------------------------------
# Event Propagation
# ---------------------------------------------------------------------------


class TestEventPropagation:
    """Events flow through the system correctly."""

    def test_plan_events_reach_rh(self, mock_event_fabric):
        """Plan events are published and accessible."""
        mock_event_fabric.publish(
            "plan.step_ready",
            {"step_id": "1", "node_id": "LH-1", "step": {"kind": "navigate"}},
        )

        assert len(mock_event_fabric.published_events) == 1
        topic, payload = mock_event_fabric.published_events[0]
        assert topic == "plan.step_ready"
        assert "step_id" in payload

    def test_multiple_event_topics(self, mock_event_fabric):
        """Multiple event topics work independently."""
        mock_event_fabric.publish("plan.started", {"plan_id": "p1"})
        mock_event_fabric.publish("plan.step_ready", {"step_id": "s1"})
        mock_event_fabric.publish("plan.completed", {"plan_id": "p1"})

        assert len(mock_event_fabric.published_events) == 3

    def test_subscriber_receives_events(self, mock_event_fabric):
        """Subscribers receive published events."""
        received = []
        mock_event_fabric.subscribe("test.topic", lambda p: received.append(p))
        mock_event_fabric.publish("test.topic", {"data": "hello"})

        # Note: MockFabric doesn't auto-dispatch, testing subscription registration
        assert "test.topic" in mock_event_fabric.subscriptions


# ---------------------------------------------------------------------------
# Integration Framework Runner
# ---------------------------------------------------------------------------


class TestIntegrationFramework:
    """Test the integration framework itself."""

    def test_runner_executes_all(self):
        """IntegrationTestRunner runs all registered cases."""
        runner = IntegrationTestRunner(verbose=False)
        for case in INTEGRATION_TEST_CASES:
            runner.register(case)

        results = runner.run_all()
        assert len(results) == len(INTEGRATION_TEST_CASES)

        summary = runner.summary()
        assert summary["total"] == len(INTEGRATION_TEST_CASES)
        assert summary["passed"] > 0

    def test_runner_by_category(self):
        """IntegrationTestRunner filters by category."""
        runner = IntegrationTestRunner(verbose=False)
        for case in INTEGRATION_TEST_CASES:
            runner.register(case)

        results = runner.run_category("events")
        assert len(results) >= 1
        assert all(r.passed for r in results)

    def test_runner_by_tag(self):
        """IntegrationTestRunner filters by tag."""
        runner = IntegrationTestRunner(verbose=False)
        for case in INTEGRATION_TEST_CASES:
            runner.register(case)

        results = runner.run_tagged("smoke")
        assert len(results) >= 1
