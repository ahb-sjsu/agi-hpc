"""
Integration tests for LH pipeline.

These tests verify the full LH pipeline with mocked downstream services.
"""

import pytest
from unittest.mock import MagicMock, patch
import grpc

from agi.lh.planner import Planner, PlanGraph
from agi.lh.plan_service import PlanService, LHPlanServiceConfig
from agi.lh.memory_client import MemoryClient
from agi.lh.safety_client import SafetyClient, SafetyResult
from agi.lh.metacog_client import MetacognitionClient, MetaReviewResult
from agi.proto_gen import plan_pb2

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_grpc_context():
    """Create a mock gRPC context."""
    ctx = MagicMock(spec=grpc.ServicerContext)
    ctx.abort = MagicMock(side_effect=grpc.RpcError("aborted"))
    return ctx


@pytest.fixture
def sample_request():
    """Create a sample PlanRequest."""
    return plan_pb2.PlanRequest(
        task=plan_pb2.Task(
            goal_id="integration-test-001",
            description="Navigate to waypoint A and pick up object",
            task_type="manipulation",
        ),
        environment=plan_pb2.EnvironmentDescriptor(
            scenario_id="warehouse-v1",
        ),
    )


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

        def close(self):
            pass

    return MockFabric()


# ---------------------------------------------------------------------------
# 3.1 LH ↔ Safety Integration Tests
# ---------------------------------------------------------------------------


class TestLHSafetyIntegration:
    """Integration tests for LH-Safety service interaction."""

    def test_safety_approval_allows_plan(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """When safety approves, plan should proceed to completion."""
        # Setup: Safety client that approves
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(
            return_value=SafetyResult(approved=True, issues=[])
        )

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            return_value=MetaReviewResult(decision="ACCEPT")
        )

        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        # Execute
        response = service.Plan(sample_request, mock_grpc_context)

        # Verify
        assert response.plan_id is not None
        mock_safety.check_plan.assert_called_once()
        # Events should be published
        assert len(mock_fabric.published_events) > 0

    def test_safety_rejection_blocks_plan(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """When safety rejects, plan should be aborted."""
        # Setup: Safety client that rejects
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(
            return_value=SafetyResult(approved=False, issues=["unsafe_operation"])
        )

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)

        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        # Execute & Verify
        with pytest.raises(grpc.RpcError):
            service.Plan(sample_request, mock_grpc_context)

        mock_safety.check_plan.assert_called_once()
        # Context.abort should have been called
        mock_grpc_context.abort.assert_called()

    def test_safety_disabled_skips_check(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """When safety is disabled, check should be skipped."""
        # Setup: Safety client that would reject, but safety is disabled
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(
            return_value=SafetyResult(approved=False, issues=["would_reject"])
        )

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            return_value=MetaReviewResult(decision="ACCEPT")
        )

        config = LHPlanServiceConfig(enable_safety=False)
        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
            config=config,
        )

        # Execute
        response = service.Plan(sample_request, mock_grpc_context)

        # Verify - safety check should NOT be called
        assert response.plan_id is not None
        mock_safety.check_plan.assert_not_called()


# ---------------------------------------------------------------------------
# 3.2 LH ↔ Metacognition Integration Tests
# ---------------------------------------------------------------------------


class TestLHMetacognitionIntegration:
    """Integration tests for LH-Metacognition service interaction."""

    def test_metacog_accept_proceeds(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """When metacognition accepts, plan should proceed."""
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(return_value=SafetyResult(approved=True))

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            return_value=MetaReviewResult(decision="ACCEPT", confidence=0.95)
        )

        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        response = service.Plan(sample_request, mock_grpc_context)

        assert response.plan_id is not None
        mock_metacog.review_plan.assert_called_once()

    def test_metacog_reject_aborts_plan(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """When metacognition rejects, plan should be aborted."""
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(return_value=SafetyResult(approved=True))

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            return_value=MetaReviewResult(decision="REJECT", issues=["critical_flaw"])
        )

        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        with pytest.raises(grpc.RpcError):
            service.Plan(sample_request, mock_grpc_context)

        mock_grpc_context.abort.assert_called()

    def test_metacog_revise_triggers_revision(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """When metacognition requests revision, revise_plan should be called."""
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(return_value=SafetyResult(approved=True))

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            return_value=MetaReviewResult(
                decision="REVISE", issues=["needs_improvement"]
            )
        )
        # revise_plan returns a modified plan
        mock_metacog.revise_plan = MagicMock(
            return_value=PlanGraph(plan_id="revised-001", goal_text="revised", steps=[])
        )

        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        response = service.Plan(sample_request, mock_grpc_context)

        assert response.plan_id is not None
        mock_metacog.review_plan.assert_called_once()
        mock_metacog.revise_plan.assert_called_once()

    def test_metacog_disabled_skips_review(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """When metacognition is disabled, review should be skipped."""
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(return_value=SafetyResult(approved=True))

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            return_value=MetaReviewResult(decision="REJECT")
        )

        config = LHPlanServiceConfig(enable_metacognition=False)
        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
            config=config,
        )

        response = service.Plan(sample_request, mock_grpc_context)

        assert response.plan_id is not None
        mock_metacog.review_plan.assert_not_called()


# ---------------------------------------------------------------------------
# 3.3 LH ↔ RH EventFabric Integration Tests
# ---------------------------------------------------------------------------


class TestLHEventFabricIntegration:
    """Integration tests for LH publishing events to RH via EventFabric."""

    def test_plan_publishes_step_ready_events(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """LH should publish plan.step_ready events for each step."""
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(return_value=SafetyResult(approved=True))

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            return_value=MetaReviewResult(decision="ACCEPT")
        )

        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        service.Plan(sample_request, mock_grpc_context)

        # Check step_ready events were published
        step_events = [
            e for e in mock_fabric.published_events if e[0] == "plan.step_ready"
        ]
        assert len(step_events) > 0

        # Each event should have required fields
        for topic, payload in step_events:
            assert "node_id" in payload
            assert "index" in payload
            assert "step" in payload

    def test_plan_publishes_completion_event(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """LH should publish plan.completed event after all steps."""
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(return_value=SafetyResult(approved=True))

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            return_value=MetaReviewResult(decision="ACCEPT")
        )

        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        service.Plan(sample_request, mock_grpc_context)

        # Check completion event was published
        completion_events = [
            e for e in mock_fabric.published_events if e[0] == "plan.completed"
        ]
        assert len(completion_events) == 1

        topic, payload = completion_events[0]
        assert "node_id" in payload
        assert "num_steps" in payload

    def test_event_payload_structure(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """Event payloads should have correct structure for RH consumption."""
        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(return_value=SafetyResult(approved=True))

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(return_value=sample_request)

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            return_value=MetaReviewResult(decision="ACCEPT")
        )

        config = LHPlanServiceConfig(node_id="LH-integration-test")
        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
            config=config,
        )

        service.Plan(sample_request, mock_grpc_context)

        # Verify node_id is correctly set in events
        for topic, payload in mock_fabric.published_events:
            assert payload["node_id"] == "LH-integration-test"


# ---------------------------------------------------------------------------
# Full Pipeline Integration Test
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end integration test of the full LH pipeline."""

    def test_full_pipeline_success(
        self, sample_request, mock_grpc_context, mock_fabric
    ):
        """Test complete pipeline: request → memory → plan → safety → metacog → events → response."""
        # Track call order
        call_order = []

        mock_memory = MagicMock(spec=MemoryClient)
        mock_memory.enrich_request = MagicMock(
            side_effect=lambda r: (call_order.append("memory"), r)[1]
        )

        mock_safety = MagicMock(spec=SafetyClient)
        mock_safety.check_plan = MagicMock(
            side_effect=lambda p: (
                call_order.append("safety"),
                SafetyResult(approved=True),
            )[1]
        )

        mock_metacog = MagicMock(spec=MetacognitionClient)
        mock_metacog.review_plan = MagicMock(
            side_effect=lambda p: (
                call_order.append("metacog"),
                MetaReviewResult(decision="ACCEPT"),
            )[1]
        )

        service = PlanService(
            planner=Planner(),
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        response = service.Plan(sample_request, mock_grpc_context)

        # Verify correct order of operations
        assert call_order == ["memory", "safety", "metacog"]

        # Verify response
        assert response.plan_id is not None
        assert len(response.steps) > 0

        # Verify events
        assert len(mock_fabric.published_events) > 0
