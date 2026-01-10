"""
Unit tests for the LH PlanService module.
"""

import pytest
from unittest.mock import MagicMock, patch
import grpc

from agi.lh.planner import Planner
from agi.lh.plan_service import PlanService, LHPlanServiceConfig
from agi.proto_gen import plan_pb2


class TestPlanServiceInit:
    """Tests for PlanService initialization."""

    def test_plan_service_initializes(
        self, planner, mock_memory, mock_safety, mock_metacog, mock_fabric
    ):
        """PlanService should initialize with all dependencies."""
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        assert service is not None

    def test_plan_service_accepts_config(
        self, planner, mock_memory, mock_safety, mock_metacog, mock_fabric
    ):
        """PlanService should accept custom config."""
        config = LHPlanServiceConfig(
            enable_safety=False,
            enable_metacognition=False,
            node_id="TestLH",
        )

        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
            config=config,
        )

        assert service._cfg.enable_safety is False
        assert service._cfg.enable_metacognition is False


class TestPlanServicePipeline:
    """Tests for the Plan RPC pipeline."""

    @pytest.fixture
    def plan_service(
        self, planner, mock_memory, mock_safety, mock_metacog, mock_fabric
    ):
        """Create PlanService with mocked dependencies."""
        return PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

    @pytest.fixture
    def mock_context(self):
        """Create a mock gRPC context."""
        ctx = MagicMock(spec=grpc.ServicerContext)
        ctx.abort = MagicMock(side_effect=grpc.RpcError("aborted"))
        return ctx

    def test_plan_returns_response(
        self, plan_service, sample_plan_request, mock_context
    ):
        """Plan RPC should return a valid PlanResponse."""
        response = plan_service.Plan(sample_plan_request, mock_context)

        assert isinstance(response, plan_pb2.PlanResponse)
        assert response.plan_id is not None

    def test_plan_publishes_steps_to_fabric(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_metacog,
        mock_fabric,
        sample_plan_request,
        mock_context,
    ):
        """Plan RPC should publish steps to EventFabric."""
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        service.Plan(sample_plan_request, mock_context)

        # Check that events were published
        step_events = [
            e for e in mock_fabric.published_events if e[0] == "plan.step_ready"
        ]
        assert len(step_events) > 0

    def test_plan_calls_memory_enrichment(
        self, plan_service, sample_plan_request, mock_context
    ):
        """Plan RPC should call memory.enrich_request."""
        with patch.object(
            plan_service._memory, "enrich_request", return_value=sample_plan_request
        ) as mock_enrich:
            plan_service.Plan(sample_plan_request, mock_context)
            mock_enrich.assert_called_once()

    def test_plan_calls_safety_check(
        self, plan_service, sample_plan_request, mock_context
    ):
        """Plan RPC should call safety.check_plan."""
        with patch.object(plan_service._safety, "check_plan") as mock_check:
            mock_check.return_value = MagicMock(approved=True)
            plan_service.Plan(sample_plan_request, mock_context)
            mock_check.assert_called_once()

    def test_plan_calls_metacognition(
        self, plan_service, sample_plan_request, mock_context
    ):
        """Plan RPC should call metacog.review_plan."""
        with patch.object(plan_service._meta, "review_plan") as mock_review:
            mock_review.return_value = MagicMock(decision="ACCEPT")
            plan_service.Plan(sample_plan_request, mock_context)
            mock_review.assert_called_once()


class TestPlanServiceSafetyGating:
    """Tests for safety gating behavior."""

    def test_safety_rejection_aborts(
        self,
        planner,
        mock_memory,
        mock_safety_reject,
        mock_metacog,
        mock_fabric,
        sample_plan_request,
    ):
        """Plan should abort when safety rejects."""
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety_reject,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        mock_context = MagicMock(spec=grpc.ServicerContext)
        mock_context.abort = MagicMock(side_effect=grpc.RpcError("aborted"))

        with pytest.raises(grpc.RpcError):
            service.Plan(sample_plan_request, mock_context)

    def test_safety_disabled_skips_check(
        self,
        planner,
        mock_memory,
        mock_safety_reject,
        mock_metacog,
        mock_fabric,
        sample_plan_request,
    ):
        """Plan should skip safety when disabled."""
        config = LHPlanServiceConfig(enable_safety=False)
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety_reject,
            metacog=mock_metacog,
            fabric=mock_fabric,
            config=config,
        )

        mock_context = MagicMock(spec=grpc.ServicerContext)

        # Should not raise even with rejecting safety client
        response = service.Plan(sample_plan_request, mock_context)
        assert response is not None


class TestPlanServiceMetacognition:
    """Tests for metacognition behavior."""

    def test_metacog_reject_aborts(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_metacog_reject,
        mock_fabric,
        sample_plan_request,
    ):
        """Plan should abort when metacognition rejects."""
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog_reject,
            fabric=mock_fabric,
        )

        mock_context = MagicMock(spec=grpc.ServicerContext)
        mock_context.abort = MagicMock(side_effect=grpc.RpcError("aborted"))

        with pytest.raises(grpc.RpcError):
            service.Plan(sample_plan_request, mock_context)

    def test_metacog_disabled_skips_review(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_metacog_reject,
        mock_fabric,
        sample_plan_request,
    ):
        """Plan should skip metacognition when disabled."""
        config = LHPlanServiceConfig(enable_metacognition=False)
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog_reject,
            fabric=mock_fabric,
            config=config,
        )

        mock_context = MagicMock(spec=grpc.ServicerContext)

        # Should not raise even with rejecting metacog client
        response = service.Plan(sample_plan_request, mock_context)
        assert response is not None


class TestPlanServiceExceptionHandling:
    """Tests for exception handling in the pipeline."""

    def test_memory_enrichment_failure_uses_raw_request(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_metacog,
        mock_fabric,
        sample_plan_request,
    ):
        """Plan should continue with raw request if memory enrichment fails."""
        # Make enrich_request raise an exception
        mock_memory.enrich_request = MagicMock(side_effect=Exception("Memory error"))

        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        mock_context = MagicMock(spec=grpc.ServicerContext)

        # Should not raise - falls back to raw request
        response = service.Plan(sample_plan_request, mock_context)
        assert response is not None

    def test_metacog_revise_triggers_revision(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_fabric,
        sample_plan_request,
    ):
        """Plan should call revise_plan when metacog returns REVISE."""
        mock_metacog = MagicMock()
        mock_metacog.review_plan = MagicMock(
            return_value=MagicMock(decision="REVISE", issues=["need_revision"])
        )
        mock_metacog.revise_plan = MagicMock(return_value=MagicMock(steps=[]))

        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        mock_context = MagicMock(spec=grpc.ServicerContext)
        service.Plan(sample_plan_request, mock_context)

        mock_metacog.revise_plan.assert_called_once()


class TestPlanServiceHelpers:
    """Tests for internal helper methods."""

    def test_publish_plan_steps_handles_empty_steps(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_metacog,
        mock_fabric,
    ):
        """_publish_plan_steps should handle plan with no steps."""
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        # Create a plan with no steps
        empty_plan = MagicMock()
        empty_plan.steps = []

        # Should not raise
        service._publish_plan_steps(empty_plan)

        # Should only have completion event (no step events)
        step_events = [
            e for e in mock_fabric.published_events if e[0] == "plan.step_ready"
        ]
        assert len(step_events) == 0

    def test_publish_plan_steps_handles_none_steps(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_metacog,
        mock_fabric,
    ):
        """_publish_plan_steps should handle plan with None steps."""
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        # Create a plan with None steps
        plan_no_steps = MagicMock(spec=[])  # No 'steps' attribute

        # Should not raise
        service._publish_plan_steps(plan_no_steps)

    def test_serialize_step_handles_dict(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_metacog,
        mock_fabric,
    ):
        """_serialize_step should pass through dict steps."""
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        dict_step = {"action": "move", "params": {"x": 1, "y": 2}}
        result = service._serialize_step(dict_step)

        assert result is dict_step

    def test_serialize_step_handles_object_with_attrs(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_metacog,
        mock_fabric,
    ):
        """_serialize_step should extract known attributes from objects."""
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        class StepObj:
            action = "grasp"
            params = {"force": 10}
            preconditions = ["object_visible"]
            postconditions = ["object_held"]

        step = StepObj()
        result = service._serialize_step(step)

        assert result["action"] == "grasp"
        assert result["params"] == {"force": 10}
        assert result["preconditions"] == ["object_visible"]
        assert result["postconditions"] == ["object_held"]

    def test_fabric_publish_error_handled(
        self,
        planner,
        mock_memory,
        mock_safety,
        mock_metacog,
        sample_plan_request,
    ):
        """Fabric publish errors should be logged but not raise."""
        # Create a fabric that raises on publish
        mock_fabric = MagicMock()
        mock_fabric.publish = MagicMock(side_effect=Exception("Fabric error"))

        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )

        mock_context = MagicMock(spec=grpc.ServicerContext)

        # Should not raise - errors are logged
        response = service.Plan(sample_plan_request, mock_context)
        assert response is not None
