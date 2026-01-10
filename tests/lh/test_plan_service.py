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
    
    def test_plan_service_initializes(self, planner, mock_memory, mock_safety, 
                                       mock_metacog, mock_fabric):
        """PlanService should initialize with all dependencies."""
        service = PlanService(
            planner=planner,
            memory=mock_memory,
            safety=mock_safety,
            metacog=mock_metacog,
            fabric=mock_fabric,
        )
        
        assert service is not None
    
    def test_plan_service_accepts_config(self, planner, mock_memory, mock_safety,
                                          mock_metacog, mock_fabric):
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
    def plan_service(self, planner, mock_memory, mock_safety, mock_metacog, mock_fabric):
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
    
    def test_plan_returns_response(self, plan_service, sample_plan_request, mock_context):
        """Plan RPC should return a valid PlanResponse."""
        response = plan_service.Plan(sample_plan_request, mock_context)
        
        assert isinstance(response, plan_pb2.PlanResponse)
        assert response.plan_id is not None
    
    def test_plan_publishes_steps_to_fabric(self, planner, mock_memory, mock_safety,
                                             mock_metacog, mock_fabric, 
                                             sample_plan_request, mock_context):
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
        step_events = [e for e in mock_fabric.published_events 
                       if e[0] == "plan.step_ready"]
        assert len(step_events) > 0
    
    def test_plan_calls_memory_enrichment(self, plan_service, sample_plan_request, 
                                           mock_context):
        """Plan RPC should call memory.enrich_request."""
        with patch.object(plan_service._memory, 'enrich_request', 
                         return_value=sample_plan_request) as mock_enrich:
            plan_service.Plan(sample_plan_request, mock_context)
            mock_enrich.assert_called_once()
    
    def test_plan_calls_safety_check(self, plan_service, sample_plan_request,
                                      mock_context):
        """Plan RPC should call safety.check_plan."""
        with patch.object(plan_service._safety, 'check_plan') as mock_check:
            mock_check.return_value = MagicMock(approved=True)
            plan_service.Plan(sample_plan_request, mock_context)
            mock_check.assert_called_once()
    
    def test_plan_calls_metacognition(self, plan_service, sample_plan_request,
                                       mock_context):
        """Plan RPC should call metacog.review_plan."""
        with patch.object(plan_service._meta, 'review_plan') as mock_review:
            mock_review.return_value = MagicMock(decision="ACCEPT")
            plan_service.Plan(sample_plan_request, mock_context)
            mock_review.assert_called_once()


class TestPlanServiceSafetyGating:
    """Tests for safety gating behavior."""
    
    def test_safety_rejection_aborts(self, planner, mock_memory, mock_safety_reject,
                                      mock_metacog, mock_fabric, sample_plan_request):
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
    
    def test_safety_disabled_skips_check(self, planner, mock_memory, mock_safety_reject,
                                          mock_metacog, mock_fabric, sample_plan_request):
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
    
    def test_metacog_reject_aborts(self, planner, mock_memory, mock_safety,
                                    mock_metacog_reject, mock_fabric, sample_plan_request):
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
    
    def test_metacog_disabled_skips_review(self, planner, mock_memory, mock_safety,
                                            mock_metacog_reject, mock_fabric, 
                                            sample_plan_request):
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
