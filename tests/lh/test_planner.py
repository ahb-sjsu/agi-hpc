"""
Unit tests for the LH Planner module.
"""
import pytest
from agi.lh.planner import Planner, PlanGraph, PlanStep


class TestPlannerGeneration:
    """Tests for plan generation."""
    
    def test_planner_generates_valid_plan_graph(self, planner, sample_plan_request):
        """Planner should generate a valid PlanGraph from a request."""
        plan = planner.generate_plan(sample_plan_request)
        
        assert isinstance(plan, PlanGraph)
        assert plan.plan_id is not None
        assert len(plan.plan_id) > 0
        assert plan.goal_text is not None
    
    def test_planner_creates_hierarchical_structure(self, planner, sample_plan_request):
        """Planner should create mission -> subgoal -> step hierarchy."""
        plan = planner.generate_plan(sample_plan_request)
        
        # Should have at least one step at each level
        levels = {step.level for step in plan.steps}
        assert 0 in levels, "Should have level 0 (mission)"
        assert 1 in levels, "Should have level 1 (subgoals)"
        assert 2 in levels, "Should have level 2 (actionable steps)"
    
    def test_planner_mission_step_exists(self, planner, sample_plan_request):
        """Planner should create exactly one mission step."""
        plan = planner.generate_plan(sample_plan_request)
        
        mission_steps = [s for s in plan.steps if s.kind == "mission"]
        assert len(mission_steps) == 1
        assert mission_steps[0].level == 0
    
    def test_planner_subgoals_have_parent(self, planner, sample_plan_request):
        """Subgoals should have parent_id pointing to mission."""
        plan = planner.generate_plan(sample_plan_request)
        
        mission = next(s for s in plan.steps if s.kind == "mission")
        subgoals = [s for s in plan.steps if s.kind == "subgoal"]
        
        for subgoal in subgoals:
            assert subgoal.parent_id == mission.step_id
    
    def test_planner_extracts_goal_text(self, planner, sample_plan_request):
        """Planner should extract goal text from task description."""
        plan = planner.generate_plan(sample_plan_request)
        
        assert "Navigate to the red cube" in plan.goal_text
    
    def test_planner_handles_empty_request(self, planner, empty_plan_request):
        """Planner should handle empty requests gracefully."""
        plan = planner.generate_plan(empty_plan_request)
        
        assert isinstance(plan, PlanGraph)
        assert len(plan.steps) > 0  # Should still create scaffold


class TestPlanStep:
    """Tests for PlanStep dataclass."""
    
    def test_plan_step_has_required_fields(self, planner, sample_plan_request):
        """Each PlanStep should have required fields populated."""
        plan = planner.generate_plan(sample_plan_request)
        
        for step in plan.steps:
            assert step.step_id is not None
            assert step.index >= 0
            assert step.level >= 0
            assert step.kind is not None
            assert step.description is not None
    
    def test_plan_steps_have_unique_ids(self, planner, sample_plan_request):
        """All step_ids should be unique."""
        plan = planner.generate_plan(sample_plan_request)
        
        ids = [step.step_id for step in plan.steps]
        assert len(ids) == len(set(ids)), "Step IDs should be unique"


class TestPlanGraph:
    """Tests for PlanGraph dataclass."""
    
    def test_plan_graph_metadata(self, planner, sample_plan_request):
        """PlanGraph should capture metadata from request."""
        plan = planner.generate_plan(sample_plan_request)
        
        assert "task_type" in plan.metadata
        assert plan.metadata["task_type"] == "manipulation"
    
    def test_plan_graph_add_step(self):
        """PlanGraph.add_step should append steps."""
        graph = PlanGraph(plan_id="test", goal_text="test goal")
        step = PlanStep(
            step_id="s1", index=0, level=0,
            kind="test", description="test step"
        )
        
        graph.add_step(step)
        
        assert len(graph.steps) == 1
        assert graph.steps[0] == step
