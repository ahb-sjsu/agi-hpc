# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Hierarchical Planning Engine for the Left Hemisphere (LH).

This module implements a minimal but architecture-aligned planning core:

    PlanRequest  ->  PlanGraph (mission → subgoals → actionable steps)

The representation matches the whitepaper's description of the LH
hierarchical planning engine:

    • multi-level plans: mission → subgoals → actionable steps → tool/env commands
    • explicit graph structure (nodes = steps, edges = dependencies)
    • annotations for required world-model checks and safety gates
    • integration points for memory and skills retrieval

At this stage, the planner is intentionally simple and deterministic so
that the rest of the system (safety, metacognition, RH, memory) can be
developed and tested. Later, this module can be upgraded to call a large
language model and richer search routines without changing the public
interface.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agi.proto_gen import plan_pb2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal plan representation
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """
    A single node in the plan graph.

    The graph itself is flattened (steps + dependency edges) rather than
    deeply nested; multi-level hierarchy is represented by:
        • level: 0 = mission, 1 = subgoal, 2+ = increasingly concrete
        • dependencies: step_ids that must complete before this one
    """

    step_id: str
    index: int
    level: int
    kind: str  # e.g. "mission", "subgoal", "skill", "control"
    description: str

    # Optional semantic metadata
    parent_id: Optional[str] = None
    requires_simulation: bool = True
    safety_tags: List[str] = field(default_factory=list)
    tool_id: Optional[str] = None
    params: Dict[str, object] = field(default_factory=dict)


@dataclass
class PlanGraph:
    """
    A full plan for a single goal.

    This is the internal representation that Safety and Metacognition
    will inspect. PlanService is responsible for converting it into the
    protobuf PlanResponse.
    """

    plan_id: str
    goal_text: str
    steps: List[PlanStep] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def add_step(self, step: PlanStep) -> None:
        self.steps.append(step)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """
    Hierarchical planner used by LH PlanService.

    Responsibilities:

        1. Interpret PlanRequest into a textual goal representation.
        2. Optionally query memory/skills to inform the plan.
        3. Construct a multi-level PlanGraph (mission → subgoals → steps).
        4. Mark which steps require RH simulation and safety attention.

    This initial implementation is deliberately simple: it creates a
    canonical "scaffold" plan suitable for end-to-end integration
    testing. Later phases can plug in a language model and richer search
    without changing the public interface.
    """

    def __init__(self) -> None:
        logger.info("[LH][Planner] Initialized minimal hierarchical planner")

    # Public API used by PlanService -------------------------------------

    def generate_plan(self, request: plan_pb2.PlanRequest) -> PlanGraph:
        """
        Entry point called by PlanService.

        The method is synchronous and deterministic; any long-running or
        stochastic behavior should be introduced later (e.g., via LLM
        calls, search, or RH simulation loops).
        """
        goal_text = self._extract_goal_text(request)
        plan_id = self._derive_plan_id(request)

        graph = PlanGraph(
            plan_id=plan_id,
            goal_text=goal_text,
            metadata={
                "task_type": getattr(getattr(request, "task", None), "task_type", ""),
                "scenario_id": getattr(
                    getattr(request, "environment", None),
                    "scenario_id",
                    "",
                ),
            },
        )

        logger.info(
            "[LH][Planner] Generating plan plan_id=%s goal=%s",
            plan_id,
            goal_text,
        )

        self._populate_plan_graph(graph)

        logger.info(
            "[LH][Planner] Generated plan plan_id=%s steps=%d",
            plan_id,
            len(graph.steps),
        )
        return graph

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _extract_goal_text(self, request: plan_pb2.PlanRequest) -> str:
        """
        Best-effort extraction of a human-readable goal from PlanRequest.

        This assumes (but does not require) a nested "task" message with
        fields like description / goal_id, as sketched in the architecture.
        """
        task = getattr(request, "task", None)
        env = getattr(request, "environment", None)

        parts: List[str] = []

        if task is not None:
            desc = getattr(task, "description", "")
            if desc:
                parts.append(desc)
            task_type = getattr(task, "task_type", "")
            if task_type:
                parts.append(f"(type: {task_type})")

        if env is not None:
            scenario_id = getattr(env, "scenario_id", "")
            if scenario_id:
                parts.append(f"(scenario: {scenario_id})")

        if not parts:
            return "unspecified-goal"

        return " ".join(parts)

    def _derive_plan_id(self, request: plan_pb2.PlanRequest) -> str:
        """
        Try to derive a stable plan_id from the request; fall back to UUID.
        """
        task = getattr(request, "task", None)
        goal_id = getattr(task, "goal_id", "") if task is not None else ""
        if goal_id:
            return f"plan_{goal_id}"
        return f"plan_{uuid.uuid4().hex[:8]}"

    def _populate_plan_graph(self, graph: PlanGraph) -> None:
        """
        Create a simple hierarchical scaffold:

            Level 0: mission
            Level 1: subgoals (analyze → plan → validate → execute)
            Level 2: concrete steps

        This structure is intentionally generic but matches the
        architecture's described planning pipeline (query memory,
        sketch plan, safety, simulation, metacognition, execution).
        """
        mission_id = self._add_mission_step(graph)
        subgoal_ids = self._add_subgoals(graph, mission_id)
        self._add_actionable_steps(graph, subgoal_ids)

    def _add_mission_step(self, graph: PlanGraph) -> str:
        step_id = self._new_step_id("mission")
        mission = PlanStep(
            step_id=step_id,
            index=len(graph.steps),
            level=0,
            kind="mission",
            description=f"Achieve goal: {graph.goal_text}",
            parent_id=None,
            requires_simulation=False,
            safety_tags=["high_level"],
        )
        graph.add_step(mission)
        return step_id

    def _add_subgoals(self, graph: PlanGraph, mission_id: str) -> Dict[str, str]:
        """
        Add canonical subgoals as children of the mission node.
        Returns a mapping name -> step_id for downstream wiring.
        """
        mapping: Dict[str, str] = {}

        def _add_subgoal(name: str, desc: str) -> str:
            sid = self._new_step_id(name)
            step = PlanStep(
                step_id=sid,
                index=len(graph.steps),
                level=1,
                kind="subgoal",
                description=desc,
                parent_id=mission_id,
                requires_simulation=False,
                safety_tags=["planning"],
                params={"subgoal_name": name},
            )
            graph.add_step(step)
            mapping[name] = sid
            return sid

        _add_subgoal("analyze_goal", "Interpret goal and constraints from request")
        _add_subgoal(
            "retrieve_knowledge",
            "Query semantic / episodic / procedural memory for relevant info",
        )
        _add_subgoal(
            "draft_plan",
            "Use reasoning model + skills to sketch candidate plan graph",
        )
        _add_subgoal(
            "validate_plan",
            "Run safety checks and request simulations for risky steps",
        )
        _add_subgoal(
            "finalize_plan",
            "Select a safe, high-confidence plan and prepare for execution",
        )

        return mapping

    def _add_actionable_steps(
        self,
        graph: PlanGraph,
        subgoal_ids: Dict[str, str],
    ) -> None:
        """
        Add level-2 actionable steps tied to subgoals.

        These are still abstract (they talk about querying memory,
        calling safety, requesting simulation) but can be mapped to actual
        tool calls or API requests later.
        """
        # Analyze goal ---------------------------------------------------
        self._add_step(
            graph,
            parent=subgoal_ids["analyze_goal"],
            level=2,
            kind="analysis",
            description="Parse natural language goal into structured task graph",
            requires_simulation=False,
            safety_tags=["planning"],
            tool_id="lh.task_interpreter",
        )

        # Retrieve knowledge ---------------------------------------------
        self._add_step(
            graph,
            parent=subgoal_ids["retrieve_knowledge"],
            level=2,
            kind="memory_query",
            description="Query semantic memory for relevant facts, tools, and rules",
            requires_simulation=False,
            safety_tags=["reads_memory"],
            tool_id="memory.semantic.query",
        )
        self._add_step(
            graph,
            parent=subgoal_ids["retrieve_knowledge"],
            level=2,
            kind="memory_query",
            description="Lookup similar past episodes in episodic memory",
            requires_simulation=False,
            safety_tags=["reads_memory"],
            tool_id="memory.episodic.lookup",
        )

        # Draft plan -----------------------------------------------------
        self._add_step(
            graph,
            parent=subgoal_ids["draft_plan"],
            level=2,
            kind="llm_planning",
            description="Use LLM + skills to draft multi-step plan",
            requires_simulation=False,
            safety_tags=["planning"],
            tool_id="lh.llm.plan",
        )

        # Validate plan (safety + RH sims) -------------------------------
        self._add_step(
            graph,
            parent=subgoal_ids["validate_plan"],
            level=2,
            kind="safety_precheck",
            description="Run pre-action safety checks on candidate plan",
            requires_simulation=False,
            safety_tags=["safety_pre_action"],
            tool_id="safety.pre_action.check_plan",
        )
        self._add_step(
            graph,
            parent=subgoal_ids["validate_plan"],
            level=2,
            kind="simulation_request",
            description="Request RH simulations for environment-changing steps",
            requires_simulation=True,
            safety_tags=["requires_simulation"],
            tool_id="rh.simulation.request",
        )

        # Finalize plan --------------------------------------------------
        self._add_step(
            graph,
            parent=subgoal_ids["finalize_plan"],
            level=2,
            kind="meta_review",
            description="Request metacognitive review and integrate feedback",
            requires_simulation=False,
            safety_tags=["metacognition"],
            tool_id="meta.review_plan",
        )
        self._add_step(
            graph,
            parent=subgoal_ids["finalize_plan"],
            level=2,
            kind="publish_steps",
            description="Publish plan.step_ready events for approved steps",
            requires_simulation=False,
            safety_tags=["execution"],
            tool_id="lh.events.publish_plan_steps",
        )

    def _add_step(
        self,
        graph: PlanGraph,
        parent: str,
        level: int,
        kind: str,
        description: str,
        requires_simulation: bool,
        safety_tags: Optional[List[str]] = None,
        tool_id: Optional[str] = None,
    ) -> str:
        step_id = self._new_step_id(kind)
        step = PlanStep(
            step_id=step_id,
            index=len(graph.steps),
            level=level,
            kind=kind,
            description=description,
            parent_id=parent,
            requires_simulation=requires_simulation,
            safety_tags=list(safety_tags or []),
            tool_id=tool_id,
        )
        graph.add_step(step)
        return step_id

    def _new_step_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
