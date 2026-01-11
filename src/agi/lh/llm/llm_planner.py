# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
LLM-Powered Hierarchical Planner for the Left Hemisphere.

This module extends the basic deterministic planner with LLM-based
plan generation. It uses the LLM adapters to generate contextual,
goal-specific hierarchical plans while falling back to the
deterministic scaffold on LLM failures.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from agi.proto_gen import plan_pb2
from agi.lh.planner import Planner, PlanGraph, PlanStep
from agi.lh.llm.adapter import LLMAdapter, LLMConfig, LLMResponse, create_adapter
from agi.lh.memory_client import PlanningContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a hierarchical planning engine for an AGI system.
Your role is to decompose high-level goals into structured, executable plans.

You must generate plans in a strict JSON format with the following structure:
- A mission (level 0) that captures the overall objective
- Subgoals (level 1) that break down the mission into major phases
- Actionable steps (level 2+) that are concrete, executable operations

Each step must include:
- step_id: unique identifier (e.g., "analyze_001")
- level: hierarchy level (0=mission, 1=subgoal, 2+=action)
- kind: type of step (mission, subgoal, analysis, skill, tool_call, control)
- description: clear description of what this step accomplishes
- parent_id: the step_id of the parent step (null for mission)
- requires_simulation: boolean indicating if RH simulation is needed
- safety_tags: list of safety-relevant tags
- tool_id: optional tool identifier for actionable steps

Output ONLY valid JSON. No explanations or markdown.
"""

PLAN_PROMPT_TEMPLATE = """Generate a hierarchical plan for the following goal:

GOAL: {goal_text}

CONTEXT:
- Task Type: {task_type}
- Scenario: {scenario_id}
- Available Tools: {available_tools}
{memory_context}
Generate a complete hierarchical plan with:
1. One mission step (level 0) capturing the overall objective
2. 3-5 subgoal steps (level 1) breaking down the mission
3. 2-4 actionable steps (level 2) for each subgoal

Use the available skills and past experiences to inform your plan.
Prefer skills with high proficiency and success rates.
Learn from past task failures to avoid similar issues.

Output as JSON with structure:
{{
  "plan_id": "plan_<id>",
  "steps": [
    {{
      "step_id": "string",
      "level": number,
      "kind": "string",
      "description": "string",
      "parent_id": "string or null",
      "requires_simulation": boolean,
      "safety_tags": ["string"],
      "tool_id": "string or null"
    }}
  ]
}}
"""

# Default tools available in the system
DEFAULT_TOOLS = [
    "lh.task_interpreter",
    "memory.semantic.query",
    "memory.episodic.lookup",
    "memory.procedural.get_skill",
    "lh.llm.plan",
    "safety.pre_action.check_plan",
    "rh.simulation.request",
    "meta.review_plan",
    "lh.events.publish_plan_steps",
]


# ---------------------------------------------------------------------------
# LLM Planner Configuration
# ---------------------------------------------------------------------------


@dataclass
class LLMPlannerConfig:
    """Configuration for LLM-powered planner."""

    use_llm: bool = True
    fallback_on_error: bool = True
    max_retries: int = 2
    temperature: float = 0.3  # Lower for more deterministic planning
    max_tokens: int = 4096
    available_tools: List[str] = None

    def __post_init__(self):
        if self.available_tools is None:
            self.available_tools = DEFAULT_TOOLS.copy()


# ---------------------------------------------------------------------------
# LLM Planner
# ---------------------------------------------------------------------------


class LLMPlanner:
    """
    LLM-powered hierarchical planner.

    Uses LLM adapters to generate contextual, goal-specific plans.
    Falls back to deterministic planning on LLM failures.
    """

    def __init__(
        self,
        config: Optional[LLMPlannerConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        llm_adapter: Optional[LLMAdapter] = None,
    ) -> None:
        self._config = config or LLMPlannerConfig()
        self._llm_config = llm_config
        self._llm: Optional[LLMAdapter] = llm_adapter
        self._fallback_planner = Planner()

        # Lazy-initialize LLM adapter
        if self._config.use_llm and self._llm is None:
            try:
                self._llm = create_adapter(llm_config)
                logger.info(
                    "[LH][LLMPlanner] Initialized with LLM: %s",
                    self._llm.model_name if self._llm else "none",
                )
            except Exception as e:
                logger.warning(
                    "[LH][LLMPlanner] Failed to initialize LLM adapter: %s. "
                    "Will use fallback planner.",
                    e,
                )
                self._llm = None

    @property
    def is_llm_available(self) -> bool:
        """Check if LLM is configured and available."""
        return self._llm is not None and self._llm.is_available()

    def generate_plan(
        self,
        request: plan_pb2.PlanRequest,
        context: Optional[PlanningContext] = None,
    ) -> PlanGraph:
        """
        Generate a hierarchical plan using LLM or fallback.

        Args:
            request: The plan request protobuf message
            context: Optional planning context from memory enrichment

        Returns:
            PlanGraph with hierarchical steps
        """
        goal_text = self._extract_goal_text(request)
        plan_id = self._derive_plan_id(request)

        logger.info(
            "[LH][LLMPlanner] Generating plan plan_id=%s goal=%s",
            plan_id,
            goal_text,
        )

        if context and context.has_context:
            logger.info(
                "[LH][LLMPlanner] Using memory context: %d facts, %d episodes, %d skills",
                len(context.facts),
                len(context.episodes),
                len(context.skills),
            )

        # Try LLM-based planning if enabled and available
        if self._config.use_llm and self.is_llm_available:
            for attempt in range(self._config.max_retries + 1):
                try:
                    graph = self._generate_llm_plan(
                        request, goal_text, plan_id, context
                    )
                    if graph and len(graph.steps) > 0:
                        logger.info(
                            "[LH][LLMPlanner] LLM plan generated: %d steps",
                            len(graph.steps),
                        )
                        return graph
                except Exception as e:
                    logger.warning(
                        "[LH][LLMPlanner] LLM planning attempt %d failed: %s",
                        attempt + 1,
                        e,
                    )
                    if attempt < self._config.max_retries:
                        continue

            if not self._config.fallback_on_error:
                raise RuntimeError("LLM planning failed and fallback is disabled")

        # Fallback to deterministic planner
        logger.info("[LH][LLMPlanner] Using fallback deterministic planner")
        return self._fallback_planner.generate_plan(request)

    def _generate_llm_plan(
        self,
        request: plan_pb2.PlanRequest,
        goal_text: str,
        plan_id: str,
        context: Optional[PlanningContext] = None,
    ) -> Optional[PlanGraph]:
        """
        Generate plan using LLM.

        Args:
            request: Plan request protobuf
            goal_text: Extracted goal text
            plan_id: Generated plan ID
            context: Optional memory context to include in prompt

        Returns:
            PlanGraph if successful, None if parsing failed
        """
        # Build context from request
        task = getattr(request, "task", None)
        env = getattr(request, "environment", None)

        task_type = getattr(task, "task_type", "general") if task else "general"
        scenario_id = getattr(env, "scenario_id", "default") if env else "default"

        # Format memory context for prompt
        memory_context = ""
        if context and context.has_context:
            memory_context = "\n" + context.to_prompt_context() + "\n"

        prompt = PLAN_PROMPT_TEMPLATE.format(
            goal_text=goal_text,
            task_type=task_type,
            scenario_id=scenario_id,
            available_tools=", ".join(self._config.available_tools),
            memory_context=memory_context,
        )

        response = self._llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        logger.debug(
            "[LH][LLMPlanner] LLM response tokens: prompt=%d completion=%d",
            response.prompt_tokens,
            response.completion_tokens,
        )

        return self._parse_llm_response(response, goal_text, plan_id)

    def _parse_llm_response(
        self,
        response: LLMResponse,
        goal_text: str,
        plan_id: str,
    ) -> Optional[PlanGraph]:
        """
        Parse LLM response into PlanGraph.

        Handles JSON extraction and validation.
        """
        content = response.content.strip()

        # Try to extract JSON from response
        json_str = self._extract_json(content)
        if not json_str:
            logger.warning("[LH][LLMPlanner] No valid JSON found in LLM response")
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("[LH][LLMPlanner] JSON parse error: %s", e)
            return None

        # Build PlanGraph from parsed data
        graph = PlanGraph(
            plan_id=data.get("plan_id", plan_id),
            goal_text=goal_text,
            metadata={"source": "llm", "model": response.model},
        )

        steps = data.get("steps", [])
        if not steps:
            logger.warning("[LH][LLMPlanner] No steps in LLM response")
            return None

        for idx, step_data in enumerate(steps):
            step = self._parse_step(step_data, idx)
            if step:
                graph.add_step(step)

        # Validate plan structure
        if not self._validate_plan(graph):
            logger.warning("[LH][LLMPlanner] Plan validation failed")
            return None

        return graph

    def _parse_step(self, data: Dict[str, Any], index: int) -> Optional[PlanStep]:
        """Parse a single step from LLM output."""
        try:
            return PlanStep(
                step_id=data.get("step_id", f"step_{uuid.uuid4().hex[:8]}"),
                index=index,
                level=int(data.get("level", 2)),
                kind=data.get("kind", "action"),
                description=data.get("description", ""),
                parent_id=data.get("parent_id"),
                requires_simulation=bool(data.get("requires_simulation", False)),
                safety_tags=list(data.get("safety_tags", [])),
                tool_id=data.get("tool_id"),
                params=data.get("params", {}),
            )
        except Exception as e:
            logger.warning("[LH][LLMPlanner] Failed to parse step: %s", e)
            return None

    def _validate_plan(self, graph: PlanGraph) -> bool:
        """
        Validate plan structure.

        Checks:
        - Has at least one mission step (level 0)
        - Has at least one subgoal (level 1)
        - Parent references are valid
        """
        if not graph.steps:
            return False

        levels = {s.level for s in graph.steps}

        # Must have mission and subgoals
        if 0 not in levels or 1 not in levels:
            return False

        # Validate parent references
        step_ids = {s.step_id for s in graph.steps}
        for step in graph.steps:
            if step.parent_id and step.parent_id not in step_ids:
                logger.warning("[LH][LLMPlanner] Invalid parent_id: %s", step.parent_id)
                return False

        return True

    def _extract_json(self, content: str) -> Optional[str]:
        """
        Extract JSON from LLM response.

        Handles cases where LLM wraps JSON in markdown or adds text.
        """
        # Try direct parse first
        content = content.strip()
        if content.startswith("{"):
            return content

        # Look for JSON in markdown code blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            return match.group(1)

        # Look for raw JSON object
        match = re.search(r"(\{[^{}]*\"steps\"[^{}]*\[.*?\]\s*\})", content, re.DOTALL)
        if match:
            return match.group(1)

        return None

    def _extract_goal_text(self, request: plan_pb2.PlanRequest) -> str:
        """Extract goal text from request."""
        task = getattr(request, "task", None)
        if task is not None:
            desc = getattr(task, "description", "")
            if desc:
                return desc
        return "unspecified-goal"

    def _derive_plan_id(self, request: plan_pb2.PlanRequest) -> str:
        """Derive plan ID from request."""
        task = getattr(request, "task", None)
        goal_id = getattr(task, "goal_id", "") if task else ""
        if goal_id:
            return f"plan_{goal_id}"
        return f"plan_{uuid.uuid4().hex[:8]}"
