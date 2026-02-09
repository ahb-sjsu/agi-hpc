# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Consistency Checker for Metacognition.

Implements Sprint 3 requirements:
- Cross-component validation
- Memory consistency checking
- RH world state verification
- Safety decision synchronization

This module ensures consistency across the cognitive architecture by:
- Verifying plan steps match memory context
- Checking RH world model aligns with perception
- Validating safety decisions are synchronized
- Detecting inconsistencies between components
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Types
# ---------------------------------------------------------------------------


class ConsistencyLevel(Enum):
    """Levels of consistency."""

    CONSISTENT = "consistent"
    MINOR_INCONSISTENCY = "minor_inconsistency"
    MAJOR_INCONSISTENCY = "major_inconsistency"
    CRITICAL_INCONSISTENCY = "critical_inconsistency"


class ComponentType(Enum):
    """Component types for consistency checking."""

    LH_PLANNER = "lh_planner"
    RH_PERCEPTION = "rh_perception"
    RH_WORLD_MODEL = "rh_world_model"
    RH_CONTROL = "rh_control"
    MEMORY_SEMANTIC = "memory_semantic"
    MEMORY_EPISODIC = "memory_episodic"
    MEMORY_PROCEDURAL = "memory_procedural"
    SAFETY = "safety"


class InconsistencyType(Enum):
    """Types of inconsistencies."""

    MEMORY_PLAN_MISMATCH = "memory_plan_mismatch"
    PERCEPTION_WORLD_MISMATCH = "perception_world_mismatch"
    SAFETY_STATE_MISMATCH = "safety_state_mismatch"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    SEMANTIC_CONTRADICTION = "semantic_contradiction"
    SKILL_AVAILABILITY_MISMATCH = "skill_availability_mismatch"
    OBJECT_STATE_MISMATCH = "object_state_mismatch"
    GOAL_STATE_MISMATCH = "goal_state_mismatch"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ComponentState:
    """State snapshot from a component."""

    component: ComponentType
    timestamp: datetime = field(default_factory=datetime.now)
    state_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    version: int = 0


@dataclass
class Inconsistency:
    """Detected inconsistency between components."""

    inconsistency_type: InconsistencyType
    level: ConsistencyLevel
    components: List[ComponentType]
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    suggested_resolution: str = ""


@dataclass
class ConsistencyCheckResult:
    """Result of a consistency check."""

    is_consistent: bool
    level: ConsistencyLevel
    inconsistencies: List[Inconsistency] = field(default_factory=list)
    checked_components: List[ComponentType] = field(default_factory=list)
    check_time_ms: int = 0
    recommendations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Consistency Checker
# ---------------------------------------------------------------------------


class ConsistencyChecker:
    """
    Cross-component consistency checker.

    Validates consistency across:
    - Memory ↔ Plan (semantic facts match plan assumptions)
    - Perception ↔ WorldModel (observed state matches predictions)
    - Safety ↔ All (safety decisions are synchronized)
    - Temporal (states are temporally coherent)
    """

    def __init__(
        self,
        tolerance_threshold: float = 0.1,
        max_temporal_drift_ms: int = 1000,
    ) -> None:
        """
        Initialize consistency checker.

        Args:
            tolerance_threshold: Acceptable difference for numerical values
            max_temporal_drift_ms: Maximum allowed temporal drift in ms
        """
        self._tolerance = tolerance_threshold
        self._max_drift_ms = max_temporal_drift_ms
        self._component_states: Dict[ComponentType, ComponentState] = {}
        self._check_count = 0

        logger.info(
            "[Meta][Consistency] initialized tolerance=%.2f drift_ms=%d",
            tolerance_threshold,
            max_temporal_drift_ms,
        )

    # ------------------------------------------------------------------ #
    # State Registration
    # ------------------------------------------------------------------ #

    def register_state(
        self,
        component: ComponentType,
        state_data: Dict[str, Any],
        state_id: str = "",
    ) -> None:
        """
        Register a component's current state for consistency checking.

        Args:
            component: Component type
            state_data: State data dictionary
            state_id: Optional state identifier
        """
        version = 0
        if component in self._component_states:
            version = self._component_states[component].version + 1

        self._component_states[component] = ComponentState(
            component=component,
            state_id=state_id,
            data=state_data,
            version=version,
        )

        logger.debug(
            "[Meta][Consistency] registered state for %s v%d",
            component.value,
            version,
        )

    def get_state(self, component: ComponentType) -> Optional[ComponentState]:
        """Get registered state for a component."""
        return self._component_states.get(component)

    # ------------------------------------------------------------------ #
    # Full Consistency Check
    # ------------------------------------------------------------------ #

    def check_all(self) -> ConsistencyCheckResult:
        """
        Perform full consistency check across all registered components.

        Returns:
            ConsistencyCheckResult with all detected inconsistencies
        """
        import time

        start = time.time()
        self._check_count += 1

        inconsistencies: List[Inconsistency] = []
        checked: Set[ComponentType] = set()

        # Check Memory ↔ Plan consistency
        if ComponentType.MEMORY_SEMANTIC in self._component_states:
            if ComponentType.LH_PLANNER in self._component_states:
                checked.add(ComponentType.MEMORY_SEMANTIC)
                checked.add(ComponentType.LH_PLANNER)
                memory_plan = self._check_memory_plan_consistency()
                inconsistencies.extend(memory_plan)

        # Check Perception ↔ WorldModel consistency
        if ComponentType.RH_PERCEPTION in self._component_states:
            if ComponentType.RH_WORLD_MODEL in self._component_states:
                checked.add(ComponentType.RH_PERCEPTION)
                checked.add(ComponentType.RH_WORLD_MODEL)
                perception_world = self._check_perception_world_consistency()
                inconsistencies.extend(perception_world)

        # Check Safety ↔ All consistency
        if ComponentType.SAFETY in self._component_states:
            checked.add(ComponentType.SAFETY)
            safety_all = self._check_safety_consistency()
            inconsistencies.extend(safety_all)

        # Check temporal consistency
        temporal = self._check_temporal_consistency()
        inconsistencies.extend(temporal)

        # Determine overall level
        level = self._determine_overall_level(inconsistencies)

        check_time = int((time.time() - start) * 1000)

        result = ConsistencyCheckResult(
            is_consistent=len(inconsistencies) == 0,
            level=level,
            inconsistencies=inconsistencies,
            checked_components=list(checked),
            check_time_ms=check_time,
            recommendations=self._generate_recommendations(inconsistencies),
        )

        logger.info(
            "[Meta][Consistency] check %d: consistent=%s level=%s issues=%d",
            self._check_count,
            result.is_consistent,
            result.level.value,
            len(inconsistencies),
        )

        return result

    # ------------------------------------------------------------------ #
    # Specific Consistency Checks
    # ------------------------------------------------------------------ #

    def check_memory_plan(
        self,
        memory_facts: Dict[str, Any],
        plan_assumptions: Dict[str, Any],
    ) -> ConsistencyCheckResult:
        """
        Check consistency between memory facts and plan assumptions.

        Args:
            memory_facts: Facts from semantic memory
            plan_assumptions: Assumptions made by the plan

        Returns:
            ConsistencyCheckResult
        """
        self.register_state(ComponentType.MEMORY_SEMANTIC, memory_facts)
        self.register_state(ComponentType.LH_PLANNER, plan_assumptions)

        inconsistencies = self._check_memory_plan_consistency()

        level = self._determine_overall_level(inconsistencies)

        return ConsistencyCheckResult(
            is_consistent=len(inconsistencies) == 0,
            level=level,
            inconsistencies=inconsistencies,
            checked_components=[
                ComponentType.MEMORY_SEMANTIC,
                ComponentType.LH_PLANNER,
            ],
            recommendations=self._generate_recommendations(inconsistencies),
        )

    def check_perception_world_model(
        self,
        perception_state: Dict[str, Any],
        world_model_state: Dict[str, Any],
    ) -> ConsistencyCheckResult:
        """
        Check consistency between perception and world model.

        Args:
            perception_state: Current perception state
            world_model_state: World model's predicted state

        Returns:
            ConsistencyCheckResult
        """
        self.register_state(ComponentType.RH_PERCEPTION, perception_state)
        self.register_state(ComponentType.RH_WORLD_MODEL, world_model_state)

        inconsistencies = self._check_perception_world_consistency()

        level = self._determine_overall_level(inconsistencies)

        return ConsistencyCheckResult(
            is_consistent=len(inconsistencies) == 0,
            level=level,
            inconsistencies=inconsistencies,
            checked_components=[
                ComponentType.RH_PERCEPTION,
                ComponentType.RH_WORLD_MODEL,
            ],
            recommendations=self._generate_recommendations(inconsistencies),
        )

    def check_safety_sync(
        self,
        safety_state: Dict[str, Any],
        component_states: Dict[ComponentType, Dict[str, Any]],
    ) -> ConsistencyCheckResult:
        """
        Check safety decisions are synchronized with component states.

        Args:
            safety_state: Current safety state
            component_states: States from other components

        Returns:
            ConsistencyCheckResult
        """
        self.register_state(ComponentType.SAFETY, safety_state)
        for comp, state in component_states.items():
            self.register_state(comp, state)

        inconsistencies = self._check_safety_consistency()

        level = self._determine_overall_level(inconsistencies)

        return ConsistencyCheckResult(
            is_consistent=len(inconsistencies) == 0,
            level=level,
            inconsistencies=inconsistencies,
            checked_components=[ComponentType.SAFETY] + list(component_states.keys()),
            recommendations=self._generate_recommendations(inconsistencies),
        )

    # ------------------------------------------------------------------ #
    # Internal Check Methods
    # ------------------------------------------------------------------ #

    def _check_memory_plan_consistency(self) -> List[Inconsistency]:
        """Check memory-plan consistency."""
        inconsistencies = []

        memory_state = self._component_states.get(ComponentType.MEMORY_SEMANTIC)
        plan_state = self._component_states.get(ComponentType.LH_PLANNER)

        if not memory_state or not plan_state:
            return inconsistencies

        memory_data = memory_state.data
        plan_data = plan_state.data

        # Check object availability
        plan_objects = set(plan_data.get("required_objects", []))
        memory_objects = set(memory_data.get("known_objects", []))

        missing_objects = plan_objects - memory_objects
        if missing_objects:
            inconsistencies.append(
                Inconsistency(
                    inconsistency_type=InconsistencyType.MEMORY_PLAN_MISMATCH,
                    level=ConsistencyLevel.MAJOR_INCONSISTENCY,
                    components=[
                        ComponentType.MEMORY_SEMANTIC,
                        ComponentType.LH_PLANNER,
                    ],
                    description=f"Plan requires objects not in memory: {missing_objects}",
                    details={"missing_objects": list(missing_objects)},
                    suggested_resolution="Query memory for object locations or update plan",
                )
            )

        # Check skill availability
        plan_skills = set(plan_data.get("required_skills", []))
        memory_skills = set(memory_data.get("available_skills", []))

        missing_skills = plan_skills - memory_skills
        if missing_skills:
            inconsistencies.append(
                Inconsistency(
                    inconsistency_type=InconsistencyType.SKILL_AVAILABILITY_MISMATCH,
                    level=ConsistencyLevel.MAJOR_INCONSISTENCY,
                    components=[
                        ComponentType.MEMORY_SEMANTIC,
                        ComponentType.LH_PLANNER,
                    ],
                    description=f"Plan requires skills not available: {missing_skills}",
                    details={"missing_skills": list(missing_skills)},
                    suggested_resolution="Learn required skills or use alternative actions",
                )
            )

        # Check semantic contradictions
        plan_facts = plan_data.get("assumed_facts", {})
        memory_facts = memory_data.get("facts", {})

        for key, plan_value in plan_facts.items():
            if key in memory_facts:
                memory_value = memory_facts[key]
                if plan_value != memory_value:
                    inconsistencies.append(
                        Inconsistency(
                            inconsistency_type=InconsistencyType.SEMANTIC_CONTRADICTION,
                            level=ConsistencyLevel.CRITICAL_INCONSISTENCY,
                            components=[
                                ComponentType.MEMORY_SEMANTIC,
                                ComponentType.LH_PLANNER,
                            ],
                            description=f"Plan assumes '{key}={plan_value}' but memory has '{key}={memory_value}'",
                            details={
                                "key": key,
                                "plan_value": plan_value,
                                "memory_value": memory_value,
                            },
                            suggested_resolution="Re-plan with correct facts from memory",
                        )
                    )

        return inconsistencies

    def _check_perception_world_consistency(self) -> List[Inconsistency]:
        """Check perception-world model consistency."""
        inconsistencies = []

        perception_state = self._component_states.get(ComponentType.RH_PERCEPTION)
        world_state = self._component_states.get(ComponentType.RH_WORLD_MODEL)

        if not perception_state or not world_state:
            return inconsistencies

        perception_data = perception_state.data
        world_data = world_state.data

        # Check object count mismatch
        perceived_objects = perception_data.get("objects", [])
        world_objects = world_data.get("objects", [])

        if len(perceived_objects) != len(world_objects):
            inconsistencies.append(
                Inconsistency(
                    inconsistency_type=InconsistencyType.OBJECT_STATE_MISMATCH,
                    level=ConsistencyLevel.MINOR_INCONSISTENCY,
                    components=[
                        ComponentType.RH_PERCEPTION,
                        ComponentType.RH_WORLD_MODEL,
                    ],
                    description=f"Perception sees {len(perceived_objects)} objects but world model has {len(world_objects)}",
                    details={
                        "perceived_count": len(perceived_objects),
                        "world_count": len(world_objects),
                    },
                    suggested_resolution="Update world model with perception data",
                )
            )

        # Check position drift
        perceived_pos = perception_data.get("position", [0, 0, 0])
        world_pos = world_data.get("position", [0, 0, 0])

        if len(perceived_pos) == len(world_pos):
            drift = (
                sum(
                    (p - w) ** 2 for p, w in zip(perceived_pos, world_pos, strict=False)
                )
                ** 0.5
            )
            if drift > self._tolerance:
                inconsistencies.append(
                    Inconsistency(
                        inconsistency_type=InconsistencyType.PERCEPTION_WORLD_MISMATCH,
                        level=ConsistencyLevel.MAJOR_INCONSISTENCY,
                        components=[
                            ComponentType.RH_PERCEPTION,
                            ComponentType.RH_WORLD_MODEL,
                        ],
                        description=f"Position drift of {drift:.3f} exceeds tolerance {self._tolerance}",
                        details={
                            "perceived_position": perceived_pos,
                            "world_position": world_pos,
                            "drift": drift,
                        },
                        suggested_resolution="Recalibrate world model or filter perception noise",
                    )
                )

        return inconsistencies

    def _check_safety_consistency(self) -> List[Inconsistency]:
        """Check safety state consistency with other components."""
        inconsistencies = []

        safety_state = self._component_states.get(ComponentType.SAFETY)
        if not safety_state:
            return inconsistencies

        safety_data = safety_state.data

        # Check if safety blocked but components are executing
        if safety_data.get("current_decision") == "BLOCK":
            # Check if any component is in active execution state
            for comp_type, comp_state in self._component_states.items():
                if comp_type == ComponentType.SAFETY:
                    continue

                if comp_state.data.get("is_executing", False):
                    inconsistencies.append(
                        Inconsistency(
                            inconsistency_type=InconsistencyType.SAFETY_STATE_MISMATCH,
                            level=ConsistencyLevel.CRITICAL_INCONSISTENCY,
                            components=[ComponentType.SAFETY, comp_type],
                            description=f"Safety blocked but {comp_type.value} is executing",
                            details={
                                "safety_decision": "BLOCK",
                                "component": comp_type.value,
                                "component_state": "executing",
                            },
                            suggested_resolution="Immediately halt component execution",
                        )
                    )

        # Check risk level consistency
        safety_risk = safety_data.get("risk_level", 0.0)
        for comp_type, comp_state in self._component_states.items():
            if comp_type == ComponentType.SAFETY:
                continue

            comp_risk = comp_state.data.get("risk_score", 0.0)
            if abs(safety_risk - comp_risk) > 0.3:
                inconsistencies.append(
                    Inconsistency(
                        inconsistency_type=InconsistencyType.SAFETY_STATE_MISMATCH,
                        level=ConsistencyLevel.MINOR_INCONSISTENCY,
                        components=[ComponentType.SAFETY, comp_type],
                        description=f"Risk level mismatch: safety={safety_risk:.2f}, {comp_type.value}={comp_risk:.2f}",
                        details={
                            "safety_risk": safety_risk,
                            "component_risk": comp_risk,
                            "component": comp_type.value,
                        },
                        suggested_resolution="Synchronize risk assessments",
                    )
                )

        return inconsistencies

    def _check_temporal_consistency(self) -> List[Inconsistency]:
        """Check temporal consistency across components."""
        inconsistencies = []

        if len(self._component_states) < 2:
            return inconsistencies

        timestamps = [
            (comp, state.timestamp) for comp, state in self._component_states.items()
        ]

        # Find max time drift
        min_time = min(t for _, t in timestamps)
        max_time = max(t for _, t in timestamps)

        drift_ms = (max_time - min_time).total_seconds() * 1000

        if drift_ms > self._max_drift_ms:
            oldest = min(timestamps, key=lambda x: x[1])
            newest = max(timestamps, key=lambda x: x[1])

            inconsistencies.append(
                Inconsistency(
                    inconsistency_type=InconsistencyType.TEMPORAL_INCONSISTENCY,
                    level=ConsistencyLevel.MAJOR_INCONSISTENCY,
                    components=[oldest[0], newest[0]],
                    description=f"Temporal drift of {drift_ms:.0f}ms exceeds limit {self._max_drift_ms}ms",
                    details={
                        "drift_ms": drift_ms,
                        "oldest": oldest[0].value,
                        "newest": newest[0].value,
                    },
                    suggested_resolution="Refresh stale component states",
                )
            )

        return inconsistencies

    # ------------------------------------------------------------------ #
    # Helper Methods
    # ------------------------------------------------------------------ #

    def _determine_overall_level(
        self, inconsistencies: List[Inconsistency]
    ) -> ConsistencyLevel:
        """Determine overall consistency level from inconsistencies."""
        if not inconsistencies:
            return ConsistencyLevel.CONSISTENT

        # Find worst level
        levels = [i.level for i in inconsistencies]

        if ConsistencyLevel.CRITICAL_INCONSISTENCY in levels:
            return ConsistencyLevel.CRITICAL_INCONSISTENCY
        if ConsistencyLevel.MAJOR_INCONSISTENCY in levels:
            return ConsistencyLevel.MAJOR_INCONSISTENCY
        if ConsistencyLevel.MINOR_INCONSISTENCY in levels:
            return ConsistencyLevel.MINOR_INCONSISTENCY

        return ConsistencyLevel.CONSISTENT

    def _generate_recommendations(
        self, inconsistencies: List[Inconsistency]
    ) -> List[str]:
        """Generate recommendations based on inconsistencies."""
        recommendations = []

        for inc in inconsistencies:
            if inc.suggested_resolution:
                recommendations.append(inc.suggested_resolution)

        # Add general recommendations based on patterns
        types = [i.inconsistency_type for i in inconsistencies]

        if InconsistencyType.TEMPORAL_INCONSISTENCY in types:
            recommendations.append("Consider increasing state refresh frequency")

        if InconsistencyType.MEMORY_PLAN_MISMATCH in types:
            recommendations.append("Re-query memory before planning")

        if InconsistencyType.SAFETY_STATE_MISMATCH in types:
            recommendations.append("Implement safety state broadcast")

        return list(set(recommendations))  # Deduplicate

    def clear_states(self) -> None:
        """Clear all registered component states."""
        self._component_states.clear()
        logger.debug("[Meta][Consistency] cleared all states")
