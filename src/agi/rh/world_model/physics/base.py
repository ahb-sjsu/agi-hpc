# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Physics Engine Protocol and Base Implementation.

Defines the interface for physics engines used in the RH world model.
Physics engines provide predictive simulation for plan evaluation,
including collision detection, dynamics, and risk estimation.

Sprint 5 Implementation:
- PhysicsEngine protocol for engine interface
- BasePhysicsEngine with common functionality
- PhysicsConfig for configuration
- PhysicsState and SimulationResult containers
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class PhysicsConfig:
    """
    Configuration for physics engines.

    Environment variables:
        AGI_RH_PHYSICS_ENGINE: Engine type (mujoco, pybullet, isaac)
        AGI_RH_PHYSICS_TIMESTEP: Simulation timestep in seconds
        AGI_RH_PHYSICS_GRAVITY: Gravity vector as comma-separated values
    """

    engine_type: str = "pybullet"
    timestep: float = 0.01  # 100 Hz
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    max_substeps: int = 10
    solver_iterations: int = 50
    enable_collisions: bool = True
    enable_constraints: bool = True
    render_mode: str = "headless"  # headless, gui, rgb_array
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollisionInfo:
    """Information about a collision event."""

    object_a: str
    object_b: str
    contact_point: Tuple[float, float, float]
    contact_normal: Tuple[float, float, float]
    penetration_depth: float
    impulse: float


@dataclass
class PhysicsState:
    """
    State of objects in the physics simulation.

    Represents positions, velocities, and other physical properties
    of all objects in the simulation.
    """

    timestamp: float
    positions: Dict[str, np.ndarray]  # object_id -> [x, y, z]
    orientations: Dict[str, np.ndarray]  # object_id -> [qx, qy, qz, qw]
    linear_velocities: Dict[str, np.ndarray]  # object_id -> [vx, vy, vz]
    angular_velocities: Dict[str, np.ndarray]  # object_id -> [wx, wy, wz]
    collisions: List[CollisionInfo] = field(default_factory=list)
    contacts: List[Tuple[str, str]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_position(self, object_id: str) -> Optional[np.ndarray]:
        """Get position of an object."""
        return self.positions.get(object_id)

    def get_velocity(self, object_id: str) -> Optional[np.ndarray]:
        """Get linear velocity of an object."""
        return self.linear_velocities.get(object_id)


@dataclass
class SimulationResult:
    """
    Result of a physics simulation rollout.

    Contains the trajectory of states, detected collisions,
    and risk assessment.
    """

    success: bool
    states: List[PhysicsState]
    duration: float
    collision_detected: bool = False
    collision_time: Optional[float] = None
    collisions: List[CollisionInfo] = field(default_factory=list)
    risk_score: float = 0.0
    violations: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_state(self) -> Optional[PhysicsState]:
        """Get the final state of the simulation."""
        return self.states[-1] if self.states else None

    @property
    def trajectory_length(self) -> int:
        """Get number of states in trajectory."""
        return len(self.states)


# ---------------------------------------------------------------------------
# Physics Engine Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PhysicsEngine(Protocol):
    """
    Protocol for physics engines.

    All physics engines must implement this interface to be used
    with the RH world model for predictive simulation.
    """

    @property
    def name(self) -> str:
        """Return the engine name/identifier."""
        ...

    def initialize(self) -> None:
        """Initialize the physics engine."""
        ...

    def reset(self, initial_state: Optional[Dict[str, Any]] = None) -> PhysicsState:
        """
        Reset the simulation to initial state.

        Args:
            initial_state: Optional initial state configuration

        Returns:
            The initial PhysicsState
        """
        ...

    def step(self, actions: Optional[Dict[str, np.ndarray]] = None) -> PhysicsState:
        """
        Step the simulation forward by one timestep.

        Args:
            actions: Optional actions to apply (object_id -> action vector)

        Returns:
            The new PhysicsState after stepping
        """
        ...

    def rollout(
        self,
        actions: List[Dict[str, np.ndarray]],
        horizon: int,
    ) -> SimulationResult:
        """
        Run simulation rollout for given actions over horizon.

        Args:
            actions: Sequence of actions to apply
            horizon: Number of steps to simulate

        Returns:
            SimulationResult with trajectory and analysis
        """
        ...

    def add_object(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        orientation: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> None:
        """
        Add an object to the simulation.

        Args:
            object_id: Unique identifier for the object
            position: Initial position (x, y, z)
            orientation: Initial orientation as quaternion (optional)
            **kwargs: Engine-specific parameters (mass, shape, etc.)
        """
        ...

    def remove_object(self, object_id: str) -> None:
        """Remove an object from the simulation."""
        ...

    def get_state(self) -> PhysicsState:
        """Get current physics state."""
        ...

    def check_collision(
        self,
        object_a: str,
        object_b: Optional[str] = None,
    ) -> List[CollisionInfo]:
        """
        Check for collisions involving object_a.

        Args:
            object_a: Object to check
            object_b: Optional specific object to check against

        Returns:
            List of CollisionInfo for detected collisions
        """
        ...

    def is_available(self) -> bool:
        """Check if the engine is available and ready."""
        ...

    def close(self) -> None:
        """Clean up and release resources."""
        ...


# ---------------------------------------------------------------------------
# Base Physics Engine
# ---------------------------------------------------------------------------


class BasePhysicsEngine(ABC):
    """
    Base class for physics engines with common functionality.

    Provides:
    - Configuration management
    - State tracking
    - Risk estimation
    - Collision analysis
    """

    def __init__(self, config: Optional[PhysicsConfig] = None) -> None:
        self._config = config or PhysicsConfig()
        self._initialized = False
        self._objects: Dict[str, Dict[str, Any]] = {}
        self._current_time = 0.0

        logger.info(
            "[RH][Physics] Initializing %s engine timestep=%.3f",
            self.__class__.__name__,
            self._config.timestep,
        )

    @property
    def name(self) -> str:
        """Return the engine name."""
        return self._config.engine_type

    @property
    def timestep(self) -> float:
        """Return the simulation timestep."""
        return self._config.timestep

    @abstractmethod
    def _init_engine(self) -> None:
        """Initialize the underlying physics engine."""
        pass

    @abstractmethod
    def _step_engine(
        self,
        actions: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Step the underlying engine forward."""
        pass

    @abstractmethod
    def _get_object_state(self, object_id: str) -> Dict[str, np.ndarray]:
        """Get state of a specific object from engine."""
        pass

    @abstractmethod
    def _add_object_to_engine(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        **kwargs,
    ) -> None:
        """Add object to underlying engine."""
        pass

    @abstractmethod
    def _remove_object_from_engine(self, object_id: str) -> None:
        """Remove object from underlying engine."""
        pass

    @abstractmethod
    def _check_collision_engine(
        self,
        object_a: str,
        object_b: Optional[str] = None,
    ) -> List[CollisionInfo]:
        """Check collisions in underlying engine."""
        pass

    def initialize(self) -> None:
        """Initialize the physics engine."""
        if not self._initialized:
            self._init_engine()
            self._initialized = True
            logger.info("[RH][Physics] Engine initialized")

    def reset(self, initial_state: Optional[Dict[str, Any]] = None) -> PhysicsState:
        """Reset the simulation."""
        if not self._initialized:
            self.initialize()

        self._current_time = 0.0

        # Apply initial state if provided
        if initial_state:
            for obj_id, obj_state in initial_state.items():
                if obj_id in self._objects:
                    # Update object position/velocity
                    pass

        return self.get_state()

    def step(self, actions: Optional[Dict[str, np.ndarray]] = None) -> PhysicsState:
        """Step the simulation forward."""
        if not self._initialized:
            self.initialize()

        self._step_engine(actions)
        self._current_time += self._config.timestep

        return self.get_state()

    def rollout(
        self,
        actions: List[Dict[str, np.ndarray]],
        horizon: int,
    ) -> SimulationResult:
        """Run simulation rollout."""
        if not self._initialized:
            self.initialize()

        states = []
        all_collisions = []
        collision_detected = False
        collision_time = None
        violations = []

        # Ensure we have enough actions
        if len(actions) < horizon:
            actions = actions + [{}] * (horizon - len(actions))

        start_time = self._current_time

        for i in range(horizon):
            action = actions[i] if i < len(actions) else {}
            state = self.step(action)
            states.append(state)

            # Check for collisions
            if state.collisions:
                all_collisions.extend(state.collisions)
                if not collision_detected:
                    collision_detected = True
                    collision_time = state.timestamp

            # Check for constraint violations
            step_violations = self._check_violations(state)
            violations.extend(step_violations)

        duration = self._current_time - start_time

        # Compute risk score
        risk_score = self._compute_risk_score(states, all_collisions, violations)

        return SimulationResult(
            success=not collision_detected and not violations,
            states=states,
            duration=duration,
            collision_detected=collision_detected,
            collision_time=collision_time,
            collisions=all_collisions,
            risk_score=risk_score,
            violations=violations,
        )

    def add_object(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        orientation: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> None:
        """Add an object to the simulation."""
        if not self._initialized:
            self.initialize()

        if orientation is None:
            orientation = (0.0, 0.0, 0.0, 1.0)  # Identity quaternion

        self._add_object_to_engine(object_id, position, orientation, **kwargs)
        self._objects[object_id] = {
            "position": position,
            "orientation": orientation,
            **kwargs,
        }

        logger.debug(
            "[RH][Physics] Added object %s at %s",
            object_id,
            position,
        )

    def remove_object(self, object_id: str) -> None:
        """Remove an object from the simulation."""
        if object_id in self._objects:
            self._remove_object_from_engine(object_id)
            del self._objects[object_id]
            logger.debug("[RH][Physics] Removed object %s", object_id)

    def get_state(self) -> PhysicsState:
        """Get current physics state."""
        positions = {}
        orientations = {}
        linear_velocities = {}
        angular_velocities = {}

        for obj_id in self._objects:
            obj_state = self._get_object_state(obj_id)
            positions[obj_id] = obj_state.get("position", np.zeros(3))
            orientations[obj_id] = obj_state.get("orientation", np.array([0, 0, 0, 1]))
            linear_velocities[obj_id] = obj_state.get("linear_velocity", np.zeros(3))
            angular_velocities[obj_id] = obj_state.get("angular_velocity", np.zeros(3))

        # Get all collisions
        collisions = []
        if self._config.enable_collisions:
            for obj_id in self._objects:
                obj_collisions = self._check_collision_engine(obj_id)
                collisions.extend(obj_collisions)

        return PhysicsState(
            timestamp=self._current_time,
            positions=positions,
            orientations=orientations,
            linear_velocities=linear_velocities,
            angular_velocities=angular_velocities,
            collisions=collisions,
        )

    def check_collision(
        self,
        object_a: str,
        object_b: Optional[str] = None,
    ) -> List[CollisionInfo]:
        """Check for collisions."""
        return self._check_collision_engine(object_a, object_b)

    def is_available(self) -> bool:
        """Check if the engine is available."""
        try:
            if not self._initialized:
                self.initialize()
            return True
        except Exception as e:
            logger.warning("[RH][Physics] Engine not available: %s", e)
            return False

    def close(self) -> None:
        """Clean up resources."""
        self._initialized = False
        self._objects.clear()
        logger.info("[RH][Physics] Engine closed")

    def _check_violations(self, state: PhysicsState) -> List[str]:
        """Check for constraint violations in state."""
        violations = []

        # Check for out-of-bounds positions
        for obj_id, pos in state.positions.items():
            if np.any(np.abs(pos) > 1000):  # Simple bounds check
                violations.append(f"Object {obj_id} out of bounds: {pos}")

        # Check for excessive velocities
        for obj_id, vel in state.linear_velocities.items():
            speed = np.linalg.norm(vel)
            if speed > 100:  # m/s limit
                violations.append(f"Object {obj_id} overspeed: {speed:.1f} m/s")

        return violations

    def _compute_risk_score(
        self,
        states: List[PhysicsState],
        collisions: List[CollisionInfo],
        violations: List[str],
    ) -> float:
        """Compute risk score from simulation results."""
        risk = 0.0

        # Collision risk
        if collisions:
            risk += min(0.5, len(collisions) * 0.1)

        # Violation risk
        if violations:
            risk += min(0.3, len(violations) * 0.1)

        # Velocity risk (based on max speed)
        max_speed = 0.0
        for state in states:
            for vel in state.linear_velocities.values():
                speed = np.linalg.norm(vel)
                max_speed = max(max_speed, speed)

        if max_speed > 10:  # m/s
            risk += min(0.2, (max_speed - 10) * 0.02)

        return min(1.0, risk)


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def create_physics_engine(
    engine_type: str = "pybullet",
    config: Optional[PhysicsConfig] = None,
) -> PhysicsEngine:
    """
    Create a physics engine by type.

    Args:
        engine_type: Type of engine (mujoco, pybullet, isaac)
        config: Optional physics configuration

    Returns:
        PhysicsEngine instance

    Raises:
        ValueError: If engine type is not supported
    """
    engine_type = engine_type.lower()

    cfg = config or PhysicsConfig(engine_type=engine_type)

    if engine_type == "mujoco":
        from agi.rh.world_model.physics.mujoco import MuJoCoEngine

        return MuJoCoEngine(cfg)
    elif engine_type == "pybullet":
        from agi.rh.world_model.physics.pybullet import PyBulletEngine

        return PyBulletEngine(cfg)
    else:
        raise ValueError(f"Unsupported physics engine: {engine_type}")
