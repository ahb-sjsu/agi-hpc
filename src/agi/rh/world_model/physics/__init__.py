# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Physics Engine interfaces and implementations.

Provides pluggable physics engines for the RH world model:
- MuJoCo for high-fidelity physics
- PyBullet for general-purpose simulation
- Isaac Gym for GPU-accelerated physics
"""

from agi.rh.world_model.physics.base import (
    PhysicsEngine,
    BasePhysicsEngine,
    PhysicsConfig,
    PhysicsState,
    SimulationResult,
    CollisionInfo,
)
from agi.rh.world_model.physics.mujoco import MuJoCoEngine
from agi.rh.world_model.physics.pybullet import PyBulletEngine

__all__ = [
    # Protocol and base
    "PhysicsEngine",
    "BasePhysicsEngine",
    "PhysicsConfig",
    "PhysicsState",
    "SimulationResult",
    "CollisionInfo",
    # Implementations
    "MuJoCoEngine",
    "PyBulletEngine",
]
