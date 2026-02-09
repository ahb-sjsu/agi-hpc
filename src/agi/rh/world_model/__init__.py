# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
RH World Model Module.

Provides predictive simulation capabilities for the Right Hemisphere:
- WorldModel: Short-horizon predictive model (stub)
- Physics engines: MuJoCo, PyBullet adapters
- World state prediction and risk estimation

Architecture references:
    - Section IV.B.2 – RH World Model
    - Section XI – Sensorimotor Loop
"""

from agi.rh.world_model.core import WorldModel, RolloutResult
from agi.rh.world_model.physics import (
    PhysicsEngine,
    BasePhysicsEngine,
    PhysicsConfig,
    PhysicsState,
    SimulationResult,
    CollisionInfo,
    MuJoCoEngine,
    PyBulletEngine,
)

__all__ = [
    # Main world model
    "WorldModel",
    "RolloutResult",
    # Physics engine protocol and base
    "PhysicsEngine",
    "BasePhysicsEngine",
    "PhysicsConfig",
    "PhysicsState",
    "SimulationResult",
    "CollisionInfo",
    # Physics implementations
    "MuJoCoEngine",
    "PyBulletEngine",
]
