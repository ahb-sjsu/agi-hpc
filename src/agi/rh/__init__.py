# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Right Hemisphere (RH) Module for AGI-HPC.

The RH implements the sensorimotor subsystem:
- Perception: sensory processing and state abstraction
- World Model: short-horizon predictive simulation
- Control: action execution and motor primitives

Architecture references:
    - Section IV.B  – RH Node
    - Section XI    – Sensorimotor Loop
    - Section XIV   – Cognitive APIs

Usage:
    from agi.rh import RHService, RHConfig

    # Start the RH service
    config = RHConfig()
    service = RHService(config)
    service.run()

    # Or use individual components
    from agi.rh import Perception, WorldModel, ControlService
"""

from agi.rh.config import (
    RHConfig,
    load_rh_config,
    PerceptionConfig,
    WorldModelConfig,
    ControlConfig as RHControlConfig,
    EventFabricConfig,
    GRPCConfig,
)
from agi.rh.perception import Perception
from agi.rh.world_model import WorldModel, RolloutResult
from agi.rh.control_service import ControlService, ControlConfig, ActionResult
from agi.rh.simulation_service import SimulationService
from agi.rh.rh_event_loop import RHEventLoop
from agi.rh.service import RHService

__all__ = [
    # Main Service
    "RHService",
    # Configuration
    "RHConfig",
    "load_rh_config",
    "PerceptionConfig",
    "WorldModelConfig",
    "RHControlConfig",
    "EventFabricConfig",
    "GRPCConfig",
    # Components
    "Perception",
    "WorldModel",
    "RolloutResult",
    "ControlService",
    "ControlConfig",
    "ActionResult",
    "SimulationService",
    "RHEventLoop",
]
