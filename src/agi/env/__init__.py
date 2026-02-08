# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Environment Interface for AGI-HPC.

Provides the bridge between the cognitive architecture and physical/simulated worlds:
- Gymnasium-compatible environment API
- Structured observations (camera, lidar, IMU, etc.)
- Structured actions (joint, cartesian, gripper, etc.)
- Environment registry and factory

Usage:
    from agi.env import Environment, create_env

    # Create environment
    env = create_env("local:simple")

    # Run episode
    async with env:
        result = await env.reset()
        while not result.done:
            action = ...  # Your policy
            result = await env.step(action)

Environment Variables:
    AGI_ENV_DEFAULT     Default environment name
    AGI_ENV_RENDER      Enable rendering (true/false)
"""

# Base types
from agi.env.base import (
    Environment,
    AsyncEnvironment,
    EnvironmentProtocol,
    EnvironmentSpec,
    EnvironmentRegistry,
    Space,
    SpaceType,
    StepResult,
    ResetResult,
    create_env,
    register_env,
)

# Observation types
from agi.env.observations import (
    Observation,
    SensorType,
    CameraObservation,
    LidarObservation,
    PointCloud,
    ImuObservation,
    JointState,
    JointStateObservation,
    ForceTorqueObservation,
    Wrench,
    ContactObservation,
    ContactPoint,
    OdometryObservation,
    Pose,
    Twist,
    RobotObservation,
)

# Action types
from agi.env.actions import (
    Action,
    ActionResult,
    ActionStatus,
    ControlMode,
    JointAction,
    JointCommand,
    CartesianAction,
    CartesianPose,
    CartesianTwist,
    GripperAction,
    GripperCommand,
    MobileBaseAction,
    BaseVelocity,
    BasePose,
    RobotAction,
    SkillAction,
)

__all__ = [
    # Base
    "Environment",
    "AsyncEnvironment",
    "EnvironmentProtocol",
    "EnvironmentSpec",
    "EnvironmentRegistry",
    "Space",
    "SpaceType",
    "StepResult",
    "ResetResult",
    "create_env",
    "register_env",
    # Observations
    "Observation",
    "SensorType",
    "CameraObservation",
    "LidarObservation",
    "PointCloud",
    "ImuObservation",
    "JointState",
    "JointStateObservation",
    "ForceTorqueObservation",
    "Wrench",
    "ContactObservation",
    "ContactPoint",
    "OdometryObservation",
    "Pose",
    "Twist",
    "RobotObservation",
    # Actions
    "Action",
    "ActionResult",
    "ActionStatus",
    "ControlMode",
    "JointAction",
    "JointCommand",
    "CartesianAction",
    "CartesianPose",
    "CartesianTwist",
    "GripperAction",
    "GripperCommand",
    "MobileBaseAction",
    "BaseVelocity",
    "BasePose",
    "RobotAction",
    "SkillAction",
]
