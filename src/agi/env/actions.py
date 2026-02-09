# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Structured action types for environment interface.

Provides standardized action representations for robot control:
- Joint commands (position, velocity, torque)
- Cartesian commands (pose, twist)
- Gripper commands
- Mobile base commands
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class ControlMode(str, Enum):
    """Joint control modes."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"


class ActionStatus(str, Enum):
    """Status of action execution."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Base Action
# ---------------------------------------------------------------------------


@dataclass
class Action:
    """Base action with metadata."""

    action_id: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result of action execution."""

    action_id: str
    status: ActionStatus
    success: bool
    message: str = ""
    duration: float = 0.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Joint Actions
# ---------------------------------------------------------------------------


@dataclass
class JointCommand:
    """Command for a single joint."""

    name: str
    value: float  # Position (rad), velocity (rad/s), or torque (Nm)
    mode: ControlMode = ControlMode.POSITION


@dataclass
class JointAction(Action):
    """Joint-space robot action."""

    commands: List[JointCommand] = field(default_factory=list)
    duration: float = 0.0  # Target duration in seconds (0 = immediate)

    @classmethod
    def from_arrays(
        cls,
        names: List[str],
        values: np.ndarray,
        mode: ControlMode = ControlMode.POSITION,
        **kwargs,
    ) -> "JointAction":
        """Create from joint names and value array."""
        commands = [
            JointCommand(name=name, value=float(value), mode=mode)
            for name, value in zip(names, values, strict=False)
        ]
        return cls(commands=commands, **kwargs)

    @property
    def joint_names(self) -> List[str]:
        return [c.name for c in self.commands]

    @property
    def values(self) -> np.ndarray:
        return np.array([c.value for c in self.commands])

    @property
    def mode(self) -> ControlMode:
        if self.commands:
            return self.commands[0].mode
        return ControlMode.POSITION


# ---------------------------------------------------------------------------
# Cartesian Actions
# ---------------------------------------------------------------------------


@dataclass
class CartesianPose:
    """Cartesian pose target."""

    position: np.ndarray  # (3,) xyz in meters
    orientation: np.ndarray  # (4,) quaternion [x,y,z,w]
    frame_id: str = "world"

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, frame_id: str = "world"
    ) -> "CartesianPose":
        """Create from 4x4 transformation matrix."""
        position = matrix[:3, 3]

        # Extract quaternion from rotation matrix
        R = matrix[:3, :3]
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return cls(
            position=position,
            orientation=np.array([x, y, z, w]),
            frame_id=frame_id,
        )


@dataclass
class CartesianTwist:
    """Cartesian velocity target."""

    linear: np.ndarray  # (3,) m/s
    angular: np.ndarray  # (3,) rad/s
    frame_id: str = "world"

    @classmethod
    def zero(cls, frame_id: str = "world") -> "CartesianTwist":
        return cls(
            linear=np.zeros(3),
            angular=np.zeros(3),
            frame_id=frame_id,
        )


@dataclass
class CartesianAction(Action):
    """Cartesian-space robot action."""

    pose: Optional[CartesianPose] = None
    twist: Optional[CartesianTwist] = None
    end_effector: str = "end_effector"
    duration: float = 0.0

    @property
    def is_pose_command(self) -> bool:
        return self.pose is not None

    @property
    def is_twist_command(self) -> bool:
        return self.twist is not None


# ---------------------------------------------------------------------------
# Gripper Actions
# ---------------------------------------------------------------------------


class GripperCommand(str, Enum):
    """Gripper command types."""

    OPEN = "open"
    CLOSE = "close"
    MOVE = "move"
    GRASP = "grasp"


@dataclass
class GripperAction(Action):
    """Gripper control action."""

    command: GripperCommand = GripperCommand.CLOSE
    position: float = 0.0  # 0 = closed, 1 = open
    speed: float = 1.0  # Normalized speed (0-1)
    force: float = 1.0  # Normalized force (0-1)
    gripper_id: str = "gripper"

    @classmethod
    def open(cls, **kwargs) -> "GripperAction":
        return cls(command=GripperCommand.OPEN, position=1.0, **kwargs)

    @classmethod
    def close(cls, **kwargs) -> "GripperAction":
        return cls(command=GripperCommand.CLOSE, position=0.0, **kwargs)

    @classmethod
    def grasp(cls, force: float = 0.8, **kwargs) -> "GripperAction":
        return cls(command=GripperCommand.GRASP, force=force, **kwargs)


# ---------------------------------------------------------------------------
# Mobile Base Actions
# ---------------------------------------------------------------------------


@dataclass
class BaseVelocity:
    """Mobile base velocity command."""

    linear_x: float = 0.0  # m/s forward
    linear_y: float = 0.0  # m/s left (for holonomic)
    angular_z: float = 0.0  # rad/s counter-clockwise


@dataclass
class BasePose:
    """Mobile base pose target."""

    x: float = 0.0  # meters
    y: float = 0.0  # meters
    theta: float = 0.0  # radians
    frame_id: str = "map"


@dataclass
class MobileBaseAction(Action):
    """Mobile base control action."""

    velocity: Optional[BaseVelocity] = None
    pose: Optional[BasePose] = None
    duration: float = 0.0

    @classmethod
    def move_forward(cls, speed: float = 0.5, **kwargs) -> "MobileBaseAction":
        return cls(
            velocity=BaseVelocity(linear_x=speed),
            **kwargs,
        )

    @classmethod
    def rotate(cls, angular_speed: float = 0.5, **kwargs) -> "MobileBaseAction":
        return cls(
            velocity=BaseVelocity(angular_z=angular_speed),
            **kwargs,
        )

    @classmethod
    def stop(cls, **kwargs) -> "MobileBaseAction":
        return cls(velocity=BaseVelocity(), **kwargs)


# ---------------------------------------------------------------------------
# Composite Actions
# ---------------------------------------------------------------------------


@dataclass
class RobotAction(Action):
    """Complete robot action combining multiple actuator commands."""

    joint_action: Optional[JointAction] = None
    cartesian_action: Optional[CartesianAction] = None
    gripper_action: Optional[GripperAction] = None
    base_action: Optional[MobileBaseAction] = None

    @property
    def has_arm_command(self) -> bool:
        return self.joint_action is not None or self.cartesian_action is not None

    @property
    def has_gripper_command(self) -> bool:
        return self.gripper_action is not None

    @property
    def has_base_command(self) -> bool:
        return self.base_action is not None


# ---------------------------------------------------------------------------
# High-Level Actions (for skills)
# ---------------------------------------------------------------------------


@dataclass
class SkillAction(Action):
    """High-level skill action."""

    skill_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_object: Optional[str] = None
    target_location: Optional[np.ndarray] = None

    @classmethod
    def pick(cls, object_id: str, **kwargs) -> "SkillAction":
        return cls(skill_name="pick", target_object=object_id, **kwargs)

    @classmethod
    def place(cls, location: np.ndarray, **kwargs) -> "SkillAction":
        return cls(skill_name="place", target_location=location, **kwargs)

    @classmethod
    def navigate(cls, location: np.ndarray, **kwargs) -> "SkillAction":
        return cls(skill_name="navigate", target_location=location, **kwargs)
