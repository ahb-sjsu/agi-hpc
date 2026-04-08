# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Robot Hardware Abstraction for AGI-HPC.

Provides a unified interface for interacting with robot hardware:
- RobotInterface protocol for generic robot access
- ROS 2 bridge for ROS-based robots
- URDF loader for robot description parsing (stub)
- HardwareAbstraction facade

Sprint 6 Implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class RobotInterface(Protocol):
    """Protocol defining the interface for robot hardware abstraction."""

    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def get_joint_positions(self) -> np.ndarray: ...
    def get_joint_velocities(self) -> np.ndarray: ...
    def get_end_effector_pose(self) -> np.ndarray: ...
    def send_joint_command(self, positions: np.ndarray) -> bool: ...
    def send_cartesian_command(self, pose: np.ndarray) -> bool: ...
    def stop(self) -> None: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def num_joints(self) -> int: ...


@dataclass
class ROS2BridgeConfig:
    """Configuration for the ROS 2 robot bridge."""

    node_name: str = "agi_rh_robot_bridge"
    namespace: str = ""
    joint_state_topic: str = "/joint_states"
    joint_command_topic: str = "/joint_commands"
    cartesian_command_topic: str = "/cartesian_commands"
    ee_pose_topic: str = "/end_effector_pose"
    num_joints: int = 7
    command_timeout: float = 5.0


class ROS2Bridge:
    """ROS 2 middleware bridge implementing the RobotInterface protocol."""

    def __init__(self, config: Optional[ROS2BridgeConfig] = None) -> None:
        self._config = config or ROS2BridgeConfig()
        self._connected: bool = False
        self._node: Any = None
        self._rclpy_available: bool = False
        self._joint_positions = np.zeros(self._config.num_joints)
        self._joint_velocities = np.zeros(self._config.num_joints)
        self._ee_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        try:
            import rclpy  # noqa: F401

            self._rclpy_available = True
        except ImportError:
            logger.warning("[rh][robot] rclpy not available; ROS2Bridge in stub mode")
        logger.info(
            "[rh][robot] ROS2Bridge initialized node=%s joints=%d rclpy=%s",
            self._config.node_name,
            self._config.num_joints,
            self._rclpy_available,
        )

    def connect(self) -> bool:
        """Establish connection to ROS 2."""
        if self._connected:
            return True
        if self._rclpy_available:
            try:
                import rclpy
                from rclpy.node import Node

                if not rclpy.ok():
                    rclpy.init()
                self._node = Node(
                    self._config.node_name, namespace=self._config.namespace or None
                )
                self._connected = True
                return True
            except Exception as exc:
                logger.error("[rh][robot] failed to initialize ROS 2: %s", exc)
                return False
        self._connected = True
        logger.info("[rh][robot] connected in stub mode")
        return True

    def disconnect(self) -> None:
        """Disconnect from ROS 2."""
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._connected = False
        logger.info("[rh][robot] disconnected")

    def get_joint_positions(self) -> np.ndarray:
        return self._joint_positions.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self._joint_velocities.copy()

    def get_end_effector_pose(self) -> np.ndarray:
        return self._ee_pose.copy()

    def send_joint_command(self, positions: np.ndarray) -> bool:
        """Send a joint position command."""
        if not self._connected:
            return False
        if len(positions) != self._config.num_joints:
            logger.error(
                "[rh][robot] joint command dimension mismatch: got %d, expected %d",
                len(positions),
                self._config.num_joints,
            )
            return False
        self._joint_positions = np.array(positions, dtype=np.float64)
        return True

    def send_cartesian_command(self, pose: np.ndarray) -> bool:
        """Send a Cartesian pose command."""
        if not self._connected:
            return False
        if len(pose) != 7:
            return False
        self._ee_pose = np.array(pose, dtype=np.float64)
        return True

    def stop(self) -> None:
        """Emergency stop."""
        if self._connected:
            self._joint_velocities = np.zeros(self._config.num_joints)
            logger.warning("[rh][robot] emergency stop triggered")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def num_joints(self) -> int:
        return self._config.num_joints


class URDFLoader:
    """URDF robot description loader (stub)."""

    def __init__(self) -> None:
        self._robot_name: str = ""
        self._joints: List[Dict[str, Any]] = []
        self._links: List[Dict[str, Any]] = []
        logger.info("[rh][robot] URDFLoader initialized (stub)")

    def load(self, urdf_path: str) -> bool:
        """Load a URDF file (stub)."""
        self._robot_name = urdf_path.split("/")[-1].replace(".urdf", "")
        return True

    def load_from_string(self, urdf_string: str) -> bool:
        """Load URDF from string (stub)."""
        return True

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def joints(self) -> List[Dict[str, Any]]:
        return list(self._joints)

    @property
    def links(self) -> List[Dict[str, Any]]:
        return list(self._links)


class HardwareAbstraction:
    """Facade over different robot interfaces."""

    def __init__(self) -> None:
        self._interfaces: Dict[str, RobotInterface] = {}
        self._active_name: Optional[str] = None
        self._active: Optional[RobotInterface] = None
        logger.info("[rh][robot] HardwareAbstraction initialized")

    def register(self, name: str, interface: RobotInterface) -> None:
        """Register a robot interface under a given name."""
        self._interfaces[name] = interface

    def set_active(self, name: str) -> bool:
        """Set the active robot interface by name."""
        if name not in self._interfaces:
            return False
        self._active_name = name
        self._active = self._interfaces[name]
        return True

    def get_active(self) -> Optional[RobotInterface]:
        return self._active

    def list_interfaces(self) -> List[str]:
        return list(self._interfaces.keys())

    def connect(self) -> bool:
        if self._active is None:
            return False
        return self._active.connect()

    def disconnect(self) -> None:
        if self._active is not None:
            self._active.disconnect()

    def get_joint_positions(self) -> np.ndarray:
        if self._active is None:
            return np.array([])
        return self._active.get_joint_positions()

    def get_joint_velocities(self) -> np.ndarray:
        if self._active is None:
            return np.array([])
        return self._active.get_joint_velocities()

    def get_end_effector_pose(self) -> np.ndarray:
        if self._active is None:
            return np.array([])
        return self._active.get_end_effector_pose()

    def send_joint_command(self, positions: np.ndarray) -> bool:
        if self._active is None:
            return False
        return self._active.send_joint_command(positions)

    def send_cartesian_command(self, pose: np.ndarray) -> bool:
        if self._active is None:
            return False
        return self._active.send_cartesian_command(pose)

    def stop(self) -> None:
        if self._active is not None:
            self._active.stop()
