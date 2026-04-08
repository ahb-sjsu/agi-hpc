# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.rh.control.robot_interface module."""

import numpy as np

from agi.rh.control.robot_interface import (
    ROS2BridgeConfig,
    ROS2Bridge,
    URDFLoader,
    HardwareAbstraction,
)


class TestROS2BridgeConfig:
    """Tests for the ROS2BridgeConfig dataclass."""

    def test_defaults(self):
        """Default values match the source definition."""
        cfg = ROS2BridgeConfig()
        assert cfg.node_name == "agi_rh_robot_bridge"
        assert cfg.namespace == ""
        assert cfg.joint_state_topic == "/joint_states"
        assert cfg.num_joints == 7
        assert cfg.command_timeout == 5.0

    def test_custom_values(self):
        """Custom values override defaults."""
        cfg = ROS2BridgeConfig(node_name="my_robot", num_joints=6, namespace="/ns")
        assert cfg.node_name == "my_robot"
        assert cfg.num_joints == 6
        assert cfg.namespace == "/ns"

    def test_all_topic_defaults(self):
        """All topic fields have the expected defaults."""
        cfg = ROS2BridgeConfig()
        assert cfg.joint_command_topic == "/joint_commands"
        assert cfg.cartesian_command_topic == "/cartesian_commands"
        assert cfg.ee_pose_topic == "/end_effector_pose"


class TestROS2Bridge:
    """Tests for the ROS2Bridge class (runs in stub mode without rclpy)."""

    def test_init_defaults(self):
        """Bridge initialises with default config and disconnected state."""
        bridge = ROS2Bridge()
        assert bridge.is_connected is False
        assert bridge.num_joints == 7
        assert bridge._joint_positions.shape == (7,)

    def test_connect_disconnect(self):
        """connect() sets connected state; disconnect() clears it."""
        bridge = ROS2Bridge()
        result = bridge.connect()
        assert result is True
        assert bridge.is_connected is True
        bridge.disconnect()
        assert bridge.is_connected is False

    def test_send_joint_command(self):
        """send_joint_command() stores positions when connected and dims match."""
        bridge = ROS2Bridge()
        bridge.connect()
        cmd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        result = bridge.send_joint_command(cmd)
        assert result is True
        np.testing.assert_allclose(bridge.get_joint_positions(), cmd)

    def test_send_joint_command_wrong_dim(self):
        """send_joint_command() returns False on dimension mismatch."""
        bridge = ROS2Bridge()
        bridge.connect()
        result = bridge.send_joint_command(np.array([1.0, 2.0]))
        assert result is False

    def test_send_joint_command_disconnected(self):
        """send_joint_command() returns False when not connected."""
        bridge = ROS2Bridge()
        result = bridge.send_joint_command(np.zeros(7))
        assert result is False

    def test_send_cartesian_command(self):
        """send_cartesian_command() accepts a 7-element pose vector."""
        bridge = ROS2Bridge()
        bridge.connect()
        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        assert bridge.send_cartesian_command(pose) is True
        np.testing.assert_allclose(bridge.get_end_effector_pose(), pose)

    def test_send_cartesian_command_wrong_dim(self):
        """send_cartesian_command() returns False for non-7-element input."""
        bridge = ROS2Bridge()
        bridge.connect()
        assert bridge.send_cartesian_command(np.zeros(3)) is False


class TestURDFLoader:
    """Tests for the URDFLoader class (stub implementation)."""

    def test_init(self):
        """URDFLoader starts with empty robot name and no joints/links."""
        loader = URDFLoader()
        assert loader.robot_name == ""
        assert loader.joints == []
        assert loader.links == []

    def test_load(self):
        """load() extracts the robot name from the path and returns True."""
        loader = URDFLoader()
        result = loader.load("/robots/panda.urdf")
        assert result is True
        assert loader.robot_name == "panda"

    def test_load_from_string(self):
        """load_from_string() returns True (stub)."""
        loader = URDFLoader()
        assert loader.load_from_string("<robot name='test'/>") is True


class TestHardwareAbstraction:
    """Tests for the HardwareAbstraction facade."""

    def test_init_empty(self):
        """HardwareAbstraction starts with no interfaces and no active."""
        ha = HardwareAbstraction()
        assert ha.list_interfaces() == []
        assert ha.get_active() is None

    def test_register_and_set_active(self):
        """register() adds an interface; set_active() makes it current."""
        ha = HardwareAbstraction()
        bridge = ROS2Bridge()
        ha.register("arm", bridge)
        assert "arm" in ha.list_interfaces()
        assert ha.set_active("arm") is True
        assert ha.get_active() is bridge

    def test_set_active_missing(self):
        """set_active() returns False for an unregistered name."""
        ha = HardwareAbstraction()
        assert ha.set_active("nonexistent") is False

    def test_delegate_connect(self):
        """connect() delegates to the active interface."""
        ha = HardwareAbstraction()
        bridge = ROS2Bridge()
        ha.register("arm", bridge)
        ha.set_active("arm")
        assert ha.connect() is True
        assert bridge.is_connected is True

    def test_no_active_returns_empty(self):
        """get_joint_positions() returns empty array with no active interface."""
        ha = HardwareAbstraction()
        result = ha.get_joint_positions()
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_send_joint_command_no_active(self):
        """send_joint_command() returns False when no active interface is set."""
        ha = HardwareAbstraction()
        assert ha.send_joint_command(np.zeros(7)) is False
